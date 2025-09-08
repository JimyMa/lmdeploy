import os
from typing import Optional, Tuple, Dict, List, Any
import argparse
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

# 全局开关：控制是否打印shape日志
PRINT_SHAPE_LOG = True

def print_shape_log(rank: int, content: str) -> None:
    """统一的shape日志打印函数"""
    if PRINT_SHAPE_LOG:
        print(f"[RANK {rank}] {content}", flush=True)


class BufferManager:
    def __init__(self, num_q_heads, head_size, lse_size, max_sp_size, max_local_sp_num, head_dim_v):
        self.max_sp_size = max_sp_size
        self.max_local_sp_num = max_local_sp_num
        self.num_q_heads = num_q_heads
        self.head_size = head_size
        self.lse_size = lse_size
        self.head_dim_v = head_dim_v
        
        # Q相关缓冲区
        self.q_gathered_in_buffer = torch.empty(
            max_local_sp_num, num_q_heads, head_size, 
            dtype=torch.bfloat16, device="cuda"
        )
        self.q_gathered_out_buffer = torch.empty(
            max_sp_size, max_local_sp_num, num_q_heads, head_size, 
            dtype=torch.bfloat16, device="cuda"
        )
        
        # 结果相关缓冲区（拼接out和lse）
        self.res_gathered_in_buffer = torch.empty(
            max_local_sp_num * max_sp_size, 1, num_q_heads, head_dim_v + lse_size, 
            dtype=torch.bfloat16, device="cuda"
        )
        self.res_gathered_out_buffer = [
            torch.empty(
                max_local_sp_num * max_sp_size, 1, num_q_heads, head_dim_v + lse_size, 
                dtype=torch.bfloat16, device="cuda"
            ) for _ in range(max_sp_size)
        ]

    def log_buffer_shapes(self, rank: int) -> None:
        """打印缓冲区形状日志"""
        print_shape_log(rank, f"BufferManager 缓冲区:")
        print_shape_log(rank, f"  q_gathered_in: {self.q_gathered_in_buffer.shape}")
        print_shape_log(rank, f"  q_gathered_out: {self.q_gathered_out_buffer.shape}")
        print_shape_log(rank, f"  res_gathered_in: {self.res_gathered_in_buffer.shape}")
        print_shape_log(rank, f"  res_gathered_out[0]: {self.res_gathered_out_buffer[0].shape}")


def dummy_merge(results_list: list[torch.Tensor]) -> torch.Tensor:
    """对SP请求结果取平均"""
    return torch.stack(results_list).mean(dim=0)


def prepare_mla_fwd(
    rank: int,
    q_batch_size: int,
    sp_groups_info_list: List[Dict],
    sp_comm_groups: Dict,
) -> Tuple[List[int], List[int], Any, List[int], int]:
    """预处理：筛选本地/SP请求并统计计数"""
    current_sp_info = sp_groups_info_list[rank]
    
    # 筛选本地和SP请求索引
    local_batches = [b for b in range(q_batch_size) if not current_sp_info.get(b, {}).get("enabled", False)]
    sp_batch_indices = [b for b, info in current_sp_info.items() if info.get("enabled", False) and rank in info["group"]]
    local_cnt = len(sp_batch_indices)
    
    # 日志输出
    print_shape_log(rank, f"prepare_mla_fwd - 本地请求: {len(local_batches)}, SP请求: {local_cnt}")
    print_shape_log(rank, f"  local_batches: {local_batches}, sp_batch_indices: {sp_batch_indices}")
    
    # 获取SP组信息和计数
    sp_group_key = next(iter(sp_comm_groups.keys()), None)
    cnt_list = []
    if sp_group_key:
        cnt_list = [sum(1 for b, info in sp_groups_info_list[r].items() 
                      if info.get("enabled", False) and r in info["group"]) 
                   for r in sp_group_key]
        print_shape_log(rank, f"  SP组: {sp_group_key}, 各组请求数: {cnt_list}")
    
    return local_batches, sp_batch_indices, sp_group_key, cnt_list, local_cnt


def all_gather_sp_q(
    buffer_manager: BufferManager,
    q: torch.Tensor,
    sp_batch_indices: List[int],
    sp_group_key: Any,
    cnt_list: List[int],
    local_cnt: int,
    sp_comm_groups: Dict
) -> torch.Tensor:
    """All-Gather收集SP请求的Q"""
    rank = dist.get_rank()
    device = q.device

    print_shape_log(rank, f"all_gather_sp_q - 输入Q: {q.shape}")
    if not sp_group_key or local_cnt == 0:
        empty_q = torch.empty(0, q.shape[1], q.shape[2], dtype=q.dtype, device=device)
        print_shape_log(rank, f"  无SP请求，返回空Q: {empty_q.shape}")
        return empty_q

    comm = sp_comm_groups[sp_group_key]
    max_cnt = max(cnt_list)

    # 1. 把本地数据拷进 in-buffer
    if local_cnt > 0:
        local_q = torch.stack([q[b] for b in sp_batch_indices])
        print_shape_log(rank, f"  本地SP Q: {local_q.shape}")
        buffer_manager.q_gathered_in_buffer[:local_cnt].copy_(local_q)

    # 2. 通信：始终用整个 out-buffer
    # all_gather_into_tensor 要求发送/接收张量连续且大小匹配
    dist.all_gather_into_tensor(
        buffer_manager.q_gathered_out_buffer.contiguous().view(-1),
        buffer_manager.q_gathered_in_buffer.contiguous(),
        group=comm
    )
    print_shape_log(rank, f"  All-Gather后 out-buffer shape: {buffer_manager.q_gathered_out_buffer.shape}")

    # 3. 用视图取出有效部分
    gathered_view = buffer_manager.q_gathered_out_buffer[:len(sp_group_key), :max_cnt]
    flat_q = torch.cat([gathered_view[i, :c] for i, c in enumerate(cnt_list) if c > 0], dim=0)
    print_shape_log(rank, f"  扁平化SP Q: {flat_q.shape}")
    return flat_q


def all_gather_sp_results(
    buffer_manager: BufferManager,
    sp_res: torch.Tensor,
    sp_lse: torch.Tensor,
    sp_group_key: Any,
    cnt_list: List[int],
    local_cnt: int,
    sp_comm_groups: Dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """All-Gather收集并合并SP结果"""
    rank = dist.get_rank()
    device = sp_res.device
    
    print_shape_log(rank, f"all_gather_sp_results - 输入: res={sp_res.shape}, lse={sp_lse.shape}")
    if not sp_group_key or local_cnt == 0:
        empty_res = torch.empty(0, *sp_res.shape[1:], device=device, dtype=sp_res.dtype)
        empty_lse = torch.empty(0, *sp_lse.shape[1:], device=device, dtype=sp_lse.dtype)
        print_shape_log(rank, f"  无SP请求，返回空结果: {empty_res.shape}, {empty_lse.shape}")
        return empty_res, empty_lse
    
    # 准备发送数据（拼接res和lse）
    comm = sp_comm_groups[sp_group_key]
    total_sp = sum(cnt_list)
    slice_sp = sp_res[:total_sp]
    slice_sp_lse_expanded = sp_lse[:total_sp].unsqueeze(1).to(slice_sp.dtype)
    combined_sp = torch.cat([slice_sp, slice_sp_lse_expanded], dim=-1)
    print_shape_log(rank, f"  拼接后数据: {combined_sp.shape}")
    
    # 填充到缓冲区并All-Gather
    if total_sp > 0:
        buffer_manager.res_gathered_in_buffer[:total_sp].copy_(combined_sp)
    dist.all_gather(buffer_manager.res_gathered_out_buffer, 
                   buffer_manager.res_gathered_in_buffer.contiguous(), 
                   group=comm)
    
    # 提取并合并当前rank负责的结果
    rank_idx = sp_group_key.index(rank)
    start_idx, end_idx = sum(cnt_list[:rank_idx]), sum(cnt_list[:rank_idx+1])
    split_idx = buffer_manager.head_dim_v
    req_results, req_results_lse = [], []
    
    for i, r in enumerate(sp_group_key):
        rank_data = buffer_manager.res_gathered_out_buffer[i]
        for global_idx in range(start_idx, end_idx):
            combined_result = rank_data[global_idx]
            out_result = combined_result[..., :split_idx]
            lse_result = combined_result[..., split_idx:].squeeze(0)
            
            local_idx = global_idx - start_idx
            if i == 0:
                req_results.append([out_result])
                req_results_lse.append([lse_result])
            else:
                req_results[local_idx].append(out_result)
                req_results_lse[local_idx].append(lse_result)
    
    # 合并结果
    final_sp = torch.stack([dummy_merge(req) for req in req_results], dim=0) if req_results else \
               torch.empty(0, *sp_res.shape[1:], device=device, dtype=sp_res.dtype)
    final_sp_lse = torch.stack([dummy_merge(req) for req in req_results_lse], dim=0) if req_results_lse else \
                   torch.empty(0, *sp_lse.shape[1:], device=device, dtype=sp_lse.dtype)
    
    print_shape_log(rank, f"  合并后: res={final_sp.shape}, lse={final_sp_lse.shape}")
    return final_sp, final_sp_lse


def flash_mla_fwd_sp(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    local_batches: List[int],
    sp_batch_indices: List[int],
    sp_group_key: Any,
    cnt_list: List[int],
    local_cnt: int,
    sp_comm_groups: Dict,
    buffer_manager: BufferManager,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SP模式下的Flash MLA前向计算"""
    rank = dist.get_rank()
    device = q.device
    print_shape_log(rank, f"\n===== flash_mla_fwd_sp 开始 =====")
    print_shape_log(rank, f"输入Q: {q.shape}, k_cache: {k_cache.shape}")

    # 收集SP Q并合并
    sp_q_single = all_gather_sp_q(buffer_manager, q, sp_batch_indices, sp_group_key, cnt_list, local_cnt, sp_comm_groups)
    local_q = q[local_batches] if local_batches else torch.empty(0, q.shape[1], q.shape[2], dtype=q.dtype, device=device)
    all_q = torch.cat([local_q, sp_q_single], dim=0)
    print_shape_log(rank, f"合并后Q: {all_q.shape}")

    # 调用MLA计算
    attn_output, lse = flash_mla_with_kvcache(
        all_q.unsqueeze(1), k_cache, block_table, cache_seqlens, head_dim_v,
        tile_scheduler_metadata, num_splits, softmax_scale, causal
    )
    print_shape_log(rank, f"MLA输出: attn={attn_output.shape}, lse={lse.shape}")

    # 拆分并合并结果
    local_cnt_total = local_q.shape[0]
    local_res, sp_res = attn_output[:local_cnt_total], attn_output[local_cnt_total:]
    local_lse, sp_lse = lse[:local_cnt_total], lse[local_cnt_total:]
    
    final_sp, final_sp_lse = all_gather_sp_results(
        buffer_manager, sp_res, sp_lse, sp_group_key, cnt_list, local_cnt, sp_comm_groups
    )

    # 最终合并
    final_out = torch.cat([local_res, final_sp], dim=0)
    final_lse = torch.cat([local_lse, final_sp_lse], dim=0)
    
    print_shape_log(rank, f"最终结果: out={final_out.shape}, lse={final_lse.shape}")
    print_shape_log(rank, f"===== flash_mla_fwd_sp 结束 =====\n")
    return final_out, final_lse


def test_flash_mla(rank: int, world_size: int, args: argparse.Namespace):
    """测试进程函数"""
    # 初始化分布式
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    print_shape_log(rank, f"分布式初始化: rank={rank}, world_size={world_size}")

    # 配置参数
    dtype = torch.bfloat16
    num_query_heads, head_size, head_dim_v = 32, 576, 512
    block_size = 64
    NUM_BLOCKS_PER_RANK = 128 * 12 + 1
    NUM_BLOCKS = NUM_BLOCKS_PER_RANK * world_size
    print_shape_log(rank, f"配置: 头数={num_query_heads}, 头尺寸={head_size}, 设备={device}")

    # 数据配置
    kv_lens_per_rank = [[2048]*8 for _ in range(world_size)]
    sp_groups_info_list = [
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': True, 'group': [0,1,2,3]}, 2: {'enabled': False}},
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': True, 'group': [0,1,2,3]}, 2: {'enabled': False}},
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': True, 'group': [0,1,2,3]}, 2: {'enabled': False}},
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': False}},
    ]
    current_sp_info = sp_groups_info_list[rank]
    q_batch_size = len(current_sp_info)

    # 初始化缓冲区
    buffer_manager = BufferManager(
        num_q_heads=num_query_heads,
        head_size=head_size,
        lse_size=1,
        max_sp_size=4,
        max_local_sp_num=2,
        head_dim_v=head_dim_v
    )
    buffer_manager.log_buffer_shapes(rank)

    # 构造输入数据
    query = torch.randn(q_batch_size, num_query_heads, head_size, dtype=dtype, device=device) / 10
    key_value_cache = torch.randn(NUM_BLOCKS, block_size, 1, head_size, dtype=dtype, device=device)
    kv_lens_this_rank = kv_lens_per_rank[rank]
    kv_batch_size = len(kv_lens_this_rank)
    block_tables = torch.randint(0, NUM_BLOCKS_PER_RANK, 
                                (kv_batch_size, NUM_BLOCKS_PER_RANK), 
                                dtype=torch.int32, device=device) if kv_batch_size > 0 else \
                   torch.empty(0, NUM_BLOCKS_PER_RANK, dtype=torch.int32, device=device)
    kv_lens_tensor = torch.tensor(kv_lens_this_rank, dtype=torch.int32, device=device) if kv_batch_size > 0 else \
                     torch.empty(0, dtype=torch.int32, device=device)

    # 元数据与通信组
    tile_scheduler_metadata, num_splits = get_mla_metadata(kv_lens_tensor, num_query_heads, 1)
    sp_comm_groups = {tuple(sorted([0,1,2,3])): dist.new_group([0,1,2,3])}

    # 预处理
    local_batches, sp_batch_indices, sp_group_key, cnt_list, local_cnt = prepare_mla_fwd(
        rank=rank,
        q_batch_size=q_batch_size,
        sp_groups_info_list=sp_groups_info_list,
        sp_comm_groups=sp_comm_groups
    )
    print_shape_log(rank, f"cnt_list: {cnt_list}")

    # 执行前向计算
    out, lse = flash_mla_fwd_sp(
        q=query,
        k_cache=key_value_cache,
        block_table=block_tables,
        cache_seqlens=kv_lens_tensor,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        local_batches=local_batches,
        sp_batch_indices=sp_batch_indices,
        sp_group_key=sp_group_key,
        cnt_list=cnt_list,
        local_cnt=local_cnt,
        sp_comm_groups=sp_comm_groups,
        buffer_manager=buffer_manager,
        softmax_scale=head_size ** -0.5,
        causal=True
    )

    # 验证结果
    assert len(out.shape) == 4 and len(lse.shape) == 3, "维度异常"
    assert out.shape[0] == q_batch_size and lse.shape[0] == q_batch_size, "batch不匹配"
    assert out.dtype == dtype, "dtype不匹配"
    print_shape_log(rank, "结果验证通过")

    # 清理
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="精简版Flash MLA SP测试")
    parser.add_argument("--num_ranks", type=int, default=4)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="12355")
    args = parser.parse_args()

    assert args.num_ranks == 4, "仅支持4个进程"
    print(f"启动测试: {args.num_ranks}进程, 地址={args.master_addr}:{args.master_port}")
    spawn(fn=test_flash_mla, args=(args.num_ranks, args), nprocs=args.num_ranks, join=True)
    print("所有进程完成")


if __name__ == "__main__":
    main()