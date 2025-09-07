import os
from typing import Optional, Tuple, Dict
import argparse
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
from flash_mla import flash_mla_with_kvcache, get_mla_metadata


def pad_to_max_len(tensor: torch.Tensor, max_len: int, dim: int = 0) -> torch.Tensor:
    pad_len = max_len - tensor.shape[dim]
    if pad_len <= 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_len
    pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=dim)


def dummy_merge(results_list: list[torch.Tensor]) -> torch.Tensor:
    """参考示例代码的合并逻辑：对每个SP请求的所有Rank结果取平均"""
    if not results_list:
        raise ValueError("合并结果列表不能为空")
    return torch.stack(results_list).mean(dim=0)


def flash_mla_fwd_sp(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    sp_groups_info: Optional[Dict] = None,
    sp_comm_groups: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rank = dist.get_rank()
    device = q.device
    sp_groups_info = sp_groups_info or {}
    sp_comm_groups = sp_comm_groups or {}
    batch_size = q.shape[0]

    print(f"\n========== [RANK {rank}] FLASH MLA SP START ==========", flush=True)

    # 1) 拆分本地/SP请求
    local_batches = [b for b in range(batch_size) if not sp_groups_info.get(b, {}).get("enabled", False)]
    sp_batches = [(b, info) for b, info in sp_groups_info.items() if info.get("enabled", False) and rank in info["group"]]
    print(f"[RANK {rank}] local_batches = {local_batches}", flush=True)
    print(f"[RANK {rank}] sp_batches = {[tpl[0] for tpl in sp_batches]}", flush=True)

    # 2) 从sp_comm_groups获取SP组信息（关键修改）
    # 确保即使当前rank没有SP请求，也能获取到需要参与的通信组
    sp_group_key = next(iter(sp_comm_groups.keys()), None)
    print(f"[RANK {rank}] 从sp_comm_groups获取的SP组key = {sp_group_key}", flush=True)

    # 3) 处理单个SP组的All-Gather
    sp_q_single = torch.empty(0, q.shape[1], q.shape[2], dtype=q.dtype, device=device)
    meta = {}  # 仅存单个组的元信息
    
    # 即使没有sp_batches，只要有sp_group_key就需要参与通信
    if sp_group_key:
        # 获取通信组
        comm = sp_comm_groups[sp_group_key]
        group_ranks = list(sp_group_key)
        local_cnt = len(sp_batches)  # 当前rank的SP请求数量
        
        # 同步组内请求数 + 填充Q
        local_cnt_tensor = torch.tensor([local_cnt], dtype=torch.long, device=device)
        cnt_list_tensor = [torch.empty_like(local_cnt_tensor) for _ in group_ranks]
        dist.all_gather(cnt_list_tensor, local_cnt_tensor, group=comm)
        cnt_list = [t.item() for t in cnt_list_tensor]
        max_cnt = max(cnt_list) if cnt_list else 0
        print(f"[RANK {rank}] SP组 {sp_group_key} - 组内请求数: {cnt_list}, 最大长度: {max_cnt}", flush=True)

        # 构建本地Q并All-Gather（如果有SP请求）
        local_q = torch.stack([q[b] for b, _ in sp_batches]) if local_cnt > 0 else torch.empty(0, q.shape[1], q.shape[2], dtype=q.dtype, device=device)
        padded_q = pad_to_max_len(local_q, max_cnt, dim=0)
        gathered = torch.empty(len(group_ranks), max_cnt, q.shape[1], q.shape[2], dtype=padded_q.dtype, device=device)
        dist.all_gather_into_tensor(gathered.view(-1), padded_q.contiguous(), group=comm)

        # 扁平化SP请求Q
        flat_q = torch.cat([gathered[i, :c] for i, c in enumerate(cnt_list) if c > 0], dim=0) if cnt_list else sp_q_single
        sp_q_single = flat_q
        meta[sp_group_key] = {"group_ranks": group_ranks, "local_cnt": local_cnt, "all_cnts": cnt_list}

    # 4) 合并本地Q与单个SP组Q
    local_q = q[local_batches] if local_batches else torch.empty(0, q.shape[1], q.shape[2], dtype=q.dtype, device=device)
    all_q = torch.cat([local_q, sp_q_single], dim=0)
    print(f"[RANK {rank}] local_q={local_q.shape}, sp_q={sp_q_single.shape}, all_q={all_q.shape}", flush=True)

    # 5) 调用Flash MLA
    all_q_expanded = all_q.unsqueeze(1)
    attn_output, lse = flash_mla_with_kvcache(
        all_q_expanded, k_cache, block_table, cache_seqlens, head_dim_v,
        tile_scheduler_metadata, num_splits, softmax_scale, causal
    )
    print(f"[RANK {rank}] attn_output shape: {attn_output.shape}, lse shape: {lse.shape}", flush=True)

    # 6) 拆分本地/SP结果（仅单个SP组）
    local_cnt = local_q.shape[0]
    local_res, sp_res = attn_output[:local_cnt], attn_output[local_cnt:]
    local_lse, sp_lse = lse[:local_cnt], lse[local_cnt:]
    print(f"[RANK {rank}] local_res={local_res.shape}, sp_res={sp_res.shape}", flush=True)

    # 7) 处理单个SP组的All2All与结果合并
    final_sp = torch.empty(0, *attn_output.shape[1:], dtype=attn_output.dtype, device=device)
    final_sp_lse = torch.empty(0, *lse.shape[1:], dtype=lse.dtype, device=device)
    
    # 即使没有sp_batches，只要有sp_group_key就需要参与通信
    if sp_group_key:
        comm = sp_comm_groups[sp_group_key]
        m = meta.get(sp_group_key, {})
        group_ranks = m.get("group_ranks", list(sp_group_key))
        local_cnt = m.get("local_cnt", 0)
        all_cnts = m.get("all_cnts", [0]*len(group_ranks))
        total_sp = sum(all_cnts)

        # 提取SP组切片并拼接out+lse
        slice_sp = sp_res[:total_sp] if total_sp > 0 else torch.empty_like(final_sp)
        slice_sp_lse = sp_lse[:total_sp] if total_sp > 0 else torch.empty_like(final_sp_lse)
        slice_sp_lse_expanded = slice_sp_lse.unsqueeze(1).to(slice_sp.dtype)
        combined_sp = torch.cat([slice_sp, slice_sp_lse_expanded], dim=-1)
        print(f"[RANK {rank}] SP组 {sp_group_key} - combined_sp shape: {combined_sp.shape}", flush=True)

        # 构造发送列表
        send_list = []
        pos = 0
        for c in all_cnts:
            if c > 0:
                send_tensor = combined_sp[pos:pos + c].contiguous()
            else:
                send_tensor = torch.empty(0, *combined_sp.shape[1:], dtype=combined_sp.dtype, device=device)
            send_list.append(send_tensor)
            pos += c

        # 构造接收列表
        recv_list = [torch.empty(local_cnt, *combined_sp.shape[1:], dtype=combined_sp.dtype, device=device) if local_cnt > 0 
                     else torch.empty(0, *combined_sp.shape[1:], dtype=combined_sp.dtype, device=device) 
                     for _ in group_ranks]

        # 单次All2All通信（即使没有数据也要参与）
        dist.all_to_all(recv_list, send_list, group=comm)

        # 合并SP结果（仅处理本地有请求的情况）
        if local_cnt > 0:
            combined_received = torch.cat([t for t in recv_list if t.numel() > 0], dim=0)
            split_idx = attn_output.shape[-1]
            sp_res_recv = combined_received[..., :split_idx]
            sp_lse_recv = combined_received[..., split_idx:].squeeze(1)

            # 按请求合并结果
            req_results = [[sp_res_recv[i + j * local_cnt] for j in range(len(group_ranks))] for i in range(local_cnt)]
            req_results_lse = [[sp_lse_recv[i + j * local_cnt] for j in range(len(group_ranks))] for i in range(local_cnt)]
            final_sp = torch.stack([dummy_merge(req) for req in req_results], dim=0)
            final_sp_lse = torch.stack([dummy_merge(req) for req in req_results_lse], dim=0)
        print(f"[RANK {rank}] SP组 {sp_group_key} - 合并后: out={final_sp.shape}, lse={final_sp_lse.shape}", flush=True)

    # 8) 最终结果合并
    final_out = torch.cat([local_res, final_sp], dim=0)
    final_lse = torch.cat([local_lse, final_sp_lse], dim=0)
    print(f"[RANK {rank}] 最终结果: final_out={final_out.shape}, final_lse={final_lse.shape}", flush=True)
    print(f"========== [RANK {rank}] FLASH MLA SP DONE ==========\n", flush=True)
    
    return final_out, final_lse


# -------------------------- 子进程测试函数 --------------------------
def test_flash_mla(rank: int, world_size: int, args: argparse.Namespace):
    # 分布式初始化
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    print(f"[RANK {rank}] 分布式初始化完成（world_size={world_size}）", flush=True)

    # 全局配置
    dtype = torch.bfloat16
    num_query_heads = 32
    head_size = 576
    head_dim_v = 512
    softmax_scale = head_size ** -0.5
    block_size = 64
    NUM_BLOCKS_PER_RANK = 128 * 12 + 1
    NUM_BLOCKS = NUM_BLOCKS_PER_RANK * world_size
    num_kv_heads = 1

    # 按rank划分的配置（不修改输入）
    kv_lens_per_rank = [
        [2048, 2048, 2048, 2048, 2048, 2048, 2048],
        [2048, 2048, 2048, 2048, 2048, 2048, 2048],
        [2048, 2048, 2048, 2048, 2048, 2048, 2048],
        [2048, 2048, 2048, 2048, 2048, 2048, 2048]
    ]
    sp_groups_info_list = [
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': True, 'group': [0,1,2,3]}, 2: {'enabled': False}},
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': True, 'group': [0,1,2,3]}, 2: {'enabled': False}},
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': True, 'group': [0,1,2,3]}, 2: {'enabled': False}},
        {0: {'enabled': False}}
    ]

    # 配置验证
    assert len(kv_lens_per_rank) == world_size, f"kv_lens_per_rank长度需等于world_size（{world_size}）"
    assert len(sp_groups_info_list) == world_size, f"sp_groups_info_list长度需等于world_size（{world_size}）"

    # 构造输入数据
    current_sp_info = sp_groups_info_list[rank]
    q_batch_size = len(current_sp_info)
    query = torch.randn(q_batch_size, num_query_heads, head_size, dtype=dtype, device=device) / 10
    key_value_cache = torch.randn(NUM_BLOCKS, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
    kv_lens_this_rank = kv_lens_per_rank[rank]
    kv_batch_size = len(kv_lens_this_rank)
    max_num_blocks_per_seq = NUM_BLOCKS_PER_RANK
    block_tables = torch.randint(0, NUM_BLOCKS_PER_RANK, (kv_batch_size, max_num_blocks_per_seq), dtype=torch.int32, device=device) if kv_batch_size > 0 else torch.empty(0, max_num_blocks_per_seq, dtype=torch.int32, device=device)
    kv_lens_tensor = torch.tensor(kv_lens_this_rank, dtype=torch.int32, device=device) if kv_batch_size > 0 else torch.empty(0, dtype=torch.int32, device=device)
    print(f"[RANK {rank}] query.shape: {query.shape}, block_tables.shape: {block_tables.shape}", flush=True)

    # 计算MLA元数据 + 创建SP通信组
    tile_scheduler_metadata, num_splits = get_mla_metadata(kv_lens_tensor, num_query_heads // num_kv_heads, num_kv_heads)
    sp_comm_groups = {tuple(sorted([0,1,2,3])): dist.new_group([0,1,2,3])}

    # 调用SP注意力函数 + 结果验证
    out, lse = flash_mla_fwd_sp(
        query, key_value_cache, block_tables, kv_lens_tensor, head_dim_v,
        tile_scheduler_metadata, num_splits, softmax_scale, causal=True,
        sp_groups_info=current_sp_info, sp_comm_groups=sp_comm_groups
    )
    assert len(out.shape) == 4 and len(lse.shape) == 3, "结果维度异常"
    assert out.shape[0] == q_batch_size and lse.shape[0] == q_batch_size, "batch_size不匹配"
    assert out.dtype == dtype, "dtype不匹配"
    print(f"[RANK {rank}] 结果验证通过！out.shape={out.shape}, lse.shape={lse.shape}", flush=True)

    # 销毁进程组
    dist.barrier()
    dist.destroy_process_group()


# -------------------------- 主函数 --------------------------
def main():
    parser = argparse.ArgumentParser(description="Flash MLA SP 分布式测试（spawn自动分配rank）")
    parser.add_argument("--num_ranks", type=int, default=4, help="进程总数（world_size）")
    parser.add_argument("--master_addr", type=str, default="localhost", help="分布式主节点地址")
    parser.add_argument("--master_port", type=str, default="12355", help="分布式主节点端口")
    args = parser.parse_args()

    assert args.num_ranks == 4, "当前配置仅支持num_ranks=4"
    print(f"启动测试：num_ranks={args.num_ranks}, master={args.master_addr}:{args.master_port}", flush=True)
    spawn(fn=test_flash_mla, args=(args.num_ranks, args), nprocs=args.num_ranks, join=True)
    print("所有进程执行完成！", flush=True)


if __name__ == "__main__":
    main()