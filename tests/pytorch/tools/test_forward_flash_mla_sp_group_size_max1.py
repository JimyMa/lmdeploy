import os
from typing import Optional, Tuple, Dict, List, Any
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
    """对每个SP请求的所有Rank结果取平均"""
    if not results_list:
        raise ValueError("合并结果列表不能为空")
    return torch.stack(results_list).mean(dim=0)


def prepare_mla_fwd(
    rank: int,
    world_size: int,
    q_batch_size: int,
    sp_groups_info_list: List[Dict],
    sp_comm_groups: Optional[Dict],
) -> Tuple[List[int], List[int], Any, List[int], int, List[int]]:
    sp_comm_groups = sp_comm_groups or {}
    current_sp_info = sp_groups_info_list[rank]
    
    local_batches = [b for b in range(q_batch_size) if not current_sp_info.get(b, {}).get("enabled", False)]
    sp_batch_indices = [b for b, info in current_sp_info.items() if info.get("enabled", False) and rank in info["group"]]
    local_cnt = len(sp_batch_indices)
    
    sp_group_key = next(iter(sp_comm_groups.keys()), None)
    cnt_list = []
    recv_cnts = []
    
    if sp_group_key:
        group_ranks = list(sp_group_key)
        cnt_list = []
        for r in group_ranks:
            sp_info = sp_groups_info_list[r]
            sp_count = sum(1 for b, info in sp_info.items() 
                          if info.get("enabled", False) and r in info["group"])
            cnt_list.append(sp_count)
        recv_cnts = [local_cnt for _ in group_ranks]  # 与原all2all保持一致
    
    print(f"\n[RANK {rank}] prepare_mla_fwd 预处理结果:")
    print(f"  1. local_batches: {local_batches}")
    print(f"  2. sp_batch_indices: {sp_batch_indices}")
    print(f"  3. sp_group_key: {sp_group_key}")
    print(f"  4. cnt_list: {cnt_list} (对应rank: {list(sp_group_key) if sp_group_key else []})")
    print(f"  5. local_cnt: {local_cnt}")
    print(f"  6. recv_cnts: {recv_cnts}")
    print(f"[RANK {rank}] 预处理完成\n")
    
    return local_batches, sp_batch_indices, sp_group_key, cnt_list, local_cnt, recv_cnts


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
    recv_cnts: List[int],
    sp_comm_groups: Optional[Dict],
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rank = dist.get_rank()
    device = q.device
    sp_comm_groups = sp_comm_groups or {}
    batch_size = q.shape[0]

    print(f"\n========== [RANK {rank}] FLASH MLA SP START ==========", flush=True)

    # SP组Q的All-Gather（保持不变）
    sp_q_single = torch.empty(0, q.shape[1], q.shape[2], dtype=q.dtype, device=device)
    group_ranks = []
    
    if sp_group_key:
        comm = sp_comm_groups[sp_group_key]
        group_ranks = list(sp_group_key)
        max_cnt = max(cnt_list) if cnt_list else 0
        print(f"[RANK {rank}] SP组 {sp_group_key} - 组内请求数: {cnt_list}, 最大长度: {max_cnt}", flush=True)

        local_q = torch.stack([q[b] for b in sp_batch_indices]) if local_cnt > 0 else torch.empty(0, q.shape[1], q.shape[2], dtype=q.dtype, device=device)
        padded_q = pad_to_max_len(local_q, max_cnt, dim=0)
        gathered = torch.empty(len(group_ranks), max_cnt, q.shape[1], q.shape[2], dtype=padded_q.dtype, device=device)
        dist.all_gather_into_tensor(gathered.view(-1), padded_q.contiguous(), group=comm)

        flat_q = torch.cat([gathered[i, :c] for i, c in enumerate(cnt_list) if c > 0], dim=0) if cnt_list else sp_q_single
        sp_q_single = flat_q

    # 合并本地Q与SP组Q（保持不变）
    local_q = q[local_batches] if local_batches else torch.empty(0, q.shape[1], q.shape[2], dtype=q.dtype, device=device)
    all_q = torch.cat([local_q, sp_q_single], dim=0)
    print(f"[RANK {rank}] local_q={local_q.shape}, sp_q={sp_q_single.shape}, all_q={all_q.shape}", flush=True)

    # 调用Flash MLA（保持不变）
    all_q_expanded = all_q.unsqueeze(1)
    attn_output, lse = flash_mla_with_kvcache(
        all_q_expanded, k_cache, block_table, cache_seqlens, head_dim_v,
        tile_scheduler_metadata, num_splits, softmax_scale, causal
    )
    print(f"[RANK {rank}] attn_output shape: {attn_output.shape}, lse shape: {lse.shape}", flush=True)

    # 拆分本地/SP结果（保持不变）
    local_cnt_total = local_q.shape[0]
    local_res, sp_res = attn_output[:local_cnt_total], attn_output[local_cnt_total:]
    local_lse, sp_lse = lse[:local_cnt_total], lse[local_cnt_total:]
    print(f"[RANK {rank}] local_res={local_res.shape}, sp_res={sp_res.shape}", flush=True)
    print(f"[RANK {rank}] local_lse={local_lse.shape}, sp_lse={sp_lse.shape}", flush=True)

    # 核心修改：按照参考代码在最后一维拼接out和lse
    final_sp = torch.empty(0, *attn_output.shape[1:], dtype=attn_output.dtype, device=device)
    final_sp_lse = torch.empty(0, *lse.shape[1:], dtype=lse.dtype, device=device)
    
    if sp_group_key:
        comm = sp_comm_groups[sp_group_key]
        total_sp = sum(cnt_list)
        group_ranks = list(sp_group_key)
        group_size = len(group_ranks)
        key = sp_group_key

        # 1. 准备当前rank的SP结果：在最后一维拼接out和lse（与参考代码一致）
        slice_sp = sp_res[:total_sp] if total_sp > 0 else torch.empty(0, *attn_output.shape[1:], dtype=attn_output.dtype, device=device)
        
        # 将lse扩展维度以匹配out的维度（与参考代码一致）
        if total_sp > 0:
            slice_sp_lse_expanded = sp_lse[:total_sp].unsqueeze(1)  # 从 (N, H, 1) 变为 (N, 1, H, 1)
        else:
            slice_sp_lse_expanded = torch.empty(0, 1, lse.shape[1], lse.shape[2], dtype=lse.dtype, device=device)
        
        # 统一dtype并在最后一维拼接（与参考代码一致）
        slice_sp_lse_expanded = slice_sp_lse_expanded.to(dtype=slice_sp.dtype)
        combined_sp = torch.cat([slice_sp, slice_sp_lse_expanded], dim=-1)
        print(f"[RANK {rank}] SP组 {key} - 拼接后combined_sp shape: {combined_sp.shape}, dtype: {combined_sp.dtype}", flush=True)

        # 2. 计算组内最大batch数，用于padding（all gather要求所有张量形状一致）
        max_batch = max(cnt_list) if cnt_list else 0
        padded_combined = pad_to_max_len(combined_sp, max_batch, dim=0)
        print(f"[RANK {rank}] SP组 {key} - 填充后combined_sp shape: {padded_combined.shape}", flush=True)

        # 3. 执行all gather
        gathered_combined = [torch.empty_like(padded_combined) for _ in range(group_size)]
        print(f"[RANK {rank}] SP组 {key} - 开始执行all gather通信（组内{group_size}个rank）...", flush=True)
        dist.all_gather(gathered_combined, padded_combined.contiguous(), group=comm)
        print(f"[RANK {rank}] SP组 {key} - all gather通信完成！", flush=True)

        # 4. 同步屏障
        dist.barrier(group=comm)
        print(f"[RANK {rank}] SP组 {key} - 所有rank均已完成通信", flush=True)

        # 5. 解析收集到的结果（与参考代码逻辑对应）
        parsed_out = []
        parsed_lse = []
        print(f"[RANK {rank}] SP组 {key} - 解析all gather结果:", flush=True)
        
        # 计算拆分维度（最后一维）
        split_dim = -1
        split_idx = slice_sp.shape[split_dim]
        
        for idx, (rank_in_group, cnt) in enumerate(zip(group_ranks, cnt_list)):
            # 提取该rank的有效数据并移除填充
            rank_combined = gathered_combined[idx][:cnt]  # [cnt, ...]
            
            # 沿最后一维拆分出out和lse（与参考代码一致）
            rank_out = rank_combined.narrow(split_dim, 0, split_idx).contiguous()
            rank_lse = rank_combined.narrow(split_dim, split_idx, rank_combined.shape[split_dim] - split_idx).contiguous()
            
            # 移除扩展的维度，恢复lse原始形状
            rank_lse = rank_lse.squeeze(1)
            
            parsed_out.append(rank_out)
            parsed_lse.append(rank_lse)
            print(f"  - 从rank {rank_in_group} 提取: out={rank_out.shape}, lse={rank_lse.shape}", flush=True)

        # 6. 合并结果（保持与原逻辑一致的平均操作）
        if local_cnt > 0:
            # 合并所有rank的out和lse
            all_sp_out = torch.cat(parsed_out, dim=0)
            all_sp_lse = torch.cat(parsed_lse, dim=0)
            print(f"[RANK {rank}] SP组 {key} - 合并所有rank数据: all_sp_out={all_sp_out.shape}, all_sp_lse={all_sp_lse.shape}", flush=True)

            # 为每个本地SP请求收集所有rank的对应结果
            req_results = []
            req_results_lse = []
            for local_idx in range(local_cnt):
                rank_out_results = [parsed_out[rank_idx][local_idx] for rank_idx in range(group_size) if parsed_out[rank_idx].shape[0] > local_idx]
                rank_lse_results = [parsed_lse[rank_idx][local_idx] for rank_idx in range(group_size) if parsed_lse[rank_idx].shape[0] > local_idx]
                req_results.append(rank_out_results)
                req_results_lse.append(rank_lse_results)

            # 执行合并（取平均）
            final_sp = torch.stack([dummy_merge(req) for req in req_results], dim=0)
            final_sp_lse = torch.stack([dummy_merge(req) for req in req_results_lse], dim=0)
        
        print(f"[RANK {rank}] SP组 {key} - 合并后: out={final_sp.shape}, lse={final_sp_lse.shape}", flush=True)

    # 验证维度一致性后再拼接
    if final_sp.numel() > 0:
        assert local_res.shape[1:] == final_sp.shape[1:], \
            f"本地结果与SP结果维度不匹配: local_res={local_res.shape[1:]}, final_sp={final_sp.shape[1:]}"
        assert local_lse.shape[1:] == final_sp_lse.shape[1:], \
            f"本地LSE与SP LSE维度不匹配: local_lse={local_lse.shape[1:]}, final_sp_lse={final_sp_lse.shape[1:]}"

    # 最终结果合并
    final_out = torch.cat([local_res, final_sp], dim=0)
    final_lse = torch.cat([local_lse, final_sp_lse], dim=0)
    print(f"[RANK {rank}] 最终结果: final_out={final_out.shape}, final_lse={final_lse.shape}", flush=True)
    print(f"========== [RANK {rank}] FLASH MLA SP DONE ==========\n", flush=True)
    
    return final_out, final_lse


def test_flash_mla(rank: int, world_size: int, args: argparse.Namespace):
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

    # 按rank划分的配置
    kv_lens_per_rank = [
        [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
        [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
        [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
        [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048]
    ]
    sp_groups_info_list = [
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': True, 'group': [0,1,2,3]}, 2: {'enabled': False}},
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': True, 'group': [0,1,2,3]}, 2: {'enabled': False}},
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': True, 'group': [0,1,2,3]}, 2: {'enabled': False}},
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': False}},
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

    # 预处理
    local_batches, sp_batch_indices, sp_group_key, cnt_list, local_cnt, recv_cnts = prepare_mla_fwd(
        rank=rank,
        world_size=world_size,
        q_batch_size=q_batch_size,
        sp_groups_info_list=sp_groups_info_list,
        sp_comm_groups=sp_comm_groups
    )

    # 调用SP注意力函数
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
        recv_cnts=recv_cnts,
        sp_comm_groups=sp_comm_groups,
        softmax_scale=softmax_scale,
        causal=True
    )

    # 结果验证
    assert len(out.shape) == 4 and len(lse.shape) == 3, "结果维度异常"
    assert out.shape[0] == q_batch_size and lse.shape[0] == q_batch_size, "batch_size不匹配"
    assert out.dtype == dtype, "dtype不匹配"
    print(f"[RANK {rank}] 结果验证通过！out.shape={out.shape}, lse.shape={lse.shape}", flush=True)

    # 销毁进程组
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Flash MLA SP 分布式测试（按照参考代码在最后一维拼接）")
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

