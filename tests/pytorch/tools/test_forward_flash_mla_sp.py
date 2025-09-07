import os
from typing import Optional, Tuple, Dict
import argparse
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
# 假设 flash_mla 相关依赖已正确导入
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
    meta = {}

    print(f"\n========== [RANK {rank}] FLASH MLA SP START ==========", flush=True)

    # 1) 拆分本地 / SP 请求
    batch_size = q.shape[0]
    local_batches, sp_batches = [], []
    for b in range(batch_size):
        info = sp_groups_info.get(b, {'enabled': False})
        if not info['enabled']:
            local_batches.append(b)
        elif rank in info['group']:
            sp_batches.append((b, info))
    print(f"[RANK {rank}] local_batches = {local_batches}", flush=True)
    print(f"[RANK {rank}] sp_batches = {[tpl[0] for tpl in sp_batches]}", flush=True)

    # 2) 建立通信组
    sp_groups = {}
    for b, info in sp_batches:
        key = tuple(sorted(info['group']))
        if key not in sp_groups:
            sp_groups[key] = []
        sp_groups[key].append((b, info))
    assert len(sp_groups) <= 1
    print(f"[RANK {rank}] sp_groups keys = {list(sp_groups.keys())}", flush=True)

    # 3) 获取所有已存在的SP通信组Key
    all_sp_group_keys = list(sp_comm_groups.keys()) if sp_comm_groups else []
    for key in sp_groups.keys():
        if key not in all_sp_group_keys:
            all_sp_group_keys.append(key)
    all_sp_group_keys = sorted(all_sp_group_keys)
    print(f"[RANK {rank}] 所有SP组（排序后） = {all_sp_group_keys}", flush=True)

    # 4) All-Gather Q：遍历所有SP组
    all_sp_q, sp_batch_indices = [], {}
    for key in all_sp_group_keys:
        # 获取/创建通信组
        comm = sp_comm_groups.get(key)
        if comm is None:
            comm = dist.new_group(list(key))
            sp_comm_groups[key] = comm
        group_ranks = list(key)

        # 处理本地SP数据
        reqs = sp_groups.get(key, [])
        local_q_list = [q[b] for b, _ in reqs] if reqs else []
        local_q = torch.stack(local_q_list) if local_q_list else \
                  torch.empty(0, q.shape[1], q.shape[2], dtype=q.dtype, device=device)
        local_cnt = len(local_q_list)

        # 同步组内所有Rank的请求数
        local_cnt_tensor = torch.tensor([local_cnt], dtype=torch.long, device=device)
        cnt_list_tensor = [torch.empty_like(local_cnt_tensor) for _ in group_ranks]
        dist.all_gather(cnt_list_tensor, local_cnt_tensor, group=comm)
        cnt_list = [t.item() for t in cnt_list_tensor]
        max_cnt = max(cnt_list) if cnt_list else 0
        print(f"[RANK {rank}] SP组 {key} - 组内各Rank请求数: {cnt_list}, 最大长度: {max_cnt}", flush=True)

        # 填充空张量确保长度一致
        padded_q = pad_to_max_len(local_q, max_cnt, dim=0)
        # All-Gather
        gathered = torch.empty(len(group_ranks), max_cnt, q.shape[1], q.shape[2],
                               dtype=padded_q.dtype, device=device)
        dist.all_gather_into_tensor(gathered.view(-1), padded_q.contiguous(), group=comm)
        print(f"[RANK {rank}] SP组 {key} - gathered_q shape: {gathered.shape}, dtype: {gathered.dtype}", flush=True)

        # 扁平化SP请求
        flat = []
        for i, c in enumerate(cnt_list):
            if c > 0:
                flat.append(gathered[i, :c])
        flat = torch.cat(flat) if flat else torch.empty(0, q.shape[1], q.shape[2], dtype=q.dtype, device=device)
        all_sp_q.append(flat)
        sp_batch_indices[key] = [b for b, _ in reqs] if reqs else []

        # 记录SP组元信息
        meta[key] = {
            'group_ranks': group_ranks,
            'local_batch_count': local_cnt,
            'all_batch_counts': cnt_list
        }

    # 5) 合并所有 Q
    local_q = q[local_batches] if local_batches else \
              torch.empty(0, q.shape[1], q.shape[2], dtype=q.dtype, device=device)
    sp_q = torch.cat(all_sp_q, dim=0) if all_sp_q else \
           torch.empty(0, q.shape[1], q.shape[2], dtype=q.dtype, device=device)
    all_q = torch.cat([local_q, sp_q], dim=0)
    print(f"[RANK {rank}] local_q={local_q.shape} (dtype:{local_q.dtype}), sp_q={sp_q.shape} (dtype:{sp_q.dtype}), all_q={all_q.shape} (dtype:{all_q.dtype})", flush=True)

    # 6) 调用Flash MLA注意力计算（保持原始维度）
    all_q_expanded = all_q.unsqueeze(1)
    attn_output, lse = flash_mla_with_kvcache(
        all_q_expanded,
        k_cache,
        block_table,
        cache_seqlens,
        head_dim_v,
        tile_scheduler_metadata,
        num_splits,
        softmax_scale,
        causal,
    )
    # 不再使用 squeeze(1)，保持 attn_output 的 4 维形状
    print(f"[RANK {rank}] attn_output shape: {attn_output.shape} (dtype: {attn_output.dtype})", flush=True)
    print(f"[RANK {rank}] 原始lse shape: {lse.shape} (dtype: {lse.dtype})", flush=True)

    # 7) 拆分结果（out和lse同步拆分）
    local_cnt = local_q.shape[0]
    local_res = attn_output[:local_cnt]   # 保持 4 维
    sp_res = attn_output[local_cnt:]      # 保持 4 维
    local_lse = lse[:local_cnt]           # 3 维
    sp_lse = lse[local_cnt:]              # 3 维
    print(f"[RANK {rank}] local_res={local_res.shape} (dtype:{local_res.dtype}), sp_res={sp_res.shape} (dtype:{sp_res.dtype})", flush=True)
    print(f"[RANK {rank}] local_lse={local_lse.shape} (dtype:{local_lse.dtype}), sp_lse={sp_lse.shape} (dtype:{sp_lse.dtype})", flush=True)

    # 8) All2All 拆分SP结果（拼接out和lse，单次通信）
    final_sp_parts = []
    final_sp_lse_parts = []
    sp_ptr = 0
    for key in all_sp_group_keys:
        comm = sp_comm_groups[key]
        m = meta[key]
        group_ranks = m['group_ranks']
        all_cnts = m['all_batch_counts']
        my_sp_cnt = m['local_batch_count']
        total_sp = sum(all_cnts)

        # 提取当前SP组的切片
        slice_sp = sp_res[sp_ptr:sp_ptr + total_sp] if total_sp > 0 else \
                   torch.empty(0, *attn_output.shape[1:], dtype=attn_output.dtype, device=device)
        slice_sp_lse = sp_lse[sp_ptr:sp_ptr + total_sp] if total_sp > 0 else \
                       torch.empty(0, *lse.shape[1:], dtype=lse.dtype, device=device)
        sp_ptr += total_sp
        print(f"[RANK {rank}] SP组 {key} - 原始切片: slice_sp={slice_sp.shape} (dtype:{slice_sp.dtype}), slice_sp_lse={slice_sp_lse.shape} (dtype:{slice_sp_lse.dtype})", flush=True)

        # 将 lse 扩展一个维度以匹配 out 的维度
        if total_sp > 0:
            slice_sp_lse_expanded = slice_sp_lse.unsqueeze(1)  # 从 (N, H, 1) 变为 (N, 1, H, 1)
        else:
            slice_sp_lse_expanded = torch.empty(0, 1, lse.shape[1], lse.shape[2], dtype=lse.dtype, device=device)
        
        # 统一dtype并拼接（最后一维拼接）
        slice_sp_lse_expanded = slice_sp_lse_expanded.to(dtype=slice_sp.dtype)
        combined_sp = torch.cat([slice_sp, slice_sp_lse_expanded], dim=-1)
        print(f"[RANK {rank}] SP组 {key} - 拼接后combined_sp shape: {combined_sp.shape}, dtype: {combined_sp.dtype}", flush=True)

        # 构造发送/接收列表
        send_cnts = all_cnts
        recv_cnts = [my_sp_cnt for _ in group_ranks]
        print(f"[RANK {rank}] SP组 {key} - send_cnts={send_cnts}, recv_cnts={recv_cnts}", flush=True)

        # 发送列表
        send_list, pos = [], 0
        for c in send_cnts:
            end = pos + c
            if c > 0 and combined_sp.numel() > 0:
                send_tensor = combined_sp[pos:end].contiguous()
            else:
                send_tensor = torch.empty(0, *combined_sp.shape[1:], dtype=combined_sp.dtype, device=device)
            send_list.append(send_tensor)
            pos = end
        # 验证发送列表dtype一致性
        send_dtypes = {t.dtype for t in send_list}
        assert len(send_dtypes) == 1, f"send_list存在多dtype: {send_dtypes}"
        print(f"[RANK {rank}] SP组 {key} - send_list所有张量dtype: {send_dtypes.pop()}", flush=True)

        # 接收列表
        recv_list = []
        for c in recv_cnts:
            if c > 0:
                recv_tensor = torch.empty(c, *combined_sp.shape[1:], dtype=combined_sp.dtype, device=device)
            else:
                recv_tensor = torch.empty(0, *combined_sp.shape[1:], dtype=combined_sp.dtype, device=device)
            recv_list.append(recv_tensor)
        # 验证接收列表dtype一致性
        recv_dtypes = {t.dtype for t in recv_list}
        assert len(recv_dtypes) == 1, f"recv_list存在多dtype: {recv_dtypes}"
        print(f"[RANK {rank}] SP组 {key} - recv_list所有张量dtype: {recv_dtypes.pop()}", flush=True)

        print(f"[RANK {rank}] SP组 {key} - send_list[0].shape: {send_list[0].shape}",flush=True)
        print(f"[RANK {rank}] SP组 {key} - recv_list[0].shape: {recv_list[0].shape}",flush=True)

        # 单次All2All通信
        dist.all_to_all(recv_list, send_list, group=comm)

        # 拆分并合并结果
        merged_sp_tensor = torch.empty(0, *attn_output.shape[1:], dtype=attn_output.dtype, device=device)
        merged_sp_tensor_lse = torch.empty(0, *lse.shape[1:], dtype=lse.dtype, device=device)
        if my_sp_cnt > 0:
            combined_received = [t for t in recv_list if t.numel() > 0]
            combined_received = torch.cat(combined_received, dim=0) if combined_received else \
                                torch.empty(0, *combined_sp.shape[1:], dtype=combined_sp.dtype, device=device)
            print(f"[RANK {rank}] SP组 {key} - 接收combined shape: {combined_received.shape}, dtype: {combined_received.dtype}", flush=True)

            # 拆分out和lse
            split_idx = attn_output.shape[-1]  # head_dim_v
            sp_res_received = combined_received[..., :split_idx]
            sp_lse_received = combined_received[..., split_idx:].squeeze(1)  # 移除添加的维度
            print(f"[RANK {rank}] SP组 {key} - 拆分后: sp_res={sp_res_received.shape} (dtype:{sp_res_received.dtype}), sp_lse={sp_lse_received.shape} (dtype:{sp_lse_received.dtype})", flush=True)

            # 合并每个SP请求的结果
            sp_request_results = [[] for _ in range(my_sp_cnt)]
            sp_request_results_lse = [[] for _ in range(my_sp_cnt)]
            current_pos = 0
            for i in range(len(group_ranks)):
                recv_c = recv_cnts[i]
                if recv_c > 0:
                    end_pos = current_pos + recv_c
                    for req_idx in range(recv_c):
                        sp_request_results[req_idx].append(sp_res_received[current_pos + req_idx])
                        sp_request_results_lse[req_idx].append(sp_lse_received[current_pos + req_idx])
                    current_pos = end_pos

            merged_sp_results = [dummy_merge(req_res) for req_res in sp_request_results]
            merged_sp_tensor = torch.stack(merged_sp_results, dim=0)
            merged_sp_results_lse = [dummy_merge(req_res) for req_res in sp_request_results_lse]
            merged_sp_tensor_lse = torch.stack(merged_sp_results_lse, dim=0)
            print(f"[RANK {rank}] SP组 {key} - 合并后: out={merged_sp_tensor.shape} (dtype:{merged_sp_tensor.dtype}), lse={merged_sp_tensor_lse.shape} (dtype:{merged_sp_tensor_lse.dtype})", flush=True)
        else:
            print(f"[RANK {rank}] SP组 {key} - 无SP请求，跳过合并", flush=True)

        final_sp_parts.append(merged_sp_tensor)
        final_sp_lse_parts.append(merged_sp_tensor_lse)

    # 9) 最终输出合并
    final_sp = torch.cat(final_sp_parts) if final_sp_parts else \
               torch.empty(0, *attn_output.shape[1:], dtype=attn_output.dtype, device=device)
    final_out = torch.cat([local_res, final_sp], dim=0)
    final_sp_lse = torch.cat(final_sp_lse_parts) if final_sp_lse_parts else \
                   torch.empty(0, *lse.shape[1:], dtype=lse.dtype, device=device)
    final_lse = torch.cat([local_lse, final_sp_lse], dim=0)
    print(f"[RANK {rank}] 最终结果: final_out={final_out.shape} (dtype:{final_out.dtype}), final_lse={final_lse.shape} (dtype:{final_lse.dtype})", flush=True)
    print(f"========== [RANK {rank}] FLASH MLA SP DONE ==========\n", flush=True)
    
    return final_out, final_lse

# -------------------------- 子进程测试函数 --------------------------
def test_flash_mla(rank: int, world_size: int, args: argparse.Namespace):
    """由spawn调用的子进程函数，rank由spawn自动分配（0 ~ world_size-1）"""
    # 1. 初始化分布式
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    print(f"[RANK {rank}] 分布式初始化完成（world_size={world_size}）", flush=True)

    # 2. 全局配置
    dtype = torch.bfloat16
    num_query_heads = 32
    head_size = 576
    head_dim_v = 512
    softmax_scale = head_size ** -0.5
    block_size = 64
    NUM_BLOCKS_PER_RANK = 128 * 12 + 1
    NUM_BLOCKS = NUM_BLOCKS_PER_RANK * world_size
    num_kv_heads = 1

    # 3. 按rank划分的配置（确保每个rank索引正确）
    kv_lens_per_rank = [
        [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
        [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
        [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
        [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048]
    ]
    sp_groups_info_list = [
        # rank0: q_batch_size=3
        {0: {'enabled': True, 'group': [0,1,2,3]},
        1: {'enabled': True, 'group': [0,1,2,3]},
        2: {'enabled': False}},
        # rank1: q_batch_size=3
        {0: {'enabled': True, 'group': [0,1,2,3]},
        1: {'enabled': True, 'group': [0,1,2,3]},
        2: {'enabled': False}},
        # rank2: q_batch_size=3
        {0: {'enabled': True, 'group': [0,1,2,3]},
        1: {'enabled': True, 'group': [0,1,2,3]},
        2: {'enabled': False}},
        # rank3: q_batch_size=3
        {0: {'enabled': True, 'group': [0,1,2,3]},
        1: {'enabled': True, 'group': [0,1,2,3]},
        2: {'enabled': False}}
    ]

    # 4. 验证配置长度
    assert len(kv_lens_per_rank) == world_size, f"kv_lens_per_rank长度需等于world_size（{world_size}）"
    assert len(sp_groups_info_list) == world_size, f"sp_groups_info_list长度需等于world_size（{world_size}）"

    # 5. 构造当前rank的输入数据
    current_sp_info = sp_groups_info_list[rank]
    q_batch_size = len(current_sp_info) if current_sp_info else 0
    # Query：显式指定dtype为bfloat16
    query = torch.randn(
        q_batch_size, num_query_heads, head_size, dtype=dtype, device=device
    ) / 10
    print(f"[RANK {rank}] query.shape: {query.shape}（q_batch_size={q_batch_size}）, dtype: {query.dtype}", flush=True)

    # KV缓存：显式指定dtype为bfloat16
    key_value_cache = torch.randn(
        NUM_BLOCKS, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )

    # Block Table
    kv_lens_this_rank = kv_lens_per_rank[rank]
    kv_batch_size = len(kv_lens_this_rank)
    max_num_blocks_per_seq_per_rank = NUM_BLOCKS_PER_RANK
    block_tables_this_rank = torch.randint(
        0, NUM_BLOCKS_PER_RANK,
        (kv_batch_size, max_num_blocks_per_seq_per_rank),
        dtype=torch.int32, device=device
    ) if kv_batch_size > 0 else torch.empty(0, max_num_blocks_per_seq_per_rank, dtype=torch.int32, device=device)
    print(f"[RANK {rank}] block_tables_this_rank.shape: {block_tables_this_rank.shape}（kv_batch_size={kv_batch_size}）", flush=True)

    # KV序列长度
    kv_lens_tensor_this_rank = torch.tensor(
        kv_lens_this_rank, dtype=torch.int32, device=device
    ) if kv_batch_size > 0 else torch.empty(0, dtype=torch.int32, device=device)

    # 6. 计算MLA元数据
    tile_scheduler_metadata, num_splits = get_mla_metadata(
        kv_lens_tensor_this_rank,
        num_query_heads // num_kv_heads,
        num_kv_heads,
    )

    # 7. 创建SP通信组
    sp_comm_groups = {}
    sp_group_key = tuple(sorted([0, 1, 2, 3]))
    comm = dist.new_group(ranks=list(sp_group_key))
    sp_comm_groups[sp_group_key] = comm

    # 8. 调用SP注意力函数
    out, lse = flash_mla_fwd_sp(
        query,
        key_value_cache,
        block_tables_this_rank,
        kv_lens_tensor_this_rank,
        head_dim_v,
        tile_scheduler_metadata,
        num_splits,
        softmax_scale,
        causal=True,
        sp_groups_info=current_sp_info,
        sp_comm_groups=sp_comm_groups
    )

    # 9. 结果验证（更新维度检查）
    assert len(out.shape) == 4, f"out应为4维，实际{len(out.shape)}维"
    assert len(lse.shape) == 3, f"lse应为3维，实际{len(lse.shape)}维"
    assert out.shape[0] == q_batch_size, f"out的batch_size不匹配：期望{q_batch_size}，实际{out.shape[0]}"
    assert lse.shape[0] == q_batch_size, f"lse的batch_size不匹配：期望{q_batch_size}，实际{lse.shape[0]}"
    assert out.dtype == dtype, f"out的dtype不匹配：期望{dtype}，实际{out.dtype}"
    print(f"[RANK {rank}] 结果验证通过！out.shape={out.shape}, lse.shape={lse.shape}", flush=True)

    # 10. 销毁进程组
    dist.barrier()
    dist.destroy_process_group()

# -------------------------- 主函数 --------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Flash MLA SP 分布式测试（spawn自动分配rank）"
    )
    parser.add_argument("--num_ranks", type=int, default=4, help="进程总数（world_size）")
    parser.add_argument("--master_addr", type=str, default="localhost", help="分布式主节点地址")
    parser.add_argument("--master_port", type=str, default="12355", help="分布式主节点端口")
    args = parser.parse_args()

    # 验证num_ranks
    assert args.num_ranks == 4, "当前配置仅支持num_ranks=4，如需修改请同步更新kv_lens_per_rank和sp_groups_info_list"

    # 启动spawn
    print(f"启动分布式测试：num_ranks={args.num_ranks}, master_addr={args.master_addr}, master_port={args.master_port}", flush=True)
    spawn(
        fn=test_flash_mla,
        args=(args.num_ranks, args),
        nprocs=args.num_ranks,
        join=True
    )
    print("所有进程执行完成！", flush=True)

if __name__ == "__main__":
    main()

