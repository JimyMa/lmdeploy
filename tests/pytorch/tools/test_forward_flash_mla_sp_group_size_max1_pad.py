import os
from typing import Optional, Tuple, Dict, List, Any
import argparse
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.multiprocessing import spawn
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

PRINT_SHAPE_LOG = True
def print_shape_log(rank: int, content: str) -> None:
    if PRINT_SHAPE_LOG:
        print(f"[RANK {rank}] {content}", flush=True)

# ---------- 分离Padding逻辑：Q和block_table/kv_lens使用不同的padding ----------
def pad_req(
    q: torch.Tensor,
    kv_lens: torch.Tensor,
    block_table: torch.Tensor,
    local_batches: List[int],
    sp_batch_indices: List[int],
    num_query_heads: int,
    pad_local: int = 128,
    pad_sp: int = 2,
    group_size: int = 1,
    cnt_list: List[int] = []
):
    device = q.device
    dtype = q.dtype
    head_dim = q.shape[2] if q.nelement() > 0 else 0
    block_num = block_table.shape[1] if block_table.nelement() > 0 else 0

    real_local, real_sp = len(local_batches), len(sp_batch_indices)
    
    # Q的padding尺寸：pad_local + pad_sp
    q_total_pad = pad_local + pad_sp
    # block_table和kv_lens的padding尺寸：pad_local + pad_sp * group_size
    bt_kv_total_pad = pad_local + pad_sp * group_size

    # --- 本地请求pad ---
    q_local_pad = torch.zeros(pad_local, num_query_heads, head_dim, dtype=dtype, device=device)
    kv_local_pad = torch.zeros(pad_local, dtype=torch.int32, device=device)
    bt_local_pad = torch.zeros(pad_local, block_num, dtype=block_table.dtype, device=device) if block_num > 0 else torch.empty(0, 0, device=device)
    if real_local > 0:
        q_local_pad[:real_local] = q[local_batches]
        kv_local_pad[:real_local] = kv_lens[local_batches]
        if block_num > 0:
            bt_local_pad[:real_local] = block_table[local_batches]

    # --- SP请求pad（Q）---
    q_sp_pad = torch.zeros(pad_sp, num_query_heads, head_dim, dtype=dtype, device=device)
    if real_sp > 0:
        q_sp_pad[:real_sp] = q[sp_batch_indices]

    # --- SP请求pad（block_table和kv_lens）---
    kv_sp_pad = torch.zeros(pad_sp * group_size, dtype=torch.int32, device=device)
    bt_sp_pad = torch.zeros(pad_sp * group_size, block_num, dtype=block_table.dtype, device=device) if block_num > 0 else torch.empty(0, 0, device=device)
    
    sp_kv_lens = kv_lens[real_local:] if (real_local < len(kv_lens)) else torch.empty(0, dtype=torch.int32, device=device)
    sp_block_table = block_table[real_local:] if (real_local < len(block_table) and block_num > 0) else torch.empty(0, block_num, dtype=block_table.dtype, device=device)

    sp_data_ptr = 0
    for rank_idx in range(group_size):

        rank_sp_start = rank_idx * pad_sp

        current_rank_valid_cnt = cnt_list[rank_idx] if (rank_idx < len(cnt_list)) else 0
        
        if current_rank_valid_cnt <= 0:
            continue
        if sp_data_ptr + current_rank_valid_cnt > len(sp_kv_lens):
            print(f"[WARN] SP数据不足：当前需分配{current_rank_valid_cnt}个，剩余{len(sp_kv_lens)-sp_data_ptr}个")
            continue

        kv_sp_pad[rank_sp_start : rank_sp_start + current_rank_valid_cnt] = sp_kv_lens[sp_data_ptr : sp_data_ptr + current_rank_valid_cnt]
        
        if block_num > 0 and sp_data_ptr + current_rank_valid_cnt <= len(sp_block_table):
            bt_sp_pad[rank_sp_start : rank_sp_start + current_rank_valid_cnt] = sp_block_table[sp_data_ptr : sp_data_ptr + current_rank_valid_cnt]
        
        sp_data_ptr += current_rank_valid_cnt

    local_mask = torch.zeros(pad_local, dtype=torch.bool, device=device)
    local_mask[:real_local] = True
    sp_mask = torch.zeros(pad_sp, dtype=torch.bool, device=device)
    sp_mask[:real_sp] = True

    q_pad = torch.cat([q_local_pad, q_sp_pad], dim=0)
    kv_lens_pad = torch.cat([kv_local_pad, kv_sp_pad], dim=0)
    block_table_pad = torch.cat([bt_local_pad, bt_sp_pad], dim=0) if block_num > 0 else torch.empty(0, 0, device=device)

    return (q_pad, 
            kv_lens_pad, block_table_pad, 
            local_mask, sp_mask, 
            real_local, real_sp,
            q_total_pad, bt_kv_total_pad)

def prepare_mla_fwd(
    rank: int,
    q: torch.Tensor,
    kv_lens: torch.Tensor,
    block_table: torch.Tensor,
    q_batch_size: int,
    num_query_heads: int,
    sp_groups_info_list: List[Dict],
    sp_comm_groups: Dict[Tuple[int, ...], ProcessGroup],
    pad_local: int = 128,
    pad_sp: int = 2,
):
    current_sp_info = sp_groups_info_list[rank]
    local_batches = [b for b in range(q_batch_size) if not current_sp_info.get(b, {}).get("enabled", False)]
    sp_batch_indices = [b for b, info in current_sp_info.items() if info.get("enabled", False) and rank in info.get("group", [])]
    real_sp = len(sp_batch_indices)

    print_shape_log(rank, f"prepare_mla_fwd - 本地请求数: {len(local_batches)}, SP请求数: {real_sp}")
    print_shape_log(rank, f"  local_batches: {local_batches}, sp_batch_indices: {sp_batch_indices}")

    sp_group_key = next(iter(sp_comm_groups.keys()), None)
    if sp_group_key is not None:
        group_size = len(sp_group_key)
        sp_comm_group = sp_comm_groups[sp_group_key]
        rank_idx_in_group = sp_group_key.index(rank)
    else:
        group_size = 0
        sp_comm_group = None
        rank_idx_in_group = 0

    cnt_list = []
    if sp_group_key:
        for r in sp_group_key:
            r_sp_cnt = 0
            for info in sp_groups_info_list[r].values():
                if isinstance(info, dict) and info.get("enabled", False) and r in info.get("group", []):
                    r_sp_cnt += 1
            cnt_list.append(r_sp_cnt)
        print_shape_log(rank, f"  SP通信组 {sp_group_key} - 各rank有效SP请求数: {cnt_list}")

    (q_pad, 
     kv_lens_pad, block_table_pad, 
     local_mask, sp_mask, 
     real_local, _,
     q_total_pad, bt_kv_total_pad) = pad_req(
        q=q,
        kv_lens=kv_lens,
        block_table=block_table,
        local_batches=local_batches,
        sp_batch_indices=sp_batch_indices,
        num_query_heads=num_query_heads,
        pad_local=pad_local,
        pad_sp=pad_sp,
        group_size=group_size
    )

    assert q_pad.shape[0] == q_total_pad, \
        f"[RANK {rank}] Q的batch size错误，应为{q_total_pad}，实际为{q_pad.shape[0]}"
    assert block_table_pad.shape[0] == bt_kv_total_pad, \
        f"[RANK {rank}] block_table的batch size错误，应为{bt_kv_total_pad}，实际为{block_table_pad.shape[0]}"
    assert kv_lens_pad.shape[0] == bt_kv_total_pad, \
        f"[RANK {rank}] cache_seqlens的batch size错误，应为{bt_kv_total_pad}，实际为{kv_lens_pad.shape[0]}"

    updated_metadata, updated_num_splits = get_mla_metadata(kv_lens_pad, num_query_heads, 1)
    print_shape_log(rank, f"prepare中metadata - 更新后: {updated_metadata.shape}")

    return (local_batches, sp_batch_indices, real_sp, cnt_list,
            q_pad, kv_lens_pad, block_table_pad,
            local_mask, sp_mask, pad_local, pad_sp, group_size,
            q_total_pad, bt_kv_total_pad,
            updated_metadata, updated_num_splits,
            sp_comm_group, rank_idx_in_group)


def all_gather_sp_q(
    q: torch.Tensor,  # 形状为[pad_local + pad_sp, ...]
    pad_local: int,
    pad_sp: int,
    sp_comm_group: Optional[ProcessGroup],
    group_size: int
) -> torch.Tensor:
    # rank = dist.get_rank()
    device = q.device
    num_head, head_dim = q.shape[1], q.shape[2]
    
    q_local = q[:pad_local]
    q_sp = q[pad_local:]
    
    gathered_sp = torch.empty(group_size, pad_sp, num_head, head_dim, dtype=q.dtype, device=device)
    dist.all_gather_into_tensor(gathered_sp.view(-1), q_sp.contiguous(), group=sp_comm_group)
    
    sp_q_all = gathered_sp.reshape(-1, num_head, head_dim)  # 形状为[pad_sp * group_size, ...]
    
    all_q = torch.cat([q_local, sp_q_all], dim=0)
    # print_shape_log(rank, f"all_gather_sp_q - 合并后Q尺寸: {all_q.shape}")
    
    return all_q


# ---------- all_gather_sp_results：无分支、无assert（依赖prepare预处理） ----------
def all_gather_sp_results(
    sp_res_all: torch.Tensor,  # 形状: [pad_sp*group_size, 1, ...]（bfloat16）
    sp_lse_all: torch.Tensor,  # 形状: [pad_sp*group_size, ...]（float32）
    sp_comm_group: Optional[ProcessGroup],  # 直接传递通信组对象
    pad_sp: int,
    group_size: int,
    rank_idx_in_group: int  # 提前从prepare获取的组内索引（无分支计算）
) -> Tuple[torch.Tensor, torch.Tensor]:
    # rank = dist.get_rank()
    device = sp_res_all.device
    head_dim_v = sp_res_all.shape[-1] if sp_res_all.nelement() > 0 else 0

    lse_original_dtype = sp_lse_all.dtype

    # 统一维度：给sp_lse_all插入第1维
    sp_lse_reshaped = sp_lse_all.unsqueeze(1)
    sp_lse_converted = sp_lse_reshaped.to(sp_res_all.dtype)
    
    combined = torch.cat([sp_res_all, sp_lse_converted], dim=-1)
    # print_shape_log(rank, f"拼接后combined.shape: {combined.shape}, dtype: {combined.dtype}")

    gathered = torch.empty(
        group_size,
        *combined.shape,
        dtype=combined.dtype, 
        device=device
    )
    dist.all_gather_into_tensor(gathered.view(-1), combined.contiguous(), group=sp_comm_group)
    # print_shape_log(rank, f"all gather后gathered.shape: {gathered.shape}, dtype: {gathered.dtype}")

    current_rank_combined = gathered[rank_idx_in_group]
    sp_slice_start = rank_idx_in_group * pad_sp
    sp_slice_end = sp_slice_start + pad_sp
    current_rank_sp_combined = current_rank_combined[sp_slice_start:sp_slice_end]
    # print_shape_log(rank, f"当前RANK对应SP切片范围: [{sp_slice_start}:{sp_slice_end}]，切片后shape: {current_rank_sp_combined.shape}")

    current_rank_sp_res = current_rank_sp_combined[..., :head_dim_v]
    current_rank_sp_lse = current_rank_sp_combined[..., head_dim_v:]

    current_rank_sp_lse = current_rank_sp_lse.to(lse_original_dtype)
    current_rank_sp_lse = current_rank_sp_lse.squeeze(1)

    # print_shape_log(rank, f"最终提取结果形状: 本地SP res={current_rank_sp_res.shape}({current_rank_sp_res.dtype}), 本地SP lse={current_rank_sp_lse.shape}({current_rank_sp_lse.dtype})")
    return current_rank_sp_res, current_rank_sp_lse


def flash_mla_fwd_sp(
    q: torch.Tensor,  # 形状为[pad_local + pad_sp, ...]
    k_cache: torch.Tensor,
    block_table: torch.Tensor,  # 形状为[pad_local + pad_sp * group_size, ...]
    cache_seqlens: torch.Tensor,  # 形状为[pad_local + pad_sp * group_size, ...]
    head_dim_v: int,
    sp_comm_group: Optional[ProcessGroup],
    # 从prepare传递的预处理参数（避免在本函数计算）
    pad_local: int,
    pad_sp: int,
    group_size: int,
    q_total_pad: int,
    bt_kv_total_pad: int,
    updated_metadata: torch.Tensor,
    updated_num_splits: torch.Tensor,
    rank_idx_in_group: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # rank = dist.get_rank()
    
    # print_shape_log(rank, f"\n===== flash_mla_fwd_sp 开始 =====")
    # print_shape_log(rank, f"Q的padding后尺寸: {q.shape} (预期: {q_total_pad})")
    # print_shape_log(rank, f"block_table的padding后尺寸: {block_table.shape} (预期: {bt_kv_total_pad})")
    # print_shape_log(rank, f"cache_seqlens的padding后尺寸: {cache_seqlens.shape} (预期: {bt_kv_total_pad})")

    # 对Q执行all gather（直接传递通信组对象，无分支）
    all_q = all_gather_sp_q(
        q=q,
        pad_local=pad_local,
        pad_sp=pad_sp,
        sp_comm_group=sp_comm_group,  # 核心修改：传通信组对象
        group_size=group_size
    )
    # print_shape_log(rank, f"all gather后Q的尺寸: {all_q.shape} (预期: {bt_kv_total_pad})")

    # MLA计算（固定逻辑，无分支）
    # print_shape_log(rank, f"MLA输入 - key_value_cache: {k_cache.shape}, block_table: {block_table.shape}, cache_seqlens: {cache_seqlens.shape}")
    attn_output, lse = flash_mla_with_kvcache(
        q=all_q.unsqueeze(1),
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=updated_metadata,
        num_splits=updated_num_splits,
        softmax_scale=softmax_scale,
        causal=causal
    )
    # print_shape_log(rank, f"MLA输出: attn={attn_output.shape}, lse={lse.shape}, attn.dtype: {attn_output.dtype}, lse.dtype: {lse.dtype}")

    local_res = attn_output[:pad_local]  # [pad_local, 1, ...]
    sp_res_all = attn_output[pad_local:]  # [pad_sp*group_size, 1, ...]
    local_lse = lse[:pad_local]  # [pad_local, ...]
    sp_lse_all = lse[pad_local:]  # [pad_sp*group_size, ...]
    # print_shape_log(rank, f"拆分结果: local_res={local_res.shape}, 全局SP_res={sp_res_all.shape}, 全局SP_lse={sp_lse_all.shape}")

    local_sp_res, local_sp_lse = all_gather_sp_results(
        sp_res_all=sp_res_all,
        sp_lse_all=sp_lse_all,
        sp_comm_group=sp_comm_group,
        pad_sp=pad_sp,
        group_size=group_size,
        rank_idx_in_group=rank_idx_in_group
    )

    # print_shape_log(rank, f"拼接前: local_res={local_res.shape}, local_sp_res={local_sp_res.shape}")
    # print_shape_log(rank, f"拼接前: local_lse={local_lse.shape}, local_sp_lse={local_sp_lse.shape}")
    final_out_pad = torch.cat([local_res, local_sp_res], dim=0)  # [pad_local+pad_sp, 1, ...]
    final_lse_pad = torch.cat([local_lse, local_sp_lse], dim=0)  # [pad_local+pad_sp, ...]
    
    # print_shape_log(rank, f"最终结果: out={final_out_pad.shape}, lse={final_lse_pad.shape}")
    # print_shape_log(rank, f"===== flash_mla_fwd_sp 结束 =====\n")
    
    return final_out_pad, final_lse_pad

def test_flash_mla(rank: int, world_size: int, args: argparse.Namespace):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    print_shape_log(rank, f"分布式初始化: rank={rank}, world_size={world_size}")

    dtype = torch.bfloat16
    num_query_heads, head_size, head_dim_v = 32, 576, 512
    block_size = 64
    NUM_BLOCKS_PER_RANK = 128 * 12 + 1  # 1537
    NUM_BLOCKS = NUM_BLOCKS_PER_RANK * world_size  # 1537*4=6148
    pad_local = 128  # 本地请求最大数量
    pad_sp = 2       # 每个RANK的SP请求最大数量（固定为2）
    print_shape_log(rank, f"配置: 头数={num_query_heads}, pad_local={pad_local}, pad_sp={pad_sp}")

    kv_lens_per_rank = [[2048]*6, [2048]*6, [2048]*6, [2048]*6]
    sp_groups_info_list = [
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': True, 'group': [0,1,2,3]}, 2: {'enabled': False}, 3: {'enabled': False}},  # RANK0：2个SP请求
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': True, 'group': [0,1,2,3]}, 2: {'enabled': False}, 3: {'enabled': False}},  # RANK1：2个SP请求
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': False}, 2: {'enabled': False}},  # RANK2：1个SP请求
        {0: {'enabled': False}, 1: {'enabled': False}}  # RANK3：0个SP请求
    ]
    current_sp_info = sp_groups_info_list[rank]
    q_batch_size = len(current_sp_info)

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

    # 通信组配置（Dict仅用于prepare解析，后续函数直接用通信组对象）
    sp_comm_groups = {tuple(sorted([0,1,2,3])): dist.new_group([0,1,2,3])}

    # 执行prepare
    prepare_outputs = prepare_mla_fwd(
        rank=rank,
        q=query,
        kv_lens=kv_lens_tensor,
        block_table=block_tables,
        q_batch_size=q_batch_size,
        num_query_heads=num_query_heads,
        sp_groups_info_list=sp_groups_info_list,
        sp_comm_groups=sp_comm_groups,
        pad_local=pad_local,
        pad_sp=pad_sp
    )
    
    (local_batches, sp_batch_indices, real_sp, cnt_list,
     q_pad, kv_lens_pad, block_table_pad,
     local_mask, sp_mask, pad_local, pad_sp, group_size,
     q_total_pad, bt_kv_total_pad,
     updated_metadata, updated_num_splits,
     sp_comm_group, rank_idx_in_group) = prepare_outputs

    # 执行MLA
    out_pad, lse_pad = flash_mla_fwd_sp(
        q=q_pad,
        k_cache=key_value_cache,
        block_table=block_table_pad,
        cache_seqlens=kv_lens_pad,
        head_dim_v=head_dim_v,
        sp_comm_group=sp_comm_group,
        pad_local=pad_local,
        pad_sp=pad_sp,
        group_size=group_size,
        q_total_pad=q_total_pad,
        bt_kv_total_pad=bt_kv_total_pad,
        updated_metadata=updated_metadata,
        updated_num_splits=updated_num_splits,
        rank_idx_in_group=rank_idx_in_group,
        softmax_scale=head_size ** -0.5,
        causal=True,
    )

    # 外部提取有效输出
    valid_local_out = out_pad[:pad_local][local_mask]
    valid_local_lse = lse_pad[:pad_local][local_mask]
    valid_sp_out = out_pad[pad_local:][sp_mask]
    valid_sp_lse = lse_pad[pad_local:][sp_mask]

    print_shape_log(rank, f"valid_local_out.shape: {valid_local_out.shape}, valid_local_lse.shape: {valid_local_lse.shape}, valid_sp_out.shape: {valid_sp_out.shape}, valid_sp_lse.shape: {valid_sp_lse.shape}")
    
    final_out = torch.cat([valid_local_out, valid_sp_out], dim=0)
    final_lse = torch.cat([valid_local_lse, valid_sp_lse], dim=0)

    # 验证结果
    assert final_out.shape[0] == len(local_batches) + len(sp_batch_indices), \
           f"[RANK {rank}] 有效结果数量错误: 实际{final_out.shape[0]} vs 预期{len(local_batches) + len(sp_batch_indices)}"
    assert final_out.dtype == dtype, f"[RANK {rank}] Dtype错误: {final_out.dtype} vs {dtype}"
    print_shape_log(rank, "✅ 结果验证通过")

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Flash MLA SP（支持CUDA Graph版本）")
    parser.add_argument("--num_ranks", type=int, default=4)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="12355")
    args = parser.parse_args()

    assert args.num_ranks == 4, "仅支持4个进程（匹配SP通信组配置）"
    print(f"启动测试: {args.num_ranks}进程, 地址={args.master_addr}:{args.master_port}")
    spawn(fn=test_flash_mla, args=(args.num_ranks, args), nprocs=args.num_ranks, join=True)
    print("所有进程完成")


if __name__ == "__main__":
    main()