import os
from typing import Optional, Tuple, Dict, List, Any
import argparse
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
# 假设 flash_mla 模块已正确导入（用户原有依赖）
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
    group_size: int = 1
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

    # --- SP请求pad（Q专用） ---
    q_sp_pad = torch.zeros(pad_sp, num_query_heads, head_dim, dtype=dtype, device=device)
    if real_sp > 0:
        q_sp_pad[:real_sp] = q[sp_batch_indices]

    # --- SP请求pad（block_table和kv_lens专用，尺寸为pad_sp * group_size） ---
    kv_sp_pad = torch.zeros(pad_sp * group_size, dtype=torch.int32, device=device)
    bt_sp_pad = torch.zeros(pad_sp * group_size, block_num, dtype=block_table.dtype, device=device) if block_num > 0 else torch.empty(0, 0, device=device)
    
    # 当前rank的SP请求放在对应位置
    rank_idx = dist.get_rank() % group_size  # 确保在组内索引有效
    sp_start = rank_idx * pad_sp
    sp_end = sp_start + pad_sp
    if real_sp > 0:
        kv_sp_pad[sp_start:sp_end][:real_sp] = kv_lens[sp_batch_indices]
        if block_num > 0:
            bt_sp_pad[sp_start:sp_end][:real_sp] = block_table[sp_batch_indices]

    # --- 有效数据掩码 ---
    local_mask = torch.zeros(pad_local, dtype=torch.bool, device=device)
    local_mask[:real_local] = True
    sp_mask = torch.zeros(pad_sp, dtype=torch.bool, device=device)
    sp_mask[:real_sp] = True

    # 拼接Q（pad_local + pad_sp）
    q_pad = torch.cat([q_local_pad, q_sp_pad], dim=0)
    # 拼接block_table和kv_lens（pad_local + pad_sp * group_size）
    kv_lens_pad = torch.cat([kv_local_pad, kv_sp_pad], dim=0)
    block_table_pad = torch.cat([bt_local_pad, bt_sp_pad], dim=0) if block_num > 0 else torch.empty(0, 0, device=device)

    return (q_pad, 
            kv_lens_pad, block_table_pad, 
            local_mask, sp_mask, 
            real_local, real_sp,
            q_total_pad, bt_kv_total_pad)

# ---------- 预处理逻辑 ----------
def prepare_mla_fwd(
    rank: int,
    q: torch.Tensor,
    kv_lens: torch.Tensor,
    block_table: torch.Tensor,
    q_batch_size: int,
    num_query_heads: int,
    sp_groups_info_list: List[Dict],
    sp_comm_groups: Dict,
    pad_local: int = 128,
    pad_sp: int = 2,
):
    current_sp_info = sp_groups_info_list[rank]
    local_batches = [b for b in range(q_batch_size) if not current_sp_info.get(b, {}).get("enabled", False)]
    sp_batch_indices = [b for b, info in current_sp_info.items() if info.get("enabled", False) and rank in info.get("group", [])]
    real_sp = len(sp_batch_indices)

    print_shape_log(rank, f"prepare_mla_fwd - 本地请求数: {len(local_batches)}, SP请求数: {real_sp}")
    print_shape_log(rank, f"  local_batches: {local_batches}, sp_batch_indices: {sp_batch_indices}")

    # 获取通信组信息
    sp_group_key = next(iter(sp_comm_groups.keys()), None)
    group_size = len(sp_group_key) if sp_group_key else 0
    rank_idx_in_group = sp_group_key.index(rank) if sp_group_key else 0

    # 计算cnt_list（每个rank的有效SP请求数）
    cnt_list = []
    if sp_group_key:
        for r in sp_group_key:
            r_sp_cnt = 0
            for info in sp_groups_info_list[r].values():
                if isinstance(info, dict) and info.get("enabled", False) and r in info.get("group", []):
                    r_sp_cnt += 1
            cnt_list.append(r_sp_cnt)
        print_shape_log(rank, f"  SP通信组 {sp_group_key} - 各rank有效SP请求数: {cnt_list}")

    # 执行分离padding
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

    # 计算更新后的metadata（使用block_table和kv_lens的尺寸）
    updated_metadata, updated_num_splits = get_mla_metadata(kv_lens_pad, num_query_heads, 1)
    print_shape_log(rank, f"prepare中metadata - 更新后: {updated_metadata.shape}")

    return (local_batches, sp_batch_indices, sp_group_key, real_sp, cnt_list,
            q_pad, kv_lens_pad, block_table_pad,
            local_mask, sp_mask, pad_local, pad_sp, group_size,
            q_total_pad, bt_kv_total_pad,
            updated_metadata, updated_num_splits)


# ---------- all_gather_sp_q：收集后Q尺寸变为pad_local + pad_sp * group_size ----------
def all_gather_sp_q(
    q: torch.Tensor,  # 形状为[pad_local + pad_sp, ...]
    pad_local: int,
    pad_sp: int,
    sp_group_key: Any,
    sp_comm_groups: Dict,
    group_size: int
) -> torch.Tensor:
    rank = dist.get_rank()
    device = q.device
    num_head, head_dim = q.shape[1], q.shape[2]
    
    # 分离本地请求和SP请求
    q_local = q[:pad_local]
    q_sp = q[pad_local:]  # 形状为[pad_sp, ...]
    
    # 收集所有rank的SP请求
    gathered_sp = torch.empty(group_size, pad_sp, num_head, head_dim, dtype=q.dtype, device=device)
    dist.all_gather_into_tensor(gathered_sp.view(-1), q_sp.contiguous(), group=sp_comm_groups[sp_group_key])
    
    # 展平收集的SP请求
    sp_q_all = gathered_sp.reshape(-1, num_head, head_dim)  # 形状为[pad_sp * group_size, ...]
    
    # 合并本地请求和所有SP请求，总尺寸为pad_local + pad_sp * group_size
    all_q = torch.cat([q_local, sp_q_all], dim=0)
    print_shape_log(rank, f"all_gather_sp_q - 合并后Q尺寸: {all_q.shape}")
    
    return all_q


# ---------- 修改all_gather_sp_results：解决数据类型不匹配问题 ----------
def all_gather_sp_results(
    sp_res_all: torch.Tensor,  # 形状: [8, 1, 32, 512]（bfloat16类型）
    sp_lse_all: torch.Tensor,  # 形状: [8, 32, 1]（float32类型）
    sp_group_key: Any,
    sp_comm_groups: Dict,
    pad_sp: int,  # 每个RANK对应的SP请求数（固定为2）
    group_size: int  # 通信组内RANK数量（如4）
) -> Tuple[torch.Tensor, torch.Tensor]:
    rank = dist.get_rank()
    device = sp_res_all.device
    head_dim_v = sp_res_all.shape[-1] if sp_res_all.nelement() > 0 else 0

    # 保存lse的原始数据类型，用于后续恢复
    lse_original_dtype = sp_lse_all.dtype
    print_shape_log(rank, f"lse原始类型: {lse_original_dtype}, sp_res类型: {sp_res_all.dtype}")

    if not sp_group_key:
        empty_res = torch.empty((pad_sp,) + sp_res_all.shape[1:], dtype=sp_res_all.dtype, device=device)
        empty_lse = torch.empty((pad_sp,) + sp_lse_all.shape[1:], dtype=lse_original_dtype, device=device)
        print_shape_log(rank, f"all_gather_sp_results - 无SP通信组，返回空结果: {empty_res.shape}, {empty_lse.shape}")
        return empty_res, empty_lse

    comm = sp_comm_groups[sp_group_key]
    # 获取当前RANK在通信组内的索引
    rank_idx_in_group = sp_group_key.index(rank)
    print_shape_log(rank, f"all_gather_sp_results - 通信组{sp_group_key}，当前RANK索引: {rank_idx_in_group}")
    
    print_shape_log(rank, f"输入形状: sp_res_all={sp_res_all.shape}, sp_lse_all={sp_lse_all.shape}")
    
    # 统一维度：给sp_lse_all插入第1维，匹配sp_res_all的维度数
    sp_lse_reshaped = sp_lse_all.unsqueeze(1)  # [8,32,1] → [8,1,32,1]
    
    # 关键修改：将lse转换为与res相同的数据类型（bfloat16）以进行拼接
    sp_lse_converted = sp_lse_reshaped.to(sp_res_all.dtype)
    print_shape_log(rank, f"lse转换后类型: {sp_lse_converted.dtype}, 形状: {sp_lse_converted.shape}")
    
    # 拼接res和lse，此时两者数据类型一致
    combined = torch.cat([sp_res_all, sp_lse_converted], dim=-1)  # [8,1,32,513]
    print_shape_log(rank, f"拼接后combined.shape: {combined.shape}, dtype: {combined.dtype}")

    # 执行all gather：收集所有RANK的combined结果
    gathered = torch.empty(
        group_size,  # 第0维：通信组内RANK数量（如4）
        *combined.shape,  # 后续维度：[8,1,32,513]
        dtype=combined.dtype, 
        device=device
    )
    dist.all_gather_into_tensor(gathered.view(-1), combined.contiguous(), group=comm)
    print_shape_log(rank, f"all gather后gathered.shape: {gathered.shape}, dtype: {gathered.dtype}")

    # 提取当前RANK对应的SP结果
    current_rank_combined = gathered[rank_idx_in_group]  # [8,1,32,513]
    sp_slice_start = rank_idx_in_group * pad_sp
    sp_slice_end = sp_slice_start + pad_sp
    current_rank_sp_combined = current_rank_combined[sp_slice_start:sp_slice_end]  # [2,1,32,513]
    print_shape_log(rank, f"当前RANK对应SP切片范围: [{sp_slice_start}:{sp_slice_end}]，切片后shape: {current_rank_sp_combined.shape}")

    # 拆分res和lse
    current_rank_sp_res = current_rank_sp_combined[..., :head_dim_v]  # [2,1,32,512]
    current_rank_sp_lse = current_rank_sp_combined[..., head_dim_v:]  # [2,1,32,1]

    # 关键修改：将lse转换回原始数据类型（float32）
    current_rank_sp_lse = current_rank_sp_lse.to(lse_original_dtype)
    print_shape_log(rank, f"lse恢复后类型: {current_rank_sp_lse.dtype}")

    # 维度还原：移除lse中多余的第1维
    current_rank_sp_lse = current_rank_sp_lse.squeeze(1)  # [2,1,32,1] → [2,32,1]

    print_shape_log(rank, f"最终提取结果形状: 本地SP res={current_rank_sp_res.shape}({current_rank_sp_res.dtype}), 本地SP lse={current_rank_sp_lse.shape}({current_rank_sp_lse.dtype})")
    return current_rank_sp_res, current_rank_sp_lse


# ---------- 核心函数：在返回结果前确保lse维度正确 ----------
def flash_mla_fwd_sp(
    q: torch.Tensor,  # 形状为[pad_local + pad_sp, ...]
    k_cache: torch.Tensor,
    block_table: torch.Tensor,  # 形状为[pad_local + pad_sp * group_size, ...]
    cache_seqlens: torch.Tensor,  # 形状为[pad_local + pad_sp * group_size, ...]
    head_dim_v: int,
    sp_group_key: Any,
    sp_comm_groups: Dict,
    # 从prepare传递的参数
    pad_local: int,
    pad_sp: int,
    group_size: int,
    q_total_pad: int,
    bt_kv_total_pad: int,
    updated_metadata: torch.Tensor,
    updated_num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rank = dist.get_rank()
    
    print_shape_log(rank, f"\n===== flash_mla_fwd_sp 开始 =====")
    print_shape_log(rank, f"Q的padding后尺寸: {q.shape} (预期: {q_total_pad})")
    print_shape_log(rank, f"block_table的padding后尺寸: {block_table.shape} (预期: {bt_kv_total_pad})")
    print_shape_log(rank, f"cache_seqlens的padding后尺寸: {cache_seqlens.shape} (预期: {bt_kv_total_pad})")
    
    # 验证padding尺寸是否正确
    assert q.shape[0] == q_total_pad, f"Q的batch size错误，应为{q_total_pad}，实际为{q.shape[0]}"
    assert block_table.shape[0] == bt_kv_total_pad, f"block_table的batch size错误，应为{bt_kv_total_pad}，实际为{block_table.shape[0]}"
    assert cache_seqlens.shape[0] == bt_kv_total_pad, f"cache_seqlens的batch size错误，应为{bt_kv_total_pad}，实际为{cache_seqlens.shape[0]}"

    # 对Q执行all gather，使其尺寸变为pad_local + pad_sp * group_size
    all_q = all_gather_sp_q(
        q=q,
        pad_local=pad_local,
        pad_sp=pad_sp,
        sp_group_key=sp_group_key,
        sp_comm_groups=sp_comm_groups,
        group_size=group_size
    )
    print_shape_log(rank, f"all gather后Q的尺寸: {all_q.shape} (预期: {bt_kv_total_pad})")
    assert all_q.shape[0] == bt_kv_total_pad, f"all gather后Q尺寸错误，应为{bt_kv_total_pad}，实际为{all_q.shape[0]}"

    # MLA计算
    print_shape_log(rank, f"MLA输入 - key_value_cache: {k_cache.shape}, block_table: {block_table.shape}, cache_seqlens: {cache_seqlens.shape}")
    
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
    print_shape_log(rank, f"MLA输出: attn={attn_output.shape}, lse={lse.shape}, attn.dtype: {attn_output.dtype}, lse.dtype: {lse.dtype}")

    # 拆分结果：本地部分 + 全局SP部分
    local_res = attn_output[:pad_local]  # [pad_local, 1, 32, 512]
    sp_res_all = attn_output[pad_local:]  # [8, 1, 32, 512]（8=pad_sp*group_size）
    local_lse = lse[:pad_local]  # [pad_local, 32, 1]
    sp_lse_all = lse[pad_local:]  # [8, 32, 1]
    print_shape_log(rank, f"拆分结果: local_res={local_res.shape}, 全局SP_res={sp_res_all.shape}, 全局SP_lse={sp_lse_all.shape}")

    # 收集SP结果（调用修改后的函数，仅返回当前RANK的2个SP结果）
    local_sp_res, local_sp_lse = all_gather_sp_results(
        sp_res_all=sp_res_all,
        sp_lse_all=sp_lse_all,
        sp_group_key=sp_group_key,
        sp_comm_groups=sp_comm_groups,
        pad_sp=pad_sp,
        group_size=group_size
    )

    # 拼接所有结果（本地结果 + 当前RANK的SP结果）
    print_shape_log(rank, f"拼接前: local_res={local_res.shape}, local_sp_res={local_sp_res.shape}")
    print_shape_log(rank, f"拼接前: local_lse={local_lse.shape}, local_sp_lse={local_sp_lse.shape}")
    final_out_pad = torch.cat([local_res, local_sp_res], dim=0)  # [pad_local+2, 1, 32, 512]
    final_lse_pad = torch.cat([local_lse, local_sp_lse], dim=0)  # [pad_local+2, 32, 1]
    
    print_shape_log(rank, f"最终结果: out={final_out_pad.shape}, lse={final_lse_pad.shape}")
    print_shape_log(rank, f"===== flash_mla_fwd_sp 结束 =====\n")
    
    # 返回完整结果和掩码，由外部提取有效输出
    return final_out_pad, final_lse_pad


# ---------- 测试 ----------
def test_flash_mla(rank: int, world_size: int, args: argparse.Namespace):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    print_shape_log(rank, f"分布式初始化: rank={rank}, world_size={world_size}")

    # 配置
    dtype = torch.bfloat16
    num_query_heads, head_size, head_dim_v = 32, 576, 512
    block_size = 64
    NUM_BLOCKS_PER_RANK = 128 * 12 + 1  # 1537
    NUM_BLOCKS = NUM_BLOCKS_PER_RANK * world_size  # 1537*4=6148
    pad_local = 128  # 本地请求最大数量
    pad_sp = 2       # 每个RANK的SP请求最大数量（固定为2，匹配需求）
    print_shape_log(rank, f"配置: 头数={num_query_heads}, pad_local={pad_local}, pad_sp={pad_sp}")

    # 测试数据（确保每个RANK的SP请求数≤2）
    kv_lens_per_rank = [[2048]*6, [2048]*6, [2048]*6, [2048]*6]
    sp_groups_info_list = [
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': True, 'group': [0,1,2,3]}, 2: {'enabled': False}},  # RANK0：2个SP请求
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': True, 'group': [0,1,2,3]}, 2: {'enabled': False}},  # RANK1：2个SP请求
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': False}},  # RANK2：1个SP请求
        {0: {'enabled': False}}  # RANK3：0个SP请求
    ]
    current_sp_info = sp_groups_info_list[rank]
    q_batch_size = len(current_sp_info)

    # 生成测试张量
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

    # 通信组（4个RANK为一组，匹配测试配置）
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
    (local_batches, sp_batch_indices, sp_group_key, real_sp, cnt_list,
     q_pad, kv_lens_pad, block_table_pad,
     local_mask, sp_mask, pad_local, pad_sp, group_size,
     q_total_pad, bt_kv_total_pad,
     updated_metadata, updated_num_splits) = prepare_outputs

    # 执行MLA
    out_pad, lse_pad = flash_mla_fwd_sp(
        q=q_pad,
        k_cache=key_value_cache,
        block_table=block_table_pad,
        cache_seqlens=kv_lens_pad,
        head_dim_v=head_dim_v,
        sp_group_key=sp_group_key,
        sp_comm_groups=sp_comm_groups,
        pad_local=pad_local,
        pad_sp=pad_sp,
        group_size=group_size,
        q_total_pad=q_total_pad,
        bt_kv_total_pad=bt_kv_total_pad,
        updated_metadata=updated_metadata,
        updated_num_splits=updated_num_splits,
        softmax_scale=head_size ** -0.5,
        causal=True,
    )

    # 在外部提取有效输出（符合CUDA Graph要求）
    valid_local_out = out_pad[:pad_local][local_mask]  # 提取本地有效结果
    valid_local_lse = lse_pad[:pad_local][local_mask]
    valid_sp_out = out_pad[pad_local:][sp_mask]  # 提取当前RANK的有效SP结果（≤2个）
    valid_sp_lse = lse_pad[pad_local:][sp_mask]

    print_shape_log(rank, f"valid_local_out.shape: {valid_local_out.shape}, valid_local_lse.shape: {valid_local_lse.shape}, valid_sp_out.shape: {valid_sp_out.shape}, valid_sp_lse.shape: {valid_sp_lse.shape}")
    
    final_out = torch.cat([valid_local_out, valid_sp_out], dim=0)
    final_lse = torch.cat([valid_local_lse, valid_sp_lse], dim=0)

    # 验证结果（有效结果数=本地请求数+当前RANK的SP请求数）
    assert final_out.shape[0] == len(local_batches) + len(sp_batch_indices), \
           f"有效结果数量错误: 实际{final_out.shape[0]} vs 预期{len(local_batches) + len(sp_batch_indices)}"
    assert final_out.dtype == dtype, f"Dtype错误: {final_out.dtype} vs {dtype}"
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