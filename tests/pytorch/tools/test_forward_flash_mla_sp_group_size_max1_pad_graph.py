import os
import math
from typing import Optional, Tuple, Dict, List, Any
import argparse
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.multiprocessing import spawn
from flash_mla import flash_mla_with_kvcache, get_mla_metadata
import torch.profiler  # 导入Torch Profiler

PRINT_SHAPE_LOG = True
def print_shape_log(rank: int, content: str) -> None:
    if PRINT_SHAPE_LOG:
        print(f"[RANK {rank}] {content}", flush=True)

# 生成所有可能的pad_local值（2的幂次，从1到128）
def generate_pad_local_values():
    return [2**i for i in range(8) if 2**i <= 128]  # 1, 2, 4, 8, 16, 32, 64, 128

# 找到大于real_local的最小2次幂pad_local值
def find_matching_pad_local(real_local: int, pad_local_candidates: List[int]) -> int:
    for pad in sorted(pad_local_candidates):
        if pad >= real_local:
            return pad
    return pad_local_candidates[-1]  # 如果都小于，返回最大的

# ---------- 分离Padding逻辑：仅用于真实数据预处理 ----------
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
    bt_sp_pad = torch.zeros(pad_sp * group_size, block_num, dtype=block_table.dtype, device=device) if block_num > 0 else torch.empty(0, block_num, dtype=block_table.dtype, device=device)
    
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

# 仅在真实数据预处理（replay前）调用
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
    device = q.device
    num_head, head_dim = q.shape[1], q.shape[2]
    
    q_local = q[:pad_local]
    q_sp = q[pad_local:]
    
    gathered_sp = torch.empty(group_size, pad_sp, num_head, head_dim, dtype=q.dtype, device=device)
    dist.all_gather_into_tensor(gathered_sp.view(-1), q_sp.contiguous(), group=sp_comm_group)
    
    sp_q_all = gathered_sp.reshape(-1, num_head, head_dim)  # 形状为[pad_sp * group_size, ...]
    
    all_q = torch.cat([q_local, sp_q_all], dim=0)
    
    return all_q


def all_gather_sp_results(
    sp_res_all: torch.Tensor,  # 形状: [pad_sp*group_size, 1, ...]（bfloat16）
    sp_lse_all: torch.Tensor,  # 形状: [pad_sp*group_size, ...]（float32）
    sp_comm_group: Optional[ProcessGroup],  # 直接传递通信组对象
    pad_sp: int,
    group_size: int,
    rank_idx_in_group: int  # 提前从prepare获取的组内索引（无分支计算）
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = sp_res_all.device
    head_dim_v = sp_res_all.shape[-1] if sp_res_all.nelement() > 0 else 0

    lse_original_dtype = sp_lse_all.dtype

    # 统一维度：给sp_lse_all插入第1维
    sp_lse_reshaped = sp_lse_all.unsqueeze(1)
    sp_lse_converted = sp_lse_reshaped.to(sp_res_all.dtype)
    
    combined = torch.cat([sp_res_all, sp_lse_converted], dim=-1)

    gathered = torch.empty(
        group_size,
        *combined.shape,
        dtype=combined.dtype, 
        device=device
    )
    dist.all_gather_into_tensor(gathered.view(-1), combined.contiguous(), group=sp_comm_group)

    current_rank_combined = gathered[rank_idx_in_group]
    sp_slice_start = rank_idx_in_group * pad_sp
    sp_slice_end = sp_slice_start + pad_sp
    current_rank_sp_combined = current_rank_combined[sp_slice_start:sp_slice_end]

    current_rank_sp_res = current_rank_sp_combined[..., :head_dim_v]
    current_rank_sp_lse = current_rank_sp_combined[..., head_dim_v:]

    current_rank_sp_lse = current_rank_sp_lse.to(lse_original_dtype)
    current_rank_sp_lse = current_rank_sp_lse.squeeze(1)

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
    updated_metadata: torch.Tensor,
    updated_num_splits: torch.Tensor,
    rank_idx_in_group: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:

    all_q = all_gather_sp_q(
        q=q,
        pad_local=pad_local,
        pad_sp=pad_sp,
        sp_comm_group=sp_comm_group,
        group_size=group_size
    )

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

    local_res = attn_output[:pad_local]  # [pad_local, 1, ...]
    sp_res_all = attn_output[pad_local:]  # [pad_sp*group_size, 1, ...]
    local_lse = lse[:pad_local]  # [pad_local, ...]
    sp_lse_all = lse[pad_local:]  # [pad_sp*group_size, ...]

    local_sp_res, local_sp_lse = all_gather_sp_results(
        sp_res_all=sp_res_all,
        sp_lse_all=sp_lse_all,
        sp_comm_group=sp_comm_group,
        pad_sp=pad_sp,
        group_size=group_size,
        rank_idx_in_group=rank_idx_in_group
    )

    final_out_pad = torch.cat([local_res, local_sp_res], dim=0)  # [pad_local+pad_sp, 1, ...]
    final_lse_pad = torch.cat([local_lse, local_sp_lse], dim=0)  # [pad_local+pad_sp, ...]
    
    return final_out_pad, final_lse_pad

# 生成完整的dummy数据（包含所有需要的参数，无需调用prepare_mla_fwd）
def generate_dummy_data_and_params(rank: int, device: torch.device, pad_local: int, pad_sp: int, 
                                  num_query_heads: int, head_size: int, 
                                  NUM_BLOCKS: int, block_size: int, 
                                  NUM_BLOCKS_PER_RANK: int, group_size: int,
                                  sp_comm_group: ProcessGroup, rank_idx_in_group: int):
    dtype = torch.bfloat16
    
    # Dummy查询数据 - 形状：[pad_local + pad_sp, num_query_heads, head_size]
    dummy_q_batch_size = pad_local + pad_sp
    dummy_query = torch.randn(dummy_q_batch_size, num_query_heads, head_size, dtype=dtype, device=device) / 10
    
    # Dummy KV缓存
    dummy_key_value_cache = torch.randn(NUM_BLOCKS, block_size, 1, head_size, dtype=dtype, device=device)
    
    # Dummy KV长度和block table - 形状：[pad_local + pad_sp * group_size, ...]
    dummy_kv_batch_size = pad_local + pad_sp * group_size
    dummy_kv_lens = torch.randint(1, 2048, (dummy_kv_batch_size,), dtype=torch.int32, device=device)
    dummy_block_tables = torch.randint(0, NUM_BLOCKS_PER_RANK,
                                     (dummy_kv_batch_size, NUM_BLOCKS_PER_RANK),
                                     dtype=torch.int32, device=device)
    
    # 生成dummy mask（全为True，因为dummy数据全部有效）
    local_mask = torch.ones(pad_local, dtype=torch.bool, device=device)
    sp_mask = torch.ones(pad_sp, dtype=torch.bool, device=device)
    
    # 生成dummy metadata
    updated_metadata, updated_num_splits = get_mla_metadata(dummy_kv_lens, num_query_heads, 1)
    
    return (dummy_query, dummy_key_value_cache, dummy_kv_lens, dummy_block_tables,
            local_mask, sp_mask, updated_metadata, updated_num_splits)


def test_flash_mla(rank: int, world_size: int, args: argparse.Namespace):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    print_shape_log(rank, f"分布式初始化: rank={rank}, world_size={world_size}")

    # 创建profiler结果目录（如果不存在）
    if args.use_profiler and rank == 0:  # 只在主进程创建目录
        os.makedirs(args.profiler_path, exist_ok=True)
        print_shape_log(rank, f"Profiler结果将保存到: {args.profiler_path}")

    # 配置参数
    dtype = torch.bfloat16
    num_query_heads, head_size, head_dim_v = 32, 576, 512
    block_size = 64
    NUM_BLOCKS_PER_RANK = 128 * 12 + 1  # 1537
    NUM_BLOCKS = NUM_BLOCKS_PER_RANK * world_size  # 1537*4=6148
    pad_sp = 2  # 固定为2
    pad_local_candidates = generate_pad_local_values()  # [1, 2, 4, 8, 16, 32, 64, 128]
    num_replays = 10  # 重放次数
    print_shape_log(rank, f"配置: 头数={num_query_heads}, pad_local候选值={pad_local_candidates}, pad_sp={pad_sp}, 重放次数={num_replays}")

    # ---------- 使用用户指定的初始配置 ----------
    # KV长度配置
    kv_lens_per_rank = [
        [2048]*7, 
        [2048]*7, 
        [2048]*7, 
        [2048]*8
    ]
    
    # SP组信息配置
    sp_groups_info_list = [
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': True, 'group': [0,1,2,3]}, 2: {'enabled': False}, 3: {'enabled': False}},  # RANK0：2个SP请求
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': True, 'group': [0,1,2,3]}, 2: {'enabled': False}, 3: {'enabled': False}},  # RANK1：2个SP请求
        {0: {'enabled': True, 'group': [0,1,2,3]}, 1: {'enabled': False}, 2: {'enabled': False}},  # RANK2：1个SP请求
        {0: {'enabled': False}, 1: {'enabled': False}, 2: {'enabled': False}}  # RANK3：0个SP请求
    ]
    
    # 通信组配置
    sp_comm_groups = {tuple(sorted([0,1,2,3])): dist.new_group([0,1,2,3])}
    sp_group_key = next(iter(sp_comm_groups.keys()))
    group_size = len(sp_group_key) if sp_group_key else 1
    sp_comm_group = sp_comm_groups[sp_group_key] if sp_group_key else None
    rank_idx_in_group = sp_group_key.index(rank) if sp_group_key else 0
    
    # 获取当前rank的配置
    current_sp_info = sp_groups_info_list[rank]
    q_batch_size = len(current_sp_info)
    kv_lens_this_rank = kv_lens_per_rank[rank]
    kv_batch_size = len(kv_lens_this_rank)

    # 计算当前rank的real_local和real_sp
    local_batches = [b for b in range(q_batch_size) if not current_sp_info.get(b, {}).get("enabled", False)]
    sp_batch_indices = [b for b, info in current_sp_info.items() if info.get("enabled", False) and rank in info.get("group", [])]
    real_local = len(local_batches)
    real_sp = len(sp_batch_indices)
    print_shape_log(rank, f"当前配置: real_local={real_local}, real_sp={real_sp}")

    # -------------------------- 1. 捕获多个CUDA Graph（仅使用dummy数据，不调用prepare_mla_fwd） --------------------------
    # 存储所有捕获的graph，键为(pad_local, pad_sp)
    mla_graphs = {}
    # 存储每个graph对应的输出缓冲区和输入缓冲区
    graph_buffers = {}
    
    print_shape_log(rank, "\n开始捕获多个CUDA Graph（不调用prepare_mla_fwd）...")
    
    for pad_local in pad_local_candidates:
        # 打印当前捕获的pad_local和pad_sp值
        print_shape_log(rank, f"\n=== 开始捕获 Graph (pad_local={pad_local}, pad_sp={pad_sp}) ===")
        
        # 生成完整的dummy数据和参数（无需调用prepare_mla_fwd）
        (dummy_query, dummy_key_value_cache, dummy_kv_lens, dummy_block_tables,
         local_mask, sp_mask, dummy_updated_metadata, dummy_updated_num_splits) = generate_dummy_data_and_params(
            rank, device, pad_local, pad_sp, num_query_heads, head_size,
            NUM_BLOCKS, block_size, NUM_BLOCKS_PER_RANK, group_size,
            sp_comm_group, rank_idx_in_group
        )
        print_shape_log(rank, f"  dummy数据形状: query={dummy_query.shape}, kv_lens={dummy_kv_lens.shape}, block_table={dummy_block_tables.shape}")
        
        # 整理flash_mla_fwd_sp的参数
        flash_kwargs = {
            "k_cache": dummy_key_value_cache,
            "block_table": dummy_block_tables,
            "cache_seqlens": dummy_kv_lens,
            "head_dim_v": head_dim_v,
            "sp_comm_group": sp_comm_group,
            "pad_local": pad_local,
            "pad_sp": pad_sp,
            "group_size": group_size,
            "updated_metadata": dummy_updated_metadata,
            "updated_num_splits": dummy_updated_num_splits,
            "rank_idx_in_group": rank_idx_in_group,
            "softmax_scale": head_size ** -0.5,
            "causal": True,
        }
        
        # 初始化Graph：预热 + 捕获
        def init_graph(input_q: torch.Tensor) -> Tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]:
            # 预热
            print_shape_log(rank, "  开始预热...")
            for _ in range(2):
                warmup_out, warmup_lse = flash_mla_fwd_sp(q=input_q, **flash_kwargs)
                torch.cuda.synchronize()
            print_shape_log(rank, "  预热完成")
            
            # 捕获
            print_shape_log(rank, "  开始捕获Graph...")
            graph = torch.cuda.CUDAGraph()
            input_q.requires_grad_(False)
            with torch.cuda.graph(graph):
                captured_out, captured_lse = flash_mla_fwd_sp(q=input_q,** flash_kwargs)
            torch.cuda.synchronize()
            print_shape_log(rank, f"  Graph捕获完成 (pad_local={pad_local}, pad_sp={pad_sp})")
            
            return graph, captured_out, captured_lse
        
        # 捕获graph（直接使用dummy_query作为输入，无需预处理）
        graph, out_buffer, lse_buffer = init_graph(dummy_query)
        
        # 存储graph和对应的缓冲区
        key = (pad_local, pad_sp)
        mla_graphs[key] = graph
        graph_buffers[key] = (
            out_buffer, lse_buffer, 
            dummy_query, dummy_kv_lens, dummy_block_tables,
            dummy_updated_metadata, dummy_updated_num_splits,
            dummy_key_value_cache, local_mask, sp_mask
        )
        
        # 每次捕获完成后同步所有进程
        print_shape_log(rank, f"  完成当前Graph (pad_local={pad_local}, pad_sp={pad_sp}) 捕获，等待所有进程同步...")
        dist.barrier()
        print_shape_log(rank, f"  所有进程已完成当前Graph (pad_local={pad_local}, pad_sp={pad_sp}) 捕获，继续执行")
    
    print_shape_log(rank, "\n所有CUDA Graph捕获完成")

    # -------------------------- 2. 准备真实数据并预处理（只执行一次） --------------------------
    # 生成真实查询数据（只生成一次，所有重放使用相同输入）
    real_query = torch.randn(q_batch_size, num_query_heads, head_size, dtype=dtype, device=device) / 10
    real_key_value_cache = torch.randn(NUM_BLOCKS, block_size, 1, head_size, dtype=dtype, device=device)
    real_kv_lens_tensor = torch.tensor(kv_lens_this_rank, dtype=torch.int32, device=device) if kv_batch_size > 0 else \
                         torch.empty(0, dtype=torch.int32, device=device)
    real_block_tables = torch.randint(0, NUM_BLOCKS_PER_RANK,
                                    (kv_batch_size, NUM_BLOCKS_PER_RANK),
                                    dtype=torch.int32, device=device) if kv_batch_size > 0 else \
                       torch.empty(0, NUM_BLOCKS_PER_RANK, dtype=torch.int32, device=device)

    # -------------------------- 3. 确定匹配的graph --------------------------
    matched_pad_local = find_matching_pad_local(real_local, pad_local_candidates)
    graph_key = (matched_pad_local, pad_sp)
    print_shape_log(rank, f"\n找到匹配的graph: {graph_key} (pad_local={matched_pad_local}, pad_sp={pad_sp})")
    
    if graph_key not in mla_graphs:
        print_shape_log(rank, f"错误: 未找到匹配的graph {graph_key}")
        dist.destroy_process_group()
        return
    
    # 获取选中的graph和缓冲区
    selected_graph = mla_graphs[graph_key]
    (out_buffer, lse_buffer, 
     dummy_query, dummy_kv_lens, dummy_block_tables,
     dummy_updated_metadata, dummy_updated_num_splits,
     dummy_key_value_cache, local_mask, sp_mask) = graph_buffers[graph_key]

    # 对真实数据调用prepare_mla_fwd进行预处理（只执行一次）
    print_shape_log(rank, "\n对真实数据调用prepare_mla_fwd进行预处理...")
    real_prepare_outputs = prepare_mla_fwd(
        rank=rank,
        q=real_query,
        kv_lens=real_kv_lens_tensor,
        block_table=real_block_tables,
        q_batch_size=q_batch_size,
        num_query_heads=num_query_heads,
        sp_groups_info_list=sp_groups_info_list,
        sp_comm_groups=sp_comm_groups,
        pad_local=matched_pad_local,
        pad_sp=pad_sp
    )
    
    (local_batches, sp_batch_indices, real_sp, cnt_list,
     real_q_pad, real_kv_lens_pad, real_block_table_pad,
     real_local_mask, real_sp_mask, _, _, _,
     _, _,
     real_updated_metadata, real_updated_num_splits,
     real_sp_comm_group, real_rank_idx_in_group) = real_prepare_outputs

    # 将真实数据复制到dummy缓冲区（只执行一次）
    print_shape_log(rank, "复制真实数据到dummy缓冲区...")
    dummy_query.copy_(real_q_pad)
    dummy_kv_lens.copy_(real_kv_lens_pad)
    dummy_block_tables.copy_(real_block_table_pad)
    dummy_updated_metadata.copy_(real_updated_metadata)
    dummy_updated_num_splits.copy_(real_updated_num_splits)
    dummy_key_value_cache.copy_(real_key_value_cache)
    torch.cuda.synchronize()

    # 预处理完成后同步所有进程
    print_shape_log(rank, "预处理完成，等待所有进程同步...")
    dist.barrier()
    print_shape_log(rank, "所有进程已完成预处理，开始重放...")

    # -------------------------- 4. 对相同配置进行多次重放（使用固定输入，Profiler只包裹replay） --------------------------
    print_shape_log(rank, f"\n开始对相同配置进行{num_replays}次重放，使用graph: {graph_key}...")
    
    # 准备profiler
    profiler = None
    profile_path = None
    if args.use_profiler:
        # 为每个rank创建不同的文件名，避免冲突
        profile_filename = f"profile_rank_{rank}"
        profile_path = os.path.join(args.profiler_path, profile_filename)
        
        print_shape_log(rank, f"启动Torch Profiler，结果将保存到: {profile_path}")
        
        # 配置profiler：等待1次迭代，热身2次，记录3次
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=2,
                active=3,
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        )
        
        # 启动profiler
        profiler.start()

    try:
        for replay_idx in range(num_replays):
            print_shape_log(rank, f"\n=== 第{replay_idx+1}次重放 ===")
            # 打印当前使用的graph key
            print_shape_log(rank, f"  使用graph: {graph_key} (pad_local={graph_key[0]}, pad_sp={graph_key[1]})")
            
            # 如果启用了profiler，在replay前准备记录
            if args.use_profiler:
                profiler.step()  # 通知profiler进入下一步
            
            # 重放选中的graph（Profiler只包裹这部分）
            print_shape_log(rank, "  执行Graph重放...")
            if args.use_profiler:
                with torch.profiler.record_function("graph_replay"):  # 添加标记
                    selected_graph.replay()
            else:
                selected_graph.replay()
            torch.cuda.synchronize()
            
            # 提取有效结果并验证
            valid_local_out = out_buffer[:matched_pad_local][real_local_mask]
            valid_local_lse = lse_buffer[:matched_pad_local][real_local_mask]
            valid_sp_out = out_buffer[matched_pad_local:][real_sp_mask]
            valid_sp_lse = lse_buffer[matched_pad_local:][real_sp_mask]
            
            final_out = torch.cat([valid_local_out, valid_sp_out], dim=0)
            final_lse = torch.cat([valid_local_lse, valid_sp_lse], dim=0)
            
            assert final_out.shape[0] == len(local_batches) + len(sp_batch_indices), \
                   f"[RANK {rank}] 有效结果数量错误: 实际{final_out.shape[0]} vs 预期{len(local_batches) + len(sp_batch_indices)}"
            assert final_out.dtype == dtype, f"[RANK {rank}] Dtype错误: {final_out.dtype} vs {dtype}"
            
            print_shape_log(rank, f"  重放结果：valid_local_out.shape={valid_local_out.shape}, valid_sp_out.shape={valid_sp_out.shape}")
            print_shape_log(rank, f"  第{replay_idx+1}次重放 ✅ 验证通过")
            
            # 每次重放完成后同步所有进程
            print_shape_log(rank, f"  完成第{replay_idx+1}次重放，等待所有进程同步...")
            dist.barrier()
            print_shape_log(rank, f"  所有进程已完成第{replay_idx+1}次重放，继续执行")
    finally:
        # 停止profiler（确保资源正确释放）
        if args.use_profiler and profiler is not None:
            profiler.stop()
            # 打印保存完成信息，包含具体路径
            print_shape_log(rank, f"Torch Profiler已停止，结果已保存到: {profile_path}")

    # 清理分布式资源
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Flash MLA SP（添加详细打印信息）")
    parser.add_argument("--num_ranks", type=int, default=4)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="12355")
    # 添加新参数：是否使用profiler
    parser.add_argument("--use_profiler", action="store_true", default=False, 
                        help="是否使用torch profiler进行性能分析，默认为False")
    # 添加新参数：profiler结果存储路径
    parser.add_argument("--profiler_path", type=str, 
                        default="/nvme2/share/chenjiefei/src/lmdeploy/tests/torch_profiler_res",
                        help="profiler结果的存储路径，默认为/nvme2/share/chenjiefei/src/lmdeploy/tests/torch_profiler_res")
    args = parser.parse_args()

    assert args.num_ranks == 4, "仅支持4个进程（匹配SP通信组配置）"
    print(f"启动测试: {args.num_ranks}进程, 地址={args.master_addr}:{args.master_port}")
    if args.use_profiler:
        print(f"将使用Torch Profiler，结果将保存到: {args.profiler_path}")
    spawn(fn=test_flash_mla, args=(args.num_ranks, args), nprocs=args.num_ranks, join=True)
    print("所有进程完成")


if __name__ == "__main__":
    main()
