import os
import torch
import torch.nn.functional as F
from typing import Optional, Dict
import torch.distributed as dist
from torch.multiprocessing import spawn

# -------------------------- 全局配置 --------------------------
NUM_HEADS     = 32
HEAD_DIM      = 576
KV_NUM_HEADS  = 8
KV_DIM        = HEAD_DIM
MAX_SEQ_LEN   = 1024
BATCH_SIZE    = 4

def empty_tensor(shape: tuple, device: torch.device) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.float32, device=device)

def pad_to_max_len(tensor: torch.Tensor, max_len: int, dim: int = 0) -> torch.Tensor:
    pad_len = max_len - tensor.shape[dim]
    if pad_len <= 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_len
    pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=dim)

def get_fixed_qkv(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """仅生成Q；K/V返回空张量保持接口兼容"""
    q = torch.zeros(BATCH_SIZE, NUM_HEADS, HEAD_DIM, device=device)
    for b in range(BATCH_SIZE):
        for h in range(NUM_HEADS):
            for d in range(HEAD_DIM):
                q[b, h, d] = b*100 + h*10 + d
    k = torch.empty(BATCH_SIZE, MAX_SEQ_LEN, KV_NUM_HEADS, KV_DIM, device=device)
    v = torch.empty(BATCH_SIZE, MAX_SEQ_LEN, KV_NUM_HEADS, KV_DIM, device=device)
    return q, k, v

def dummy_merge(results_list: list[torch.Tensor]) -> torch.Tensor:
    """参考示例代码的合并逻辑：对每个SP请求的所有Rank结果取平均"""
    if not results_list:
        raise ValueError("合并结果列表不能为空")
    # 堆叠所有Rank的结果并按Rank维度取平均（dim=0为Rank维度）
    return torch.stack(results_list).mean(dim=0)

# -------------------------- forward --------------------------
def forward(
    hidden_states: torch.Tensor,
    sp_groups_info: Optional[Dict] = None,
    sp_comm_groups: Optional[Dict] = None,
) -> tuple[torch.Tensor, Dict]:
    rank   = dist.get_rank()
    device = hidden_states.device
    meta   = {}

    print(f"\n========== [RANK {rank}] START ==========",flush=True)

    # 1) 获取 Q
    query_states, _, _ = get_fixed_qkv(device)
    print(f"[RANK {rank}] query_states  -> {query_states.shape}",flush=True)

    # 2) 拆分本地 / SP 请求
    local_batches, sp_batches = [], []
    for b in range(BATCH_SIZE):
        info = sp_groups_info.get(b, {'enabled': False})
        if not info['enabled']:
            local_batches.append(b)
        elif rank in info['group']:
            sp_batches.append((b, info))
    print(f"[RANK {rank}] local_batches = {local_batches}",flush=True)
    print(f"[RANK {rank}] sp_batches    = {[tpl[0] for tpl in sp_batches]}",flush=True)

    # 3) 建立通信组
    sp_groups = {}
    for b, info in sp_batches:
        key = tuple(sorted(info['group']))
        if key not in sp_groups:
            sp_groups[key] = []
        sp_groups[key].append((b, info))
    assert len(sp_groups) <= 1
    print(f"[RANK {rank}] sp_groups keys = {list(sp_groups.keys())}",flush=True)

    # 4) All-Gather Q
    all_sp_q, sp_batch_indices = [], {}
    for key, reqs in sp_groups.items():
        comm = sp_comm_groups.get(key)
        if comm is None:
            comm = dist.new_group(list(key))
            sp_comm_groups[key] = comm

        local_q_list = [query_states[b] for b, _ in reqs]
        local_q      = torch.stack(local_q_list) if local_q_list else \
                       torch.empty(0, NUM_HEADS, HEAD_DIM, device=device)
        print(f"[RANK {rank}] local_q for group {key} -> {local_q.shape}",flush=True)

        # 同步 batch 数
        local_cnt = torch.tensor([len(local_q_list)], dtype=torch.long, device=device)
        cnt_list  = [torch.empty_like(local_cnt) for _ in key]
        dist.all_gather(cnt_list, local_cnt, group=comm)
        cnt_list  = [c.item() for c in cnt_list]
        max_cnt   = max(cnt_list)
        print(f"[RANK {rank}] group {key} cnt_list={cnt_list}, max_cnt={max_cnt}",flush=True)

        # 填充+All-Gather
        padded_q = pad_to_max_len(local_q, max_cnt, dim=0)
        gathered = torch.empty(len(key), max_cnt, NUM_HEADS, HEAD_DIM,
                               dtype=padded_q.dtype, device=device)
        dist.all_gather_into_tensor(gathered.view(-1), padded_q.contiguous(), group=comm)
        print(f"[RANK {rank}] gathered_q shape -> {gathered.shape}",flush=True)

        # 扁平化
        flat = []
        for i, c in enumerate(cnt_list):
            if c > 0:
                flat.append(gathered[i, :c])
        flat = torch.cat(flat) if flat else torch.empty(0, NUM_HEADS, HEAD_DIM, device=device)
        print(f"[RANK {rank}] flat_sp_q after cat -> {flat.shape}",flush=True)

        all_sp_q.append(flat)
        # 记录当前Rank在该SP组中的批次索引（后续用于匹配SP请求）
        sp_batch_indices[key] = [b for b, _ in reqs]

        meta[key] = {
            'group_ranks': list(key),
            'master_rank': reqs[0][1]['master_rank'],
            'local_batch_count': len(local_q_list),  # 当前Rank的SP请求数（my_sp_cnt）
            'all_batch_counts': cnt_list,
            'is_master': (rank == reqs[0][1]['master_rank'])
        }

    # 5) 合并所有 Q
    local_q = query_states[local_batches] if local_batches else \
              torch.empty(0, NUM_HEADS, HEAD_DIM, device=device)
    sp_q    = torch.cat(all_sp_q, dim=0) if all_sp_q else \
              torch.empty(0, NUM_HEADS, HEAD_DIM, device=device)
    all_q   = torch.cat([local_q, sp_q], dim=0)
    print(f"[RANK {rank}] local_q={local_q.shape}, sp_q={sp_q.shape}, all_q={all_q.shape}",flush=True)

    # 6) 原 K/V 逻辑已删除
    # 7) 用一次矩阵乘占位（模拟注意力计算）
    attn_output = torch.randn(all_q.shape[0], NUM_HEADS, KV_DIM, device=device)
    print(f"[RANK {rank}] attn_output (GEMM) -> {attn_output.shape}",flush=True)

    # 8) 拆分结果
    local_cnt   = local_q.shape[0]
    local_res   = attn_output[:local_cnt]  # Local请求结果（无需合并）
    sp_res      = attn_output[local_cnt:]  # SP请求结果（需All2All后合并）
    print(f"[RANK {rank}] local_res={local_res.shape}, sp_res={sp_res.shape}",flush=True)

    # 9) All2All 拆分 SP 结果（核心修改部分）
    final_sp_parts = []
    sp_ptr = 0
    for key, m in meta.items():
        comm        = sp_comm_groups[key]
        ranks       = m['group_ranks']  # SP组内所有Rank
        all_cnts    = m['all_batch_counts']  # 组内每个Rank的原始SP请求数
        my_sp_cnt   = m['local_batch_count']  # 当前Rank的SP请求数（需合并的请求数）
        rank_idx    = ranks.index(rank)
        total_sp    = sum(all_cnts)  # 组内所有SP请求总数

        # 截取当前SP组对应的结果切片
        slice_sp = sp_res[sp_ptr:sp_ptr + total_sp]
        sp_ptr  += total_sp
        print(f"[RANK {rank}] slice_sp for {key} -> {slice_sp.shape}",flush=True)

        # 构造发送/接收长度（send_cnts：发给每个Rank的数量；recv_cnts：从每个Rank接收的数量）
        send_cnts = all_cnts  # 发给Rank i的数量 = Rank i的原始SP请求数
        recv_cnts = [all_cnts[rank_idx] for _ in ranks]  # 从每个Rank接收的数量 = 当前Rank的SP请求数
        print(f"[RANK {rank}] send_cnts={send_cnts}, recv_cnts={recv_cnts}",flush=True)

        # 构造发送列表（按Rank拆分slice_sp）
        send_list, pos = [], 0
        for c in send_cnts:
            end = pos + c
            if c > 0 and slice_sp.numel() > 0:
                send_list.append(slice_sp[pos:end].contiguous())
            else:
                send_list.append(torch.empty(0, NUM_HEADS, KV_DIM, device=device))
            pos = end

        # 构造接收列表（预分配内存）
        recv_list = []
        for c in recv_cnts:
            if c > 0:
                recv_tensor = torch.empty(c, NUM_HEADS, KV_DIM, dtype=torch.float32, device=device)
                recv_list.append(recv_tensor)
            else:
                recv_list.append(torch.empty(0, NUM_HEADS, KV_DIM, device=device))

        # 执行All2All通信（交换SP请求结果）
        dist.all_to_all(recv_list, send_list, group=comm)

        # -------------------------- 核心修改：SP请求结果合并 --------------------------
        merged_sp_results = []
        if my_sp_cnt > 0:
            # 1. 合并接收的所有非空张量（按Rank维度）
            sp_results_received = []
            for tensor in recv_list:
                if tensor.numel() > 0:
                    sp_results_received.append(tensor)
            sp_results_received = torch.cat(sp_results_received, dim=0) if sp_results_received else \
                                  torch.empty(0, NUM_HEADS, KV_DIM, device=device)
            print(f"[RANK {rank}] all2all received sp results -> {sp_results_received.shape}",flush=True)

            # 2. 为每个SP请求分配所有Rank的结果（按发送方Rank拆分）
            sp_request_results = [[] for _ in range(my_sp_cnt)]  # 每个请求对应一个结果列表
            current_pos = 0
            for i in range(len(ranks)):
                recv_c = recv_cnts[i]  # 从Rank i接收的数量
                if recv_c > 0:
                    end_pos = current_pos + recv_c
                    # 将Rank i的结果分配到对应请求的列表中
                    for req_idx in range(recv_c):
                        sp_request_results[req_idx].append(sp_results_received[current_pos + req_idx])
                    current_pos = end_pos

            # 3. 合并每个SP请求的所有Rank结果（使用dummy_merge）
            for req_idx in range(my_sp_cnt):
                merged = dummy_merge(sp_request_results[req_idx])
                merged_sp_results.append(merged)

            # 4. 转为张量（my_sp_cnt, NUM_HEADS, KV_DIM）
            merged_sp_tensor = torch.stack(merged_sp_results, dim=0)
            print(f"[RANK {rank}] merged sp results -> {merged_sp_tensor.shape}",flush=True)
            final_sp_parts.append(merged_sp_tensor)
        else:
            # 无SP请求时添加空张量
            final_sp_parts.append(torch.empty(0, NUM_HEADS, KV_DIM, device=device))

    # 10) 最终输出
    final_sp = torch.cat(final_sp_parts) if final_sp_parts else \
               torch.empty(0, NUM_HEADS, KV_DIM, device=device)
    final    = torch.cat([local_res, final_sp], dim=0)  # Local结果 + 合并后的SP结果
    print(f"[RANK {rank}] final_sp={final_sp.shape}, final={final.shape}",flush=True)
    print(f"========== [RANK {rank}] DONE ==========\n",flush=True)
    return final, meta

# -------------------------- 分布式测试 --------------------------
def init_distributed(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12362'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def verify_results(rank: int, final_output: torch.Tensor, sp_groups_info: Dict) -> bool:
    # 计算预期结果数量：local请求数 + 当前Rank的SP请求数
    expected_cnt = 0
    for b, info in sp_groups_info.items():
        if not info['enabled']:
            expected_cnt += 1  # Local请求
        elif rank in info['group']:
            expected_cnt += 1  # 当前Rank的SP请求
    expected_shape = (expected_cnt, NUM_HEADS, KV_DIM)
    
    # 校验形状和非零（排除全零无效结果）
    if final_output.shape != expected_shape:
        print(f"❌ Rank {rank} 期望 {expected_shape}, 实际 {final_output.shape}",flush=True)
        return False
    if final_output.numel() > 0 and torch.all(final_output == 0):
        print(f"❌ Rank {rank} 输出全零",flush=True)
        return False
    print(f"✅ Rank {rank} 校验通过",flush=True)
    return True

def test_forward(rank: int, world_size: int):
    init_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    hidden_states = torch.randn(BATCH_SIZE, MAX_SEQ_LEN, HEAD_DIM, device=device)

    # 每个Rank的SP组配置：Batch 0/1为SP请求（组[0,1,2,3]），Batch 2/3为Local请求
    sp_groups_info_list = [
        {0: {'enabled': True, 'group': [0,1,2,3], 'master_rank': 0},
         1: {'enabled': True, 'group': [0,1,2,3], 'master_rank': 0},
         2: {'enabled': False}, 3: {'enabled': False}},
        {0: {'enabled': True, 'group': [0,1,2,3], 'master_rank': 1},
         1: {'enabled': True, 'group': [0,1,2,3], 'master_rank': 1},
         2: {'enabled': False}, 3: {'enabled': False}},
        {0: {'enabled': True, 'group': [0,1,2,3], 'master_rank': 2},
         1: {'enabled': True, 'group': [0,1,2,3], 'master_rank': 2},
         2: {'enabled': False}, 3: {'enabled': False}},
        {0: {'enabled': True, 'group': [0,1,2,3], 'master_rank': 3},
         1: {'enabled': True, 'group': [0,1,2,3], 'master_rank': 3},
         2: {'enabled': False}, 3: {'enabled': False}},
    ]

    # 预创建SP通信组（避免重复创建）
    sp_comm_groups = {}
    if rank in [0,1,2,3]:
        sp_comm_groups[tuple(sorted([0,1,2,3]))] = dist.new_group(ranks=[0,1,2,3])

    # 执行forward
    final_output, _ = forward(
        hidden_states=hidden_states,
        sp_groups_info=sp_groups_info_list[rank],
        sp_comm_groups=sp_comm_groups
    )

    # 校验结果
    ok = verify_results(rank, final_output, sp_groups_info_list[rank])
    dist.barrier()

    # 汇总所有Rank的校验结果
    if rank == 0:
        all_ok = torch.tensor(1, device='cuda:0')
        for r in range(1, world_size):
            buf = torch.tensor(0, device='cuda:0')
            dist.recv(buf, src=r)
            all_ok &= buf
        print("\n🎉 所有Rank校验通过！" if all_ok else "\n❌ 部分Rank校验失败！",flush=True)
    else:
        dist.send(torch.tensor(1 if ok else 0, device='cuda'), dst=0)
    
    dist.barrier()
    dist.destroy_process_group()

def main():
    WORLD_SIZE = 4  # 4卡分布式测试
    spawn(fn=test_forward, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

if __name__ == '__main__':
    main()