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

def get_fixed_qkv(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """仅生成Q；K/V返回空张量保持接口兼容"""
    q = torch.zeros(batch_size, NUM_HEADS, HEAD_DIM, device=device)
    for b in range(batch_size):
        for h in range(NUM_HEADS):
            for d in range(HEAD_DIM):
                q[b, h, d] = b*100 + h*10 + d
    k = torch.empty(batch_size, MAX_SEQ_LEN, KV_NUM_HEADS, KV_DIM, device=device)
    v = torch.empty(batch_size, MAX_SEQ_LEN, KV_NUM_HEADS, KV_DIM, device=device)
    return q, k, v

def dummy_merge(results_list: list[torch.Tensor]) -> torch.Tensor:
    """参考示例代码的合并逻辑：对每个SP请求的所有Rank结果取平均"""
    if not results_list:
        raise ValueError("合并结果列表不能为空")
    return torch.stack(results_list).mean(dim=0)

# -------------------------- forward（核心修改） --------------------------
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
    query_states, _, _ = get_fixed_qkv(hidden_states.shape[0], device)
    print(f"[RANK {rank}] query_states  -> {query_states.shape}",flush=True)

    # 2) 拆分本地 / SP 请求
    local_batches, sp_batches = [], []
    for b in range(query_states.shape[0]):
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

    # -------------------------- 关键修改1：获取所有已存在的SP通信组Key --------------------------
    # 即使无SP请求，也需处理sp_comm_groups中所有SP组（避免通信遗漏）
    all_sp_group_keys = list(sp_comm_groups.keys()) if sp_comm_groups else []
    # 补充当前Rank自身的SP组Key（防止sp_comm_groups未预创建）
    for key in sp_groups.keys():
        if key not in all_sp_group_keys:
            all_sp_group_keys.append(key)
    print(f"[RANK {rank}] 需处理的所有SP组 = {all_sp_group_keys}",flush=True)

    # 4) All-Gather Q：遍历所有SP组（无论是否有本地SP请求）
    all_sp_q, sp_batch_indices = [], {}
    for key in all_sp_group_keys:
        # 获取/创建通信组
        comm = sp_comm_groups.get(key)
        if comm is None:
            comm = dist.new_group(list(key))
            sp_comm_groups[key] = comm
        group_ranks = list(key)

        # -------------------------- 关键修改2：补全本地SP数据（无请求则为空） --------------------------
        reqs = sp_groups.get(key, [])  # 无请求则为空列表
        local_q_list = [query_states[b] for b, _ in reqs] if reqs else []
        local_q = torch.stack(local_q_list) if local_q_list else \
                  torch.empty(0, NUM_HEADS, HEAD_DIM, device=device)
        local_cnt = len(local_q_list)  # 0或正整数
        print(f"[RANK {rank}] SP组 {key} - local_q shape: {local_q.shape}, 本地请求数: {local_cnt}",flush=True)

        # 同步组内所有Rank的请求数（所有Rank必须参与，即使local_cnt=0）
        local_cnt_tensor = torch.tensor([local_cnt], dtype=torch.long, device=device)
        cnt_list_tensor = [torch.empty_like(local_cnt_tensor) for _ in group_ranks]
        dist.all_gather(cnt_list_tensor, local_cnt_tensor, group=comm)
        cnt_list = [t.item() for t in cnt_list_tensor]
        max_cnt = max(cnt_list) if cnt_list else 0
        print(f"[RANK {rank}] SP组 {key} - 组内各Rank请求数: {cnt_list}, 最大长度: {max_cnt}",flush=True)

        # 填充空张量（确保所有Rank发送数据长度一致）
        padded_q = pad_to_max_len(local_q, max_cnt, dim=0)
        # All-Gather：所有Rank必须执行（空数据也需发送）
        gathered = torch.empty(len(group_ranks), max_cnt, NUM_HEADS, HEAD_DIM,
                               dtype=padded_q.dtype, device=device)
        dist.all_gather_into_tensor(gathered.view(-1), padded_q.contiguous(), group=comm)
        print(f"[RANK {rank}] SP组 {key} - gathered_q shape: {gathered.shape}",flush=True)

        # 扁平化SP请求（过滤空数据）
        flat = []
        for i, c in enumerate(cnt_list):
            if c > 0:
                flat.append(gathered[i, :c])
        flat = torch.cat(flat) if flat else torch.empty(0, NUM_HEADS, HEAD_DIM, device=device)
        all_sp_q.append(flat)
        sp_batch_indices[key] = [b for b, _ in reqs] if reqs else []

        # -------------------------- 关键修改3：补全SP组元信息（无请求也需记录） --------------------------
        master_rank = reqs[0][1]['master_rank'] if reqs else group_ranks[0]
        meta[key] = {
            'group_ranks': group_ranks,
            'master_rank': master_rank,
            'local_batch_count': local_cnt,  # 当前Rank的SP请求数（0或正整数）
            'all_batch_counts': cnt_list,    # 组内所有Rank的请求数
            'is_master': (rank == master_rank)
        }

    # 5) 合并所有 Q
    local_q = query_states[local_batches] if local_batches else \
              torch.empty(0, NUM_HEADS, HEAD_DIM, device=device)
    sp_q    = torch.cat(all_sp_q, dim=0) if all_sp_q else \
              torch.empty(0, NUM_HEADS, HEAD_DIM, device=device)
    all_q   = torch.cat([local_q, sp_q], dim=0)
    print(f"[RANK {rank}] local_q={local_q.shape}, sp_q={sp_q.shape}, all_q={all_q.shape}",flush=True)

    # 6) 模拟注意力计算（占位）
    attn_output = torch.randn(all_q.shape[0], NUM_HEADS, KV_DIM, device=device)
    print(f"[RANK {rank}] attn_output shape: {attn_output.shape}",flush=True)

    # 7) 拆分结果
    local_cnt   = local_q.shape[0]
    local_res   = attn_output[:local_cnt]
    sp_res      = attn_output[local_cnt:]
    print(f"[RANK {rank}] local_res={local_res.shape}, sp_res={sp_res.shape}",flush=True)

    # 8) All2All 拆分SP结果：遍历所有SP组（无论是否有本地SP请求）
    final_sp_parts = []
    sp_ptr = 0
    for key in all_sp_group_keys:
        comm = sp_comm_groups[key]
        m = meta[key]
        group_ranks = m['group_ranks']
        all_cnts = m['all_batch_counts']
        my_sp_cnt = m['local_batch_count']
        rank_idx = group_ranks.index(rank)
        total_sp = sum(all_cnts)

        # -------------------------- 关键修改4：补全SP结果切片（无请求则为空） --------------------------
        slice_sp = sp_res[sp_ptr:sp_ptr + total_sp] if total_sp > 0 else \
                   torch.empty(0, NUM_HEADS, KV_DIM, device=device)
        sp_ptr += total_sp
        print(f"[RANK {rank}] SP组 {key} - slice_sp shape: {slice_sp.shape}",flush=True)

        # 构造发送/接收长度（无请求时recv_cnts=0）
        send_cnts = all_cnts
        recv_cnts = [my_sp_cnt for _ in group_ranks]
        print(f"[RANK {rank}] SP组 {key} - send_cnts={send_cnts}, recv_cnts={recv_cnts}",flush=True)

        # 构造发送列表（空数据也需构造空张量）
        send_list, pos = [], 0
        for c in send_cnts:
            end = pos + c
            send_tensor = slice_sp[pos:end].contiguous() if (c > 0 and slice_sp.numel() > 0) else \
                          torch.empty(0, NUM_HEADS, KV_DIM, device=device)
            send_list.append(send_tensor)
            pos = end

        # 构造接收列表（空数据也需构造空张量）
        recv_list = []
        for c in recv_cnts:
            recv_tensor = torch.empty(c, NUM_HEADS, KV_DIM, device=device) if c > 0 else \
                          torch.empty(0, NUM_HEADS, KV_DIM, device=device)
            recv_list.append(recv_tensor)

        # -------------------------- 关键修改5：所有Rank必须执行All-to-All --------------------------
        dist.all_to_all(recv_list, send_list, group=comm)

        # 合并SP结果（无请求则添加空张量）
        merged_sp_tensor = torch.empty(0, NUM_HEADS, KV_DIM, device=device)
        if my_sp_cnt > 0:
            # 合并接收的非空张量
            sp_results_received = [t for t in recv_list if t.numel() > 0]
            sp_results_received = torch.cat(sp_results_received, dim=0) if sp_results_received else \
                                  torch.empty(0, NUM_HEADS, KV_DIM, device=device)
            print(f"[RANK {rank}] SP组 {key} - 接收结果 shape: {sp_results_received.shape}",flush=True)

            # 分配并合并每个SP请求的结果
            sp_request_results = [[] for _ in range(my_sp_cnt)]
            current_pos = 0
            for i in range(len(group_ranks)):
                recv_c = recv_cnts[i]
                if recv_c > 0:
                    end_pos = current_pos + recv_c
                    for req_idx in range(recv_c):
                        sp_request_results[req_idx].append(sp_results_received[current_pos + req_idx])
                    current_pos = end_pos

            merged_sp_results = [dummy_merge(req_res) for req_res in sp_request_results]
            merged_sp_tensor = torch.stack(merged_sp_results, dim=0)
            print(f"[RANK {rank}] SP组 {key} - 合并结果 shape: {merged_sp_tensor.shape}",flush=True)
        else:
            print(f"[RANK {rank}] SP组 {key} - 无SP请求，跳过合并",flush=True)

        final_sp_parts.append(merged_sp_tensor)

    # 9) 最终输出
    final_sp = torch.cat(final_sp_parts) if final_sp_parts else \
               torch.empty(0, NUM_HEADS, KV_DIM, device=device)
    final    = torch.cat([local_res, final_sp], dim=0)
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
    # 计算预期结果数量：本地请求数 + 当前Rank的SP请求数
    expected_cnt = 0
    for b, info in sp_groups_info.items():
        if not info['enabled']:
            expected_cnt += 1
        elif rank in info['group']:
            expected_cnt += 1
    expected_shape = (expected_cnt, NUM_HEADS, KV_DIM)

    # 校验形状和有效性
    if final_output.shape != expected_shape:
        print(f"❌ Rank {rank} 形状不匹配：期望 {expected_shape}, 实际 {final_output.shape}",flush=True)
        return False
    if final_output.numel() > 0 and torch.all(final_output == 0):
        print(f"❌ Rank {rank} 输出全零（无效）",flush=True)
        return False
    print(f"✅ Rank {rank} 校验通过",flush=True)
    return True

def test_forward(rank: int, world_size: int):
    init_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # 测试用例：Rank 3无SP请求（仅2个本地请求）
    sp_groups_info_list = [
        {0: {'enabled': True, 'group': [0,1,2,3], 'master_rank': 0},
         1: {'enabled': True, 'group': [0,1,2,3], 'master_rank': 0},
         2: {'enabled': False}, 3: {'enabled': False}},  # Rank 0：2SP+2Local
        {0: {'enabled': True, 'group': [0,1,2,3], 'master_rank': 1},
         1: {'enabled': True, 'group': [0,1,2,3], 'master_rank': 1},
         2: {'enabled': False}, 3: {'enabled': False}},  # Rank 1：2SP+2Local
        {0: {'enabled': True, 'group': [0,1,2,3], 'master_rank': 2},
         1: {'enabled': True, 'group': [0,1,2,3], 'master_rank': 2},
         2: {'enabled': False}, 3: {'enabled': False}},  # Rank 2：2SP+2Local
        {0: {'enabled': False}, 1: {'enabled': False}}   # Rank 3：0SP+2Local（关键测试）
    ]

    # 生成输入数据
    bs_current_rank = len(sp_groups_info_list[rank])
    hidden_states = torch.randn(bs_current_rank, MAX_SEQ_LEN, HEAD_DIM, device=device)

    # 预创建SP通信组（所有Rank都需知道SP组[0,1,2,3]）
    sp_comm_groups = {}
    sp_group_key = tuple(sorted([0,1,2,3]))
    if sp_group_key not in sp_comm_groups:
        comm = dist.new_group(ranks=[0,1,2,3])
        sp_comm_groups[sp_group_key] = comm

    # 执行forward
    final_output, _ = forward(
        hidden_states=hidden_states,
        sp_groups_info=sp_groups_info_list[rank],
        sp_comm_groups=sp_comm_groups
    )

    # 校验结果
    ok = verify_results(rank, final_output, sp_groups_info_list[rank])
    dist.barrier()

    # 汇总校验结果（Rank 0收集所有结果）
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