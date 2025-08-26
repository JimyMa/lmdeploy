import os
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

C, D = 4, 6
D_OUT = D // 2

def empty(device):
    return torch.empty(0, C, D_OUT, dtype=torch.float32, device=device)

def pad_and_gather(local_reqs, max_len, world_size):
    L = local_reqs.size(0)
    if L < max_len:
        pad = torch.zeros(max_len - L, C, D, device=local_reqs.device)
        padded = torch.cat([local_reqs, pad], dim=0)
    else:
        padded = local_reqs
    gathered = torch.empty(world_size * max_len, C, D,
                           dtype=padded.dtype,
                           device=padded.device)
    dist.all_gather_into_tensor(gathered, padded)
    return gathered.view(world_size, max_len, C, D)   # (world_size, max_len, C, D)

def dummy_merge(results_list):
    if not results_list:
        return None
    return torch.stack(results_list).mean(dim=0)

def run(rank, world_size):
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12361'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    device = torch.device('cuda', local_rank)

    # 1. 本地请求
    if rank == 2:
        L_rank = 0
    else:
        L_rank = 2 + rank
    local_reqs = torch.randn(L_rank, C, D, device=device) * 10

    # 2. 同步各 rank 的 L_rank
    send_len = torch.tensor([L_rank], dtype=torch.long, device=device)
    all_len = torch.empty(world_size, dtype=torch.long, device=device)
    dist.all_gather_into_tensor(all_len, send_len)
    all_len = all_len.tolist()
    max_len = max(all_len) if world_size > 0 else 0

    # 3. All-Gather
    global_reqs = pad_and_gather(local_reqs, max_len, world_size)

    # 4. 计算
    flat = [global_reqs[i][:l] for i, l in enumerate(all_len)]
    flat = torch.cat(flat, dim=0) if any(all_len) else torch.empty(0, C, D, device=device)
    W = torch.randn(D, D_OUT, device=device)
    results = torch.matmul(flat, W) if flat.numel() else torch.empty(0, C, D_OUT, device=device)

    # 5. 准备 all_to_all_single 的元信息
    send_counts = torch.tensor(all_len, dtype=torch.long, device=device)  # 每个 rank 要发给谁多少条
    recv_counts = torch.empty(world_size, dtype=torch.long, device=device)
    dist.all_to_all_single(recv_counts, send_counts)                      # 交换长度

    # 6. 构造发送张量：把要发给 rank i 的数据连续拼接
    send_displ = [0] + torch.cumsum(send_counts, dim=0)[:-1].tolist()
    send_tensor = torch.empty(send_counts.sum(), C, D_OUT, device=device)
    for i in range(world_size):
        start_out = send_displ[i]
        end_out   = start_out + send_counts[i]
        start_in  = sum(all_len[:i])          # results 中属于 rank i 的起始
        end_in    = start_in + all_len[i]
        if send_counts[i] > 0:
            send_tensor[start_out:end_out] = results[start_in:end_in]

    # 7. 构造接收张量
    recv_tensor = torch.empty(recv_counts.sum(), C, D_OUT, device=device)

    # 8. all_to_all_single 通信
    dist.all_to_all_single(
        recv_tensor, send_tensor,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes =send_counts.tolist()
    )

    # 9. 拆分 recv_tensor -> 每个请求的结果
    #    需要知道：来自 rank i 的结果在 recv_tensor 中的起始位置
    recv_displ = [0] + torch.cumsum(recv_counts, dim=0)[:-1].tolist()
    my_request_count = all_len[rank]
    merged_results = []

    if my_request_count > 0:
        request_results = [[] for _ in range(my_request_count)]
        for i in range(world_size):
            cnt = recv_counts[i].item()
            if cnt == 0:
                continue
            start = recv_displ[i]
            end   = start + cnt
            slice_ = recv_tensor[start:end]
            for req_idx in range(cnt):
                request_results[req_idx].append(slice_[req_idx])
        for req_idx in range(my_request_count):
            merged = dummy_merge(request_results[req_idx])
            merged_results.append((req_idx, merged))

    # 10. 打印
    print(f"Rank {rank} 发送长度: {send_counts.tolist()}")
    print(f"Rank {rank} 接收长度: {recv_counts.tolist()}")
    print(f"Rank {rank} 合并了 {len(merged_results)} 个请求结果")

    dist.barrier()
    dist.destroy_process_group()

def main():
    world_size = 4
    spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()