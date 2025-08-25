import os
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn


def empty(device):
    """返回一个 0 元素的 CUDA 张量，用于占位"""
    return torch.empty(0, dtype=torch.float32, device=device)


def run(rank, world_size):
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    device = torch.device('cuda', local_rank)

    # ---------------- 1. 各 rank 的发送大小 ----------------
    # Rank 2 完全不参与收发（除了自己给自己发 0 元素）
    if rank == 2:
        send_sizes = [0] * world_size
    else:
        # 其他 rank 随便写点非零数
        send_sizes = [1, 1, 0, 2] if rank == 0 else [2, 1, 0, 1]

    # ---------------- 2. 元信息同步 ----------------
    send_sizes_t = torch.tensor(send_sizes, dtype=torch.long, device=device)
    recv_sizes_all = torch.empty(world_size, world_size, dtype=torch.long, device=device)
    dist.all_gather_into_tensor(recv_sizes_all.view(-1), send_sizes_t)
    recv_sizes = recv_sizes_all[:, rank].tolist()

    # ---------------- 3. 构造 input / output list ----------------
    # 真实要发的数据
    dummy_data = torch.arange(sum(send_sizes), dtype=torch.float32, device=device)
    input_list = list(dummy_data.split(send_sizes)) if sum(send_sizes) else \
                 [empty(device) for _ in range(world_size)]

    output_list = [empty(device) if n == 0 else
                   torch.empty(n, dtype=torch.float32, device=device)
                   for n in recv_sizes]

    print(f'Rank {rank} send_sizes={send_sizes} recv_sizes={recv_sizes}', flush=True)

    # ---------------- 4. all-to-all ----------------
    dist.all_to_all(output_list, input_list)

    # ---------------- 5. 打印结果 ----------------
    print(f'Rank {rank} received: {[t.tolist() if t.numel() else [] for t in output_list]}')

    dist.destroy_process_group()


def main():
    world_size = 4
    spawn(run, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()