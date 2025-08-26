import os
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

def pad_to_3d(tensor, target_len_dim0, value=0.0):
    """
    只在第 0 维做零填充，使 tensor.shape[0] == target_len_dim0。
    tensor 为 3-D：(L, C, D)
    """
    current_len_dim0 = tensor.size(0)
    if current_len_dim0 == target_len_dim0:
        return tensor

    pad_shape = (target_len_dim0 - current_len_dim0, tensor.size(1), tensor.size(2))
    padding = tensor.new_full(pad_shape, value)
    return torch.cat([tensor, padding], dim=0)

def run(rank, size):
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=size)

    # 1. 每个 rank 造一个 3-D 张量：第 0 维长度随 rank 变化，其余两维固定
    dim0_len = 2 + rank          # rank0=2, rank1=3 ...
    dim1_len = 4                 # 固定
    dim2_len = 3                 # 固定
    local_tensor = torch.arange(
        dim0_len * dim1_len * dim2_len,
        dtype=torch.float32,
        device='cuda'
    ).view(dim0_len, dim1_len, dim2_len)
    # 为了让不同 rank 的数据可区分，加偏移
    local_tensor += rank * 100

    # ---------- 阶段1：All-Gather 元信息（第 0 维长度） ----------
    meta_tensor = torch.tensor([local_tensor.shape[0]], dtype=torch.long, device='cuda')
    gathered_meta = [torch.zeros(1, dtype=torch.long, device='cuda') for _ in range(size)]
    dist.all_gather(gathered_meta, meta_tensor)
    all_lengths_dim0 = [int(t.item()) for t in gathered_meta]
    max_len_dim0 = max(all_lengths_dim0)

    # ---------- 阶段2：就地零填充 ----------
    padded_tensor = pad_to_3d(local_tensor, max_len_dim0)

    # ---------- 阶段3：All-Gather 完整张量 ----------
    gathered_padded = [torch.zeros_like(padded_tensor) for _ in range(size)]
    dist.all_gather(gathered_padded, padded_tensor)

    # ---------- 阶段4：根据真实长度切片 ----------
    gathered_tensors = [t[:l] for t, l in zip(gathered_padded, all_lengths_dim0)]

    # ---------- 打印 & 简单校验 ----------
    print(f"Rank {rank} 原始形状={tuple(local_tensor.shape)}")
    print(f"Rank {rank} 收集后 shapes:",
          [tuple(t.shape) for t in gathered_tensors])

    # 校验：每个 rank 应该收到自己期望的 3-D 张量
    for i, t in enumerate(gathered_tensors):
        expected = torch.arange(
            all_lengths_dim0[i] * dim1_len * dim2_len,
            dtype=torch.float32,
            device='cuda'
        ).view(all_lengths_dim0[i], dim1_len, dim2_len) + i * 100
        if not torch.equal(t, expected):
            print(f"Rank {rank} 发现第 {i} 个张量校验失败！")
            break
    else:
        print(f"Rank {rank} 校验通过 ✓")

    dist.destroy_process_group()

def main():
    world_size = 4
    spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()