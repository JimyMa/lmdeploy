import os
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

def pad_to(tensor, target_shape, value=0.0):
    """将 tensor 零填充到 target_shape（支持 1-D）。"""
    current_shape = tensor.shape
    if current_shape == target_shape:
        return tensor
    pad_size = target_shape[0] - current_shape[0]
    return torch.cat([tensor, tensor.new_full((pad_size,), value)])

def run(rank, size):
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=size)

    # 1. 每个 rank 造一个长度随机的张量
    local_len = 2 + rank  # 举例：rank0=2, rank1=3...
    local_tensor = torch.arange(local_len, dtype=torch.float32, device='cuda')

    # ---------- 阶段1：All-Gather 元信息（shape） ----------
    meta_tensor = torch.tensor([local_tensor.shape[0]], dtype=torch.long, device='cuda')
    gathered_meta = [torch.zeros(1, dtype=torch.long, device='cuda') for _ in range(size)]
    dist.all_gather(gathered_meta, meta_tensor)
    # 把 list[tensor] -> list[int]
    all_lengths = [int(t.item()) for t in gathered_meta]
    max_len = max(all_lengths)
    target_shape = (max_len,)

    # ---------- 阶段2：就地零填充 ----------
    padded_tensor = pad_to(local_tensor, target_shape)

    # ---------- 阶段3：All-Gather 完整张量 ----------
    gathered_padded = [torch.zeros_like(padded_tensor) for _ in range(size)]
    dist.all_gather(gathered_padded, padded_tensor)

    # ---------- 阶段4：根据真实长度切片 ----------
    gathered_tensors = [t[:l] for t, l in zip(gathered_padded, all_lengths)]

    # 打印结果
    print(f"Rank {rank} 原始长度={local_len} 原始值={local_tensor}")
    print(f"Rank {rank} 收集后：{gathered_tensors}")

    # 简单校验：每个 rank 应该收到一个 [0,1,2,...,i-1] 的递增张量
    for i, t in enumerate(gathered_tensors):
        expected = torch.arange(all_lengths[i], dtype=torch.float32, device='cuda')
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
    