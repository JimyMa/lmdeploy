import os
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

def run(rank, size):
    """
    每个进程执行的函数（使用 NCCL）
    """
    # 1. 选择 GPU
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # 2. 初始化进程组（NCCL 后端）
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=size)

    # 3. 创建 GPU 张量
    local_tensor = torch.tensor([rank] * 3, dtype=torch.float32, device='cuda')

    print(f"Rank {rank} 原始张量: {local_tensor}", flush=True)

    # 4. All-gather
    gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(size)]
    dist.all_gather(gathered_tensors, local_tensor)

    print(f"Rank {rank} 收集后的张量列表: {gathered_tensors}", flush=True)

    # 5. 校验
    valid = all(torch.all(t == i) for i, t in enumerate(gathered_tensors))
    print(f"Rank {rank} 验证{'成功' if valid else '失败'}", flush=True)

    # 6. 清理
    dist.destroy_process_group()

def main():
    world_size = 4
    spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

