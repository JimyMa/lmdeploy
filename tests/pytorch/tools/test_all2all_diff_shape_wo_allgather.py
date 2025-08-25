import os
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn


def run(rank: int, world_size: int):
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    # 1. 各 rank 的原始数据
    raw_data = {
        0: torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float32, device='cuda'),
        1: torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=torch.float32, device='cuda'),
        2: torch.tensor([20, 21, 22, 23, 24], dtype=torch.float32, device='cuda'),
        3: torch.tensor([30, 31, 32, 33, 34, 35, 36], dtype=torch.float32, device='cuda')
    }

    # 2. 各 rank 的 split 长度（按列出的表格）
    input_splits_tbl = {
        0: [2, 2, 1, 1],
        1: [3, 2, 2, 2],
        2: [2, 1, 1, 1],
        3: [2, 2, 2, 1]
    }
    output_splits_tbl = {
        0: [2, 3, 2, 2],
        1: [2, 2, 1, 2],
        2: [1, 2, 1, 2],
        3: [1, 2, 1, 1]
    }

    # 3. 构造 input list
    input_splits = input_splits_tbl[rank]
    input_tensor = raw_data[rank]
    input_list = list(input_tensor.split(input_splits))

    # 4. 根据 output_splits 提前创建接收 buffer
    output_splits = output_splits_tbl[rank]
    output_list = [torch.empty(n, dtype=torch.float32, device='cuda')
                   for n in output_splits]

    print(f'Rank {rank} input_list len={[t.numel() for t in input_list]} -> '
          f'output_list len={[t.numel() for t in output_list]}', flush=True)

    # 5. all-to-all
    dist.all_to_all(output_list, input_list)

    # 6. 打印结果验证
    print(f'Rank {rank} received: {[t.tolist() for t in output_list]}', flush=True)

    # 7. 清理
    dist.destroy_process_group()


def main():
    world_size = 4
    spawn(run, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()