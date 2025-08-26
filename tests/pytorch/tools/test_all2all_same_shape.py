import os
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn


def run(rank, world_size):
    """
    每个进程执行的函数（NCCL 后端）。
    使用 dist.all_to_all（list 形式）完成 all-to-all。
    """
    # 1. 选择 GPU
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # 2. 初始化进程组
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'   # 换一个端口避免冲突
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    # 3. 构造输入 list
    #    每个进程准备 world_size 个张量，分别要发给对应 rank
    sub_size = 3
    input_list = [
        torch.full((sub_size,), rank * 10 + i, dtype=torch.float32, device='cuda')
        for i in range(world_size)
    ]
    print(f"Rank {rank} 输入 list: {[t.tolist() for t in input_list]}", flush=True)

    # 4. 创建接收 list
    output_list = [torch.empty(sub_size, dtype=torch.float32, device='cuda')
                   for _ in range(world_size)]

    # 5. all-to-all
    dist.all_to_all(output_list, input_list)

    print(f"Rank {rank} 输出 list: {[t.tolist() for t in output_list]}", flush=True)

    # 6. 简单校验
    expected = [torch.full_like(output_list[rank], i * 10 + rank)
                for i in range(world_size)]
    passed = all(torch.equal(out, exp) for out, exp in zip(output_list, expected))
    print(f"Rank {rank} 验证{'成功' if passed else '失败'}", flush=True)

    # 7. 清理
    dist.destroy_process_group()


def main():
    world_size = 4
    spawn(run, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()