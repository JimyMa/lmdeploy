import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
import os

def run(rank, size):
    """
    每个进程执行的函数
    
    Args:
        rank: 进程的唯一标识（0, 1, ..., size-1）
        size: 总进程数
    """
    # 配置分布式环境
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=size)
    
    # 每个进程创建自己的张量
    # 这里让每个进程的张量值等于其rank
    local_tensor = torch.tensor([rank] * 3, dtype=torch.float32)
    
    # 打印all_gather之前的张量
    print(f"Rank {rank} 原始张量: {local_tensor}", flush=True)
    
    # 为收集结果创建存储空间
    # 结果将是一个列表，包含所有进程的张量
    gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(size)]
    
    # 执行all_gather操作
    dist.all_gather(gathered_tensors, local_tensor)
    
    # 打印all_gather之后的结果
    print(f"Rank {rank} 收集后的张量列表: {gathered_tensors}", flush=True)
    
    # 验证结果是否正确
    # 检查收集到的第i个张量是否全为i
    valid = True
    for i in range(size):
        if not torch.all(gathered_tensors[i] == i):
            valid = False
            break
    
    if valid:
        print(f"Rank {rank} 验证成功: 收集结果正确", flush=True)
    else:
        print(f"Rank {rank} 验证失败: 收集结果不正确", flush=True)
    
    # 清理分布式环境
    dist.destroy_process_group()

def main():
    # 启动4个进程
    world_size = 4
    spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

