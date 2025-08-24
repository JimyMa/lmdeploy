import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
import os

def run(rank, size):
    """每个进程执行的函数"""
    # 配置分布式环境
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group('gloo', rank=rank, world_size=size)
    
    # 不同Rank创建不同形状的张量
    if rank == 0:
        local_tensor = torch.tensor([[rank, rank+1]], dtype=torch.float32)
    elif rank == 1:
        local_tensor = torch.tensor([[rank, rank+1], 
                                    [rank+2, rank+3]], dtype=torch.float32)
    elif rank == 2:
        local_tensor = torch.tensor([[rank, rank+1], 
                                    [rank+2, rank+3],
                                    [rank+4, rank+5]], dtype=torch.float32)
    else:  # rank == 3
        local_tensor = torch.tensor([[rank, rank+1], 
                                    [rank+2, rank+3],
                                    [rank+4, rank+5],
                                    [rank+6, rank+7]], dtype=torch.float32)
    
    # 打印all_gather之前的张量及其形状
    print(f"Rank {rank} 原始张量: \n{local_tensor}", flush=True)
    print(f"Rank {rank} 原始张量形状: {local_tensor.shape}\n", flush=True)
    
    # 使用all_gather_object收集不同形状的张量
    gathered_objects = [None for _ in range(size)]
    dist.all_gather_object(gathered_objects, local_tensor)
    
    # 构建整合的结果字符串
    result_str = f"Rank {rank} 使用all_gather_object收集的结果: "
    for i, tensor in enumerate(gathered_objects):
        # 将张量数据转换为列表以便更清晰地展示
        tensor_data = tensor.tolist()
        result_str += f"[Rank {i}: 形状{tensor.shape}, 内容{tensor_data}] "
    
    # 一次性打印整合后的结果
    print(result_str, flush=True)
    
    # 验证结果是否正确
    valid = True
    for i in range(size):
        expected_rows = i + 1
        if gathered_objects[i].shape[0] != expected_rows or gathered_objects[i].shape[1] != 2:
            valid = False
            break
    
    print(f"Rank {rank} 验证结果: {'成功' if valid else '失败'}\n", flush=True)
    
    # 清理分布式环境
    dist.destroy_process_group()

def main():
    # 启动4个进程
    world_size = 4
    spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
    