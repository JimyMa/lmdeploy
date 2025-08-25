import os
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

# 分布式参数配置
WORLD_SIZE = 4  # 总进程数
MASTER_ADDR = 'localhost'  # 主节点地址
MASTER_PORT = '12361'      # 主节点端口

# 数据维度参数
C, D = 4, 6
D_OUT = D // 2     # 结果最后一维

def empty(device):
    return torch.empty(0, C, D_OUT, dtype=torch.float32, device=device)

def pad_and_gather(local_reqs, max_len, world_size):
    """仅填充第 0 维，然后 All-Gather"""
    L = local_reqs.size(0)
    if L < max_len:
        pad = torch.zeros(max_len - L, C, D, device=local_reqs.device)
        padded = torch.cat([local_reqs, pad], dim=0)
    else:
        padded = local_reqs
    gathered = torch.empty(world_size, max_len, C, D,
                           dtype=padded.dtype,
                           device=padded.device)
    dist.all_gather_into_tensor(gathered.view(-1), padded)
    return gathered          # (world_size, max_len, C, D)

def dummy_merge(results_list):
    """简单的合并操作：取平均值"""
    if not results_list:
        return None
    # 按元素取平均
    merged = torch.stack(results_list).mean(dim=0)
    return merged

def run(rank, world_size):
    # 设置本地GPU
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    
    # 配置分布式环境
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    device = torch.device('cuda', local_rank)

    # 1. 每个 rank 自己的请求，只属于自己
    # 测试包含0的情况，让rank=2的请求数为0
    if rank == 2:
        L_rank = 0
    else:
        L_rank = 2 + rank                        # 第 0 维长度随 rank 变化
    local_reqs = torch.randn(L_rank, C, D, device=device) * 10

    # 2. 同步各 rank 的 L_rank
    send_len = torch.tensor([L_rank], dtype=torch.long, device=device)
    all_len = torch.empty(world_size, dtype=torch.long, device=device)
    dist.all_gather_into_tensor(all_len, send_len)
    all_len = all_len.tolist()
    max_len = max(all_len) if world_size > 0 else 0

    # 3. All-Gather 请求：形状 (world_size, max_len, C, D)
    global_reqs = pad_and_gather(local_reqs, max_len, world_size)

    # 4. 计算：扁平化后批量矩阵乘
    # 真实扁平化：把 (world_size, max_len, C, D) -> (L_total, C, D)
    flat = [global_reqs[i][:l] for i, l in enumerate(all_len)]
    flat = torch.cat(flat, dim=0) if any(all_len) else torch.empty(0, C, D, device=device)  # (L_total, C, D)

    W = torch.randn(D, D_OUT, device=device)
    results = torch.matmul(flat, W) if flat.numel() > 0 else torch.empty(0, C, D_OUT, device=device)  # (L_total, C, D_OUT)

    # 5. 把 results 再按“原始 owner rank”拆分
    #    start_idx[i] 表示 rank i 的结果在 results 中的起始位置
    start_idx = [0]
    for l in all_len[:-1]:
        start_idx.append(start_idx[-1] + l)
    
    # 6. 准备all-to-all通信的元信息
    # 定义发送长度：当前rank要发送给每个目标rank的数据量（第0维长度）
    send_lens = all_len.copy()  # send_lens[i] = 要发送给rank i的数据量
    
    # 同步元信息：获取每个rank要从其他rank接收的数据量
    send_lens_t = torch.tensor(send_lens, dtype=torch.long, device=device)
    recv_lens_all = torch.empty(world_size, world_size, dtype=torch.long, device=device)
    dist.all_gather_into_tensor(recv_lens_all.view(-1), send_lens_t)
    recv_lens = recv_lens_all[:, rank].tolist()  # 第i个元素是从rank i要接收的数据量

    # 7. 构造发送数据列表
    total_send = sum(send_lens)
    input_list = []
    current_pos = 0
    
    for i in range(world_size):
        end_pos = current_pos + send_lens[i]
        if send_lens[i] > 0 and total_send > 0:
            input_slice = results[current_pos:end_pos].contiguous()
            input_list.append(input_slice)
        else:
            input_list.append(empty(device))
        current_pos = end_pos

    # 8. 构造接收数据列表（预先分配内存）
    output_list = []
    for i in range(world_size):
        if recv_lens[i] > 0:
            recv_tensor = torch.empty(recv_lens[i], C, D_OUT, dtype=torch.float32, device=device)
            output_list.append(recv_tensor)
        else:
            output_list.append(empty(device))

    # 9. 执行all-to-all通信
    dist.all_to_all(output_list, input_list)

    # 10. 收集并合并结果
    local_results = []
    for tensor in output_list:
        if tensor.numel() > 0:
            local_results.append(tensor)
    
    if local_results:
        local_results = torch.cat(local_results, dim=0)
    else:
        local_results = torch.empty(0, C, D_OUT, device=device)

    # 11. 对于每个请求，收集所有rank的计算结果并合并
    my_request_count = all_len[rank]
    merged_results = []
    
    if my_request_count > 0 and local_results.numel() > 0:
        # 为每个请求创建结果列表
        request_results = [[] for _ in range(my_request_count)]
        
        # 按发送方rank拆分结果
        current = 0
        for i in range(world_size):
            count = recv_lens[i]
            if count > 0:
                end = current + count
                # 将来自rank i的结果分配到对应的请求
                for req_idx in range(count):
                    request_results[req_idx].append(local_results[current + req_idx])
                current = end
        
        # 合并每个请求的结果
        for req_idx in range(my_request_count):
            merged = dummy_merge(request_results[req_idx])
            merged_results.append((req_idx, merged))

    # 12. 输出统计信息
    print(f"Rank {rank} 发送长度: {send_lens}")
    print(f"Rank {rank} 接收长度: {recv_lens}")
    print(f"Rank {rank} 合并了 {len(merged_results)} 个请求结果")
    
    # 同步所有进程
    dist.barrier()
    
    # 13. 清理
    dist.destroy_process_group()

def main():
    # 使用代码中定义的参数启动分布式进程
    spawn(run, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

if __name__ == '__main__':
    main()
