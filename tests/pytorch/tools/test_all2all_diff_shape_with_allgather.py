import os
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn


def run(rank: int, world_size: int):
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    # ----------------------------------------------------------
    # 1) 各 rank 造自己的“原始数据”，这里为了演示仍然用固定值，
    #    但你可以替换成任意长度、任意内容，只要保证每个 peer
    #    知道自己要发给谁多少元素即可。本例沿用你之前的数据。
    # ----------------------------------------------------------
    raw_data = {
        0: torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float32, device='cuda'),
        1: torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=torch.float32, device='cuda'),
        2: torch.tensor([20, 21, 22, 23, 24], dtype=torch.float32, device='cuda'),
        3: torch.tensor([30, 31, 32, 33, 34, 35, 36], dtype=torch.float32, device='cuda')
    }
    tensor = raw_data[rank]

    # ----------------------------------------------------------
    # 2) 动态决定发送长度：这里用“输入 split”举例，
    #    你也可以用任意逻辑计算 send_sizes[i]。
    # ----------------------------------------------------------
    input_splits = {
        0: [2, 2, 1, 1],
        1: [3, 2, 2, 2],
        2: [2, 1, 1, 1],
        3: [2, 2, 2, 1]
    }
    send_sizes = torch.tensor(input_splits[rank], dtype=torch.long, device='cuda')  # shape [world_size]

    # ----------------------------------------------------------
    # 3) 元信息同步：做一次 all_gather，得到全局 recv_sizes
    #    recv_sizes[i][j] 表示 rank j 从 rank i 接收的元素个数
    # ----------------------------------------------------------
    recv_sizes = torch.empty(world_size, world_size, dtype=torch.long, device='cuda')
    dist.all_gather_into_tensor(recv_sizes.view(-1), send_sizes)   # 1-D gather 再 reshape
    # 现在 recv_sizes[i] 就是 rank i 发出的 send_sizes，即 rank j 需要用的
    my_recv_sizes = recv_sizes[:, rank]  # shape [world_size]

    # ----------------------------------------------------------
    # 4) 根据 my_recv_sizes 创建接收缓冲区
    # ----------------------------------------------------------
    input_list = list(tensor.split(send_sizes.tolist()))
    output_list = [torch.empty(n.item(), dtype=torch.float32, device='cuda')
                   for n in my_recv_sizes]

    print(f'Rank {rank} send_sizes={send_sizes.tolist()} '
          f'recv_sizes={my_recv_sizes.tolist()}', flush=True)

    # ----------------------------------------------------------
    # 5) 真正的可变长度 all-to-all
    # ----------------------------------------------------------
    dist.all_to_all(output_list, input_list)

    print(f'Rank {rank} received: {[t.tolist() for t in output_list]}', flush=True)

    # ----------------------------------------------------------
    # 6) 清理
    # ----------------------------------------------------------
    dist.destroy_process_group()


def main():
    world_size = 4
    spawn(run, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()