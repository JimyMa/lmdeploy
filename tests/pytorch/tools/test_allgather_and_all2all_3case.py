import os
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

# -------------------------- 1. 全局配置（含测试用例参数） --------------------------
# 分布式基础参数
WORLD_SIZE = 4
MASTER_ADDR = 'localhost'
MASTER_PORT = '12361'

# 数据维度参数
C, D = 4, 6
D_OUT = D // 2  # 结果最后一维（3）

# 测试用例配置（3组场景，可通过修改 TEST_CASE 切换）
TEST_CASE = 3  # 1: 基础正常场景 | 2: 含空请求场景 | 3: 非对称请求场景
if TEST_CASE == 1:
    # 用例1：基础正常场景（所有RANK均有请求，数量递增）
    RANK_REQUEST_COUNTS = [2, 3, 4, 5]  # rank0:2条, rank1:3条, rank2:4条, rank3:5条
elif TEST_CASE == 2:
    # 用例2：含空请求场景（rank2无请求，验证0数据处理）
    RANK_REQUEST_COUNTS = [2, 3, 0, 5]  # rank2: 0条请求
else:
    # 用例3：非对称请求场景（rank0请求数最少，rank3最多，验证极端长度适配）
    RANK_REQUEST_COUNTS = [1, 4, 2, 6]  # 非递增、非对称请求数

# 固定权重矩阵W（定值，确保所有RANK计算结果一致）
def get_fixed_W(device):
    """固定权重矩阵：每行值为 [行号, 行号+1, 行号+2]（因D_OUT=3）"""
    W = torch.zeros(D, D_OUT, device=device, dtype=torch.float32)
    for i in range(D):
        W[i] = torch.tensor([i, i+1, i+2], device=device)  # 可手动计算预期结果
    return W

# -------------------------- 2. 工具函数 --------------------------
def empty(device):
    """返回空张量（形状(0, C, D_OUT)）"""
    return torch.empty(0, C, D_OUT, dtype=torch.float32, device=device)

def pad_and_gather(local_reqs, max_len, world_size):
    """All Gather前的填充逻辑（保持原逻辑不变）"""
    L = local_reqs.size(0)
    if L < max_len:
        pad = torch.zeros(max_len - L, C, D, device=local_reqs.device)
        padded = torch.cat([local_reqs, pad], dim=0)
    else:
        padded = local_reqs
    gathered = torch.empty(world_size, max_len, C, D, dtype=padded.dtype, device=padded.device)
    dist.all_gather_into_tensor(gathered.view(-1), padded)
    return gathered  # (world_size, max_len, C, D)

def dummy_merge(results_list):
    """固定合并逻辑：取所有RANK结果的平均值（便于计算预期值）"""
    if not results_list:
        return None
    return torch.stack(results_list).mean(dim=0)

def get_expected_single_result(req_tensor, W):
    """计算单条请求的预期结果（矩阵乘法手动展开，用于校验）"""
    # req_tensor: (C, D)，W: (D, D_OUT)，结果: (C, D_OUT)
    expected = torch.matmul(req_tensor, W)
    return expected

def get_expected_merged_result(req_tensor, W, world_size):
    """计算单条请求的预期合并结果（所有RANK计算相同值，平均值=单值）"""
    single_result = get_expected_single_result(req_tensor, W)
    # 因所有RANK计算结果相同，合并后的值 = 单条结果（平均值=自身）
    return single_result

# -------------------------- 3. 核心运行逻辑 --------------------------
def run(rank, world_size):
    # 1. 初始化设备与分布式环境
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    device = torch.device('cuda', local_rank)

    # 2. 生成当前RANK的本地请求（固定值，便于校验）
    L_rank = RANK_REQUEST_COUNTS[rank]  # 从测试用例获取请求数
    # 固定请求值：req_idx条请求的第c个通道第d维 = rank * 100 + req_idx * 10 + c * 1 + d
    local_reqs = torch.zeros(L_rank, C, D, device=device, dtype=torch.float32)
    for req_idx in range(L_rank):
        for c in range(C):
            for d in range(D):
                local_reqs[req_idx, c, d] = rank * 100 + req_idx * 10 + c * 1 + d

    # 3. 同步所有RANK的请求数量（All Gather元信息）
    send_len = torch.tensor([L_rank], dtype=torch.long, device=device)
    all_len = torch.empty(world_size, dtype=torch.long, device=device)
    dist.all_gather_into_tensor(all_len, send_len)
    all_len = all_len.tolist()
    max_len = max(all_len) if world_size > 0 else 0

    # 4. All Gather所有请求（获取全局请求）
    global_reqs = pad_and_gather(local_reqs, max_len, world_size)  # (world_size, max_len, C, D)

    # 5. 分布式计算（固定W，确保所有RANK计算一致）
    W = get_fixed_W(device)  # 固定权重矩阵
    # 扁平化全局请求：(world_size, max_len, C, D) → (L_total, C, D)
    flat = [global_reqs[i][:l] for i, l in enumerate(all_len)]
    flat = torch.cat(flat, dim=0) if any(all_len) else torch.empty(0, C, D, device=device)
    # 矩阵乘计算（所有RANK计算相同的全局结果）
    results = torch.matmul(flat, W) if flat.numel() > 0 else torch.empty(0, C, D_OUT, device=device)

    # 6. 按原始RANK拆分计算结果（记录每个切片的归属）
    start_idx = [0]
    for l in all_len[:-1]:
        start_idx.append(start_idx[-1] + l)
    slices = [slice(s, s + l) for s, l in zip(start_idx, all_len)]
    slice_ranks = list(range(world_size))  # 每个切片对应的原始RANK

    # 7. All2All通信：将结果发送回原始RANK
    # 7.1 同步All2All元信息（每个RANK需接收的数据量）
    send_lens = all_len.copy()  # 发送给rank i的数据量 = rank i的请求数
    send_lens_t = torch.tensor(send_lens, dtype=torch.long, device=device)
    recv_lens_all = torch.empty(world_size, world_size, dtype=torch.long, device=device)
    dist.all_gather_into_tensor(recv_lens_all.view(-1), send_lens_t)
    recv_lens = recv_lens_all[:, rank].tolist()  # 从rank i接收的数据量

    # 7.2 构造发送/接收列表
    input_list = []
    current_pos = 0
    for i in range(world_size):
        end_pos = current_pos + send_lens[i]
        if send_lens[i] > 0 and results.numel() > 0:
            input_list.append(results[current_pos:end_pos].contiguous())
        else:
            input_list.append(empty(device))
        current_pos = end_pos

    output_list = []
    for i in range(world_size):
        if recv_lens[i] > 0:
            output_list.append(torch.empty(recv_lens[i], C, D_OUT, device=device))
        else:
            output_list.append(empty(device))

    # 7.3 执行All2All通信
    dist.all_to_all(output_list, input_list)

    # 8. 合并结果并校验正确性
    # 8.1 收集当前RANK作为主RANK的所有结果
    local_results = []
    for tensor in output_list:
        if tensor.numel() > 0:
            local_results.append(tensor)
    local_results = torch.cat(local_results, dim=0) if local_results else empty(device)

    # 8.2 结果校验（核心逻辑）
    is_success = True
    my_request_count = all_len[rank]  # 当前RANK作为主RANK的请求数

    if my_request_count > 0:
        # 遍历当前RANK的每条请求，校验合并结果
        for req_idx in range(my_request_count):
            # 步骤1：获取原始请求（用于计算预期结果）
            original_req = local_reqs[req_idx]  # (C, D)
            
            # 步骤2：计算预期合并结果（固定逻辑，可手动验证）
            expected_merged = get_expected_merged_result(original_req, W, world_size)
            
            # 步骤3：获取实际合并结果（从local_results中提取）
            actual_merged = local_results[req_idx]  # (C, D_OUT)
            
            # 步骤4：校验（误差允许1e-5，因浮点计算）
            if not torch.allclose(actual_merged, expected_merged, atol=1e-5):
                print(f"❌ Rank {rank} 校验失败！请求{req_idx}")
                print(f"  原始请求: {original_req}")
                print(f"  预期合并结果: {expected_merged}")
                print(f"  实际合并结果: {actual_merged}")
                is_success = False
                break
    else:
        # 空请求场景：校验local_results是否为空
        if local_results.numel() != 0:
            print(f"❌ Rank {rank} 校验失败！无请求但收到数据")
            is_success = False

    # 8.3 输出校验结果
    if is_success:
        print(f"✅ Rank {rank} 测试用例{TEST_CASE}校验通过！")
        print(f"  - 本地请求数: {my_request_count}")
        print(f"  - 发送长度: {send_lens}")
        print(f"  - 接收长度: {recv_lens}")
    else:
        print(f"❌ Rank {rank} 测试用例{TEST_CASE}校验失败！")

    # 同步所有进程，确保日志输出完整
    dist.barrier()
    dist.destroy_process_group()

# -------------------------- 4. 启动入口 --------------------------
def main():
    print(f"=== 启动测试用例{TEST_CASE}（总进程数：{WORLD_SIZE}）===")
    spawn(run, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

if __name__ == '__main__':
    main()