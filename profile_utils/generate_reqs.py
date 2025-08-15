import json

def generate_rank_requests(
    rank_total: int,
    short_input_len: list,
    short_output_len: list,
    long_input_len: list,
    long_output_len: list,
    short_req_num: list,
    long_req_num: list,
    output_file: str = "rank_requests.json"
) -> None:
    """
    生成RANK请求配置并保存到JSON文件
    
    参数:
        rank_total: RANK总数
        short_input_len: 每个RANK的短请求输入长度列表
        short_output_len: 每个RANK的短请求输出长度列表
        long_input_len: 每个RANK的长请求输入长度列表
        long_output_len: 每个RANK的长请求输出长度列表
        short_req_num: 每个RANK的短请求数量列表
        long_req_num: 每个RANK的长请求数量列表
        output_file: 输出JSON文件名
    """
    # 验证输入列表长度是否一致
    assert len(short_input_len) == rank_total
    assert len(short_output_len) == rank_total
    assert len(long_input_len) == rank_total
    assert len(long_output_len) == rank_total
    assert len(short_req_num) == rank_total
    assert len(long_req_num) == rank_total

    requests = []
    total_short_requests = 0
    total_long_requests = 0
    rank_token_counts = []  # 存储每个RANK的token数量
    
    # 为每个RANK生成请求
    for rank in range(rank_total):
        # 计算当前RANK的token数量
        short_tokens = short_req_num[rank] * (short_input_len[rank] + short_output_len[rank])
        long_tokens = long_req_num[rank] * (long_input_len[rank] + long_output_len[rank])
        total_tokens = short_tokens + long_tokens
        rank_token_counts.append(total_tokens)
        
        # 生成当前RANK的短请求
        for _ in range(short_req_num[rank]):
            requests.append({
                "rank": rank,
                "input_length": short_input_len[rank],
                "output_length": short_output_len[rank],
                "request_type": "short"
            })
            total_short_requests += 1
        
        # 生成当前RANK的长请求
        for _ in range(long_req_num[rank]):
            requests.append({
                "rank": rank,
                "input_length": long_input_len[rank],
                "output_length": long_output_len[rank],
                "request_type": "long"
            })
            total_long_requests += 1
    
    # 按照rank从小到大排序
    requests.sort(key=lambda x: x["rank"])
    
    # 计算统计信息
    total_requests = total_short_requests + total_long_requests
    avg_batch_size = total_requests / rank_total
    max_token_count = max(rank_token_counts)
    
    # 打印统计信息
    print(f"总请求数: {total_requests}")
    print(f"短请求数: {total_short_requests}")
    print(f"长请求数: {total_long_requests}")
    print(f"平均batch size (每个rank的平均请求数): {avg_batch_size:.2f}")
    print(f"RANK token数量最大值: {max_token_count}")
    
    # 保存到JSON文件
    with open(output_file, 'w') as f:
        json.dump(requests, f, indent=2)
    
    print(f"成功生成 {len(requests)} 个请求配置到 {output_file}")

# 示例用法
if __name__ == "__main__":
    # 示例参数（假设有32个RANK）
    RANK_TOTAL = 32
    
    # 每个RANK的短请求输入长度
    SHORT_INPUT_LEN = [512] * RANK_TOTAL
    # 每个RANK的短请求输出长度
    SHORT_OUTPUT_LEN = [100] * RANK_TOTAL
    # 每个RANK的长请求输入长度
    LONG_INPUT_LEN = [250016] * RANK_TOTAL
    # 每个RANK的长请求输出长度
    LONG_OUTPUT_LEN = [100] * RANK_TOTAL
    # 每个RANK的短请求数量
    SHORT_REQ_NUM = [67, 60, 26, 256, 234, 256, 23, 256,
                     51, 33, 55, 165, 64, 134, 20, 20,
                     70, 136, 20, 21, 20, 119, 31, 145,
                     22, 23, 22, 217, 243, 76, 32, 51]
    # 每个RANK的长请求数量
    LONG_REQ_NUM = [0, 0, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 1, 1,
                    0, 0, 1, 0, 0, 0, 0, 0,
                    1, 1, 1, 0, 0, 0, 0, 0]
    
    # 生成请求配置
    generate_rank_requests(
        rank_total=RANK_TOTAL,
        short_input_len=SHORT_INPUT_LEN,
        short_output_len=SHORT_OUTPUT_LEN,
        long_input_len=LONG_INPUT_LEN,
        long_output_len=LONG_OUTPUT_LEN,
        short_req_num=SHORT_REQ_NUM,
        long_req_num=LONG_REQ_NUM,
        output_file="/nvme4/share/chenjiefei/src/sglang/profile_utils/raw_requests.json"
    )

# 2025-08-13 01:25:00 - Node: 0, KV Cache Usage: 0.24, Total Tokens: 776707, Running Requests: 67, Waiting Requests: 83, Batch Size: 150
# 2025-08-13 01:25:00 - Node: 1, KV Cache Usage: 0.22, Total Tokens: 808476, Running Requests: 60, Waiting Requests: 182, Batch Size: 242
# 2025-08-13 01:25:00 - Node: 2, KV Cache Usage: 0.09, Total Tokens: 775370, Running Requests: 26, Waiting Requests: 135, Batch Size: 161
# 2025-08-13 01:25:00 - Node: 3, KV Cache Usage: 0.93, Total Tokens: 827136, Running Requests: 256, Waiting Requests: 439, Batch Size: 695
# 2025-08-13 01:25:00 - Node: 4, KV Cache Usage: 0.85, Total Tokens: 867882, Running Requests: 234, Waiting Requests: 76, Batch Size: 310
# 2025-08-13 01:25:00 - Node: 5, KV Cache Usage: 0.93, Total Tokens: 981280, Running Requests: 256, Waiting Requests: 245, Batch Size: 501
# 2025-08-13 01:25:00 - Node: 6, KV Cache Usage: 1.00, Total Tokens: 1083944, Running Requests: 24, Waiting Requests: 98, Batch Size: 122
# 2025-08-13 01:25:00 - Node: 7, KV Cache Usage: 0.93, Total Tokens: 1074464, Running Requests: 256, Waiting Requests: 243, Batch Size: 499
# 2025-08-13 01:25:00 - Node: 8, KV Cache Usage: 0.19, Total Tokens: 776387, Running Requests: 51, Waiting Requests: 77, Batch Size: 128
# 2025-08-13 01:25:00 - Node: 9, KV Cache Usage: 0.12, Total Tokens: 775489, Running Requests: 33, Waiting Requests: 108, Batch Size: 141
# 2025-08-13 01:25:00 - Node: 10, KV Cache Usage: 0.20, Total Tokens: 776199, Running Requests: 55, Waiting Requests: 73, Batch Size: 128
# 2025-08-13 01:25:00 - Node: 11, KV Cache Usage: 0.60, Total Tokens: 864533, Running Requests: 165, Waiting Requests: 91, Batch Size: 256
# 2025-08-13 01:25:00 - Node: 12, KV Cache Usage: 0.23, Total Tokens: 783952, Running Requests: 64, Waiting Requests: 130, Batch Size: 194
# 2025-08-13 01:25:00 - Node: 13, KV Cache Usage: 0.49, Total Tokens: 852070, Running Requests: 134, Waiting Requests: 7, Batch Size: 141
# 2025-08-13 01:25:00 - Node: 14, KV Cache Usage: 1.00, Total Tokens: 1076309, Running Requests: 21, Waiting Requests: 88, Batch Size: 109
# 2025-08-13 01:25:00 - Node: 15, KV Cache Usage: 1.00, Total Tokens: 775541, Running Requests: 21, Waiting Requests: 161, Batch Size: 182
# 2025-08-13 01:25:00 - Node: 16, KV Cache Usage: 0.26, Total Tokens: 776998, Running Requests: 70, Waiting Requests: 76, Batch Size: 146
# 2025-08-13 01:25:00 - Node: 17, KV Cache Usage: 0.50, Total Tokens: 838536, Running Requests: 136, Waiting Requests: 78, Batch Size: 214
# 2025-08-13 01:25:00 - Node: 18, KV Cache Usage: 1.00, Total Tokens: 1047445, Running Requests: 21, Waiting Requests: 29, Batch Size: 50
# 2025-08-13 01:25:00 - Node: 19, KV Cache Usage: 0.08, Total Tokens: 775557, Running Requests: 21, Waiting Requests: 147, Batch Size: 168
# 2025-08-13 01:25:00 - Node: 20, KV Cache Usage: 1.00, Total Tokens: 804773, Running Requests: 21, Waiting Requests: 224, Batch Size: 245
# 2025-08-13 01:25:00 - Node: 21, KV Cache Usage: 0.43, Total Tokens: 857015, Running Requests: 119, Waiting Requests: 50, Batch Size: 169
# 2025-08-13 01:25:00 - Node: 22, KV Cache Usage: 0.11, Total Tokens: 775855, Running Requests: 31, Waiting Requests: 120, Batch Size: 151
# 2025-08-13 01:25:00 - Node: 23, KV Cache Usage: 0.53, Total Tokens: 868813, Running Requests: 145, Waiting Requests: 101, Batch Size: 226
# 2025-08-13 01:25:00 - Node: 24, KV Cache Usage: 1.00, Total Tokens: 1064615, Running Requests: 23, Waiting Requests: 58, Batch Size: 81
# 2025-08-13 01:25:00 - Node: 25, KV Cache Usage: 1.00, Total Tokens: 791384, Running Requests: 24, Waiting Requests: 188, Batch Size: 212
# 2025-08-13 01:25:00 - Node: 26, KV Cache Usage: 1.00, Total Tokens: 711111, Running Requests: 23, Waiting Requests: 13, Batch Size: 62
# 2025-08-13 01:25:00 - Node: 27, KV Cache Usage: 0.79, Total Tokens: 940041, Running Requests: 217, Waiting Requests: 39, Batch Size: 256
# 2025-08-13 01:25:00 - Node: 28, KV Cache Usage: 0.88, Total Tokens: 1001971, Running Requests: 243, Waiting Requests: 181, Batch Size: 424
# 2025-08-13 01:25:00 - Node: 29, KV Cache Usage: 0.28, Total Tokens: 819916, Running Requests: 76, Waiting Requests: 97, Batch Size: 173
# 2025-08-13 01:25:00 - Node: 30, KV Cache Usage: 0.12, Total Tokens: 1079872, Running Requests: 32, Waiting Requests: 56, Batch Size: 88
# 2025-08-13 01:25:00 - Node: 31, KV Cache Usage: 0.18, Total Tokens: 791795, Running Requests: 51, Waiting Requests: 142, Batch Size: 193