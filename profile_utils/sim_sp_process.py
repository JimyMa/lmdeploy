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
    # 每个RANK的长请求输入长度, 32 = 5*4+4*3
    LONG_INPUT_LEN = [50003, 50003, 50003, 50003, 50003, 50003, 50003, 50003,
                      50003, 50003, 50003, 50003, 50003, 50003, 50003, 50003,
                      50003, 50003, 62504, 62504, 62504, 62504, 62504, 62504,
                      62504, 62504, 62504, 62504, 62504, 62504, 62504, 62504,]
    # 每个RANK的长请求输出长度
    LONG_OUTPUT_LEN = [100] * RANK_TOTAL
    # 每个RANK的短请求数量
    SHORT_REQ_NUM = [92, 92, 92, 92, 92, 92, 92, 92,
                     93, 93, 93, 93, 93, 93, 93, 93,
                     93, 93, 93, 93, 93, 93, 93, 93,
                     93, 93, 93, 93, 93, 93, 93, 93]
    # 每个RANK的长请求数量
    LONG_REQ_NUM = [1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1]
    
    # 生成请求配置
    generate_rank_requests(
        rank_total=RANK_TOTAL,
        short_input_len=SHORT_INPUT_LEN,
        short_output_len=SHORT_OUTPUT_LEN,
        long_input_len=LONG_INPUT_LEN,
        long_output_len=LONG_OUTPUT_LEN,
        short_req_num=SHORT_REQ_NUM,
        long_req_num=LONG_REQ_NUM,
        output_file="/nvme4/share/chenjiefei/src/sglang/profile_utils/scheduled_requests.json"
    )