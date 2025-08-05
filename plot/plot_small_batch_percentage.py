import os
import re
import datetime
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def parse_log_file(log_file):
    # 初始化数据结构
    batch_data = defaultdict(list)
    waiting_data = defaultdict(list)
    kv_cache_data = defaultdict(list)
    timestamps = defaultdict(list)
    
    # 日志解析正则表达式
    timestamp_pattern = re.compile(
        r'Timestamp: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})'
    )
    data_pattern = re.compile(
        r'(\d+),(\d+),(\d+),([\d.]+)'
    )

    with open(log_file, 'r') as f:
        for line in f:
            # 查找时间戳
            ts_match = timestamp_pattern.search(line)
            if not ts_match:
                continue
                
            timestamp_str = ts_match.group(1)
            try:
                timestamp = datetime.datetime.strptime(
                    timestamp_str, "%Y-%m-%d %H:%M:%S.%f"
                )
            except ValueError:
                continue
                
            # 提取所有数据条目
            data_entries = data_pattern.findall(line)
            
            for entry in data_entries:
                rank = int(entry[0])
                batch_size = int(entry[1])
                num_waiting = int(entry[2])
                kv_cache_usage = float(entry[3])
                
                # 存储数据
                timestamps[rank].append(timestamp)
                batch_data[rank].append(batch_size)
                waiting_data[rank].append(num_waiting)
                kv_cache_data[rank].append(kv_cache_usage)
    
    return timestamps, batch_data, waiting_data, kv_cache_data

def calculate_small_batch_rank_percentage(batch_data, kv_cache_data, kv_cache_threshold=0.85, batch_size_threshold=30):
    # 初始化统计变量
    total_ranks = 0
    small_batch_ranks = 0
    
    # 遍历所有rank
    for rank in batch_data:
        total_ranks += 1
        # 检查是否满足small batch rank条件
        if any(kv_cache_usage > kv_cache_threshold and batch_size < batch_size_threshold 
               for batch_size, kv_cache_usage in zip(batch_data[rank], kv_cache_data[rank])):
            small_batch_ranks += 1
    
    # 计算占比
    if total_ranks == 0:
        return 0.0
    return (small_batch_ranks / total_ranks) * 100

def plot_small_batch_rank_percentage_over_time(timestamps, batch_data, kv_cache_data, output_dir, kv_cache_threshold=0.85, batch_size_threshold=30):
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化时间戳和占比列表
    all_timestamps = []
    percentages = []
    
    # 遍历每个时间戳，计算当时的small batch rank占比
    for ts in sorted(set(ts for sublist in timestamps.values() for ts in sublist)):
        all_timestamps.append(ts)
        current_batch_data = {rank: [bs for t, bs in zip(timestamps[rank], batch_data[rank]) if t <= ts] for rank in batch_data}
        current_kv_cache_data = {rank: [kv for t, kv in zip(timestamps[rank], kv_cache_data[rank]) if t <= ts] for rank in kv_cache_data}
        percentage = calculate_small_batch_rank_percentage(current_batch_data, current_kv_cache_data, kv_cache_threshold, batch_size_threshold)
        percentages.append(percentage)
    
    # 绘制图表
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.plot(all_timestamps, percentages, marker='o', linestyle='-', color='b')
    plt.title('Small Batch Rank Percentage Over Time', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Small Batch Rank Percentage (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 保存图表
    output_filename = os.path.join(output_dir, 'small_batch_rank_percentage_over_time.png')
    plt.savefig(output_filename)
    plt.close()
    
    print(f"Saved small batch rank percentage over time plot to: {os.path.abspath(output_filename)}")

def main():
    parser = argparse.ArgumentParser(description='Calculate and plot small batch rank percentage over time from log file.')
    parser.add_argument('log_file', nargs='?', help='Path to the log file', default="/nvme4/share/chenjiefei/scripts/server/2p4d_decode_256_0.65_2025-08-05_08-59/dp32ep32_Decode_node0.log")
    parser.add_argument('output_dir', nargs='?', help='Directory to save the plot', default="/nvme4/share/chenjiefei/src/lmdeploy/plot")
    parser.add_argument('--kv_cache_threshold', type=float, default=0.85, help='KV Cache usage threshold')
    parser.add_argument('--batch_size_threshold', type=int, default=30, help='Batch size threshold')
    parser.add_argument('--delete_warmup_data', type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to delete the earliest data entry (warmup data)')

    args = parser.parse_args()
    
    # 解析日志文件
    timestamps, batch_data, waiting_data, kv_cache_data = parse_log_file(args.log_file)
    
    # 如果需要删除warmup数据，则删除时间最早的一条数据
    if args.delete_warmup_data:
        for rank in timestamps:
            if timestamps[rank]:
                earliest_timestamp = min(timestamps[rank])
                earliest_index = timestamps[rank].index(earliest_timestamp)
                timestamps[rank].pop(earliest_index)
                batch_data[rank].pop(earliest_index)
                waiting_data[rank].pop(earliest_index)
                kv_cache_data[rank].pop(earliest_index)
    
    # 绘制small batch rank占比随时间变化图
    plot_small_batch_rank_percentage_over_time(timestamps, batch_data, kv_cache_data, args.output_dir, args.kv_cache_threshold, args.batch_size_threshold)

if __name__ == "__main__":
    main()