import os
import re
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from collections import defaultdict
from matplotlib.ticker import MaxNLocator
import math

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

def plot_metrics(output_dir, timestamps, batch_data, waiting_data, kv_cache_data):
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置学术风格
    plt.style.use('seaborn-v0_8') 
    sns.set_style("whitegrid", {
        'grid.linestyle': '--',
        'grid.alpha': 0.4
    })
    plt.rcParams['font.family'] = 'DejaVu Sans'  # 学术常用字体
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    
    # 创建三个图表
    metrics = [
        ('Batch Size', batch_data, 'batch_size', 'Batch Size'),
        ('Num Waiting', waiting_data, 'num_waiting', 'Number of Waiting Requests'),
        ('KV Cache Usage', kv_cache_data, 'kv_cache_usage', 'KV Cache Usage (%)')
    ]
    
    saved_files = []
    
    for title, data_dict, filename_prefix, ylabel in metrics:
        # 获取所有rank并按顺序排序
        ranks = sorted(data_dict.keys())
        total_ranks = len(ranks)
        
        # 计算需要多少张图
        num_plots = math.ceil(total_ranks / 8)
        
        for plot_idx in range(num_plots):
            # 获取当前图中要显示的rank
            start_idx = plot_idx * 8
            end_idx = min((plot_idx + 1) * 8, total_ranks)
            current_ranks = ranks[start_idx:end_idx]
            
            fig, ax = plt.subplots(figsize=(8, 4.5))  # 更适合论文的宽高比
            
            # 使用学术友好的调色板
            palette = sns.color_palette("husl", len(current_ranks))
            
            for i, rank in enumerate(current_ranks):
                if rank in timestamps and timestamps[rank]:
                    # 确保时间戳和指标数据长度匹配
                    x = timestamps[rank][:len(data_dict[rank])]
                    y = data_dict[rank][:len(x)]
                    
                    # 使用更清晰的线条和标记
                    ax.plot(x, y, 
                           label=f'Rank {rank}',
                           color=palette[i],
                           linewidth=1.5,
                           marker='o',
                           markersize=4,
                           markeredgewidth=0.5,
                           markeredgecolor='white')
            
            # 优化X轴时间显示
            plt.xticks(rotation=30, ha='right')
            
            # 设置Y轴为整数刻度（适用于Batch和Num Waiting）
            if title != 'KV Cache Usage':
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            # 添加标签和标题
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f'{title} (Ranks {current_ranks[0]}-{current_ranks[-1]})', fontsize=11, pad=10)
            
            # 优化图例位置和样式
            legend = ax.legend(bbox_to_anchor=(1.05, 1), 
                              loc='upper left',
                              borderaxespad=0.,
                              frameon=True,
                              fontsize=8,
                              title='Rank',
                              title_fontsize=9)
            legend.get_frame().set_alpha(0.8)
            
            # 调整边距
            plt.tight_layout()
            plt.subplots_adjust(right=0.8)  # 为图例留出空间
            
            # 保存图像
            output_filename = f"{filename_prefix}_ranks_{current_ranks[0]}-{current_ranks[-1]}.png"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            
            # 打印保存路径
            print(f"Saved {title} plot (Ranks {current_ranks[0]}-{current_ranks[-1]}) to: {os.path.abspath(output_path)}")
            saved_files.append(os.path.abspath(output_path))
    
    return saved_files

def main():
    parser = argparse.ArgumentParser(description='Parse metrics from log file and generate plots.')
    parser.add_argument('log_file', nargs='?', help='Path to the log file', default="/nvme4/share/chenjiefei/scripts/server/2p4d_decode_256_0.65_2025-08-05_07-33/dp32ep32_Decode_node0.log")
    parser.add_argument('output_dir', nargs='?', help='Directory to save the plots', default="/nvme4/share/chenjiefei/src/lmdeploy/plot")
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
    
    # 生成图表
    saved_files = plot_metrics(args.output_dir, timestamps, batch_data, waiting_data, kv_cache_data)
    
    print("\nAll plots saved successfully:")
    for file_path in saved_files:
        print(f"- {file_path}")

if __name__ == "__main__":
    main()