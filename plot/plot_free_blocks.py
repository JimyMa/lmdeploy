import re
import os
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_log_and_plot(log_file_path, num_ranks, save_folder, time_interval=9.5, skip_head=0, skip_tail=0, show_instance_legend=False):
    # 确保保存文件夹存在
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # 读取log文件
    with open(log_file_path, 'r') as f:
        lines = f.readlines()

    # 正则表达式匹配时间戳（格式如[2025-08-20 21:32:23.2]）
    timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d)\]'
    
    # 存储每个时间戳下的节点行
    timestamp_data = defaultdict(list)
    
    # 存储当前时间戳
    current_timestamp = None
    
    for line in lines:
        # 检查行是否包含时间戳
        match = re.search(timestamp_pattern, line)
        if match:
            current_timestamp = match.group(1)
        
        # 如果当前时间戳不为空且行包含"Node:"，则添加到对应时间戳的列表
        if current_timestamp and "Node:" in line:
            timestamp_data[current_timestamp].append(line)

    # 只保留那些节点数量等于num_ranks的时间戳
    valid_timestamps = {}
    for ts, node_lines in timestamp_data.items():
        if len(node_lines) == num_ranks:
            valid_timestamps[ts] = node_lines

    # 如果没有有效数据，则退出
    if not valid_timestamps:
        print(f"No valid timestamps found with exactly {num_ranks} ranks.")
        print(f"Found timestamps with node counts: {[(ts, len(nodes)) for ts, nodes in timestamp_data.items()]}")
        return

    # 初始化数据结构存储指标数据
    free_blocks_total = []  # 存储每个时间戳的free_blocks总和
    wait_first_block_data = defaultdict(list)  # key: rank, value: list of values for each timestamp
    wait_num_block_data = defaultdict(list)    # key: rank, value: list of values for each timestamp
    
    # 存储所有时间戳字符串和对应的datetime对象
    all_timestamp_strings = []
    all_timestamp_objects = []

    # 函数解析时间戳字符串为datetime对象
    def parse_timestamp(ts_str):
        # 添加秒的小数部分（如果缺少）
        if '.' not in ts_str:
            ts_str += '.0'
        
        # 确保小数部分只有一位
        base, fraction = ts_str.split('.')
        fraction = fraction[0]  # 只取第一位小数
        
        # 构建完整的时间字符串
        full_ts_str = f"{base}.{fraction}"
        
        try:
            return datetime.strptime(full_ts_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            # 如果解析失败，尝试不使用微秒
            return datetime.strptime(base, "%Y-%m-%d %H:%M:%S")

    # 获取所有有效时间戳并排序
    sorted_timestamps = sorted(valid_timestamps.keys(), key=lambda x: parse_timestamp(x))
    
    # 存储原始时间戳对象
    original_timestamp_objects = [parse_timestamp(ts) for ts in sorted_timestamps]
    
    # 应用skip_head和skip_tail
    if skip_head > 0:
        sorted_timestamps = sorted_timestamps[skip_head:]
        original_timestamp_objects = original_timestamp_objects[skip_head:]
    if skip_tail > 0 and len(sorted_timestamps) > skip_tail:
        sorted_timestamps = sorted_timestamps[:-skip_tail]
        original_timestamp_objects = original_timestamp_objects[:-skip_tail]
    
    # 如果没有剩余时间戳，则退出
    if not sorted_timestamps:
        print(f"No timestamps remaining after skipping head ({skip_head}) and tail ({skip_tail}).")
        return
    
    # 提取每个有效时间戳下的指标数据
    for ts_str in sorted_timestamps:
        node_lines = valid_timestamps[ts_str]
        total_free_blocks = 0
        
        # 记录时间戳
        all_timestamp_strings.append(ts_str)
        all_timestamp_objects.append(parse_timestamp(ts_str))
        
        for line in node_lines:
            # 提取RANK
            rank_match = re.search(r'Node: (\d+)', line)
            if not rank_match:
                continue
            rank = int(rank_match.group(1))
            
            # 提取free_blocks
            free_blocks_match = re.search(r'Free Blocks: (\d+)', line)
            if free_blocks_match:
                free_blocks_value = int(free_blocks_match.group(1))
                total_free_blocks += free_blocks_value
            
            # 提取wait_first_block
            wait_first_block_match = re.search(r'Wait First Block: (\d+)', line)
            if wait_first_block_match:
                wait_first_block_value = int(wait_first_block_match.group(1))
                wait_first_block_data[rank].append(wait_first_block_value)
            
            # 提取wait_num_block
            wait_num_block_match = re.search(r'Wait Num Block: (\d+)', line)
            if wait_num_block_match:
                wait_num_block_value = int(wait_num_block_match.group(1))
                wait_num_block_data[rank].append(wait_num_block_value)
        
        free_blocks_total.append(total_free_blocks)

    # 打印free blocks信息
    if free_blocks_total:
        max_free_blocks = max(free_blocks_total)
        total_records = len(free_blocks_total)
        print(f"Free Blocks - Max value: {max_free_blocks}, Total records: {total_records}")
    else:
        print("No free blocks data found")
        return

    # 计算相对时间（秒），以第一个保留的时间戳为起点
    if all_timestamp_objects:
        first_timestamp_after_skip = all_timestamp_objects[0]
        relative_times = []
        for ts_dt in all_timestamp_objects:
            sec_from_start = (ts_dt - first_timestamp_after_skip).total_seconds()
            relative_times.append(sec_from_start)
        
        print(f"Time range after skipping: {relative_times[0]:.1f}s to {relative_times[-1]:.1f}s")
    else:
        print("No timestamps available after skipping")
        return

    # 准备堆叠柱状图数据
    wait_first_stacked = []
    wait_num_stacked = []
    
    # 对于每个RANK，提取其值
    for rank in sorted(wait_first_block_data.keys()):
        wait_first_stacked.append(wait_first_block_data[rank])
        wait_num_stacked.append(wait_num_block_data[rank])
    
    # 转换为numpy数组便于处理
    wait_first_stacked = np.array(wait_first_stacked)
    wait_num_stacked = np.array(wait_num_stacked)
    
    # 计算每个时间戳的总和
    wait_first_total = np.sum(wait_first_stacked, axis=0)
    wait_num_total = np.sum(wait_num_stacked, axis=0)

    # 创建颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, num_ranks))

    # 构建保存文件路径
    first_plot_path = os.path.join(save_folder, 'free_blocks_and_wait_first.png')
    second_plot_path = os.path.join(save_folder, 'free_blocks_and_wait_num.png')

    # 计算第一个图的最大数据值，用于对齐刻度
    max_free_blocks = max(free_blocks_total) if free_blocks_total else 0
    max_wait_first = max(wait_first_total) if len(wait_first_total) > 0 else 0
    first_plot_max = max(max_free_blocks, max_wait_first) * 1.1  # 增加10%的余量

    # 绘制第一个图：Free Blocks折线 + Wait First Block堆叠柱状图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制Wait First Block堆叠柱状图
    bottom = np.zeros(len(relative_times))
    for i, rank in enumerate(sorted(wait_first_block_data.keys())):
        # 只有当需要显示图例时才设置label
        label = f'Instance {rank}' if show_instance_legend else None
        ax.bar(relative_times, wait_first_block_data[rank], bottom=bottom, 
               color=colors[i], width=5, label=label, alpha=0.7, zorder=1)
        bottom += wait_first_block_data[rank]
    
    # 绘制Free Blocks折线（使用较高的zorder确保在顶层，使用虚线，统一线宽为3）
    ax.plot(relative_times, free_blocks_total, 'b--', linewidth=3, label='Total Free Blocks', zorder=2)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Num of Blocks')
    ax.set_ylim(0, first_plot_max)  # 使用统一的最大值
    
    # 添加图例
    if show_instance_legend:
        ax.legend(loc='upper left')
    else:
        # 只显示折线图的图例
        handles, labels = ax.get_legend_handles_labels()
        # 找到折线图的句柄
        line_handle = None
        for handle, label in zip(handles, labels):
            if label == 'Total Free Blocks':
                line_handle = handle
                break
        if line_handle:
            ax.legend([line_handle], ['Total Free Blocks'], loc='upper left')
    
    plt.title(f'Free Blocks and Wait First Block Over Time\n(Skipped head: {skip_head}, tail: {skip_tail})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(first_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 计算第二个图的最大数据值，用于对齐刻度
    max_wait_num = max(wait_num_total) if len(wait_num_total) > 0 else 0
    second_plot_max = max(max_free_blocks, max_wait_num) * 1.1  # 增加10%的余量

    # 绘制第二个图：Free Blocks折线 + Wait Num Block堆叠柱状图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制Wait Num Block堆叠柱状图
    bottom = np.zeros(len(relative_times))
    for i, rank in enumerate(sorted(wait_num_block_data.keys())):
        # 只有当需要显示图例时才设置label
        label = f'Instance {rank}' if show_instance_legend else None
        ax.bar(relative_times, wait_num_block_data[rank], bottom=bottom, 
               color=colors[i], width=5, label=label, alpha=0.7, zorder=1)
        bottom += wait_num_block_data[rank]
    
    # 绘制Free Blocks折线（使用较高的zorder确保在顶层，使用虚线，统一线宽为3）
    ax.plot(relative_times, free_blocks_total, 'b--', linewidth=3, label='Total Free Blocks', zorder=2)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Num of Blocks')
    ax.set_ylim(0, second_plot_max)  # 使用统一的最大值
    
    # 添加图例
    if show_instance_legend:
        ax.legend(loc='upper left')
    else:
        # 只显示折线图的图例
        handles, labels = ax.get_legend_handles_labels()
        # 找到折线图的句柄
        line_handle = None
        for handle, label in zip(handles, labels):
            if label == 'Total Free Blocks':
                line_handle = handle
                break
        if line_handle:
            ax.legend([line_handle], ['Total Free Blocks'], loc='upper left')
    
    plt.title(f'Free Blocks and Wait Num Block Over Time\n(Skipped head: {skip_head}, tail: {skip_tail})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(second_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {save_folder}:")
    print(f"  - {first_plot_path}")
    print(f"  - {second_plot_path}")

# 主函数，添加命令行参数解析
def main():
    parser = argparse.ArgumentParser(description='Parse proxy log and plot metrics.')
    parser.add_argument('--log_file', type=str, 
                        default='/nvme2/share/chenjiefei/scripts/proxy_log/proxy_res_20250820_230400_501.log',
                        help='Path to the log file')
    parser.add_argument('--num_ranks', type=int, default=32,
                        help='Number of ranks/nodes')
    parser.add_argument('--save_folder', type=str, 
                        default='/nvme2/share/chenjiefei/src/lmdeploy/plot/test',
                        help='Directory to save output plots')
    parser.add_argument('--time_interval', type=float, default=9.5,
                        help='Time interval in minutes for each plot (default: 9.5)')
    parser.add_argument('--skip_head', type=int, default=600,
                        help='Skip the first N records for each rank/node (default: 0)')
    parser.add_argument('--skip_tail', type=int, default=600,
                        help='Skip the last N records for each rank/node (default: 0)')
    parser.add_argument('--show_instance_legend', action='store_true',
                        help='Show legend for individual instances')
    
    args = parser.parse_args()
    
    parse_log_and_plot(
        log_file_path=args.log_file,
        num_ranks=args.num_ranks,
        save_folder=args.save_folder,
        time_interval=args.time_interval,
        skip_head=args.skip_head,
        skip_tail=args.skip_tail,
        show_instance_legend=args.show_instance_legend
    )

# 示例使用
if __name__ == '__main__':
    # 使用命令行参数
    main()
    
    # 或者直接调用（保留原有接口）
    # log_file_path = "/nvme2/share/chenjiefei/scripts/proxy_log/proxy_res_20250820_230400_501.log"
    # num_ranks = 32
    # save_folder = "/nvme2/share/chenjiefei/src/lmdeploy/plot/test"
    # parse_log_and_plot(log_file_path, num_ranks, save_folder)