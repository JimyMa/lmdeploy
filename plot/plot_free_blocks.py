import re
import os
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def parse_log_and_plot(log_file_path, num_ranks, save_folder, show_instance_legend=False):
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
    
    # 存储所有时间戳
    all_timestamps = []

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
    
    # 提取每个有效时间戳下的指标数据
    for ts_str in sorted_timestamps:
        node_lines = valid_timestamps[ts_str]
        total_free_blocks = 0
        
        # 记录时间戳
        all_timestamps.append(ts_str)
        
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

    # 获取第一个时间戳用于计算相对时间
    first_timestamp = parse_timestamp(sorted_timestamps[0])
    
    # 计算相对时间（秒）
    relative_times = []
    for ts_str in sorted_timestamps:
        ts_dt = parse_timestamp(ts_str)
        sec_from_start = (ts_dt - first_timestamp).total_seconds()
        relative_times.append(sec_from_start)

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
    
    # 绘制Free Blocks折线（使用较高的zorder确保在顶层）
    ax.plot(relative_times, free_blocks_total, 'b-', linewidth=3, label='Total Free Blocks', zorder=2)
    
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
    
    plt.title('Free Blocks and Wait First Block Over Time')
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
    
    # 绘制Free Blocks折线（使用较高的zorder确保在顶层）
    ax.plot(relative_times, free_blocks_total, 'b-', linewidth=2, label='Total Free Blocks', zorder=2)
    
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
    
    plt.title('Free Blocks and Wait Num Block Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(second_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {save_folder}:")
    print(f"  - {first_plot_path}")
    print(f"  - {second_plot_path}")

# 示例使用
if __name__ == '__main__':
    log_file_path = "/nvme2/share/chenjiefei/scripts/proxy_log/proxy_res_20250820_230400_501.log"  # 请替换为您的log文件路径
    num_ranks = 32             # 请替换为您的RANK数量
    save_folder = "/nvme2/share/chenjiefei/src/lmdeploy/plot/test"      # 请替换为您想要保存的文件夹路径
    
    # 默认不显示instance图例
    parse_log_and_plot(log_file_path, num_ranks, save_folder)
    
    # 如果需要显示instance图例，可以这样调用
    # parse_log_and_plot(log_file_path, num_ranks, save_folder, show_instance_legend=True)
    
    # 如果需要显示instance图例，可以这样调用
    # parse_log_and_plot(log_file_path, num_ranks, save_folder, show_instance_legend=True)
    
# /nvme2/share/chenjiefei/scripts/proxy_log/proxy_res_20250820_213220_214.log