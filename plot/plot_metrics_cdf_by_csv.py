import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def plot_metrics_comparison(csv_files, output_dir):
    """
    绘制三种调度策略的server_inference_tpot、queueing_latency和tpot的CDF对比图
    
    参数:
    csv_files: 包含三个CSV文件路径的列表，顺序为[kvcache, batchsize, roundrobin]
    output_dir: 图片保存目录
    """
    # 验证输入
    if len(csv_files) != 3:
        raise ValueError("需要提供三个CSV文件路径 [kvcache, batchsize, roundrobin]")
    
    # 策略名称和对应的颜色/线型 - 使用更专业的调色板
    strategies = {
        'kvcache': {'name': 'KVCache Balance', 'color': '#4E79A7', 'linestyle': '-', 'marker': 'o'},
        'batchsize': {'name': 'Batch Size Balance', 'color': '#F28E2B', 'linestyle': '--', 'marker': 's'},
        'roundrobin': {'name': 'Round Robin', 'color': '#E15759', 'linestyle': '-.', 'marker': '^'}
    }
    
    # 初始化数据结构
    metrics_data = {
        'server_inference_tpot': {key: [] for key in strategies},
        'queueing_latency': {key: [] for key in strategies},
        'tpot': {key: [] for key in strategies}
    }
    
    # 读取并处理CSV文件
    for i, file_path in enumerate(csv_files):
        strategy = list(strategies.keys())[i]
        print(f"正在处理策略: {strategies[strategy]['name']}, 文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 处理server_inference_tpot - 转换为毫秒
                try:
                    tpot_val = float(row['server_inference_tpot']) * 1000  # 秒转毫秒
                    if tpot_val > 0:  # 过滤无效值
                        metrics_data['server_inference_tpot'][strategy].append(tpot_val)
                except (ValueError, KeyError):
                    pass
                
                # 处理queueing_latency - 保持秒为单位
                try:
                    queue_val = float(row['queueing_latency'])
                    if queue_val >= 0:  # 过滤负值
                        metrics_data['queueing_latency'][strategy].append(queue_val)
                except (ValueError, KeyError):
                    pass
                
                # 处理tpot - 转换为毫秒
                try:
                    tpot_val = float(row['tpot']) * 1000  # 秒转毫秒
                    if tpot_val > 0:  # 过滤无效值
                        metrics_data['tpot'][strategy].append(tpot_val)
                except (ValueError, KeyError):
                    pass
    
    # 设置更专业的绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.figsize': (10, 6.5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.25,
        'axes.linewidth': 0.8
    })
    
    # 为每个指标绘制CDF图
    for metric in metrics_data:
        plt.figure()
        
        # 自定义标题和标签
        title_map = {
            'server_inference_tpot': "Server Inference TPOT (Time Per Output Token)",
            'queueing_latency': "Queueing Latency",
            'tpot': "TPOT (Total Time Per Output Token)"
        }
        
        # 根据指标设置不同的单位
        if metric == 'queueing_latency':
            unit = "seconds"
        else:
            unit = "milliseconds"
            
        label_map = {
            'server_inference_tpot': f"Time ({unit})",
            'queueing_latency': f"Time ({unit})",
            'tpot': f"Time ({unit})"
        }
        
        plt.title(f"{title_map[metric]} CDF Comparison", pad=15)
        plt.xlabel(label_map[metric])
        plt.ylabel("Cumulative Probability")
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        
        # 只保留主要网格线，简化背景
        plt.grid(True, axis='y', linestyle='-', alpha=0.3)
        plt.grid(True, axis='x', linestyle='-', alpha=0.1)
        
        # 收集所有策略的数据用于设置坐标轴范围
        all_data = []
        
        for strategy in strategies:
            data = metrics_data[metric][strategy]
            if not data:
                print(f"警告: {strategies[strategy]['name']} 策略的 {metric} 无有效数据")
                continue
            
            # 计算CDF
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            all_data.extend(sorted_data)
            
            # 绘制曲线
            plt.plot(
                sorted_data, 
                cdf, 
                label=strategies[strategy]['name'],
                color=strategies[strategy]['color'],
                linestyle=strategies[strategy]['linestyle'],
                linewidth=2.5
            )
        
        # 设置图例和比例尺
        plt.legend(loc='lower right', frameon=True, framealpha=0.9)
        plt.ylim(0, 1)
        
        # 设置X轴范围 - 确保从最小值开始
        if all_data:
            min_val = max(min(all_data) * 0.95, 0.001)  # 避免0值问题，留5%边距
            max_val = max(all_data) * 1.05  # 留5%边距
            
            # 根据数据范围决定是否使用对数坐标
            log_scale_needed = max_val / min_val > 100
            if log_scale_needed:
                plt.xscale('log')
                plt.xlim(min_val, max_val)
            else:
                plt.xscale('linear')
                plt.xlim(min_val, max_val)
        
        # 保存图片
        save_path = os.path.join(output_dir, f"{metric}_cdf_comparison.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"已保存图片: {save_path}")
        plt.close()

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际CSV文件路径
    csv_files = [
        "/nvme4/share/chenjiefei/scripts/csv/request_details_lmdeploy_86518964eaef84e3fdd98e9861759a1384f9c29d_rate_70.0_40000req_20250806_212121_40000_kvcache_balance.csv",      # KVCache Balance策略
        "/nvme4/share/chenjiefei/scripts/csv/request_details_lmdeploy_86518964eaef84e3fdd98e9861759a1384f9c29d_rate_70.0_40000req_20250806_215206_40000_batchsize_balance.csv",    # Batch Size Balance策略
        "/nvme4/share/chenjiefei/scripts/csv/request_details_lmdeploy_86518964eaef84e3fdd98e9861759a1384f9c29d_rate_70.0_40000req_20250806_201803_40000_roundrobin.csv"    # Round Robin策略
    ]
    
    # 替换为你想保存图片的目录
    output_dir = "/nvme4/share/chenjiefei/src/lmdeploy/plot/cdf/"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制对比图
    plot_metrics_comparison(csv_files, output_dir)