import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def plot_metrics_comparison(strategy_files, output_dir):
    """
    绘制多种调度策略的server_inference_tpot、queueing_latency和tpot的CDF对比图
    并打印每个指标的不同调度策略的P50，P95和P99值
    
    参数:
    strategy_files: 字典，格式为 {策略标识: CSV文件路径}，策略标识可以是 'kvcache', 'batchsize', 'roundrobin' 或自定义标识
    output_dir: 图片保存目录
    """
    # 策略名称和对应的颜色/线型 - 使用更专业的调色板
    default_strategies = {
        'kvcache': {'name': 'KVCache Balance', 'color': '#4E79A7', 'linestyle': '-', 'marker': 'o'},
        'batchsize': {'name': 'Batch Size Balance', 'color': '#F28E2B', 'linestyle': '--', 'marker': 's'},
        'roundrobin': {'name': 'Round Robin', 'color': '#E15759', 'linestyle': '-.', 'marker': '^'}
    }
    
    # 创建实际使用的策略配置，允许自定义策略
    strategies = {}
    for strategy_id, file_path in strategy_files.items():
        if strategy_id in default_strategies:
            strategies[strategy_id] = default_strategies[strategy_id]
        else:
            # 为自定义策略分配默认样式（循环使用）
            colors = ['#59A14F', '#8CD17D', '#B6992D', '#499894', '#79706E']
            linestyles = ['-', '--', '-.', ':']
            markers = ['D', 'X', '*', 'P', 'H']
            
            idx = len(strategies)
            strategies[strategy_id] = {
                'name': strategy_id.capitalize(),
                'color': colors[idx % len(colors)],
                'linestyle': linestyles[idx % len(linestyles)],
                'marker': markers[idx % len(markers)]
            }
    
    # 初始化数据结构
    metrics_data = {
        'server_inference_tpot': {key: [] for key in strategies},
        'queueing_latency': {key: [] for key in strategies},
        'tpot': {key: [] for key in strategies}
    }
    
    # 读取并处理CSV文件
    for strategy_id, file_path in strategy_files.items():
        print(f"正在处理策略: {strategies[strategy_id]['name']}, 文件: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 处理server_inference_tpot - 转换为毫秒
                    try:
                        tpot_val = float(row['server_inference_tpot']) * 1000  # 秒转毫秒
                        if tpot_val > 0:  # 过滤无效值
                            metrics_data['server_inference_tpot'][strategy_id].append(tpot_val)
                    except (ValueError, KeyError):
                        pass
                    
                    # 处理queueing_latency - 保持秒为单位
                    try:
                        queue_val = float(row['queueing_latency'])
                        if queue_val >= 0:  # 过滤负值
                            metrics_data['queueing_latency'][strategy_id].append(queue_val)
                    except (ValueError, KeyError):
                        pass
                    
                    # 处理tpot - 转换为毫秒
                    try:
                        tpot_val = float(row['tpot']) * 1000  # 秒转毫秒
                        if tpot_val > 0:  # 过滤无效值
                            metrics_data['tpot'][strategy_id].append(tpot_val)
                    except (ValueError, KeyError):
                        pass
        except FileNotFoundError:
            print(f"警告: 文件不存在 {file_path}")
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
    
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
        
        for strategy_id in strategies:
            data = metrics_data[metric][strategy_id]
            if not data:
                print(f"警告: {strategies[strategy_id]['name']} 策略的 {metric} 无有效数据")
                continue
            
            # 计算CDF
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            all_data.extend(sorted_data)
            
            # 绘制曲线
            plt.plot(
                sorted_data, 
                cdf, 
                label=strategies[strategy_id]['name'],
                color=strategies[strategy_id]['color'],
                linestyle=strategies[strategy_id]['linestyle'],
                linewidth=2.5
            )
        
        # 如果没有有效数据，跳过此指标
        if not all_data:
            print(f"警告: 指标 {metric} 无有效数据，跳过绘图")
            plt.close()
            continue
            
        # 设置图例和比例尺
        plt.legend(loc='lower right', frameon=True, framealpha=0.9)
        plt.ylim(0, 1)
        
        # 设置X轴范围 - 确保从最小值开始
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
    
    # 打印每个指标的不同调度策略的P50，P95和P99值
    print("\n" + "="*80)
    print("各指标在不同调度策略下的百分位数统计 (P50, P95, P99)")
    print("="*80)
    
    # 定义要计算的百分位数
    percentiles = [50, 95, 99]
    
    for metric in metrics_data:
        # 根据指标设置单位描述
        if metric == 'queueing_latency':
            unit = "秒"
        else:
            unit = "毫秒"
            
        title_map = {
            'server_inference_tpot': "Server Inference TPOT",
            'queueing_latency': "Queueing Latency",
            'tpot': "Total TPOT"
        }
        
        print(f"\n{title_map[metric]} ({unit}):")
        print("-" * (len(title_map[metric]) + len(unit) + 4))
        
        # 打印表头
        header = f"{'策略名称':<20}"
        for p in percentiles:
            header += f"P{p:<8}"
        print(header)
        print("-" * len(header))
        
        # 打印每个策略的百分位数
        for strategy_id in strategies:
            data = metrics_data[metric][strategy_id]
            if not data:
                print(f"{strategies[strategy_id]['name']:<20} 无有效数据")
                continue
            
            # 计算百分位数
            p_values = np.percentile(data, percentiles)
            
            # 格式化输出
            line = f"{strategies[strategy_id]['name']:<20}"
            for p, val in zip(percentiles, p_values):
                # 根据数值大小调整显示精度
                if val < 1:
                    line += f"{val:.4f}   "
                elif val < 10:
                    line += f"{val:.3f}   "
                else:
                    line += f"{val:.2f}   "
            print(line)

# 使用示例
if __name__ == "__main__":
    # 创建策略文件字典，可自由组合策略
    strategy_files = {
        'kvcache': "/nvme2/share/chenjiefei/scripts/csv/request_details_lmdeploy_86518964eaef84e3fdd98e9861759a1384f9c29d_rate_56.0_90000req_20250821_125521_rate56_kvcache_balance_0821.csv",
        'batchsize': "/nvme2/share/chenjiefei/scripts/csv/request_details_lmdeploy_86518964eaef84e3fdd98e9861759a1384f9c29d_rate_56.0_90000req_20250821_030426_rate56_batchsize_balance_0821.csv",
        # 'roundrobin': "/nvme4/share/chenjiefei/scripts/csv/request_details_lmdeploy_..._roundrobin.csv"
        # 可以添加自定义策略: 'custom': "/path/to/custom_strategy.csv"
    }
    
    # 替换为你想保存图片的目录
    output_dir = "/nvme2/share/chenjiefei/src/lmdeploy/plot/cdf/ratio_0.01_rate_56/"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制对比图并打印百分位数
    plot_metrics_comparison(strategy_files, output_dir)
