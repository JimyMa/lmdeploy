import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import json

# ========== 配置部分 ==========
JSON_PATH = "/nvme4/share/chenjiefei/dataset/tokenized_sharegpt/sharegpt_data_filtered.json"  # 请替换为您的实际路径
SAVE_DIR = "/nvme4/share/chenjiefei/src/lmdeploy/dataset_process/plot"  # 保存图片的目录路径
# =============================

def extract_prefix(path):
    """从文件路径提取前缀（文件名不带扩展名）"""
    return Path(path).stem

def plot_cdf(series, title, filename, color, save_dir=SAVE_DIR):
    plt.figure(figsize=(10, 6))
    
    # 计算CDF
    sorted_data = np.sort(series.dropna())
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    
    # 绘制主曲线
    sns.lineplot(x=sorted_data, y=cdf, color=color, linewidth=2.5, label='CDF')
    
    # 添加统计标记 - 新增了均值(mean)的计算和显示
    median = np.median(sorted_data)
    mean_val = np.mean(sorted_data)  # 计算均值
    p90 = np.percentile(sorted_data, 90)
    
    plt.axvline(median, color='red', linestyle='--', linewidth=1, alpha=0.7)
    plt.axvline(mean_val, color='green', linestyle='-.', linewidth=1, alpha=0.7)  # 均值线
    plt.axvline(p90, color='purple', linestyle=':', linewidth=1, alpha=0.7)
    
    # 标注关键点 - 新增了均值的标注
    plt.text(median*1.05, 0.45, f'Median: {median:.1f}', color='red', fontsize=10)
    plt.text(mean_val*1.05, 0.6, f'Mean: {mean_val:.1f}', color='green', fontsize=10)  # 均值标注
    plt.text(p90*1.05, 0.8, f'P90: {p90:.1f}', color='purple', fontsize=10)
    
    # 美化样式
    plt.title(f'CDF of {title} Tokens', fontsize=14, pad=15)
    plt.xlabel(f'Number of {title} Tokens', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.grid(True, alpha=0.2)
    sns.despine()
    
    # 确保保存目录存在
    save_path = Path(save_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 打印完整保存路径
    print(f"图表已保存至: {save_path.resolve()}")

# 主流程
try:
    # 读取JSON数据
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取prompt_len和output_len字段
    prompt_lengths = [item['prompt_len'] for item in data if 'prompt_len' in item]
    output_lengths = [item['output_len'] for item in data if 'output_len' in item]
    
    # 创建包含提取数据的DataFrame
    df = pd.DataFrame({
        'prompt_len': prompt_lengths,
        'output_len': output_lengths
    })
    
    prefix = extract_prefix(JSON_PATH)
    
    # 绘制prompt长度的CDF图
    plot_cdf(df['prompt_len'], 
             "Prompt Length", 
             f"{prefix}_prompt_length_cdf.png",
             color='#1f77b4')  # 蓝色系
    
    # 绘制output长度的CDF图
    plot_cdf(df['output_len'], 
             "Output Length", 
             f"{prefix}_output_length_cdf.png",
             color='#ff7f0e')  # 橙色系
    
    print(f"成功生成图表: {prefix}_prompt_length_cdf.png 和 {prefix}_output_length_cdf.png")
    
except Exception as e:
    print(f"处理失败: {str(e)}")
