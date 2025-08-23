#!/usr/bin/env python3
# draw_bs.py
import re
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import interpolate

# --------------------------------------------------
# 1. 解决中文标题乱码：全局字体
# --------------------------------------------------
plt.rcParams["font.family"] = "Noto Sans CJK SC"   # 没有就改成 "PingFang SC" 或英文标题
sns.set_theme(
    style="whitegrid",
    rc={
        "font.size": 11,

        # 坐标轴
        "axes.linewidth": 0.8,
        "axes.edgecolor": ".2",

        # 网格
        "grid.linewidth": 0.3,
        "grid.alpha": 0.3,

        # 刻度
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        # 图例
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fontsize": 10,
        "legend.edgecolor": "0.8",

        # 线条
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
    },
)

# 使用更专业的配色方案
PAL = sns.color_palette("husl", 8)  # 更鲜明的配色


# --------------------------------------------------
# 2. 日志解析
# --------------------------------------------------
def extract_batch_sizes(log_file_path: Path):
    pattern = re.compile(r"RANK0 rebalance_batch_size: (\d+), max_batch_size: (\d+)")

    rebalance_sizes, max_sizes, iterations = [], [], []
    with log_file_path.open() as f:
        for idx, line in enumerate(f, 1):
            m = pattern.search(line)
            if m:
                rebalance_sizes.append(int(m.group(1)))
                max_sizes.append(int(m.group(2)))
                iterations.append(idx)
    return iterations, rebalance_sizes, max_sizes


# --------------------------------------------------
# 3. 绘图 - 改进版本
# --------------------------------------------------
def plot_batch_sizes(
    iterations,
    rebalance_sizes,
    max_sizes,
    *,
    title: str,
    output_file: Path,
):
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    
    # 如果数据点太多，可以适当稀疏标记点
    markevery = max(1, len(iterations) // 15)
    
    # 绘制 Rebalance Batch Size - 使用更粗的实线
    ax.plot(
        iterations,
        rebalance_sizes,
        label="Rebalance Batch Size",
        color=PAL[2],  # 蓝色系
        marker="o",
        markevery=markevery,
        linestyle="-",
        linewidth=2.2,
        markersize=5,
        markerfacecolor='white',
        markeredgewidth=1.5,
        alpha=0.9
    )
    
    # 绘制 Max Batch Size - 使用虚线样式
    ax.plot(
        iterations,
        max_sizes,
        label="Max Batch Size",
        color=PAL[0],  # 红色系
        marker="s",
        markevery=markevery,
        linestyle="--",
        linewidth=2.0,
        markersize=5,
        markerfacecolor='white',
        markeredgewidth=1.5,
        alpha=0.9
    )
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Batch Size", fontsize=12)
    
    # 美化坐标轴
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # 添加网格
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # 设置图例
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, ncol=2)
    
    # 自动调整y轴范围，留出一些边距
    y_min = min(min(rebalance_sizes), min(max_sizes))
    y_max = max(max(rebalance_sizes), max(max_sizes))
    ax.set_ylim(y_min * 0.95, y_max * 1.05)
    
    # 如果迭代次数很多，可以适当调整x轴显示
    if len(iterations) > 50:
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    
    sns.despine(ax=ax, left=True, bottom=True)
    fig.tight_layout()
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"✅ 已保存 → {output_file}")


# --------------------------------------------------
# 4. 命令行入口
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="对比两种调度策略的 batch size 变化"
    )
    parser.add_argument(
        "--kv-cache-log",
        type=Path,
        default=Path(
            "/nvme2/share/chenjiefei/scripts/server/2p4d_decode_256_0.85_2025-08-21_04-13_rate56_kvcache_balance_0821/dp32ep32_Decode_node0.log"
        ),
        help="KV Cache 均衡调度策略日志",
    )
    parser.add_argument(
        "--batch-size-log",
        type=Path,
        default=Path(
            "/nvme2/share/chenjiefei/scripts/server/2p4d_decode_256_0.85_2025-08-20_18-27_rate56_batchsize_balance_0821/dp32ep32_Decode_node0.log"
        ),
        help="Batch Size 均衡调度策略日志",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/nvme2/share/chenjiefei/src/lmdeploy/plot/rebalance_bs/ratio_0.01_rate_56"),
        help="图表输出文件夹",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # KV Cache 策略
    iters, reb, max_ = extract_batch_sizes(args.kv_cache_log)
    if iters:
        plot_batch_sizes(
            iters,
            reb,
            max_,
            title="KV Cache Balance Strategy - Batch Size",
            # 若中文仍有问题，可换成英文：
            # title="KV-Cache Balanced Scheduling – Batch Size Trend",
            output_file=args.output_dir / "kv_cache_strategy_batch_sizes.png",
        )
    else:
        print("⚠️ KV Cache 日志为空")

    # Batch Size 策略
    iters, reb, max_ = extract_batch_sizes(args.batch_size_log)
    if iters:
        plot_batch_sizes(
            iters,
            reb,
            max_,
            title="Batch Size Balance Strategy - Batch Size",
            # title="Batch-Size Balanced Scheduling – Batch Size Trend",
            output_file=args.output_dir / "batch_size_strategy_batch_sizes.png",
        )
    else:
        print("⚠️ Batch Size 日志为空")


if __name__ == "__main__":
    main()