#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import matplotlib.dates as mdates
import math
import argparse
import os

# -------------------------------------------------
# 字体配置
# -------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'SimSun', 'WenQuanYi Micro Hei', 'Heiti TC'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.8
})

# -------------------------------------------------
# 工具函数
# -------------------------------------------------
def get_distinct_colors(count):
    base_colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
        '#FFA500', '#800080', '#008000', '#000080', '#808000', '#800000',
        '#008080', '#C0C0C0', '#000000'
    ]
    if count <= len(base_colors):
        return base_colors[:count]
    colors = []
    for i in range(count):
        hue = i / count
        rgb = plt.cm.hsv(hue)
        colors.append(rgb)
    return colors

# -------------------------------------------------
# 日志解析
# -------------------------------------------------
def parse_log_data(log_str, skip_head=0, skip_tail=0):
    pattern = (r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - '
               r'Node: (\d+), '
               r'KV Cache Usage: ([\d.]+), '
               r'Total Tokens: (\d+), '
               r'Running Requests: (\d+), '
               r'Waiting Requests: (\d+), '
               r'Batch Size: (\d+)')

    node_raw = {}
    for line in log_str.strip().split('\n'):
        match = re.search(pattern, line)
        if not match:
            continue
        ts_str, nid, kv, total_tokens, run, wait, batch = match.groups()
        ts          = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
        nid         = int(nid)
        kv          = float(kv)
        total_tokens= int(total_tokens)
        run         = int(run)
        wait        = int(wait)
        batch       = int(batch)
        node_raw.setdefault(nid, []).append((ts, kv, total_tokens, run, wait, batch))

    node_data = {}
    for nid, records in node_raw.items():
        records.sort(key=lambda x: x[0])
        total = len(records)
        head  = max(0, min(skip_head, total))
        tail  = max(0, min(skip_tail, total - head))
        sliced = records[head: total - tail]
        if not sliced:
            continue
        node_data[nid] = {
            'timestamps':   [r[0] for r in sliced],
            'kv_cache':     [r[1] for r in sliced],
            'total_tokens': [r[2] for r in sliced],
            'running':      [r[3] for r in sliced],
            'waiting':      [r[4] for r in sliced],
            'batch_size':   [r[5] for r in sliced],
        }

    data = {
        'timestamps': [],
        'nodes': {},
        'metrics': {
            'kv_cache':      {},
            'total_tokens':  {},
            'running':       {},
            'waiting':       {},
            'batch_size':    {}
        }
    }
    all_ts = []
    for nid, node in node_data.items():
        all_ts.extend(node['timestamps'])
        data['nodes'][nid] = node
        data['metrics']['kv_cache'][nid]     = node['kv_cache']
        data['metrics']['total_tokens'][nid] = node['total_tokens']
        data['metrics']['running'][nid]      = node['running']
        data['metrics']['waiting'][nid]      = node['waiting']
        data['metrics']['batch_size'][nid]   = node['batch_size']
    data['timestamps'] = sorted(all_ts)
    return data

# -------------------------------------------------
# 时间区间切分 & 数据过滤
# -------------------------------------------------
def get_time_intervals(all_timestamps, interval_minutes):
    if not all_timestamps:
        return []
    start_time = min(all_timestamps)
    end_time   = max(all_timestamps)
    delta      = timedelta(minutes=interval_minutes)
    intervals  = []
    current    = start_time
    while current < end_time:
        nxt = current + delta
        intervals.append((current, nxt))
        current = nxt
    return intervals

def filter_data_by_time(timestamps, values, start_time, end_time):
    filtered_ts, filtered_vals = [], []
    for ts, val in zip(timestamps, values):
        if start_time <= ts < end_time:
            filtered_ts.append(ts)
            filtered_vals.append(val)
    return filtered_ts, filtered_vals

# -------------------------------------------------
# 绘图
# -------------------------------------------------
def plot_metrics(data, metric_name, ylabel, title, output_dir,
                 filename_prefix=None, ranks_per_plot=8, time_interval=3):
    if not data['nodes']:
        print(f"No valid data to plot for {metric_name}")
        return
    os.makedirs(output_dir, exist_ok=True)
    node_ids = sorted(data['metrics'][metric_name].keys())
    total_nodes = len(node_ids)
    num_groups = math.ceil(total_nodes / ranks_per_plot)

    all_ts = []
    for node in data['nodes'].values():
        all_ts.extend(node['timestamps'])
    if not all_ts:
        print(f"No valid timestamps found for {metric_name}")
        return
    intervals = get_time_intervals(all_ts, time_interval)
    print(f"Generated {len(intervals)} time intervals of {time_interval} minutes each for {metric_name}")

    for idx, (start, end) in enumerate(intervals):
        start_str = start.strftime("%Y%m%d_%H%M%S")
        end_str   = end.strftime("%Y%m%d_%H%M%S")
        for g in range(num_groups):
            start_rank = g * ranks_per_plot
            end_rank   = min((g + 1) * ranks_per_plot, total_nodes)
            ranks = node_ids[start_rank:end_rank]
            has_data = False
            for r in ranks:
                ts, _ = filter_data_by_time(data['nodes'][r]['timestamps'],
                                            data['metrics'][metric_name][r],
                                            start, end)
                if ts:
                    has_data = True
                    break
            if not has_data:
                continue
            colors = get_distinct_colors(len(ranks))
            plt.figure(figsize=(14, 8), dpi=300)
            ax = plt.gca()
            plt.style.use('ggplot')
            cur_ts = []
            for i, r in enumerate(ranks):
                ts, vals = filter_data_by_time(data['nodes'][r]['timestamps'],
                                               data['metrics'][metric_name][r],
                                               start, end)
                if ts:
                    cur_ts.extend(ts)
                    plt.plot(ts, vals, marker='o', markersize=4, linewidth=1.5,
                             label=f'Node {r}', color=colors[i], alpha=0.8)
            if not cur_ts:
                plt.close()
                continue
            plt.title(f"{title}\n(Time: {start.strftime('%H:%M')}-{end.strftime('%H:%M')}, "
                      f"Nodes {ranks[0]}-{ranks[-1]})", fontweight='bold', pad=20)
            plt.ylabel(ylabel, labelpad=15)
            plt.xlabel('Time', labelpad=15)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xlim(min(cur_ts), max(cur_ts))
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            ncol = 1 if len(ranks) <= 10 else 2 if len(ranks) <= 20 else 3
            plt.legend(ncol=ncol, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                       fancybox=True, shadow=True)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)
            if filename_prefix:
                out = (f"{filename_prefix}_time_{start_str}-{end_str}_"
                       f"nodes_{ranks[0]}-{ranks[-1]}.png")
                out_path = os.path.join(output_dir, out)
                plt.savefig(out_path, bbox_inches='tight', dpi=300)
                print(f"Saved plot to {out_path}")
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='Parse and visualize proxy log metrics.')
    parser.add_argument('--log_file', type=str,
                        default='/nvme4/share/chenjiefei/scripts/proxy_log/proxy_res_20250806_195705_40000_rounrobin.log',
                        help='Path to the log file')
    parser.add_argument('--output_dir', type=str,
                        default='/nvme4/share/chenjiefei/src/lmdeploy/plot/kvcache_balance_proxy',
                        help='Directory to save output plots')
    parser.add_argument('--ranks_per_plot', type=int, default=4,
                        help='Number of ranks/nodes to display per plot (default: 8)')
    parser.add_argument('--time_interval', type=int, default=12,
                        help='Time interval in minutes for each plot (default: 3)')
    parser.add_argument('--skip_head', type=int, default=20,
                        help='Skip the first N records for each rank/node (default: 0)')
    parser.add_argument('--skip_tail', type=int, default=0,
                        help='Skip the last N records for each rank/node (default: 0)')
    args = parser.parse_args()

    try:
        with open(args.log_file, 'r') as f:
            log_str = f.read()
        parsed = parse_log_data(log_str, args.skip_head, args.skip_tail)
        if not parsed['nodes']:
            print("No valid data found in logs")
            exit()

        plot_metrics(parsed, 'kv_cache',     'KV Cache Usage',         'KV Cache Usage Across Nodes',   args.output_dir, 'kv_cache_usage',    args.ranks_per_plot, args.time_interval)
        plot_metrics(parsed, 'total_tokens', 'Total Tokens',           'Total Tokens Across Nodes',     args.output_dir, 'total_tokens',      args.ranks_per_plot, args.time_interval)
        plot_metrics(parsed, 'running',      'Number of Running Requests', 'Running Requests Across Nodes', args.output_dir, 'running_requests',  args.ranks_per_plot, args.time_interval)
        plot_metrics(parsed, 'waiting',      'Number of Waiting Requests', 'Waiting Requests Across Nodes', args.output_dir, 'waiting_requests',  args.ranks_per_plot, args.time_interval)
        plot_metrics(parsed, 'batch_size',   'Server recv requests',   'Server recv requests Across Nodes', args.output_dir, 'server_recv_requests', args.ranks_per_plot, args.time_interval)

    except FileNotFoundError:
        print(f"Error: The log file was not found at {args.log_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# rm -rf /nvme4/share/chenjiefei/src/lmdeploy/plot/kvcache_balance_proxy/*
if __name__ == "__main__":
    main()