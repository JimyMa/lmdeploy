import json
import os
import pandas as pd
from pandas.errors import ParserError
import random  # 导入random模块用于打乱数据

def process_csv(input_path, output_dir, max_total_tokens):
    """
    处理CSV文件，过滤出num_total_tokens小于指定值的记录，打乱后转换格式保存
    
    参数:
        input_path: 输入CSV文件路径
        output_dir: 输出目录
        max_total_tokens: num_total_tokens的最大值阈值
    """
    try:
        # 读取输入CSV文件
        df = pd.read_csv(input_path, encoding='utf-8')
        
        # 记录过滤前的总数量
        total_before = len(df)
        print(f"过滤前总记录数: {total_before}")
        
        # 检查是否包含必要字段
        required_fields = ['num_prefill_tokens', 'num_decode_tokens', 'num_total_tokens']
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            print(f"警告：CSV文件缺少必要字段: {', '.join(missing_fields)}")
            return
        
        # 过滤num_total_tokens小于阈值的记录
        filtered_df = df[df['num_total_tokens'] < max_total_tokens]
        
        # 对过滤后的结果进行随机打乱
        # 使用sample方法打乱DataFrame，frac=1表示返回所有行
        shuffled_df = filtered_df.sample(frac=1, random_state=None)
        # 也可以使用random.shuffle方法，但需要先转换为列表
        # shuffled_list = filtered_df.to_dict('records')
        # random.shuffle(shuffled_list)
        
        # 转换格式
        processed_data = shuffled_df.apply(lambda row: {
            "prompt_len": row['num_prefill_tokens'],
            "output_len": row['num_decode_tokens']
        }, axis=1).tolist()
        
        # 记录过滤后的数量
        total_after = len(processed_data)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存处理后的数据为JSON
        output_path = os.path.join(output_dir, 'gemini_issues_processed_data.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"过滤后保留记录数: {total_after}")
        print(f"数据已打乱并保存{total_after}条记录到{output_path}")
        
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_path}")
    except ParserError:
        print(f"错误：输入文件不是有效的CSV格式")
    except Exception as e:
        print(f"处理过程中发生错误：{str(e)}")

# 使用示例
if __name__ == "__main__":
    # 输入CSV文件路径（请根据实际情况修改）
    input_file = "/nvme4/share/chenjiefei/dataset/medha/Gemini Issues Stats.csv"
    # 输出目录（请根据实际情况修改）
    output_directory = "/nvme4/share/chenjiefei/dataset/filtered_medha"
    # num_total_tokens的最大值阈值（请根据实际需求修改）
    max_total = 340000  # 示例值
    
    process_csv(input_file, output_directory, max_total)

