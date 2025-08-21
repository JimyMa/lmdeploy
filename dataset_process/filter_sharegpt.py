import os
import json
import glob
import random

def main():
    root_dir = "/nvme4/share/chenjiefei/dataset/tokenized_sharegpt"
    json_files = glob.glob(os.path.join(root_dir, "*.json"))

    for fp in json_files:
        if fp.endswith("_filtered.json"):
            continue  # 避免重复处理

        with open(fp, "r", encoding="utf-8") as fr:
            data = json.load(fr)

        before_count = len(data)
        filtered = []

        for item in data:
            prompt_len = item.get("prompt_len", 0)
            output_len = item.get("output_len", 0)

            # 判断条件
            if output_len + 400 >= 450:

                if output_len < 450 and prompt_len + output_len + 350 < 8192:
                    filtered.append({
                        "prompt_len": prompt_len,
                        "output_len": output_len + 400
                    })
                if output_len >= 450 and prompt_len + output_len + 100 < 8192:
                    filtered.append({
                        "prompt_len": prompt_len,
                        "output_len": output_len + 100
                    })
                
        random.shuffle(filtered)

        after_count = len(filtered)

        base, ext = os.path.splitext(fp)
        out_fp = f"{base}_filtered{ext}"

        with open(out_fp, "w", encoding="utf-8") as fw:
            json.dump(filtered, fw, ensure_ascii=False, indent=2)

        # 打印统计信息
        print(f"Input file : {os.path.basename(fp)}")
        print(f"  过滤前样本数 : {before_count}")
        print(f"  过滤后样本数 : {after_count}")
        print(f"  输出文件     : {os.path.basename(out_fp)}\n")

if __name__ == "__main__":
    main()