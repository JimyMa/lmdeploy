import json
import os

from datasets import Dataset

from utils import DATASETS_META, RAW_DATASETS_PATH, TOKENIZERS

# --------------------------------------------------
# 1. ShareGPT Dataset Preparation
# --------------------------------------------------
SHAREGPT_RAW_PATH = os.path.join(
    RAW_DATASETS_PATH,
    DATASETS_META["Aeala/ShareGPT_Vicuna_unfiltered"]["raw_file_name"]
)

print("Preprocessing ShareGPT text completion dataset...")
sharegpt_tc_data = []
with open(SHAREGPT_RAW_PATH, "r", encoding="utf-8") as f:
    for line in f:
        sharegpt_tc_data.append(json.loads(line))

for idx, data in enumerate(sharegpt_tc_data):
    # 确保 index 0 为人类，index 1 为 GPT
    sharegpt_tc_data[idx]["input"]  = data["conversations"][0]["value"]
    sharegpt_tc_data[idx]["output"] = data["conversations"][1]["value"]

    del sharegpt_tc_data[idx]["id"]
    del sharegpt_tc_data[idx]["conversations"]

sharegpt_tc_data = Dataset.from_list(sharegpt_tc_data)
print("Finished.")

# --------------------------------------------------
# 2. Tokenize & Save as JSON
# --------------------------------------------------
def encode(row):
    row["input_ids"]  = tokenizer.encode(row["input"])
    row["output_ids"] = tokenizer.encode(row["output"])
    # 计算长度并存储
    row["input_lens"] = len(row["input_ids"])
    row["output_lens"] = len(row["output_ids"])
    return row

def max_len_seqs(row):
    max_len = tokenizer.model_max_length
    return len(row["input"]) < max_len and len(row["output"]) < max_len

DATASETS = [sharegpt_tc_data]

for name, tokenizer in TOKENIZERS.items():
    print(f"Using {name} tokenizer.")
    save_dir = "/nvme4/share/chenjiefei/dataset/tokenized_sharegpt"
    os.makedirs(save_dir, exist_ok=True)

    for dataset, meta in zip(DATASETS, DATASETS_META.values()):
        out_path = os.path.join(save_dir, meta["dataset_file_name"].replace(".pkl", ".json"))

        print(f"Preparing {meta['dataset_file_name']} ...")
        dataset = (
            dataset
            .filter(max_len_seqs)
            .map(encode)
            .remove_columns(["input", "output"])  # 保留input_ids, output_ids, input_lens, output_lens
        )

        # 组装成包含input_lens和output_lens的字典列表
        json_list = [
            {
                "prompt_len": ex["input_lens"],
                "output_len": ex["output_lens"]
            } 
            for ex in dataset
        ]

        print(f"Saving {out_path} ...")
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump(json_list, fw, ensure_ascii=False)
        print("Finished.")