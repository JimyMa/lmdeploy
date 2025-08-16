import os

from transformers import AutoTokenizer, LlamaTokenizer

RAW_DATASETS_PATH = "/nvme4/share/chenjiefei/dataset/raw"

DATASETS_META = {
    "Aeala/ShareGPT_Vicuna_unfiltered": {
        "version": None,
        "raw_file_name": "sharegpt_data.json",
        "dataset_file_name": "sharegpt_data.pkl",
        "chat_dataset_file_name": "sharegpt_chat_data.pkl",
    },
}

TOKENIZERS = {
    "llama-7": AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf"),
}