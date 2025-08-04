#!/bin/bash

# LOCAL_IP=`hostname -I | awk '{print $1}'`
LOCAL_IP="10.130.8.139"
PORT=$1

num_prompts=$2
backend="lmdeploy"
dataset_name="random"
dataset_path="/mnt/137_nvme2/ShareGPT_V3_unfiltered_cleaned_split.json"

echo ">>> num_prompts: ${num_prompts}, dataset: ${dataset_name}"

for in_len in $3
do
    echo "input len: ${in_len}"

    for out_len in $4
    do
        echo "output len: ${out_len}"

        python3 /nvme4/share/${USER}/src/lmdeploy/benchmark/profile_restful_api.py \
            --backend ${backend} \
            --dataset-name ${dataset_name}  \
            --dataset-path ${dataset_path}  \
            --num-prompts ${num_prompts}    \
            --random-input-len ${in_len}    \
            --random-output-len ${out_len}  \
            --random-range-ratio 1          \
            --host ${LOCAL_IP}              \
            --port ${PORT}                  \
            --request-rate 64
    done

done