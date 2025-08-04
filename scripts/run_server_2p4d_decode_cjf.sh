#!/bin/bash

export ROLE='Decode'

#### hybper parameter begin ############

# Model Setting
export MODEL_PATH='/mnt/137_nvme2/huggingface_hub/hub/models--deepseek-ai--DeepSeek-V3/snapshots/86518964eaef84e3fdd98e9861759a1384f9c29d'
# export MODEL_PATH='/mnt/137_nvme3/InternVL3-235B-Qwen3MoE-20250702d-7500-cpt-data-0628-sft-science-data-0710-d027025-slow-tokenize-lr-8e5-rjob-h200-0711/'

# distributed setting
export NODE_RANK=$1
export GPU_NUMS=32
export MASTER_ADDR='10.130.8.147'
export MASTER_PORT=29555

# proxy setting
export PROXY_URL='10.130.8.139:8050'

# batch setting
export MAX_BATCH_SIZE='256'
export MAX_PREFILL_TOKEN_NUM='256'

# cache setting
export CACHE_MAX_ENTRY_COUNT='0.65'

# TBO setting
# export ENABLE_MICROBATCH="--enable-microbatch"

# vision encoder setting
# export DISABLE_VISION_ENCODER="--disable-vision-encoder"

# export ENABLE_MICROBATCH_PREFILL_TOKEN_THRESHOLD='8190'

# Memory Pool Setting
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False'

# log setting
export server_start_time=`date '+%Y-%m-%d_%H-%M'`
export LOG_LEVEL=ERROR
export LOG_DIR="2p4d_decode_${MAX_BATCH_SIZE}_${CACHE_MAX_ENTRY_COUNT}_${server_start_time}"
mkdir -p ${LOG_DIR}

#### hybper parameter end ############

export USER=root

ray stop --force
ray start --head --port 6677 --disable-usage-stats --temp-dir="/nvme2/share/root/ray_logs"
sleep 2

let NNODES=${GPU_NUMS}/8
export LMDEPLOY_FAKE_EPLB=TRUE
export TRANSFORMERS_OFFLINE=1
export LMDEPLOY_DP_MASTER_ADDR=${MASTER_ADDR}
export LMDEPLOY_DP_MASTER_PORT=29500

export HOME=/root/

export DG_JIT_CACHE_DIR=/root/

# Print all parameters before starting
echo "========== Configuration Parameters =========="
echo "ROLE: $ROLE"
echo "NODE_RANK: $NODE_RANK"
echo "GPU_NUMS: $GPU_NUMS"
echo "PROXY_URL: $PROXY_URL"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "MAX_BATCH_SIZE: $MAX_BATCH_SIZE"
echo "MAX_PREFILL_TOKEN_NUM: $MAX_PREFILL_TOKEN_NUM"
echo "CACHE_MAX_ENTRY_COUNT: $CACHE_MAX_ENTRY_COUNT"
echo "DEEPEP_MAX_BATCH_SIZE: $DEEPEP_MAX_BATCH_SIZE"
echo "ENABLE_MICROBATCH: $ENABLE_MICROBATCH"
echo "ENABLE_MICROBATCH_PREFILL_TOKEN_THRESHOLD: $ENABLE_MICROBATCH_PREFILL_TOKEN_THRESHOLD"
echo "ENABLE_MICROBATCH_PREFILL_BATCHSIZE_THRESHOLD: $ENABLE_MICROBATCH_PREFILL_BATCHSIZE_THRESHOLD"
echo "ENABLE_MICROBATCH_DECODE_BATCHSIZE_THRESHOLD: $ENABLE_MICROBATCH_DECODE_BATCHSIZE_THRESHOLD"
echo "USER: $USER"
echo "MODEL_PATH: $MODEL_PATH"
echo "LMDEPLOY_DP_MASTER_ADDR: $LMDEPLOY_DP_MASTER_ADDR"
echo "LMDEPLOY_DP_MASTER_PORT: $LMDEPLOY_DP_MASTER_PORT"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "DG_JIT_CACHE_DIR: $DG_JIT_CACHE_DIR"
echo "HOME: $HOME"
echo "TRANSFORMERS_OFFLINE: $TRANSFORMERS_OFFLINE"
echo "LMDEPLOY_FAKE_EPLB: $LMDEPLOY_FAKE_EPLB"
echo "=============================================="

env \
TRANSFORMERS_OFFLINE=1 \
LMDEPLOY_FAKE_EPLB=TRUE \
DG_JIT_CACHE_DIR=/root/ \
DEEPEP_MAX_BATCH_SIZE='256' \
ENABLE_MICROBATCH_PREFILL_BATCHSIZE_THRESHOLD='4' \
ENABLE_MICROBATCH_DECODE_BATCHSIZE_THRESHOLD='128' \
/opt/py3/bin/python3 /opt/py3/bin/lmdeploy serve api_server \
    ${MODEL_PATH}                                    \
    --backend pytorch                                \
    --ep ${GPU_NUMS}                                 \
    --dp ${GPU_NUMS}                                 \
    --proxy-url http://${PROXY_URL}                  \
    --nnodes ${NNODES}                               \
    --cache-max-entry-count ${CACHE_MAX_ENTRY_COUNT} \
    --max-prefill-token-num ${MAX_PREFILL_TOKEN_NUM} \
    --role ${ROLE}                                   \
    --node-rank ${NODE_RANK} ${DISABLE_VISION_ENCODER} ${ENABLE_MICROBATCH} --session-len 400000 --model-format fp8 --log-level ${LOG_LEVEL} --enable-metrics --max-batch-size ${MAX_BATCH_SIZE}  2>&1 | tee ${LOG_DIR}/dp${GPU_NUMS}ep${GPU_NUMS}_${ROLE}_node${NODE_RANK}.log