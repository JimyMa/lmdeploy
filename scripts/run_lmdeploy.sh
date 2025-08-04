MODEL_PATH='/mnt/137_nvme2/huggingface_hub/hub/models--deepseek-ai--DeepSeek-V3/'
CONTAINER_MODEL_PATH='/mnt/137_nvme2/huggingface_hub/hub/models--deepseek-ai--DeepSeek-V3/'

EARTH=/nvme4/share/$USER
MARS=/nvme2/share/root

SCRIPTS_PATH=$EARTH/scripts/
SRC_PATH=$EARTH/src/
DEEP_GEMM_PATH=$EARTH/.deep_gemm/
LOG_PATH=$EARTH/logs/

DOCKER_IMG_NAME='pjlab-shanghai-acr-registry.cn-shanghai.cr.aliyuncs.com/pjlab-eflops/llm:lmdeploy-v0.7.3-cu12-deepep-20250422-shenhao-aliepandibverbandslime-20250716'

sudo docker run -it \
    --gpus all \
    --cap-add ALL \
    --cap-add SYS_ADMIN --cap-add SYS_PTRACE \
    --network host \
    --ipc host \
    --name ${USER}_lmdeploy_with_mnt \
    --privileged \
    -v /mnt:/mnt \
    -v $EARTH/scripts/:$MARS/scripts/ \
    -v $EARTH/src/:$MARS/src/ \
    -v $EARTH/logs/:$MARS/logs/ \
    $device_args \
    --entrypoint "/bin/bash" \
    $DOCKER_IMG_NAME \
    -c "/bin/bash"