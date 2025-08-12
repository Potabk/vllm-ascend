#!/bin/bash

ENV_FILE="/root/actions-runner/.cache/set_env.sh"

docker exec -i ascend_ci_a3 bash -lc "source $ENV_FILE && exec bash"

if [ $? -ne 0 ]; then
    echo "Failed to execute command in the container. Please check if the container is running."
    exit 1
fi

# config mirror
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi

cd $GITHUB_WORKSPACE/vllm_empty
VLLM_TARGET_DEVICE=empty pip install -e .

cd $GITHUB_WORKSPACE
pip install -e .

# show
pip list | grep vllm
