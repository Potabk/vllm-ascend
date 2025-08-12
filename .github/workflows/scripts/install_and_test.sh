#!/bin/bash

# set -euo pipefail


# config mirror
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi

sleep 1000

# cd $GITHUB_WORKSPACE/vllm_empty
# VLLM_TARGET_DEVICE=empty pip install -e .

# cd $GITHUB_WORKSPACE
# pip install -e .

# # show
# pip list | grep vllm
