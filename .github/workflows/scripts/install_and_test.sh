#!/bin/bash

# set -euo pipefail

# check env
echo "====> Check environment"
echo "nic_name: $GLOO_SOCKET_IFNAME"
echo "local_ip: $HCCL_IF_IP"
# config mirror
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi

export WORKSPACE="/root/workspace"

cd $WORKSPACE
# install sys dependencies
apt-get update -y
apt-get -y install `cat packages.txt`
apt-get -y install gcc g++ cmake libnuma-dev

# install vllm
cd $WORKSPACE/vllm-empty
VLLM_TARGET_DEVICE=empty pip install -e .

# install vllm-ascend
cd $WORKSPACE
pip install -e .


