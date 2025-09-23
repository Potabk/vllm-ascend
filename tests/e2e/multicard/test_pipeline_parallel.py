# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import asyncio

import pytest
import pytest_asyncio

from tests.e2e.conftest import RemoteOpenAIServer

MODELS = [
    "Qwen/Qwen3-0.6B",
]

TENSOR_PARALLELS = [1]
PIPELINE_PARALLELS = [2]
DIST_EXECUTOR_BACKEND = ["mp", "ray"]

prompts = [
    "Hello, my name is",
    "The future of AI is",
]

# General request argument values for these tests
api_keyword_args = {
    # Greedy sampling ensures that requests which receive the `target_token`
    # arg will decode it in every step
    "temperature": 0.0,
    # Since EOS will never be decoded (unless `target_token` is EOS)
    "max_tokens": 20,
    # Return decoded token logprobs (as a way of getting token id)
    "logprobs": 0,
}


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("pp_size", PIPELINE_PARALLELS)
@pytest.mark.parametrize("distributed_executor_backend", DIST_EXECUTOR_BACKEND)
async def test_models(model: str, tp_size: int, pp_size: int,
                      distributed_executor_backend: str) -> None:
    server_args = [
        "--tensor-parallel-size",
        str(tp_size), "--pipeline-parallel-size",
        str(pp_size), "--distributed-executor-backend",
        distributed_executor_backend, "--gpu-memory-utilization", "0.7"
    ]
    with RemoteOpenAIServer(model, server_args) as server:
        chat_input = [{"role": "user", "content": "Write a long story"}]
        client = server.get_async_client()
        tasks = [
            asyncio.create_task(
                client.chat.completions.create(
                    messages=chat_input,
                    model=model,
                    max_tokens=64,
                )) for _ in range(200)
        ]

        results = []
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                results.append(result)
            except Exception as e:
                results.append(e)

        print(results)
