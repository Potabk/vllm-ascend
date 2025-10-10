# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

from tests.e2e.conftest import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "Qwen/Qwen3-0.6B"

prompts = [
    "San Francisco is a",
]

request_keyword_args = {
    "max_tokens": 10,
}


@pytest.fixture(scope="module")
def server():
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--max-num-seqs",
        "128",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [MODEL_NAME])
async def test_models(model: str, client: openai.AsyncOpenAI):
    batch = await client.completions.create(
        model=model,
        prompt=prompts,
        **request_keyword_args,
    )
    choices: list[openai.types.CompletionChoice] = batch.choices
    assert choices[0].text, "empty response"
