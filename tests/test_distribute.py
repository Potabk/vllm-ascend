# SPDX-License-Identifier: Apache-2.0
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/basic_correctness/test_basic_correctness.py`.
"""
import os
import weakref

import pytest

from vllm import LLM
from vllm.platforms import current_platform

from ..conftest import VllmRunner
from ..models.utils import check_outputs_equal
from ..utils import multi_gpu_test

MODELS = [
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
]

TARGET_TEST_SUITE = os.environ.get("TARGET_TEST_SUITE", "NPU")


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model, distributed_executor_backend, attention_backend"[
        ("Qwen/QwQ-32B", "mp", "ASCEND"),
    ])
def test_models_distributed(
    monkeypatch: pytest.MonkeyPatch,
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    distributed_executor_backend: str,
    attention_backend: str,
) -> None:
    with monkeypatch.context() as monkeypatch_context:
        if attention_backend:
            monkeypatch_context.setenv(
                "VLLM_ATTENTION_BACKEND",
                attention_backend,
            )
        max_tokens = 5

        # NOTE: take care of the order. run vLLM first, and then run HF.
        # vLLM needs a fresh new process without cuda initialization.
        # if we run HF first, the cuda initialization will be done and it
        # will hurt multiprocessing backend with fork method
        # (the default method).
        with vllm_runner(
                model,
                tensor_parallel_size=2,
                distributed_executor_backend=distributed_executor_backend,
        ) as vllm_model:
            vllm_outputs = vllm_model.generate_greedy(example_prompts,
                                                      max_tokens)

        with hf_runner(model) as hf_model:
            hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

        check_outputs_equal(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )
