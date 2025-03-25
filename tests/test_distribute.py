# SPDX-License-Identifier: Apache-2.0
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/basic_correctness/test_basic_correctness.py`.
"""
import os
import weakref

import pytest

from vllm import LLM
from vllm.platforms import current_platform

from conftest import VllmRunner
from model_utils import check_outputs_equal


TARGET_TEST_SUITE = os.environ.get("TARGET_TEST_SUITE", "NPU")


@pytest.mark.parametrize(
    "model, distributed_executor_backend,  tp", [
        ("Qwen/QwQ-32B", "mp", 4),
        #("Qwen/QwQ-32B", "mp", 4),
    ])
def test_models_distributed(
    monkeypatch: pytest.MonkeyPatch,
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    distributed_executor_backend: str,
    tp: str,
) -> None:
    with monkeypatch.context() as monkeypatch_context:
        max_tokens = 5
        # NOTE: take care of the order. run vLLM first, and then run HF.
        # vLLM needs a fresh new process without cuda initialization.
        # if we run HF first, the cuda initialization will be done and it
        # will hurt multiprocessing backend with fork method
        # (the default method).
        with vllm_runner(
                model,
                tensor_parallel_size=tp,
                distributed_executor_backend=distributed_executor_backend,
        ) as vllm_model:
            vllm_outputs = vllm_model.generate_greedy(example_prompts,
                                                      max_tokens)
        print(vllm_outputs)