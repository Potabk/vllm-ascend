#
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
import functools
import sys
from enum import Enum
from unittest.mock import MagicMock

import pytest
import torch

try:
    _npu_available = torch.npu.is_available()
except AttributeError:
    _npu_available = False

if not _npu_available:
    triton_runtime = MagicMock()
    triton_runtime.driver.active.utils.get_device_properties.return_value = {
        "num_aic": 8,
        "num_vectorcore": 8,
    }
    sys.modules["triton.runtime"] = triton_runtime
    # triton and torch_npu is not available in the environment, so we need to mock them
    if "torch_npu" not in sys.modules:
        sys.modules["torch_npu"] = MagicMock()
    sys.modules["torch_npu"].npu.current_device = MagicMock(return_value=0)
    sys.modules["torch_npu._inductor"] = MagicMock()

from vllm_ascend.utils import adapt_patch  # noqa E402
from vllm_ascend.utils import register_ascend_customop  # noqa E402


adapt_patch()
adapt_patch(True)

# register Ascend CustomOp here because uts will use this
register_ascend_customop()


class RunnerDeviceType(str, Enum):
    """Chip types — values match runner_label.json "chip" field exactly.

    Shared by:
      - tests/ut/conftest.py (npu_test decorator)
      - .github/workflows/scripts/determine_smart_e2e_scope.py (AST parser)
    """

    A2 = "a2"
    A3 = "a3"
    _310P = "310p"
    CPU = "cpu"


def npu_test(num_npus: int = 1, npu_type: RunnerDeviceType = RunnerDeviceType.A2):
    """Decorator that marks a test with NPU resource requirements.

    Serves two purposes:
      1. **CI routing** — the AST parser in determine_smart_e2e_scope.py reads
         the decorator keyword arguments (num_npus, npu_type) to group tests
         by runner type. The parameter names and decorator name must stay in
         sync with the parser.
      2. **Runtime gating** — at test time the decorator skips the test when
         the current environment lacks the required NPU hardware.

    Args:
        num_npus: Number of NPU devices required (default 1).
        npu_type: The NPU chip type required (default A2).
    """
    if not isinstance(npu_type, RunnerDeviceType):
        npu_type = RunnerDeviceType(npu_type)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if npu_type == RunnerDeviceType.CPU:
                return func(*args, **kwargs)
            if not torch.npu.is_available():
                pytest.skip(f"NPU not available (need {npu_type.value} x{num_npus})")
            device_count = torch.npu.device_count()
            if device_count < num_npus:
                pytest.skip(f"Not enough NPUs: need {num_npus}, have {device_count}")
            return func(*args, **kwargs)

        return wrapper

    return decorator
