#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""
Tests for NPUModelRunner.initialize_attn_backend with different KVCacheConfig
inputs constructed from various KVCacheSpec types.

Covers scenarios from the attention backend design doc:
- Standard attention (FullAttentionSpec): MHA/GQA/MQA with different params
- MLA attention (MLAAttentionSpec): DeepSeek-style
- Sliding window attention (SlidingWindowSpec)
- Multiple KV cache groups (heterogeneous specs)
- Backend deduplication within a single group
"""

import pytest
import torch
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

import vllm_ascend.compilation.acl_graph as acl_graph
from tests.ut.conftest import RunnerDeviceType, npu_test
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

BLOCK_SIZE = 128
NUM_BLOCKS = 10
DEVICE_TYPE = current_platform.device_type


# ---------------------------------------------------------------------------
# VllmConfig builders
# ---------------------------------------------------------------------------


def get_vllm_config():
    model_config = ModelConfig(
        model="facebook/opt-125m",
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig()
    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )


def get_mla_vllm_config():
    """VllmConfig with MLA attributes patched onto hf_text_config."""
    vllm_config = get_vllm_config()
    hf_cfg = vllm_config.model_config.hf_text_config
    hf_cfg.kv_lora_rank = 512
    hf_cfg.qk_nope_head_dim = 128
    hf_cfg.qk_rope_head_dim = 64
    hf_cfg.v_head_dim = 128
    hf_cfg.q_lora_rank = 1536
    return vllm_config


# ---------------------------------------------------------------------------
# Thin adapter for MLA layers
#
# Creating a real MLAAttention layer requires DeepSeek model components
# (ColumnParallelLinear, etc.) which aren't available in unit tests.
# This adapter wraps the REAL AscendMLABackend so the grouping and
# metadata-builder creation in initialize_attn_backend uses real NPU code.
# ---------------------------------------------------------------------------


class MLAAttentionLayerAdapter(AttentionLayerBase):
    """Thin adapter that returns the real AscendMLABackend."""

    def get_attn_backend(self):
        return AscendMLABackend

    def get_kv_cache_spec(self, vllm_config):
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_runner(vllm_config, layer_map):
    """
    Create an NPUModelRunner with Attention layers registered.

    Args:
        vllm_config: VllmConfig to use.
        layer_map: dict mapping layer_name -> layer instance (already created
                   inside a set_current_vllm_config context).
    Returns:
        NPUModelRunner ready for initialize_attn_backend.
    """
    for name, layer in layer_map.items():
        vllm_config.compilation_config.static_forward_context[name] = layer
    return NPUModelRunner(vllm_config, DEVICE_TYPE)


def _make_full_attn_spec(num_kv_heads, head_size, **overrides):
    defaults = dict(
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
    )
    defaults.update(overrides)
    return FullAttentionSpec(**defaults)


def _make_mla_spec(**overrides):
    defaults = dict(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=576,  # kv_lora_rank(512) + qk_rope_head_dim(64)
        dtype=torch.float16,
    )
    defaults.update(overrides)
    return MLAAttentionSpec(**defaults)


def _make_sliding_window_spec(num_kv_heads, head_size, **overrides):
    defaults = dict(
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
        sliding_window=4096,
    )
    defaults.update(overrides)
    return SlidingWindowSpec(**defaults)


def _build_kv_cache_config(groups):
    """
    Build a KVCacheConfig from a list of (layer_names, kv_cache_spec) tuples.
    """
    kv_cache_tensors = []
    kv_cache_groups = []
    for layer_names, spec in groups:
        tensor_size = spec.page_size_bytes * NUM_BLOCKS
        kv_cache_tensors.append(KVCacheTensor(size=tensor_size, shared_by=layer_names))
        kv_cache_groups.append(KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec))
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


# ---------------------------------------------------------------------------
# Tests — Standard Attention (FullAttentionSpec)
# ---------------------------------------------------------------------------


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_single_layer_full_attention():
    """Single layer with FullAttentionSpec — simplest case."""
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        mc = vllm_config.model_config
        num_kv_heads = mc.get_num_kv_heads(vllm_config.parallel_config)
        head_size = mc.get_head_size()

        layer = Attention(num_kv_heads, head_size, 0.1)
        runner = _create_runner(vllm_config, {"layer.0": layer})

        spec = _make_full_attn_spec(num_kv_heads, head_size)
        kv_cache_config = _build_kv_cache_config(
            [
                (["layer.0"], spec),
            ]
        )
        runner.initialize_attn_backend(kv_cache_config)

    assert len(runner.attn_groups) == 1
    groups = runner.attn_groups[0]
    assert len(groups) == 1
    group = groups[0]
    assert group.layer_names == ["layer.0"]
    assert isinstance(group.kv_cache_spec, FullAttentionSpec)
    assert group.kv_cache_group_id == 0
    # NPU version creates metadata builders eagerly
    assert len(group.metadata_builders) == 1

    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_multiple_layers_same_backend():
    """Multiple layers with the same backend are deduplicated into one AttentionGroup."""
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        mc = vllm_config.model_config
        num_kv_heads = mc.get_num_kv_heads(vllm_config.parallel_config)
        head_size = mc.get_head_size()

        layer_names = ["layer.0", "layer.1", "layer.2"]
        layers = {n: Attention(num_kv_heads, head_size, 0.1) for n in layer_names}
        runner = _create_runner(vllm_config, layers)

        spec = _make_full_attn_spec(num_kv_heads, head_size)
        kv_cache_config = _build_kv_cache_config(
            [
                (layer_names, spec),
            ]
        )
        runner.initialize_attn_backend(kv_cache_config)

    assert len(runner.attn_groups) == 1
    groups = runner.attn_groups[0]
    assert len(groups) == 1
    assert groups[0].layer_names == layer_names

    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_full_attention_gqa_params():
    """GQA: num_kv_heads differs from default, spec reflects it."""
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        gqa_kv_heads = 4
        head_size = 128
        layer = Attention(gqa_kv_heads, head_size, 0.1)
        runner = _create_runner(vllm_config, {"layer.0": layer})

        spec = _make_full_attn_spec(gqa_kv_heads, head_size)
        kv_cache_config = _build_kv_cache_config(
            [
                (["layer.0"], spec),
            ]
        )
        runner.initialize_attn_backend(kv_cache_config)

    group = runner.attn_groups[0][0]
    assert group.kv_cache_spec.num_kv_heads == 4
    assert group.kv_cache_spec.head_size == 128

    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_full_attention_bf16():
    """FullAttentionSpec with bf16 dtype."""
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        mc = vllm_config.model_config
        num_kv_heads = mc.get_num_kv_heads(vllm_config.parallel_config)
        head_size = mc.get_head_size()

        layer = Attention(num_kv_heads, head_size, 0.1)
        runner = _create_runner(vllm_config, {"layer.0": layer})

        spec = _make_full_attn_spec(num_kv_heads, head_size, dtype=torch.bfloat16)
        kv_cache_config = _build_kv_cache_config(
            [
                (["layer.0"], spec),
            ]
        )
        runner.initialize_attn_backend(kv_cache_config)

    group = runner.attn_groups[0][0]
    assert group.kv_cache_spec.dtype == torch.bfloat16

    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
@pytest.mark.parametrize("block_size", [16, 64, 256])
def test_full_attention_various_block_sizes(block_size):
    """FullAttentionSpec with different block sizes."""
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        mc = vllm_config.model_config
        num_kv_heads = mc.get_num_kv_heads(vllm_config.parallel_config)
        head_size = mc.get_head_size()

        layer = Attention(num_kv_heads, head_size, 0.1)
        runner = _create_runner(vllm_config, {"layer.0": layer})

        spec = _make_full_attn_spec(num_kv_heads, head_size, block_size=block_size)
        kv_cache_config = _build_kv_cache_config(
            [
                (["layer.0"], spec),
            ]
        )
        runner.initialize_attn_backend(kv_cache_config)

    group = runner.attn_groups[0][0]
    assert group.kv_cache_spec.block_size == block_size

    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_many_layers_single_group():
    """32 layers in a single group are handled correctly."""
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        mc = vllm_config.model_config
        num_kv_heads = mc.get_num_kv_heads(vllm_config.parallel_config)
        head_size = mc.get_head_size()

        layer_names = [f"layer.{i}" for i in range(32)]
        layers = {n: Attention(num_kv_heads, head_size, 0.1) for n in layer_names}
        runner = _create_runner(vllm_config, layers)

        spec = _make_full_attn_spec(num_kv_heads, head_size)
        kv_cache_config = _build_kv_cache_config(
            [
                (layer_names, spec),
            ]
        )
        runner.initialize_attn_backend(kv_cache_config)

    assert len(runner.attn_groups) == 1
    groups = runner.attn_groups[0]
    assert len(groups) == 1
    assert len(groups[0].layer_names) == 32

    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None


# ---------------------------------------------------------------------------
# Tests — Sliding Window (SlidingWindowSpec)
# ---------------------------------------------------------------------------


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_single_layer_sliding_window():
    """Sliding window attention with SlidingWindowSpec."""
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        mc = vllm_config.model_config
        num_kv_heads = mc.get_num_kv_heads(vllm_config.parallel_config)
        head_size = mc.get_head_size()

        # On NPU the backend is still AscendAttentionBackend;
        # the sliding window behavior is in the spec, not the backend.
        layer = Attention(num_kv_heads, head_size, 0.1, per_layer_sliding_window=2048)
        runner = _create_runner(vllm_config, {"layer.0": layer})

        spec = _make_sliding_window_spec(num_kv_heads, head_size, sliding_window=2048)
        kv_cache_config = _build_kv_cache_config(
            [
                (["layer.0"], spec),
            ]
        )
        runner.initialize_attn_backend(kv_cache_config)

    assert len(runner.attn_groups) == 1
    group = runner.attn_groups[0][0]
    assert isinstance(group.kv_cache_spec, SlidingWindowSpec)
    assert group.kv_cache_spec.sliding_window == 2048

    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None


# ---------------------------------------------------------------------------
# Tests — MLA Attention (MLAAttentionSpec)
#
# Uses a thin adapter for the attention layer (creating a real MLAAttention
# requires DeepSeek model components), but the backend (AscendMLABackend)
# and metadata builder (AscendMLAMetadataBuilder) are fully real.
# ---------------------------------------------------------------------------


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_single_layer_mla_attention():
    """MLA attention with MLAAttentionSpec and real AscendMLABackend."""
    vllm_config = get_mla_vllm_config()
    with set_current_vllm_config(vllm_config):
        layer = MLAAttentionLayerAdapter()
        runner = _create_runner(vllm_config, {"layer.0": layer})

        spec = _make_mla_spec()
        kv_cache_config = _build_kv_cache_config(
            [
                (["layer.0"], spec),
            ]
        )
        runner.initialize_attn_backend(kv_cache_config)

    assert len(runner.attn_groups) == 1
    group = runner.attn_groups[0][0]
    assert group.backend is AscendMLABackend
    assert isinstance(group.kv_cache_spec, MLAAttentionSpec)
    assert group.kv_cache_spec.num_kv_heads == 1
    assert group.kv_cache_spec.head_size == 576
    assert len(group.metadata_builders) == 1

    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_multiple_layers_mla_attention():
    """Multiple MLA layers grouped together."""
    vllm_config = get_mla_vllm_config()
    with set_current_vllm_config(vllm_config):
        layer_names = ["layer.0", "layer.1", "layer.2", "layer.3"]
        layers = {n: MLAAttentionLayerAdapter() for n in layer_names}
        runner = _create_runner(vllm_config, layers)

        spec = _make_mla_spec()
        kv_cache_config = _build_kv_cache_config(
            [
                (layer_names, spec),
            ]
        )
        runner.initialize_attn_backend(kv_cache_config)

    assert len(runner.attn_groups) == 1
    groups = runner.attn_groups[0]
    assert len(groups) == 1
    assert groups[0].layer_names == layer_names
    assert groups[0].backend is AscendMLABackend

    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None


# ---------------------------------------------------------------------------
# Tests — Multiple KV Cache Groups
# ---------------------------------------------------------------------------


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_two_groups_full_and_sliding_window():
    """Two KV cache groups: full attention + sliding window."""
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        mc = vllm_config.model_config
        num_kv_heads = mc.get_num_kv_heads(vllm_config.parallel_config)
        head_size = mc.get_head_size()

        full_layers = ["layer.0", "layer.1"]
        sw_layers = ["layer.2", "layer.3"]
        layer_map = {}
        for n in full_layers:
            layer_map[n] = Attention(num_kv_heads, head_size, 0.1)
        for n in sw_layers:
            layer_map[n] = Attention(num_kv_heads, head_size, 0.1, per_layer_sliding_window=2048)
        runner = _create_runner(vllm_config, layer_map)

        full_spec = _make_full_attn_spec(num_kv_heads, head_size)
        sw_spec = _make_sliding_window_spec(num_kv_heads, head_size, sliding_window=2048)
        kv_cache_config = _build_kv_cache_config(
            [
                (full_layers, full_spec),
                (sw_layers, sw_spec),
            ]
        )
        runner.initialize_attn_backend(kv_cache_config)

    assert len(runner.attn_groups) == 2

    # Group 0: full attention
    assert len(runner.attn_groups[0]) == 1
    assert runner.attn_groups[0][0].layer_names == full_layers
    assert runner.attn_groups[0][0].kv_cache_group_id == 0
    assert isinstance(runner.attn_groups[0][0].kv_cache_spec, FullAttentionSpec)

    # Group 1: sliding window
    assert len(runner.attn_groups[1]) == 1
    assert runner.attn_groups[1][0].layer_names == sw_layers
    assert runner.attn_groups[1][0].kv_cache_group_id == 1
    assert isinstance(runner.attn_groups[1][0].kv_cache_spec, SlidingWindowSpec)

    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_two_groups_full_and_mla():
    """Two KV cache groups: standard attention + MLA attention."""
    vllm_config = get_mla_vllm_config()
    with set_current_vllm_config(vllm_config):
        mc = vllm_config.model_config
        num_kv_heads = mc.get_num_kv_heads(vllm_config.parallel_config)
        head_size = mc.get_head_size()

        runner = _create_runner(
            vllm_config,
            {
                "layer.0": Attention(num_kv_heads, head_size, 0.1),
                "layer.1": MLAAttentionLayerAdapter(),
            },
        )

        full_spec = _make_full_attn_spec(num_kv_heads, head_size)
        mla_spec = _make_mla_spec()
        kv_cache_config = _build_kv_cache_config(
            [
                (["layer.0"], full_spec),
                (["layer.1"], mla_spec),
            ]
        )
        runner.initialize_attn_backend(kv_cache_config)

    assert len(runner.attn_groups) == 2
    assert isinstance(runner.attn_groups[0][0].kv_cache_spec, FullAttentionSpec)
    assert runner.attn_groups[0][0].kv_cache_group_id == 0
    assert isinstance(runner.attn_groups[1][0].kv_cache_spec, MLAAttentionSpec)
    assert runner.attn_groups[1][0].backend is AscendMLABackend
    assert runner.attn_groups[1][0].kv_cache_group_id == 1

    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_three_groups():
    """Three KV cache groups: full + sliding window + MLA."""
    vllm_config = get_mla_vllm_config()
    with set_current_vllm_config(vllm_config):
        mc = vllm_config.model_config
        num_kv_heads = mc.get_num_kv_heads(vllm_config.parallel_config)
        head_size = mc.get_head_size()

        runner = _create_runner(
            vllm_config,
            {
                "layer.0": Attention(num_kv_heads, head_size, 0.1),
                "layer.1": Attention(num_kv_heads, head_size, 0.1, per_layer_sliding_window=2048),
                "layer.2": MLAAttentionLayerAdapter(),
            },
        )

        full_spec = _make_full_attn_spec(num_kv_heads, head_size)
        sw_spec = _make_sliding_window_spec(num_kv_heads, head_size, sliding_window=2048)
        mla_spec = _make_mla_spec()
        kv_cache_config = _build_kv_cache_config(
            [
                (["layer.0"], full_spec),
                (["layer.1"], sw_spec),
                (["layer.2"], mla_spec),
            ]
        )
        runner.initialize_attn_backend(kv_cache_config)

    assert len(runner.attn_groups) == 3
    assert runner.attn_groups[0][0].kv_cache_group_id == 0
    assert runner.attn_groups[1][0].kv_cache_group_id == 1
    assert runner.attn_groups[2][0].kv_cache_group_id == 2

    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None


# ---------------------------------------------------------------------------
# Tests — Backend Dedup and Grouping
# ---------------------------------------------------------------------------


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_mixed_backends_in_single_group():
    """
    When layers in the same KV cache group use different backends,
    they are split into separate AttentionGroup objects.
    """
    vllm_config = get_mla_vllm_config()
    with set_current_vllm_config(vllm_config):
        mc = vllm_config.model_config
        num_kv_heads = mc.get_num_kv_heads(vllm_config.parallel_config)
        head_size = mc.get_head_size()

        runner = _create_runner(
            vllm_config,
            {
                "layer.0": Attention(num_kv_heads, head_size, 0.1),
                "layer.1": MLAAttentionLayerAdapter(),
            },
        )

        spec = _make_full_attn_spec(num_kv_heads, head_size)
        kv_cache_config = _build_kv_cache_config(
            [
                (["layer.0", "layer.1"], spec),
            ]
        )
        runner.initialize_attn_backend(kv_cache_config)

    # One kv_cache_group, but two different backends → two AttentionGroups
    assert len(runner.attn_groups) == 1
    groups = runner.attn_groups[0]
    assert len(groups) == 2
    backends = {g.backend for g in groups}
    assert AscendMLABackend in backends

    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_metadata_builders_created_for_each_group():
    """Each AttentionGroup gets its own metadata builder (eagerly on NPU)."""
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        mc = vllm_config.model_config
        num_kv_heads = mc.get_num_kv_heads(vllm_config.parallel_config)
        head_size = mc.get_head_size()

        runner = _create_runner(
            vllm_config,
            {
                "layer.0": Attention(num_kv_heads, head_size, 0.1),
                "layer.1": Attention(num_kv_heads, head_size, 0.1, per_layer_sliding_window=2048),
            },
        )

        full_spec = _make_full_attn_spec(num_kv_heads, head_size)
        sw_spec = _make_sliding_window_spec(num_kv_heads, head_size, sliding_window=2048)
        kv_cache_config = _build_kv_cache_config(
            [
                (["layer.0"], full_spec),
                (["layer.1"], sw_spec),
            ]
        )
        runner.initialize_attn_backend(kv_cache_config)

    for group_list in runner.attn_groups:
        for group in group_list:
            assert len(group.metadata_builders) == 1

    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None


# ---------------------------------------------------------------------------
# Tests — Edge Cases
# ---------------------------------------------------------------------------


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_duplicate_initialization_raises():
    """Calling initialize_attn_backend twice raises AssertionError."""
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        mc = vllm_config.model_config
        num_kv_heads = mc.get_num_kv_heads(vllm_config.parallel_config)
        head_size = mc.get_head_size()

        layer = Attention(num_kv_heads, head_size, 0.1)
        runner = _create_runner(vllm_config, {"layer.0": layer})

        spec = _make_full_attn_spec(num_kv_heads, head_size)
        kv_cache_config = _build_kv_cache_config(
            [
                (["layer.0"], spec),
            ]
        )
        runner.initialize_attn_backend(kv_cache_config)

        with pytest.raises(AssertionError, match="already initialized"):
            runner.initialize_attn_backend(kv_cache_config)

    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_num_blocks_does_not_affect_grouping():
    """Different num_blocks values produce the same attn group structure."""
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        mc = vllm_config.model_config
        num_kv_heads = mc.get_num_kv_heads(vllm_config.parallel_config)
        head_size = mc.get_head_size()

        layer = Attention(num_kv_heads, head_size, 0.1)
        runner = _create_runner(vllm_config, {"layer.0": layer})

        spec = _make_full_attn_spec(num_kv_heads, head_size)
        kv_cache_config = KVCacheConfig(
            num_blocks=500,
            kv_cache_tensors=[
                KVCacheTensor(size=spec.page_size_bytes * 500, shared_by=["layer.0"]),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=spec),
            ],
        )
        runner.initialize_attn_backend(kv_cache_config)

    assert len(runner.attn_groups) == 1
    assert len(runner.attn_groups[0]) == 1

    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None
