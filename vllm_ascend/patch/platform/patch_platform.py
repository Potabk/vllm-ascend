import os

# todo: please remove it when solve cuda hard code in vllm
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import vllm_version_is


def get_attn_backend_cls(
    cls,
    selected_backend,
    head_size,
    dtype,
    kv_cache_dtype,
    block_size,
    use_mla,
    has_sink=False,
    use_sparse=False,
    attn_type: str | None = None,
):
    ascend_config = get_ascend_config()

    if use_mla and ascend_config.enable_shared_expert_dp:
        if use_mla and use_sparse:
            return "vllm_ascend.torchair.torchair_sfa.AscendSFATorchairBackend"

    use_torchair = ascend_config.torchair_graph_config.enabled
    # choose attention backend based on use_mla and use_torchair
    backend_map = {
        (True, False, True):
        "vllm_ascend.torchair.torchair_mla.AscendMLATorchairBackend",
        (True, False, False):
        "vllm_ascend.attention.mla_v1.AscendMLABackend",
        (False, False, True):
        "vllm_ascend.torchair.torchair_attention.AscendAttentionTorchairBackend",
        (False, False, False):
        "vllm_ascend.attention.attention_v1.AscendAttentionBackend",
        (True, True, False):
        "vllm_ascend.attention.sfa_v1.AscendSFABackend",
        (True, True, True):
        "vllm_ascend.torchair.torchair_sfa.AscendSFATorchairBackend",
    }
    return backend_map[(use_mla, use_sparse, use_torchair)]


if vllm_version_is("0.12.0"):
    NPUPlatform.get_attn_backend_cls = classmethod(get_attn_backend_cls)
