"""Compare fused CUDA cached GDN against the pinned HF implementation over 24 steps."""

import copy
import os
import sys
from pathlib import Path
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import Qwen3_5TextConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
from bonsai_lens.cuda_gdn import forward

torch.manual_seed(13)
for batch in (1, 2):
    cfg = Qwen3_5TextConfig(
        hidden_size=64,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    reference = Qwen3_5GatedDeltaNet(cfg, 0).half().cuda().eval()
    # Constructor uses uninitialized decay parameters; initialize bounded, nontrivial gates.
    with torch.no_grad():
        reference.A_log.copy_(torch.randn_like(reference.A_log) * 0.2)
        reference.dt_bias.copy_(torch.randn_like(reference.dt_bias) * 0.2)
    fused = copy.deepcopy(reference)
    reference.forward = MethodType(forward, reference)
    fused.forward = MethodType(forward, fused)
    refcache, cache = DynamicCache(config=cfg), DynamicCache(config=cfg)
    with torch.inference_mode():
        x = torch.randn(batch, 8, 64, device="cuda", dtype=torch.float16)
        os.environ["BONSAI_CUDA_LEGACY_GDN"] = "1"
        expected = reference(x, cache_params=refcache)
        actual = fused(x, cache_params=cache)
        torch.testing.assert_close(actual, expected)
        for step in range(24):
            x = torch.randn(batch, 1, 64, device="cuda", dtype=torch.float16)
            os.environ["BONSAI_CUDA_LEGACY_GDN"] = "1"
            expected = reference(x, cache_params=refcache)
            os.environ.pop("BONSAI_CUDA_LEGACY_GDN", None)
            actual = fused(x, cache_params=cache)
            torch.testing.assert_close(actual, expected, atol=0.0003, rtol=0.01)
            torch.testing.assert_close(
                cache.layers[0].conv_states,
                refcache.layers[0].conv_states,
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                cache.layers[0].recurrent_states,
                refcache.layers[0].recurrent_states,
                atol=0.0003,
                rtol=0.01,
            )
print(
    "PASS: fused GDN convolution, recurrence, gates and norm; 24 cached steps, batches 1/2"
)
