"""Check cached GDN chunks against whole-prompt and single-token execution on CPU."""

import sys
from pathlib import Path
from types import MethodType

import torch
from transformers import Qwen3_5TextConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bonsai_lens.cuda_gdn import forward

torch.manual_seed(13)
cfg = Qwen3_5TextConfig(
    hidden_size=16,
    num_hidden_layers=2,
    layer_types=["linear_attention", "full_attention"],
    linear_num_key_heads=1,
    linear_num_value_heads=2,
    linear_key_head_dim=4,
    linear_value_head_dim=4,
    linear_conv_kernel_dim=4,
)
layer = Qwen3_5GatedDeltaNet(cfg, 0).eval()
x = torch.randn(1, 15, 16)
with torch.inference_mode():
    reference = layer(x, cache_params=DynamicCache(config=cfg))
    layer.forward = MethodType(forward, layer)
    for sizes in ((6, 7, 2), (1,) * 15, (3, 1, 5, 6)):
        cache = DynamicCache(config=cfg)
        parts = []
        offset = 0
        for size in sizes:
            parts.append(layer(x[:, offset : offset + size], cache_params=cache))
            offset += size
        torch.testing.assert_close(
            torch.cat(parts, dim=1), reference, rtol=2e-5, atol=2e-6
        )
print(
    "PASS: cached multi-token GDN chunks and single-token decoding match whole-prompt reference"
)
