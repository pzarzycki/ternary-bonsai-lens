"""Validate the CUDA backend's real, host-resident vision weights without loading text matrices.

uv run python tests/check_vision.py
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
    Qwen2VLImageProcessorPil,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5VisionModel,
    Qwen3_5VisionRotaryEmbedding,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bonsai_lens.packed_file import PackedFile

pack = ROOT / "models" / "bonsai2-27b-mlx"
config = json.loads((pack / "config.json").read_text(encoding="utf-8"))
cfg = Qwen3_5VisionConfig(**config["vision_config"])
cfg._attn_implementation = "sdpa"
with torch.device("meta"):
    vision = Qwen3_5VisionModel(cfg)
weights = {}
with PackedFile(pack / "model.safetensors") as f:
    for key in f.keys():
        if not key.startswith("vision_tower."):
            continue
        name = key.removeprefix("vision_tower.")
        value = f.get_tensor(key)
        if name == "patch_embed.proj.weight":
            value = value.permute(0, 4, 1, 2, 3)
        weights[name] = value.half()
vision.load_state_dict(weights, assign=True)
vision.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(
    cfg.hidden_size // cfg.num_heads // 2
)
vision.eval()
processor = Qwen2VLImageProcessorPil.from_pretrained(str(pack))
inputs = processor(images=[Image.new("RGB", (96, 96), "red")], return_tensors="pt")
with torch.inference_mode():
    result = vision(inputs["pixel_values"].half(), grid_thw=inputs["image_grid_thw"])
assert result.pooler_output.shape[-1] == config["text_config"]["hidden_size"]
assert torch.isfinite(result.pooler_output).all()
print(
    "PASS: real vision weights, CPU fp16 encoder, finite merged image embeddings",
    tuple(result.pooler_output.shape),
)
