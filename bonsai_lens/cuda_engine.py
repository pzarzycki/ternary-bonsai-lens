"""CUDA loader for the existing Prism Hadamard MLX pack.

Packed matrices stay 2-bit. NVRTC kernels consume them directly; only matrices
that exceed available VRAM stream from RAM. Stock Transformers supplies Qwen's
attention, recurrent cache and vision tower. No remote model code is executed.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .packed_file import PackedFile
from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
    Qwen2VLImageProcessorPil,
)
from .common import PromptMixin, Readout, Session
from .cuda_kernels import matmul, rotate


def to_numpy(x):
    return x.detach().float().cpu().numpy()


def dequantize(weight, scales, biases):
    """Used for selected embedding rows and numerical tests, never full linears."""
    shifts = torch.arange(0, 32, 2, device=weight.device, dtype=torch.int64)
    values = ((weight.to(torch.int64)[..., None] >> shifts) & 3).reshape(
        weight.shape[0], -1
    )
    return (
        (
            values.reshape(weight.shape[0], -1, 128) * scales[..., None]
            + (-scales if biases is None else biases)[..., None]
        )
        .reshape(weight.shape[0], -1)
        .half()
    )


class Packed(nn.Module):
    def __init__(self, arrays, block, signs, embedding, device):
        super().__init__()
        self.weight, self.scales, self.biases = arrays
        self.block, self.embedding, self.device = block, embedding, device
        self.signs = (
            signs.to(device=device, dtype=torch.float16) if signs is not None else None
        )

    def forward(self, x):
        if self.embedding:
            ids = x.flatten().to(self.weight.device)
            w, s, b = (
                a[ids].to(self.device) if a is not None else None
                for a in (self.weight, self.scales, self.biases)
            )
            out = dequantize(w, s, b).reshape(*x.shape, -1)
            return rotate(out, self.block, self.signs, True) if self.block else out
        if self.block:
            x = rotate(x, self.block, self.signs)
        arrays = (self.weight, self.scales, self.biases)
        if self.weight.device.type != "cuda":
            arrays = tuple(a.to(self.device) if a is not None else None for a in arrays)
        return matmul(x, *arrays)


class HostVision(nn.Module):
    """Keep the occasional vision encoder in host memory, prioritizing text weights on CUDA."""

    def __init__(self, inner, device):
        super().__init__()
        self.inner, self.device = inner, device
        self.spatial_merge_size = inner.spatial_merge_size
        self.config = inner.config

    @property
    def dtype(self):
        return self.inner.dtype

    def forward(self, *args, **kwargs):
        args = tuple(a.cpu() if isinstance(a, torch.Tensor) else a for a in args)
        kwargs = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
        }
        result = self.inner(*args, **kwargs)
        result.pooler_output = result.pooler_output.to(self.device)
        result.last_hidden_state = result.last_hidden_state.to(self.device)
        return result


class NormTap(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls = 0
        self.keep_full = False
        self.full_post = None

    def forward(self, h):
        out = self.inner(h)
        self.calls += 1
        self.pre, self.post = h[:, -1, :], out[:, -1, :]
        self.full_post = out if self.keep_full else None
        return out


class Engine(PromptMixin):
    backend = "cuda"

    def __init__(self, pack):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Install an NVIDIA driver and run uv sync (CUDA PyTorch is required)."
            )
        self.device = torch.device("cuda:0")
        self.pack = Path(pack)
        self.config = json.loads(
            (self.pack / "config.json").read_text(encoding="utf-8")
        )
        if (
            self.config.get("model_type") != "prism_hadamard_qwen35"
            or self.config.get("base_model_type") != "qwen3_5"
        ):
            raise ValueError("Expected a Prism Hadamard Qwen3.5 vision pack")
        torch.backends.cuda.matmul.allow_tf32 = True
        cfg = Qwen3_5Config(
            **{
                k: v
                for k, v in self.config.items()
                if k
                in (
                    "text_config",
                    "vision_config",
                    "image_token_id",
                    "video_token_id",
                    "vision_start_token_id",
                    "vision_end_token_id",
                )
            }
        )
        cfg._attn_implementation = "sdpa"
        with torch.device("meta"):
            self.model = Qwen3_5ForConditionalGeneration(cfg)
        self.lm = self.model.model.language_model
        from types import MethodType
        from .cuda_gdn import forward as cached_gdn_forward

        for layer in self.lm.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.forward = MethodType(
                    cached_gdn_forward, layer.linear_attn
                )
        self._template = None
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.pack))
        self.image_processor = Qwen2VLImageProcessorPil.from_pretrained(str(self.pack))
        self.hidden_size = cfg.text_config.hidden_size
        self.vocab_size = cfg.text_config.vocab_size
        self.image_token_id = cfg.image_token_id
        eos = json.loads(
            (self.pack / "generation_config.json").read_text(encoding="utf-8")
        ).get("eos_token_id", [])
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        for token in ("<|im_end|>", "<|endoftext|>"):
            i = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(i, int) and i >= 0:
                self.eos_ids.add(i)
        print("Loading packed CUDA weights...", flush=True)
        self._load_weights()
        if os.environ.get("BONSAI_CUDA_MLP_GRAPHS", "1") == "1":
            from .cuda_graphs import GraphedMLP

            for layer in self.lm.layers:
                # Host-streamed weights cannot be captured in a CUDA graph.
                if all(
                    proj.weight.is_cuda
                    for proj in (
                        layer.mlp.gate_proj,
                        layer.mlp.up_proj,
                        layer.mlp.down_proj,
                    )
                ):
                    layer.mlp = GraphedMLP(layer.mlp)
        self.tap = NormTap(self.lm.norm)
        self.lm.norm = self.tap
        self.model.eval()
        self.runtime_info = {
            "backend": "cuda",
            "lens_readouts": True,
            "gpu": torch.cuda.get_device_name(),
            "packed_gpu_gb": round(self.gpu_bytes / 2**30, 3),
            "packed_cpu_gb": round(self.cpu_bytes / 2**30, 3),
            "vision_device": "cpu",
            "mlp_graphs": os.environ.get("BONSAI_CUDA_MLP_GRAPHS", "1") == "1",
            "host_projections": self.host_projections,
            "prefill_tile_stride": 136
            if os.environ.get("BONSAI_CUDA_MMA_PADDING", "1") == "1"
            else 128,
        }
        print("CUDA runtime:", self.runtime_info, flush=True)

    def _load_weights(self):
        def target_name(name):
            if name.startswith("language_model.model."):
                return "model.language_model." + name[len("language_model.model.") :]
            if name.startswith("language_model.lm_head."):
                return "lm_head." + name[len("language_model.lm_head.") :]
            if name.startswith("vision_tower."):
                return "model.visual." + name[len("vision_tower.") :]
            raise ValueError(f"Unmapped tensor {name}")

        records = {"language_model." + r["path"]: r for r in self.config["modules"]}
        packed_keys = {
            key + "." + suffix
            for key in records
            for suffix in ("weight", "scales", "biases", "signs")
        }
        self.gpu_bytes = self.cpu_bytes = 0
        self.host_projections = []
        with PackedFile(self.pack / "model.safetensors") as f:
            dense = {}
            for key in f.keys():
                if key in packed_keys:
                    continue
                name = target_name(key)
                value = f.get_tensor(key)
                if name.endswith("linear_attn.conv1d.weight"):
                    value = value.transpose(1, 2)
                if name == "model.visual.patch_embed.proj.weight":
                    value = value.permute(0, 4, 1, 2, 3)
                # MLX stores RMSNorm's actual multiplier; HF stores multiplier minus one.
                module = self.model.get_submodule(name.rsplit(".", 1)[0])
                if module.__class__.__name__ == "Qwen3_5RMSNorm":
                    value = value.float() - 1
                dense[name] = value.to(
                    device="cpu" if name.startswith("model.visual.") else self.device,
                    dtype=torch.float16,
                )
            result = self.model.load_state_dict(dense, strict=False, assign=True)
            if result.unexpected_keys:
                raise ValueError(f"Unexpected weights: {result.unexpected_keys}")
            del dense
            # Leave space for recurrent/KV states, vision activations and streamed projections.
            free, _ = torch.cuda.mem_get_info()
            budget = max(
                0,
                free
                - int(float(os.environ.get("BONSAI_CUDA_RESERVE_GB", "0.45")) * 2**30),
            )
            print(
                "Dense CUDA weights loaded; packed VRAM budget:",
                round(budget / 2**30, 2),
                "GiB",
                flush=True,
            )
            for index, (key, record) in enumerate(records.items()):
                if index % 50 == 0:
                    print("Loading packed module", index, "/", len(records), flush=True)
                path = target_name(key + ".weight").rsplit(".", 1)[0]
                old = self.model.get_submodule(path)
                arrays = [
                    f.get_tensor(key + "." + s) for s in ("weight", "scales", "biases")
                ]
                width = arrays[0].shape[1] * 16
                if tuple(old.weight.shape) != (arrays[0].shape[0], width):
                    raise ValueError(f"Packed shape mismatch: {path}")
                if not torch.equal(arrays[2], -arrays[1]):
                    raise ValueError(f"{path}: expected ternary bias = -scale")
                # Reject the unused fourth code. Validate in bounded host-memory chunks.
                bits = arrays[0].numpy().view(np.uint32)
                for a in range(0, bits.shape[0], 2048):
                    w = bits[a : a + 2048]
                    if np.any(w & (w >> np.uint32(1)) & np.uint32(0x55555555)):
                        raise ValueError(f"{path}: fourth code in ternary weights")
                # The documented ternary bias is redundant: consume exactly {-s,0,+s}.
                arrays = [arrays[0].view(torch.int32), arrays[1].half(), None]
                size = sum(
                    a.numel() * a.element_size() for a in arrays if a is not None
                )
                if size <= budget and not record["embedding"]:
                    arrays = [
                        a.to(self.device) if a is not None else None for a in arrays
                    ]
                    budget -= size
                    self.gpu_bytes += size
                else:
                    self.cpu_bytes += size
                    if not record["embedding"]:
                        self.host_projections.append(path)
                signs = f.get_tensor(key + ".signs") if record["block"] else None
                if signs is not None and (
                    signs.numel() != width
                    or not torch.all((signs == 1) | (signs == -1))
                ):
                    raise ValueError(f"Invalid Hadamard signs: {path}")
                parent, _, attr = path.rpartition(".")
                setattr(
                    self.model.get_submodule(parent) if parent else self.model,
                    attr,
                    Packed(
                        arrays, record["block"], signs, record["embedding"], self.device
                    ),
                )
        remaining = [n for n, p in self.model.named_parameters() if p.is_meta]
        if remaining:
            raise ValueError(f"Checkpoint is missing parameters: {remaining}")
        # Rotary frequencies are non-persistent buffers, initialized on meta with the model.
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5TextRotaryEmbedding,
            Qwen3_5VisionRotaryEmbedding,
        )

        self.lm.rotary_emb = Qwen3_5TextRotaryEmbedding(
            self.model.config.text_config, device=self.device
        )
        visual = self.model.model.visual
        visual.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(
            visual.config.hidden_size // visual.config.num_heads // 2
        )
        self.model.model.visual = HostVision(visual, self.device)

    def new_session(self, text, images=None):
        if not images:
            return Session(
                tokens=list(self.tokenizer.encode(text, add_special_tokens=False))
            )
        from PIL import Image

        with_images = []
        for path in images:
            with Image.open(path) as image:
                with_images.append(image.convert("RGB"))
        inputs = self.image_processor(images=with_images, return_tensors="pt")
        grids = inputs["image_grid_thw"]
        merge = self.config["vision_config"]["spatial_merge_size"]
        for grid in grids:
            text = text.replace(
                "<|image_pad|>",
                "<|placeholder|>" * int(grid.prod().item() // merge**2),
                1,
            )
        text = text.replace("<|placeholder|>", "<|image_pad|>")
        kwargs = {k: v.to(self.device) for k, v in inputs.items()}
        return Session(
            tokens=list(self.tokenizer.encode(text, add_special_tokens=False)),
            image_kwargs=kwargs,
            images=list(images),
        )

    @torch.inference_mode()
    def forward(self, session):
        new = session.tokens[session.fed :]
        if not new:
            raise ValueError("nothing to feed: append a token first")
        first = session.fed == 0
        x = torch.tensor([new], device=self.device)
        model = self.model.model
        model.rope_deltas = session.rope_state[0]
        kwargs = dict(session.image_kwargs or {}) if first else {}
        # HF requires multimodal type ids to compute the image M-RoPE positions.
        if first and kwargs:
            kwargs["mm_token_type_ids"] = (x == self.image_token_id).long()
        chunk = max(1, int(os.environ.get("BONSAI_PREFILL_CHUNK", "256")))
        image_states = []
        embeddings = None
        if first and kwargs:
            embeddings = self.lm.embed_tokens(x)
            features = torch.cat(
                model.get_image_features(
                    pixel_values=kwargs["pixel_values"],
                    image_grid_thw=kwargs["image_grid_thw"],
                ).pooler_output
            )
            mask, _ = model.get_placeholder_mask(
                x, inputs_embeds=embeddings, image_features=features
            )
            embeddings = embeddings.masked_scatter(mask, features.to(embeddings.dtype))
            positions, model.rope_deltas = model.get_rope_index(
                x,
                image_grid_thw=kwargs["image_grid_thw"],
                mm_token_type_ids=kwargs["mm_token_type_ids"],
            )
        else:
            delta = model.rope_deltas if model.rope_deltas is not None else 0
            positions = (
                torch.arange(session.fed, session.fed + len(new), device=self.device)[
                    None, None, :
                ].expand(3, 1, -1)
                + delta
            )
        self.tap.keep_full = bool(first and kwargs)
        try:
            for a in range(0, len(new), chunk):
                b = min(a + chunk, len(new))
                emb = (
                    embeddings[:, a:b]
                    if embeddings is not None
                    else self.lm.embed_tokens(x[:, a:b])
                )
                output = self.lm(
                    inputs_embeds=emb,
                    position_ids=positions[..., a:b],
                    past_key_values=session.cache,
                    use_cache=True,
                )
                session.cache = output.past_key_values
                if self.tap.keep_full:
                    image_mask = x[0, a:b] == self.image_token_id
                    if image_mask.any():
                        image_states.append(self.tap.full_post[0, image_mask].clone())
            logits = self.model.lm_head(self.tap.post[:, None, :])[0, -1].float()
            if image_states:
                session.image_final = torch.cat(image_states)
            session.fed = len(session.tokens)
            session.rope_state = (model.rope_deltas, None)
            session.next_logits = to_numpy(logits)[None, :]
            if first or len(new) > 1:
                # Release prefill scratch buffers before decoding on small WDDM GPUs.
                torch.cuda.empty_cache()
            return Readout(
                to_numpy(logits), to_numpy(self.tap.pre[0]), to_numpy(self.tap.post[0])
            )
        finally:
            self.tap.keep_full = False
            self.tap.full_post = None

    def generate(self, session, temperature=0.0, top_p=0.95, top_k=20, min_p=0.0):
        while True:
            logits = (
                session.next_logits[0]
                if session.fed == len(session.tokens)
                and session.next_logits is not None
                else self.forward(session).logits
            )
            token = self.sample(logits, temperature, top_p, top_k, min_p)
            session.tokens.append(token)
            yield token

    @staticmethod
    def sample(logits, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0):
        logits = torch.as_tensor(logits, device="cuda", dtype=torch.float32).flatten()
        if temperature <= 0:
            return int(logits.argmax().item())
        logits = logits / temperature
        order = logits.argsort(descending=True)
        probs = logits[order].softmax(-1)
        if top_k > 0:
            probs[top_k:] = 0
        if min_p > 0:
            probs[probs < probs.max() * min_p] = 0
        if top_p < 1:
            normalized = probs / probs.sum()
            probs[normalized.cumsum(-1) - normalized >= top_p] = 0
        return int(order[torch.multinomial(probs, 1)].item())

    @torch.inference_mode()
    def lens_topk(self, h, apply_norm, k=5, chunk=16):
        ids, probs, ents = [], [], []
        for part in h.split(chunk):
            if apply_norm:
                part = self.tap.inner(part.half())
            logp = self.model.lm_head(part.half()).float().log_softmax(-1)
            values, indices = logp.topk(k)
            ids.append(indices.cpu().numpy())
            probs.append(values.exp().cpu().numpy())
            ents.append(to_numpy(-(logp.exp() * logp).sum(-1)))
        return np.concatenate(ids), np.concatenate(probs), np.concatenate(ents)

    @torch.inference_mode()
    def image_details(self, session):
        if session.image_info is not None:
            return session.image_info
        if not session.image_kwargs:
            return []
        kw = session.image_kwargs
        emb = torch.cat(self.model.model.get_image_features(**kw).pooler_output)
        norms = to_numpy(emb.float().norm(dim=-1))
        inp = self.lens_topk(emb, True)
        out = (
            self.lens_topk(session.image_final, False)
            if session.image_final is not None
            else None
        )
        outnorm = to_numpy(session.image_final.float().norm(dim=-1)) if out else None
        x = torch.tensor([session.tokens], device=self.device)
        positions, _ = self.model.model.get_rope_index(
            x,
            image_grid_thw=kw["image_grid_thw"],
            mm_token_type_ids=(x == self.image_token_id).long(),
        )
        positions = positions[:, 0].cpu().numpy()
        pads = np.where(np.asarray(session.tokens) == self.image_token_id)[0]
        tc = self.config["vision_config"]
        merge, patch = tc["spatial_merge_size"], tc["patch_size"]
        txt = x[0][x[0] != self.image_token_id][:256]
        text_norm = float(self.lm.embed_tokens(txt).float().norm(dim=-1).mean().item())
        result = []
        offset = 0
        for index, grid in enumerate(kw["image_grid_thw"].tolist()):
            t, gh, gw = grid
            rows, cols = gh // merge, gw // merge
            n = t * rows * cols
            seq = pads[offset : offset + n]
            tokens = []
            vocab = set(inp[0][offset : offset + n].ravel().tolist())
            if out:
                vocab.update(out[0][offset : offset + n].ravel().tolist())
            for j, p in enumerate(seq):
                g = offset + j
                tok = {
                    "seq": int(p),
                    "row": j // cols % rows,
                    "col": j % cols,
                    "mrope": positions[:, p].tolist(),
                    "norm": float(norms[g]),
                    "in_top": [
                        [int(a), float(b)] for a, b in zip(inp[0][g], inp[1][g])
                    ],
                    "in_entropy": float(inp[2][g]),
                }
                if out:
                    tok.update(
                        out_top=[
                            [int(a), float(b)] for a, b in zip(out[0][g], out[1][g])
                        ],
                        out_entropy=float(out[2][g]),
                        out_norm=float(outnorm[g]),
                    )
                tokens.append(tok)
            result.append(
                dict(
                    index=index,
                    source=session.images[index],
                    grid_thw=grid,
                    patch=patch,
                    merge=merge,
                    rows=rows,
                    cols=cols,
                    width=gw * patch,
                    height=gh * patch,
                    n_tokens=n,
                    n_patches=t * gh * gw,
                    patch_dim=int(kw["pixel_values"].shape[-1]),
                    span=[int(seq[0]) - 1, int(seq[-1]) + 1],
                    text_embed_norm=text_norm,
                    has_final=out is not None,
                    vocab={str(i): self.readable(i) for i in vocab},
                    tokens=tokens,
                )
            )
            offset += n
        if out:
            session.image_info = result
        return result
