"""Shared MLX engine for Ternary Bonsai 2 27B: loading, prompting, incremental forward, readouts.

Readout points (last token only):
  * pre_norm  - residual stream after the last decoder block, before the final RMSNorm   [5120]
  * post_norm - output of the final RMSNorm = exact input of the unembedding (lm_head)    [5120]
  * logits    - lm_head(post_norm)                                                         [248320]

The pack stores linear weights in a Hadamard-rotated basis, but every `Packed` module (lm_head
included) rotates its own input, and the embedding applies the inverse rotation. So the residual
stream - and therefore pre_norm/post_norm - lives in the model's natural, unrotated basis.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACK = ROOT / "models" / "bonsai2-27b-mlx"


class NormTap(nn.Module):
    """Wraps the final RMSNorm and keeps the last-position input/output of the latest call."""

    def __init__(self, norm: nn.Module):
        super().__init__()
        self.inner = norm
        self.calls = 0
        self.pre: mx.array | None = None
        self.post: mx.array | None = None
        self.keep_full = False  # also keep the whole-sequence output (used for image prefill)
        self.full_post: mx.array | None = None

    def __call__(self, h):
        out = self.inner(h)
        self.calls += 1
        self.pre = h[:, -1, :]
        self.post = out[:, -1, :]
        if self.keep_full:
            self.full_post = out
        return out


@dataclass
class Readout:
    logits: mx.array  # [vocab] float32
    pre_norm: mx.array  # [hidden] float32
    post_norm: mx.array  # [hidden] float32


@dataclass
class Session:
    """Token history + model cache. `fed` tokens are already in the cache."""

    tokens: list[int] = field(default_factory=list)
    fed: int = 0
    cache: list | None = None
    image_kwargs: dict | None = None  # pixel_values etc. for the first (prefill) forward
    rope_state: tuple = (None, None)  # (_rope_deltas, _position_ids) owned by this session
    images: list[str] = field(default_factory=list)  # source image paths, in prompt order
    image_final: mx.array | None = None  # final-layer post-norm hidden at every <|image_pad|> [N, hidden]
    image_info: list | None = None  # cached image_details() result
    next_logits: mx.array | None = None  # [1, vocab] prediction after the last *fed* token (lets decoding resume)


class Engine:
    def __init__(self, pack: Path | str = DEFAULT_PACK):
        self.pack = Path(pack)
        sys.path.insert(0, str(self.pack / "runtime"))
        from vision_artifact import generation_eos_ids, load_vl_model

        self.model, self.processor, self.config = load_vl_model(self.pack)
        self.lm = self.model.language_model
        self.tap = NormTap(self.lm.model.norm)
        self.lm.model.norm = self.tap
        self.tokenizer = self.processor.tokenizer
        self.eos_ids = set(generation_eos_ids(self.pack))
        for t in ("<|im_end|>", "<|endoftext|>"):
            i = self.tokenizer.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0:
                self.eos_ids.add(i)
        self.image_token_id = self.config["image_token_id"]
        self.hidden_size = self.config["text_config"]["hidden_size"]
        self.vocab_size = self.config["text_config"]["vocab_size"]
        self._template = None

    # ---------------------------------------------------------------- prompting
    def render(
        self, messages: list[dict], thinking: bool = False, add_generation_prompt=True, reasoning_effort="xhigh",
        tools: list[dict] | None = None,
    ) -> str:
        """Apply the chat template. With thinking on, the template injects a system block whose text
        depends on `reasoning_effort` (xhigh | medium | low); "medium" injects nothing."""
        if self._template is None:
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
            # transformers' tojson (what the model was trained with): jinja's built-in one sorts keys
            # (listing a tool's "content" parameter before "path") and HTML-escapes ' < > &.
            env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False: json.dumps(
                x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)
            self._template = env.from_string((self.pack / "chat_template.jinja").read_text(encoding="utf-8"))
        return self._template.render(
            messages=messages, add_generation_prompt=add_generation_prompt,
            enable_thinking=thinking,
            reasoning_effort=reasoning_effort,
            tools=tools,
        )

    def new_session(self, text: str, images: list[str] | None = None) -> Session:
        """Tokenize `text` (already templated, or raw) into a fresh session.

        With `images`, `text` must contain one `<|vision_start|><|image_pad|><|vision_end|>`
        placeholder per image, in order (the chat template emits one per `{"type": "image"}` item).
        """
        if not images:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return Session(tokens=list(ids))
        from mlx_vlm.utils import prepare_inputs

        inputs = prepare_inputs(self.processor, images=list(images), prompts=text)
        ids = inputs["input_ids"][0].tolist()
        extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        return Session(tokens=ids, image_kwargs=extra, images=list(images))

    def extend(self, session: Session, text: str):
        """Append text (e.g. next user turn) to the session without resetting the cache."""
        session.tokens += self.tokenizer.encode(text, add_special_tokens=False)

    # ---------------------------------------------------------------- forward
    # Prefill is fed in chunks: MLX's attention materializes a [heads, chunk, context] fp32 score
    # matrix, so feeding a long prompt at once needs O(L^2) memory (32k tokens -> 103 GB).
    ATTN_SCORE_BUDGET = 4 * 2**30  # bytes for one chunk's attention scores
    MAX_CHUNK, MIN_CHUNK = 2048, 128

    def _chunk_size(self, context_len: int) -> int:
        heads = self.config["text_config"]["num_attention_heads"]
        c = self.ATTN_SCORE_BUDGET // (heads * 4 * max(context_len, 1))
        return int(max(self.MIN_CHUNK, min(self.MAX_CHUNK, c)))

    def forward(self, session: Session) -> Readout:
        """Feed all not-yet-fed tokens (chunked); return readouts for the last position."""
        if session.cache is None:
            session.cache = self.lm.make_cache()
            session.fed = 0
            session.rope_state = (None, None)
        new = session.tokens[session.fed :]
        if not new:
            raise ValueError("nothing to feed: append a token first")
        x = mx.array([new])
        chunk = self._chunk_size(len(session.tokens))
        has_images = session.fed == 0 and bool(session.image_kwargs)
        # mrope state lives on the model object; swap in this session's so sessions can interleave.
        self.lm._rope_deltas, self.lm._position_ids = session.rope_state
        calls = self.tap.calls
        image_states = []

        if session.fed == 0:
            # Prefill: build input embeddings (scattering vision features into <|image_pad|> slots)
            # and 3D M-RoPE positions for the whole prompt once, then feed it chunk by chunk.
            if has_images:
                kw = session.image_kwargs
                emb = self.model.get_input_embeddings(
                    x, kw["pixel_values"], image_grid_thw=kw.get("image_grid_thw")
                ).inputs_embeds
                positions, deltas = self.lm.get_rope_index(x, kw.get("image_grid_thw"), None, None)
            else:
                emb = self.lm.model.embed_tokens(x)
                positions, _ = self.lm.get_rope_index(x, None, None, None)
                deltas = mx.zeros((1, 1), dtype=mx.int32)
            self.tap.keep_full = has_images
            try:
                for a in range(0, len(new), chunk):
                    b = min(a + chunk, len(new))
                    self.lm(
                        x[:, a:b],
                        inputs_embeds=emb[:, a:b],
                        cache=session.cache,
                        position_ids=positions[..., a:b],
                        skip_logits=True,
                    )
                    if has_images:  # keep the last-layer states at image positions for inspection
                        idx = [i - a for i in range(a, b) if new[i] == self.image_token_id]
                        if idx:
                            image_states.append(self.tap.full_post[0][mx.array(idx)])
                    mx.eval(self.tap.post, *image_states[-1:])
            finally:
                self.tap.keep_full = False
                self.tap.full_post = None
            # Decode steps derive positions as cache_offset + delta.
            self.lm._rope_deltas, self.lm._position_ids = deltas, None
            if image_states:
                session.image_final = mx.concatenate(image_states)
                session.image_info = None
        else:
            # Continuation (decode step or appended turn): positions follow from cache offset + delta.
            for a in range(0, len(new), chunk):
                self.model(x[:, a : a + chunk], cache=session.cache, skip_logits=True)
                if len(new) > chunk:
                    mx.eval(self.tap.post)
        assert self.tap.calls > calls, "final-norm tap was not hit"

        post = self.tap.post
        logits = self.lm.lm_head(post[:, None, :])[0, -1].astype(mx.float32)
        pre, post = self.tap.pre[0].astype(mx.float32), post[0].astype(mx.float32)
        mx.eval(logits, pre, post)
        session.fed = len(session.tokens)
        session.rope_state = (self.lm._rope_deltas, self.lm._position_ids)
        session.next_logits = logits[None, :]
        return Readout(logits=logits, pre_norm=pre, post_norm=post)

    # ---------------------------------------------------------------- image inspection
    def lens_topk(self, h: mx.array, apply_norm: bool, k: int = 5, chunk: int = 256):
        """Project hidden vectors [N, hidden] through (final norm +) lm_head; top-k ids/probs + entropy."""
        ids, probs, ents = [], [], []
        for i in range(0, h.shape[0], chunk):
            x = h[i : i + chunk].astype(mx.float16)
            if apply_norm:
                x = self.tap.inner(x)
            logits = self.lm.lm_head(x).astype(mx.float32)
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            top = mx.argpartition(-logits, k, axis=-1)[:, :k]
            tp = mx.exp(mx.take_along_axis(logp, top, axis=-1))
            order = mx.argsort(-tp, axis=-1)
            top, tp = mx.take_along_axis(top, order, axis=-1), mx.take_along_axis(tp, order, axis=-1)
            ent = -(mx.exp(logp) * logp).sum(axis=-1)
            mx.eval(top, tp, ent)
            ids.append(np.asarray(top))
            probs.append(np.asarray(tp))
            ents.append(np.asarray(ent))
        return np.concatenate(ids), np.concatenate(probs), np.concatenate(ents)

    def image_details(self, session: Session) -> list[dict]:
        """Per-image breakdown of how each image became tokens (computed once per session)."""
        if session.image_info is not None:
            return session.image_info
        if not session.image_kwargs:
            return []
        kw = session.image_kwargs
        grids = np.asarray(kw["image_grid_thw"]).tolist()  # [[t, h_patches, w_patches], ...]
        merge = self.config["vision_config"]["spatial_merge_size"]
        patch = self.config["vision_config"]["patch_size"]
        # Vision tower output = the embeddings scattered over the <|image_pad|> slots.
        emb, _ = self.model.vision_tower(kw["pixel_values"].astype(mx.float16), kw["image_grid_thw"])
        emb = emb.astype(mx.float32)
        norms = np.asarray(mx.linalg.norm(emb, axis=-1))
        in_ids, in_p, in_ent = self.lens_topk(emb, apply_norm=True)
        out = None
        if session.image_final is not None:
            out = self.lens_topk(session.image_final, apply_norm=False)
            out_norms = np.asarray(mx.linalg.norm(session.image_final.astype(mx.float32), axis=-1))
        pos, _ = self.lm.get_rope_index(mx.array([session.tokens]), kw["image_grid_thw"], None, None)
        pos = np.asarray(pos)[:, 0]  # [3, L]
        toks = np.asarray(session.tokens)
        pad_idx = np.where(toks == self.image_token_id)[0]
        text_norm = float(np.asarray(mx.linalg.norm(self.lm.model.embed_tokens(mx.array(toks[toks != self.image_token_id][:256])).astype(mx.float32), axis=-1)).mean())

        result, offset = [], 0
        for n_img, (t, gh, gw) in enumerate(grids):
            rows, cols = gh // merge, gw // merge
            n = t * rows * cols
            sl = slice(offset, offset + n)
            seq = pad_idx[sl]
            vocab = set(in_ids[sl].ravel().tolist()) | (set(out[0][sl].ravel().tolist()) if out else set())
            tokens = []
            for j in range(n):
                g = offset + j
                tok = {
                    "seq": int(seq[j]),
                    "row": (j % (rows * cols)) // cols,
                    "col": j % cols,
                    "mrope": pos[:, seq[j]].tolist(),
                    "norm": float(norms[g]),
                    "in_top": [[int(a), float(b)] for a, b in zip(in_ids[g], in_p[g])],
                    "in_entropy": float(in_ent[g]),
                }
                if out:
                    tok["out_top"] = [[int(a), float(b)] for a, b in zip(out[0][g], out[1][g])]
                    tok["out_entropy"] = float(out[2][g])
                    tok["out_norm"] = float(out_norms[g])
                tokens.append(tok)
            result.append({
                "index": n_img,
                "source": session.images[n_img] if n_img < len(session.images) else None,
                "grid_thw": [t, gh, gw],
                "patch": patch,
                "merge": merge,
                "rows": rows,
                "cols": cols,
                "width": gw * patch,
                "height": gh * patch,
                "n_tokens": n,
                "n_patches": t * gh * gw,
                "patch_dim": int(kw["pixel_values"].shape[-1]),
                "span": [int(seq[0]) - 1, int(seq[-1]) + 1],  # vision_start .. vision_end
                "text_embed_norm": text_norm,
                "has_final": out is not None,
                "vocab": {str(i): self.readable(i) for i in vocab},
                "tokens": tokens,
            })
            offset += n
        if out is not None:  # final-layer part only exists after the prefill step
            session.image_info = result
        return result

    # ---------------------------------------------------------------- pipelined decoding
    def generate(self, session: Session, temperature=0.0, top_p=0.95, top_k=20, min_p=0.0):
        """Yield generated token ids, one per decode step, with the CPU and GPU overlapped.

        From the second token on, step n+1's graph is built and queued (mx.async_eval) on the
        still-lazy token n before token n is read back, so building the graph in Python overlaps GPU work
        (+15% tok/s). Yielded tokens are appended to `session.tokens`; `session.fed` tracks what
        the cache holds. The caller stops by breaking out of the loop.
        """
        if temperature <= 0:
            sample = lambda logits: mx.argmax(logits, axis=-1)
        else:
            from mlx_lm.sample_utils import make_sampler

            sample = make_sampler(temp=temperature, top_p=top_p, top_k=top_k, min_p=min_p)

        if session.fed == len(session.tokens) and session.next_logits is not None:
            first_logits = session.next_logits  # resume: everything is in the cache already
        else:
            first_logits = self.forward(session).logits[None, :]  # prefill / pending tokens
        self.lm._rope_deltas, self.lm._position_ids = session.rope_state

        def step(y):  # queue: feed token y, predict the next one (lazy)
            self.lm(y, cache=session.cache, skip_logits=True)
            logits = self.lm.lm_head(self.tap.post[:, None, :])[:, -1, :].astype(mx.float32)
            session.next_logits = logits
            return sample(logits).reshape(1, 1)

        try:
            # First token: read it back immediately so TTFT doesn't wait for a queued step.
            y = sample(first_logits).reshape(1, 1)
            session.tokens.append(int(y.item()))
            yield session.tokens[-1]  # not fed yet (fed == len - 1) if the caller stops here
            y = step(y)
            mx.async_eval(y)
            session.fed = len(session.tokens)
            while True:
                next_y = step(y)  # queued before y is read back: overlaps graph building with GPU work
                mx.async_eval(next_y)
                session.tokens.append(int(y.item()))
                session.fed = len(session.tokens)
                yield session.tokens[-1]
                y = next_y
        finally:
            session.rope_state = (self.lm._rope_deltas, self.lm._position_ids)

    # ---------------------------------------------------------------- sampling
    @staticmethod
    def sample(logits: mx.array, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0) -> int:
        if temperature <= 0:
            return int(mx.argmax(logits).item())
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
        return int(sampler(logits[None, :]).item())

    # ---------------------------------------------------------------- token helpers
    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def readable(self, i: int) -> str:
        """Decoded text, or the raw BPE piece when the token is a partial UTF-8 byte sequence."""
        t = self.decode([i])
        return f"‹{self.piece(i)}›" if "\ufffd" in t else t

    def piece(self, i: int) -> str:
        """Raw vocabulary entry (byte-level BPE form)."""
        return self.tokenizer.convert_ids_to_tokens(int(i))


def summarize(readout: Readout, k: int = 50) -> dict:
    """Compact JSON-friendly summary of a readout (top-k, stats, full hidden vectors)."""
    logits = np.asarray(readout.logits)
    z = logits - logits.max()
    p = np.exp(z)
    p /= p.sum()
    logp = np.log(np.maximum(p, 1e-30))
    top = np.argpartition(-logits, k)[:k]
    top = top[np.argsort(-logits[top])]
    pre, post = np.asarray(readout.pre_norm), np.asarray(readout.post_norm)

    def vstats(v):
        return {
            "l2": float(np.linalg.norm(v)),
            "mean": float(v.mean()),
            "std": float(v.std()),
            "min": float(v.min()),
            "max": float(v.max()),
            "argmax_abs": int(np.abs(v).argmax()),
        }

    hist, edges = np.histogram(logits, bins=60)
    return {
        "top_ids": top.tolist(),
        "top_logits": logits[top].tolist(),
        "top_probs": p[top].tolist(),
        "entropy": float(-(p * logp).sum()),
        "logit_stats": {
            "mean": float(logits.mean()),
            "std": float(logits.std()),
            "min": float(logits.min()),
            "max": float(logits.max()),
        },
        "logit_hist": {"counts": hist.tolist(), "edges": edges.tolist()},
        "pre_norm": pre.tolist(),
        "post_norm": post.tolist(),
        "pre_stats": vstats(pre),
        "post_stats": vstats(post),
    }


def load_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default
