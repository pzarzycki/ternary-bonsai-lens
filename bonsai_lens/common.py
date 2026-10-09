"""Backend-independent sessions, prompting and readout serialization."""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Readout:
    logits: np.ndarray  # [vocab] float32
    pre_norm: np.ndarray  # [hidden] float32
    post_norm: np.ndarray  # [hidden] float32


@dataclass
class Session:
    """Token history + model cache. `fed` tokens are already in the cache."""

    tokens: list[int] = field(default_factory=list)
    fed: int = 0
    cache: list | None = None
    image_kwargs: dict | None = (
        None  # pixel_values etc. for the first (prefill) forward
    )
    rope_state: tuple = (
        None,
        None,
    )  # (_rope_deltas, _position_ids) owned by this session
    images: list[str] = field(
        default_factory=list
    )  # source image paths, in prompt order
    image_final: np.ndarray | None = (
        None  # final-layer post-norm hidden at every <|image_pad|> [N, hidden]
    )
    image_info: list | None = None  # cached image_details() result
    next_logits: np.ndarray | None = (
        None  # [1, vocab] prediction after the last *fed* token (lets decoding resume)
    )


class PromptMixin:
    def render(
        self,
        messages: list[dict],
        thinking: bool = False,
        add_generation_prompt=True,
        reasoning_effort="xhigh",
        tools: list[dict] | None = None,
    ) -> str:
        """Apply the chat template. With thinking on, the template injects a system block whose text
        depends on `reasoning_effort` (xhigh | medium | low); "medium" injects nothing."""
        if self._template is None:
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
            # transformers' tojson (what the model was trained with): jinja's built-in one sorts keys
            # (listing a tool's "content" parameter before "path") and HTML-escapes ' < > &.
            env.filters["tojson"] = (
                lambda x, indent=None, separators=None, sort_keys=False: json.dumps(
                    x,
                    ensure_ascii=False,
                    indent=indent,
                    separators=separators,
                    sort_keys=sort_keys,
                )
            )
            self._template = env.from_string(
                (self.pack / "chat_template.jinja").read_text(encoding="utf-8")
            )
        return self._template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=thinking,
            reasoning_effort=reasoning_effort,
            tools=tools,
        )

    def extend(self, session: Session, text: str):
        """Append text (e.g. next user turn) to the session without resetting the cache."""
        session.tokens += self.tokenizer.encode(text, add_special_tokens=False)

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
    k = min(k, logits.size)
    top = np.argpartition(-logits, k - 1)[:k]
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
