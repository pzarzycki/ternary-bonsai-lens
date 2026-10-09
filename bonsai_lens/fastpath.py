"""Opt-in decode optimizations for the Hadamard-packed model (numerically equivalent up to fp16 rounding).

1. Fused projections. Projections that read the same input (MLP gate|up, attention q|k|v,
   GatedDeltaNet qkv|z) are concatenated into one 2-bit weight matrix: one Hadamard rotation of
   the input and one quantized matmul instead of two or three.
2. Compiled rotation. sign-flip + blockwise FWHT + casts are fused into one compiled kernel.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


@mx.compile
def _rotate(x, signs):
    """sign-flip + normalized blockwise Walsh-Hadamard over the last axis (block = len(signs) ≤ width)."""
    shape, dtype = x.shape, x.dtype
    block = 1024
    y = x.astype(mx.float32) * signs
    y = mx.hadamard_transform(y.reshape(-1, block), scale=1 / math.sqrt(block)).reshape(shape)
    return y.astype(dtype)


def rotate(x, block, signs):
    if block != 1024:  # compiled kernel hardcodes the pack's block size
        from runtime import fwht

        return fwht(x, block, signs)
    return _rotate(x, signs)


class FusedGroup:
    """Holds the concatenated weights of several Packed linears sharing one input."""

    def __init__(self, members):
        p0 = members[0]
        for p in members[1:]:
            if p.block != p0.block or not mx.array_equal(p.signs, p0.signs):
                raise ValueError("members do not share one input rotation")
        self.block, self.signs = p0.block, p0.signs
        self.weight = mx.concatenate([p.weight for p in members])
        self.scales = mx.concatenate([p.scales for p in members])
        self.biases = mx.concatenate([p.biases for p in members])
        sizes = [p.weight.shape[0] for p in members]
        self.splits = [sum(sizes[: i + 1]) for i in range(len(sizes) - 1)]
        mx.eval(self.weight, self.scales, self.biases)
        self._x = None  # last input (kept to compare identity) and its outputs
        self._outs = None

    def output(self, x, index):
        if self._x is not x:
            y = mx.quantized_matmul(
                rotate(x, self.block, self.signs), self.weight, self.scales, self.biases,
                transpose=True, group_size=128, bits=2,
            )
            self._x, self._outs = x, mx.split(y, self.splits, axis=-1)
        out = self._outs[index]
        if index == len(self.splits):  # last member consumed: drop references
            self._x = self._outs = None
        return out


class FusedMember(nn.Module):
    def __init__(self, group: FusedGroup, index: int):
        super().__init__()
        self._group, self._index = group, index

    def __call__(self, x):
        return self._group.output(x, self._index)


def _fuse(module, names):
    members = [getattr(module, n) for n in names]
    group = FusedGroup(members)
    for i, n in enumerate(names):
        setattr(module, n, FusedMember(group, i))


def apply(engine) -> dict:
    """Patch the engine's language model in place. Returns counts of what was changed."""
    import runtime  # the pack's runtime dir is on sys.path (Engine adds it)

    stats = {"mlp": 0, "attn": 0, "gdn": 0}
    for layer in engine.lm.model.layers:
        _fuse(layer.mlp, ["gate_proj", "up_proj"])
        stats["mlp"] += 1
        if getattr(layer, "is_linear", False):
            _fuse(layer.linear_attn, ["in_proj_qkv", "in_proj_z"])
            stats["gdn"] += 1
        else:
            _fuse(layer.self_attn, ["q_proj", "k_proj", "v_proj"])
            stats["attn"] += 1

    # Remaining Packed linears (down_proj, o_proj, out_proj, lm_head) use the compiled rotation.
    original = runtime.fwht

    def fwht(x, block, signs, inverse=False):
        if inverse or block != 1024:
            return original(x, block, signs, inverse)
        return _rotate(x, signs)

    runtime.fwht = fwht
    mx.clear_cache()
    return stats
