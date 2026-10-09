"""Preserve GDN convolution and recurrent states for cached multi-token chunks.

Transformers 5.5.0 only uses cached GDN states when seq_len == 1. Lens also
appends turns and prefills in chunks, so both states must carry across those calls.
The projections, head ordering and recurrence remain the pinned implementation.
"""

import os
import torch
from torch.nn import functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    apply_mask_to_padding_states,
    torch_causal_conv1d_update,
)


def forward(self, hidden_states, cache_params=None, attention_mask=None):
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch, length, _ = hidden_states.shape
    cached = cache_params is not None and cache_params.has_previous_state(
        self.layer_idx
    )
    initial_state = (
        cache_params.layers[self.layer_idx].recurrent_states if cached else None
    )
    mixed = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(batch, length, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    if (
        cached
        and length == 1
        and hidden_states.is_cuda
        and os.environ.get("BONSAI_CUDA_LEGACY_GDN") != "1"
    ):
        from .cuda_recurrent import cached_update

        return self.out_proj(cached_update(self, mixed, z, a, b, cache_params))
    beta = b.sigmoid()
    if cached:
        # The portable update supports arbitrary sequence lengths, unlike optional fused updates.
        mixed = torch_causal_conv1d_update(
            mixed,
            cache_params.layers[self.layer_idx].conv_states,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        if cache_params is not None:
            state = F.pad(mixed, (self.conv_kernel_size - length, 0))
            cache_params.update_conv_state(state, self.layer_idx)
        mixed = F.silu(self.conv1d(mixed)[:, :, :length])
    query, key, value = torch.split(
        mixed.transpose(1, 2), [self.key_dim, self.key_dim, self.value_dim], dim=-1
    )
    query = query.reshape(batch, length, -1, self.head_k_dim)
    key = key.reshape(batch, length, -1, self.head_k_dim)
    value = value.reshape(batch, length, -1, self.head_v_dim)
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    repeats = self.num_v_heads // self.num_k_heads
    if repeats > 1:
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
    rule = (
        self.recurrent_gated_delta_rule
        if cached and length == 1
        else self.chunk_gated_delta_rule
    )
    output, state = rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=cache_params is not None,
        use_qk_l2norm_in_kernel=True,
    )
    if cache_params is not None:
        cache_params.update_recurrent_state(state, self.layer_idx)
    output = self.norm(
        output.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
    )
    return self.out_proj(output.reshape(batch, length, -1))
