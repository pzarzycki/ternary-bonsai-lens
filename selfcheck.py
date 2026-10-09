#!/usr/bin/env python3
"""Sanity checks for the engine and readout taps.   uv run selfcheck.py"""

import time

import numpy as np

from bonsai_lens.engine import Engine

t0 = time.perf_counter()
e = Engine()
print(f"loaded in {time.perf_counter() - t0:.1f}s")
if getattr(e, "backend", "mlx") == "llama":
    s = e.new_session(
        e.render(
            [
                {
                    "role": "user",
                    "content": "What is 2 plus 2? Answer with the number only.",
                }
            ]
        )
    )
    output = []
    gen = e.generate(s, temperature=0)
    try:
        for token in gen:
            if token in e.eos_ids:
                break
            output.append(token)
            if len(output) >= 8:
                break
    finally:
        gen.close()
        e.close()
    answer = e.decode(output)
    print("greedy answer:", repr(answer))
    print("PASS" if answer.strip() == "4" else "FAIL")
    raise SystemExit(0 if answer.strip() == "4" else 1)

prompt = e.render(
    [{"role": "user", "content": "What is 2 plus 2? Answer with the number only."}]
)
s = e.new_session(prompt)
print("prompt tokens:", len(s.tokens))

# 1) Reference: the stock path computes lm_head over all positions; compare its last row to our tap.
if getattr(e, "backend", "mlx") == "cuda":
    import torch

    with torch.inference_mode():
        e.model.model.rope_deltas = None
        output = e.model(
            input_ids=torch.tensor([s.tokens], device=e.device),
            use_cache=True,
            logits_to_keep=1,
        )
        ref = output.logits[0, -1].float().cpu().numpy()
        del output
else:
    import mlx.core as mx

    e.lm._rope_deltas = e.lm._position_ids = None
    ref = (
        e.model(mx.array([s.tokens]), cache=e.lm.make_cache())
        .logits[0, -1]
        .astype(mx.float32)
    )
    mx.eval(ref)

calls = e.tap.calls
r = e.forward(s)
d_logits = float(np.max(np.abs(np.asarray(r.logits) - np.asarray(ref))))
if getattr(e, "backend", "mlx") == "cuda":
    with torch.inference_mode():
        norm = (
            e.tap.inner(
                torch.tensor(r.pre_norm[None], device=e.device, dtype=torch.float16)
            )[0]
            .float()
            .cpu()
            .numpy()
        )
else:
    norm = np.asarray(
        e.tap.inner(r.pre_norm[None].astype(mx.float16))[0].astype(mx.float32)
    )
d_norm = float(np.max(np.abs(norm - np.asarray(r.post_norm))))
print(f"max|lm_head(tap_post) - stock_logits| = {d_logits:.4g}")
print(f"max|norm(tap_pre) - tap_post|        = {d_norm:.4g}")
print(f"hidden {r.post_norm.shape}, logits {r.logits.shape}")

# 2) Greedy decode a few steps; tap must fire once per forward.
out = []
tok = e.sample(r.logits, 0)
for _ in range(8):
    if tok in e.eos_ids:
        break
    out.append(tok)
    s.tokens.append(tok)
    r = e.forward(s)
    tok = e.sample(r.logits, 0)
print("tap calls per forward:", (e.tap.calls - calls) / (len(out) + 1))
answer = e.decode(out)
print("greedy answer:", repr(answer))

ok = d_logits < 0.05 and d_norm < 0.05 and "4" in answer
print("PASS" if ok else "FAIL")
print(
    "top-5 at step 0:",
    [
        (e.decode([i]), round(float(x), 2))
        for i, x in zip(
            *(lambda l: (np.argsort(-l)[:5], np.sort(l)[::-1][:5]))(np.asarray(ref))
        )
    ],
)

raise SystemExit(0 if ok else 1)
