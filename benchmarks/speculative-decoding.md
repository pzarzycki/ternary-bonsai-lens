# Speculative decoding: investigated, not pursued

**Goal:** at least 2× decode speed over the baseline (38.5 tok/s steady at short context, see `baseline.md`).
**Verdict:** speculative decoding can't reliably reach 2× on this stack. It was stopped before integration.

## Measurements (M5 Max, this pack)

Verifying *k* tokens in one target forward pass (`verify(k)`), stock MLX kernels:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| ms | 24.6 | 42.4 | 47.2 | 52.4 | 58.9 | 70.4 | 83.5 | 97.2 |
| × decode step | 1.0 | 1.72 | 1.92 | 2.13 | 2.39 | 2.86 | 3.39 | 3.95 |

The cause is the 2-bit `quantized_matmul`:
- It is memory-bound for a single row: 0.045 ms for 5120→17408.
- For 2–16 rows it switches to a kernel limited by unpacking, not memory: 0.19 ms at 8 rows, 4.3× slower.

Non-matmul work in a verify pass (GDN recurrence with state capture, attention, norms) also grows with *k*: about 21 ms at k=8, against about 11 ms at k=1.

## Why 2× is out of reach

Expected rate = tokens accepted per cycle ÷ (verify(k) + draft time). The ternary-tuned DFlash2 drafter (`naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2`) accepts these many tokens per 8-token cycle, as reported by its author: about 2.6 for chat, 3.3–4.1 for code, about 4.6 for raw code completion.

| verify kernel | verify(8) | chat | code | raw code |
|---|---:|---:|---:|---:|
| stock MLX (measured) | 83.5 ms | 0.7× | 0.9–1.1× | 1.2× |
| custom 8-row kernel, as reported by the drafter's author (2.3× decode step) | about 57 ms | 1.0× | 1.3–1.6× | 1.8× |
| perfect kernel (matmuls at 1-row cost) | about 35 ms | 1.7× | 2.1–2.7× | 3.0× |

All rows assume a 4-bit drafter at about 3 ms per cycle.
- The drafter's author measured **1.2–1.5× on code and math, and break-even or a slight loss on chat** on an M4 Pro (`NakliTechie/dflash-mlx-bonsai2`).
- Even a theoretically perfect verify kernel reaches 2× only on code-heavy output.
- Sampling at temperature above 0 lowers acceptance further.

## What was tried

- **mlx-vlm's speculative path.** It supports DFlash v1 and MTP for `qwen3_5`, including GDN-state rollback. Its multi-row verify kernel handles only 4- and 5-bit weights, and DFlash2's candidate selector is not implemented.
- **A 2-bit adaptation of that verify kernel.** It was numerically correct (max error 0.006–0.012) but 25× slower than stock at 8 rows, because the per-row input buffers overflow registers. It was discarded.

## Single-stream ceiling

Decode streams 6.8 GB of weights per token at about 500 GB/s, which takes about 14 ms of the 25 ms step. Removing every other overhead gives at most about 1.75×; true ternary packing (about 1.7 instead of 2.25 bits per weight) brings that to about 2.3×, but neither is reachable in practice (see the README).
