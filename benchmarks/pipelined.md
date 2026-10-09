# Generation performance: `pipelined`

- Date: 2026-10-05T16:40:03
- Machine: Apple M5 Max, 64 GB
- Software: mlx 0.32.0, mlx-vlm 0.6.3, fastpath=off, decode loop: pipelined
- Model: prism-ml/Ternary-Bonsai-2-27B-mlx-2bit (Hadamard-packed 2-bit, vision tower loaded)
- Model load: 1.2 s; sampling: greedy; prefill: budgeted (4 GB attention scores per chunk)

## Results

| case | prompt tok | TTFT (median) | TTFT range | prefill tok/s | 1st decode token | next-token min | avg | max | steady tok/s | all 10: avg | peak mem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text 128 | 128 | 187 ms | 186 ms–187 ms | 686 | 44.7 ms | 22.3 ms | 22.6 ms | 25.1 ms | 44.3 | 24.8 ms (40.2 tok/s) | 9.2 GB |
| text 1024 | 1,024 | 1.46 s | 1.45 s–1.46 s | 702 | 53.1 ms | 22.2 ms | 22.4 ms | 22.7 ms | 44.7 | 25.5 ms (39.3 tok/s) | 11.7 GB |
| text 4096 | 4,096 | 6.80 s | 6.50 s–7.25 s | 603 | 83.3 ms | 23.2 ms | 23.6 ms | 24.4 ms | 42.5 | 29.5 ms (33.9 tok/s) | 14.8 GB |
| text 16384 | 16,384 | 34.67 s | 33.99 s–38.00 s | 473 | 124.5 ms | 26.8 ms | 28.5 ms | 29.9 ms | 35.1 | 38.1 ms (26.3 tok/s) | 18.6 GB |
| text 32768 | 32,768 | 1.7 min | 1.7 min–1.7 min | 313 | 94.2 ms | 30.9 ms | 31.1 ms | 31.5 ms | 32.2 | 37.4 ms (26.7 tok/s) | 20.4 GB |
| text 65536 | 65,536 | 3.7 min | 3.7 min–3.7 min | 299 | 1.42 s | 39.4 ms | 39.8 ms | 40.4 ms | 25.1 | 178.1 ms (5.6 tok/s) | 23.5 GB |
| image 1024x768 | 786 | 1.93 s | 1.93 s–4.51 s | 407 | 80.4 ms | 24.5 ms | 25.4 ms | 26.7 ms | 39.4 | 30.9 ms (32.3 tok/s) | 11.2 GB |

## Method

- **TTFT** is measured from the start of tokenization to the first sampled token: tokenize, then a chunked prefill forward pass, then greedy sampling.
- **Next-token latency** is the time for each of the following 10 tokens: one forward pass plus sampling, with the real generated token fed back.
  - **1st decode token** is the first of those, measured right after prefill (median over repetitions). It is reported on its own because it can stall; see the findings below.
  - **Next-token min/avg/max** and **steady tok/s** cover the remaining 9 tokens across all repetitions.
  - **All 10: avg** averages over all 10 tokens, including the first.
- **Prompts** are built by `bench.py`:
  - Text prompts are chat-templated, with thinking off. Their body is filler source code (mlx-vlm `language.py`, repeated), trimmed so the prompt has exactly *N* tokens, and ends with a question.
  - The image case is a synthetic 1024×768 scene with a one-line prompt.
- **Repetitions and warm-up:** cases up to 16k run 3 times (the median TTFT is reported); 32k and 65k run once. A warm-up run comes first, so Metal kernel compilation is excluded.
- **Isolation:** nothing else used the GPU during the run (the web server was stopped).

Raw records: `benchmarks/results.jsonl`.

## Compared with `baseline.md`

**Change:** `Engine.generate` overlaps graph building with GPU work. From the 2nd token on, step *n*+1's graph is built and queued with `mx.async_eval` before token *n* is read back. The first token is read immediately, so TTFT is unaffected. Greedy output is identical to the step-by-step loop, verified on 60 tokens for a text prompt and an image prompt.

| case | steady tok/s, baseline → pipelined | gain | next-token avg | TTFT |
|---|---|---:|---|---|
| text 128 | 38.5 → **44.3** | ×1.15 | 26.0 → 22.6 ms | 187 → 190 ms |
| text 1024 | 38.5 → **44.7** | ×1.16 | 26.0 → 22.4 ms | 1.46 → 1.46 s |
| text 4096 | 37.0 → **42.5** | ×1.15 | 27.0 → 23.6 ms | 6.56 → 6.80 s |
| text 16384 | 33.0 → **35.1** | ×1.06 | 30.3 → 28.5 ms | 32.0 → 34.7 s |
| text 32768 | 26.9 → **32.2** | ×1.19 | 37.1 → 31.1 ms | 1.3 → 1.7 min* |
| text 65536 | 21.5 → **25.1** | ×1.17 | 46.5 → 39.8 ms | 3.2 → 3.7 min* |
| image 1024x768 | 34.5 → **39.4** | ×1.14 | 29.0 → 25.4 ms | 1.89 → 1.93 s |

\* The prefill code is unchanged. The long-context TTFT spread is the same run-to-run variance documented in `baseline.md`, where an earlier identical baseline run measured 1.8 and 5.5 min.

**Trade-offs:**
- **The gap between the 1st and 2nd tokens grows by about one step,** for example 26 → 45 ms at 128. With the pipeline, reading a token back waits for the step queued after it. It's a one-time cost of about 20 ms per reply, after which every token arrives faster.
- **The first-decode stall after a long prefill (MLX buffer-pool reclaim) still happens intermittently:** 1.4 s at 65k in this run, 40 ms in the previous one. The fix is to cap the buffer cache during prefill; not done yet.

**Ruled out:**
- **fastpath (fused projections and compiled rotation):** adds only about 1% on top of the pipelining.
- **Removing every Hadamard rotation**, an upper bound with wrong numerics measured for timing only, would reach 47.6 tok/s. So custom rotation kernels can't add more than about 5%.
