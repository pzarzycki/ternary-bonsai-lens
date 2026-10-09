# Generation performance: `baseline`

- Date: 2026-10-05T15:50:13
- Machine: Apple M5 Max, 64 GB
- Software: mlx 0.32.0, mlx-vlm 0.6.3, fastpath=off
- Model: prism-ml/Ternary-Bonsai-2-27B-mlx-2bit (Hadamard-packed 2-bit, vision tower loaded)
- Model load: 1.9 s; sampling: greedy; prefill: budgeted (4 GB attention scores per chunk)

## Results

| case | prompt tok | TTFT (median) | TTFT range | prefill tok/s | 1st decode token | next-token min | avg | max | steady tok/s | all 10: avg | peak mem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text 128 | 128 | 187 ms | 187 ms–188 ms | 684 | 26.2 ms | 25.4 ms | 26.0 ms | 28.4 ms | 38.5 | 26.0 ms (38.5 tok/s) | 9.2 GB |
| text 1024 | 1,024 | 1.46 s | 1.46 s–1.46 s | 700 | 39.0 ms | 25.5 ms | 26.0 ms | 26.6 ms | 38.5 | 27.4 ms (36.5 tok/s) | 11.7 GB |
| text 4096 | 4,096 | 6.56 s | 6.43 s–6.61 s | 624 | 63.2 ms | 26.4 ms | 27.0 ms | 28.7 ms | 37.0 | 30.7 ms (32.6 tok/s) | 14.8 GB |
| text 16384 | 16,384 | 31.99 s | 30.29 s–33.09 s | 512 | 98.5 ms | 29.7 ms | 30.3 ms | 31.3 ms | 33.0 | 37.1 ms (26.9 tok/s) | 18.6 GB |
| text 32768 | 32,768 | 1.3 min | 1.3 min–1.3 min | 427 | 40.7 ms | 36.2 ms | 37.1 ms | 38.7 ms | 26.9 | 37.5 ms (26.7 tok/s) | 20.4 GB |
| text 65536 | 65,536 | 3.2 min | 3.2 min–3.2 min | 341 | 50.8 ms | 45.9 ms | 46.5 ms | 46.9 ms | 21.5 | 47.0 ms (21.3 tok/s) | 23.5 GB |
| image 1024x768 | 786 | 1.89 s | 1.88 s–2.02 s | 416 | 49.5 ms | 27.7 ms | 29.0 ms | 30.2 ms | 34.5 | 31.1 ms (32.2 tok/s) | 11.2 GB |

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

## Findings

1. **Steady decode is about 38.5 tok/s up to 1k context.** It falls as context grows: 37.0 tok/s at 4k, 33.0 at 16k, 26.9 at 32k and 21.5 at 65k. The extra time per token goes to reading the KV cache.
   - Only 16 of the 64 layers have a KV cache; the 48 GatedDeltaNet layers keep a fixed-size state.
   - **The KV cache is stored in float32**, which doubles its memory and bandwidth. That makes it an optimization target.
2. **Prefill throughput** falls from about 700 tok/s (1k) to about 340 tok/s (65k), because attention cost grows quadratically.
3. **The first decode token right after prefill is slower** than the steady state. Across all repetitions it costs about +13 ms at 1k, +36 ms at 4k and +68 ms at 16k, and +20 ms after an image prompt.
   - In an earlier run of this same suite, it **stalled for 1.1 s at 32k and 5.5 s at 65k**.
   - That stall is linked to MLX's buffer pool: about 35 GB of freed buffers stay cached after a long prefill.
   - In a separate test at 32k, capping the pool with `mx.set_cache_limit(4 GB)` cut the first token from 1.8 s to 43 ms and TTFT from 99.5 to 93.9 s, at a cost of about +3 ms per token in steady state.
4. **Long-context numbers vary a lot between runs.** The earlier run of the identical suite measured:

   | case | TTFT | prefill tok/s | steady decode |
   |---|---|---|---|
   | 32k | 1.8 min | 312 | 7.1 tok/s (all-10 avg, stall included) |
   | 65k | 5.5 min | 198 | 1.7 tok/s (all-10 avg, stall included) |

   This run measured 1.3 min and 3.2 min TTFT. That earlier run's results above 16k were probably hurt by memory-pool pressure. When comparing optimizations at 32k and above, run them more than once.
5. **Images:** a 1024×768 image becomes 768 tokens. Its TTFT is 1.89 s, against 1.46 s for a 1024-token text prompt, so encoding through the vision tower adds about 0.5–0.7 s.

## Bottleneck breakdown (decode step at short context, about 25 ms)

| part | time | note |
|---|---:|---|
| Streaming the 6.8 GB of 2-bit weights | about 14 ms | the 2-bit matmul kernel reaches about 500 GB/s (the chip's peak is about 545 GB/s) |
| Hadamard rotations (about 400 per token) | about 3.5–4 ms | small kernel dispatches |
| GatedDeltaNet recurrence, conv, gating (48 layers) | about 3.4 ms | |
| Attention core, norms, residuals | about 3 ms | |
| Building the Python graph | about 2 ms | could overlap with GPU work |

Measured with isolated component timings, not a Metal trace.
