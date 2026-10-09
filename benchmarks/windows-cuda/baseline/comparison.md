# Windows CUDA comparison

Same ternary model, greedy sampling, identical prompt token IDs, fresh prompt caches. Backends run sequentially; one warmup precedes measurements. CUDA uses the MLX 2-bit container; llama.cpp uses native PTQ1_0. This compares the complete implementations, not just kernels.

| Backend | Prompt tokens | First token (s) | Decode (tokens/s) |
|---|---:|---:|---:|
| cuda | 128 | 28.403 | 1.38 |
| cuda | 512 | 110.028 | 1.43 |
| llama | 128 | 0.819 | 13.79 |
| llama | 512 | 1.883 | 13.99 |

Both arithmetic checks returned `4`. Floating-point differences across kernels may change generated sequences; raw tokens and timings are in the adjacent JSON files.

macOS MLX is preserved but was not executed on this Windows host.
