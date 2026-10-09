# Windows CUDA comparison

Same ternary model, greedy sampling, identical prompt token IDs, fresh prompt caches. Backends run sequentially; one warmup precedes measurements. CUDA uses the MLX 2-bit container; llama.cpp uses native PTQ1_0. This compares the complete implementations, not just kernels.

| Backend | Prompt tokens | First token (s) | Decode (tokens/s) |
|---|---:|---:|---:|
| cuda | 128 | 2.777 | 1.81 |
| cuda | 512 | 11.912 | 1.69 |

Both arithmetic checks returned `4`. Floating-point differences across kernels may change generated sequences; raw tokens and timings are in the adjacent JSON files.

macOS MLX is preserved but was not executed on this Windows host.
