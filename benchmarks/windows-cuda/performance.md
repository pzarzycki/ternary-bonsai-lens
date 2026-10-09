# Windows ternary performance

Measured on an RTX 3070 Ti Laptop (8 GB), i9-12900H and 16 GB RAM. Both implementations use Ternary Bonsai 2 27B. Direct CUDA reads the original MLX 2-bit pack; Prism llama.cpp reads the official PTQ1_0 GGUF. No dense replacement checkpoint is used.

## Update: fast native readouts

Prism now provides **real final-layer readouts through its existing DLL capture API**. The app's small bridge uses the unchanged official CUDA kernels. It exports the layer-63 residual, masked normalized hidden vector and full logits for the last position, without a second model or recomputing projections in Python. The two original backends remain available; Windows `auto` prefers this native implementation when the bridge is built.

Three-trial Turbo medians, 32 generated tokens and the same prompt IDs as before, **capturing full readouts on every token**:

| Prompt tokens | First token (seconds) | Decode tokens/s |
|---|---:|---:|
| 128 | 0.229 | 40.05 |
| 512 | 0.822 | 40.55 |

[Raw native readout timings and token IDs](prism-readouts/llama.json). This is about 22–25x faster than the last direct CUDA Turbo run below. These timings cover native computation and vector copies into Python, not HTTP serialization/rendering of every token in the UI.

The bridge configures one output row and only registers capture when its mode changes. Re-registering it every token caused graph reservation overhead and initially limited decode to 16–17 tokens/s; that overhead was removed. Chat/Code disable hidden-state capture. The pinned [Qwen3.5 graph](https://github.com/PrismML-Eng/llama.cpp/blob/2459f68b5c0eb26261fd5a81682004b93cd645ba/src/models/qwen35.cpp) assigns `h_nextn` after final RMSNorm; its name/comments must not be interpreted as a pre-norm tap. Pre-norm comes from the separate final-layer capture.

Native checks found **exact capture-on/off logit equality** for the arithmetic prompt and RMSNorm agreement within 0.000008, plus correct recurrent session restoration. The remaining checks and app validation are recorded below. These establish that the vectors come from the actual native forward pass, not model-quality equivalence with the separate direct CUDA implementation.

The same bridge with capture disabled reached **42.89 / 43.09 tokens/s**, with first-token times **0.202 / 0.729 seconds** ([raw Chat measurements](prism-chat/llama.json)). Readout capture therefore cost about **6–7%** of decode throughput in these runs. All three 32-token greedy sequences matched exactly between capture enabled/disabled for each prompt length.

Additional real-model checks passed at 128/512 prompt tokens with exact full-logit parity, correct RMSNorm vectors and restored recurrent sessions. Native CPU vision identified a red image; final-layer readouts after that same image prompt were finite and selected `Red`. Text benchmarks ran before lazy-loading vision. Individual image-token grid/lens inspection remains a direct CUDA/MLX feature; the native UI disables that inspector. The web production build and nine model-independent agent checks passed.

The local app was then restarted with `BONSAI_BACKEND=llama`. Its live API/NPZ test passed: native `4`, then EOS, finite full logits and pre/post-norm vectors, with exact JSON/NPZ hidden-vector agreement. The app remains at `http://localhost:8000/#lens`, reporting `lens_readouts: true`. The old backend-name restriction in `/api/reset` was replaced with a readout-capability check.

## Protocol

`compare_backends.py` runs the engines sequentially, with identical 128- and 512-token prompt IDs, greedy sampling, fresh prompt caches, one warmup and 32 generated tokens per trial. Tables use medians of three trials. First-token time includes prompt processing; decode rate excludes it. Model loading is excluded. The direct CUDA timings include its readout path; native timings include the local server transport. This measures complete implementations rather than equivalent isolated kernels.

## Software improvements

Earlier Turbo comparison, before the native readout bridge, with the other desktop applications left running:

| Backend | Prompt tokens | First token (seconds) | Decode tokens/s | Direct layer readouts |
|---|---:|---:|---:|---|
| Direct CUDA | 128 | 3.626 | 1.63 | Yes |
| Direct CUDA | 512 | 22.881 | 1.87 | Yes |
| Prism llama.cpp | 128 | 0.762 | 26.04 | No |
| Prism llama.cpp | 512 | 1.081 | 27.65 | No |

[Raw Turbo results](turbo/comparison.md). Turbo removed the reported software thermal slowdown, but CUDA was slower than its earlier Silent run. During this run VRAM reached roughly 7.98 GB and free system RAM fell below 0.5 GB; memory pressure is a plausible cause, not an established diagnosis. These are not isolated-machine peak measurements. The native engine remains much faster for Chat/Code, while direct CUDA retains the requested real layer readouts.

Before tuning, direct CUDA took 28.40 / 110.03 seconds to produce the first token for the 128 / 512-token prompts, then decoded at 1.38 / 1.43 tokens/s. Native build 10743 decoded at 13.79 / 13.99 tokens/s. These baseline runs used two trials in the original Silent profile; see [raw baseline](baseline/comparison.md).

Changes retained after testing:

- Direct CUDA: packed ternary tensor-core prompt tiles, shared-memory padding, fused cached GDN convolution/recurrence/normalization, and single-token MLP CUDA graphs. Text projection weights now all fit in VRAM on this host; embeddings and vision stay on CPU. Weights remain packed, with only small shared-memory tiles unpacked for multiplication.
- Corrected cached multi-token GDN state handling so appended prompts reuse both convolution and recurrent state correctly.
- Native: upgraded to official Prism build 10754, increased microbatch to 512, retained CUDA graphs and full text GPU offload, and moved the vision encoder to CPU to preserve text VRAM.

The final sustained Silent runs achieved CUDA 2.43 / 2.21 tokens/s and native 22.22 / 24.02 tokens/s. First-token times were CUDA 4.01 / 14.18 seconds and native 0.64 / 1.58 seconds. NVIDIA reported software thermal slowdown around 70 C and a GPU clock near 795 MHz. A temporary ASUS Recommended experiment overlapped the user's switch to Turbo, so its mixed-profile results are excluded from the main comparison.

## Native parameter sweep

Separate `llama-bench` runs used a 512-token prompt, 64 generated tokens and two repetitions. They use a smaller text-only benchmark allocation, so these rates must not be substituted for the full application's numbers.

| Configuration | Prompt tokens/s | Decode tokens/s |
|---|---:|---:|
| PTQ1_0, microbatch 128, 4 threads | 335.95 | 28.21 |
| PTQ1_0, microbatch 128, 14 threads | 337.77 | 28.65 |
| PTQ1_0, microbatch 512, 4 threads | 361.55 | 29.16 |
| PTQ1_0, microbatch 512, 14 threads | 377.10 | 29.71 |
| PTQ1_0, Q8 KV, microbatch 512, 4 threads | 347.75 | 28.91 |
| PTQ1_0, CUDA graphs disabled | 349.77 | 18.94 |
| PQ2_0, Q8 KV, microbatch 512, 4 threads | 313.31 | 21.09 |

Thus PTQ1_0, microbatch 512 and graphs remain the defaults. Larger PQ2_0 packing and Q8 KV did not improve this host's tested workload. [Raw sweep](native-sweep/summary.json).

A final Turbo CUDA headroom test increased `BONSAI_CUDA_RESERVE_GB` from 0.45 to 0.7, leaving 6.195 GiB of packed weights on GPU and streaming nine projections. Two-trial medians were 1.81 / 1.69 tokens/s and 2.78 / 11.91 seconds to first token. Prompt latency improved, but decode did not improve consistently, so 0.45 remains the default. This experiment does not establish that memory pressure caused the earlier regression. [Raw headroom test](turbo-headroom/comparison.md).

## Remaining opportunities, ranked

1. **Cooling and operating mode.** The measured software thermal slowdown makes the laptop's power/fan profile consequential. The user enabled Turbo for the final comparison. ASUS documents its [operating modes](https://www.asus.com/support/faq/1043747/); no overclock or manual power-limit change was applied.
2. **Prism's pending small-batch PTQ kernel.** [PR 218](https://github.com/PrismML-Eng/llama.cpp/pull/218) reports about a 1.5x decode improvement on an RTX 3060 with a new Q8 activation layout and DP4A path. It was open and unmerged when researched. It is not in the pinned official build and has not been compiled or measured here. A source build and controlled correctness/performance comparison are the most promising next native experiment.
3. **Direct CUDA matrix-vector multiplication.** Profiling places roughly 88% of GPU time in packed ternary matrix-vector kernels. More graph capture alone cannot remove that cost. An integer dot-product path could address it, but changes activation precision and requires numerical and model-quality validation. A tested alternative that grouped ternary sums regressed full-model decode and was removed.
4. **Avoid unsuitable cache and speculative settings.** Prism's [known issues](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/main/KNOWN_ISSUES.md) warn about Q5 KV performance, ignored n-gram flags and MTP compatibility requirements. F16 KV and one native slot remain the tested defaults; speculative speedups are not assumed.

The implemented tensor-core/padding and graph changes follow [NVIDIA's CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) and [PyTorch's CUDA graph guidance](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/). Packing and runtime compatibility were checked against the [original MLX documentation](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit), [official GGUF documentation](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) and [pinned Prism release](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10754-2459f68).

## Validation and boundaries

Packed multiplication, tensor-core tails, Hadamard transforms, cached GDN state, and CUDA graph replay checks passed. Both engines answered the arithmetic check with `4`; native CPU vision identified a red test image. Kernel/state comparisons and arithmetic are limited correctness checks, not a broad quality evaluation. Direct CUDA exposes real pre/post-norm vectors and logits; the native server does not expose these taps. macOS MLX is preserved but cannot be executed on this Windows machine. Long-context memory limits were not established by these 128/512-token benchmarks.

After benchmarking, the local app was restarted with direct CUDA. `tests/check_readout_api.py` passed against the running app: greedy `4`, then EOS, finite 248,320-element logits and 5,120-element pre/post-norm vectors, with exact hidden-vector agreement between API JSON and the downloadable NPZ. The app remains at `http://localhost:8000/#lens` with real readouts enabled.

Reproduce with:

```powershell
uv run compare_backends.py --contexts 128 512 --decode 32 --reps 3 --profile --output-dir benchmarks/windows-cuda/recheck
uv run tune_llama.py
uv run bench_cuda_kernels.py
```
