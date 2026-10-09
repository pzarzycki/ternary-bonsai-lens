# Bonsai Lens

A local runtime for **Ternary Bonsai 2 27B** with Apple Silicon MLX and two Windows/NVIDIA implementations. Both Windows backends run the actual ternary weights; neither substitutes a dense Qwen checkpoint.

| Backend | Weights | Chat / Code / images | Layer readouts |
|---|---|---|---|
| `mlx` | Original MLX 2-bit pack | Yes | Yes |
| `cuda` | Same MLX pack, direct packed CUDA kernels | Yes | Yes |
| `llama` | Official PTQ1_0 GGUF, Prism llama.cpp CUDA | Yes | Yes, with the native readout bridge |

`auto` selects MLX on Apple Silicon and fast Prism CUDA on Windows when the readout bridge is built, otherwise direct CUDA. Both Windows implementations remain available explicitly. Use one model process at a time on an 8 GB GPU.

## Measured Windows performance

On an RTX 3070 Ti Laptop GPU (8 GB), i9-12900H and 16 GB RAM, with ASUS Turbo enabled:

| Implementation | Decode, 128-token prompt | Decode, 512-token prompt |
|---|---:|---:|
| **Prism CUDA with full readouts every token** | **40.05 tok/s** | **40.55 tok/s** |
| Prism CUDA Chat, hidden capture disabled | 42.89 tok/s | 43.09 tok/s |
| Direct packed CUDA with readouts | 1.63 tok/s | 1.87 tok/s |

Native readout first-token times were **0.229 s / 0.822 s** for the two prompt lengths. Measurements use identical prompt IDs, greedy sampling, fresh caches, 32 generated tokens and medians of three trials. Loading is excluded; readout timings include copies of full logits and both hidden vectors into Python, but exclude web JSON serialization and rendering. Text runs preceded lazy-loading vision. The direct CUDA measurements were earlier runs on the same host; memory pressure and power/thermal conditions affect results.

Native capture preserved every tested logit exactly and cost about 6–7% of decode throughput. See [the performance report](benchmarks/windows-cuda/performance.md) for raw results, validation and optimization research. macOS MLX is preserved; it was not benchmarked on this Windows host.

## Windows + NVIDIA setup

Install `uv`, Git and Node.js/npm, plus an NVIDIA driver. Native Lens readouts also require Visual Studio Build Tools with **Desktop development with C++** to build the small bridge. The native path uses bundled CUDA runtime DLLs; it does not require compiling CUDA kernels or installing the CUDA toolkit. Direct packed CUDA additionally requires CUDA 12.x toolkit/NVRTC; CUDA 12.8 is preferred when installed.

Recommended fast native setup:

```powershell
.\setup-windows.ps1 -Backend llama
$env:BONSAI_BACKEND = 'llama'
uv run server.py                  # http://localhost:8000/#lens
```

Use `.\setup-windows.ps1 -Backend both` to install both Windows implementations, or `-Backend cuda` for only the direct packed path. The script creates the local Python environment, pins Prism's `prism-b10754-2459f68` CUDA build, upgrades an older installed binary, downloads PTQ1_0 and the Q8 vision projector, verifies new GGUF downloads against Hugging Face SHA-256 metadata, and builds the UI. Interrupted downloads resume when rerun. Native-only setup downloads the tokenizer/config files but does not need the full MLX weights.

Native readouts use a small C++ bridge linked to the **existing official Prism DLLs**. Setup builds it when Visual Studio C++ Build Tools are installed. To build it separately, run `.\native\build-prism-readouts.ps1`. The pinned headers and DLL version must match. Without the bridge, explicit `llama` retains server-based Chat/Code without Lens; `-SkipReadouts` selects that setup. `auto` falls back to direct CUDA when the bridge is absent, which requires the full MLX pack.

The bridge exports the actual layer-63 residual, the final RMSNorm output and full logits from Prism's graph. Only the last output position is copied; Chat/Code disable hidden-vector capture. Session changes save/restore the native recurrent/KV state in RAM. Native final-layer readouts also work after an image prompt; the detailed image-token inspector remains available through direct CUDA/MLX.

Select a backend explicitly:

```powershell
$env:BONSAI_BACKEND = 'llama'       # optimized official CUDA runtime
uv run chat.py
# Stop the running model before launching a different backend.
$env:BONSAI_BACKEND = 'cuda'        # direct packed CUDA with Lens readouts
uv run server.py
```

### Runtime settings

| Native setting | Default | Purpose |
|---|---|---|
| `BONSAI_CTX` | `8192` | Total native prompt + reply context |
| `BONSAI_NGL` | `99` | Offload all text layers to GPU |
| `BONSAI_BATCH` / `BONSAI_UBATCH` | `512` / `512` | Prompt batch sizes |
| `BONSAI_THREADS` | `14` in the bridge | CPU threads |
| `BONSAI_KV_TYPE` | `f16` | Bridge also accepts `q8_0` or `q4_0` |
| `BONSAI_VISION_GPU` | `0` | Lazy-load vision on CPU; `1` offloads it |

`BONSAI_GGUF` overrides the native model; `BONSAI_READOUT_DLL` overrides the bridge. `BONSAI_LLAMA_API=server`, `BONSAI_LLAMA_BIN` or `BONSAI_LLAMA_URL` selects the original server path without taps. Its private endpoint defaults to `127.0.0.1:8081`; only a server started by this app is shut down with it. `BONSAI_PACK` overrides the MLX pack/tokenizer directory for either implementation. Docker Desktop is required only for the Code sandbox.

Direct CUDA verifies the three ternary codes and redundant affine bias, applies the specified Hadamard transforms, and keeps language matrices packed. It uses tensor-core prompt tiles, fused cached GDN kernels and single-token MLP graphs. Embeddings/vision stay in RAM; projections stream from RAM if VRAM is insufficient. `BONSAI_CUDA_RESERVE_GB=0.45` reserves working memory, `BONSAI_PREFILL_CHUNK=256` bounds prompt chunks, and `BONSAI_CUDA_MLP_GRAPHS=0` disables graph replay. Use this implementation for the detailed image-token inspector or direct-kernel development.

The native Code agent compacts earlier to fit the configured context and reserves 2k tokens for a response. The Chat reply selector respects the native context size; prompt and reply share that context.

### Validation and benchmarks

Stop the model server before running checks that load their own model:

```powershell
uv run selfcheck.py                        # greedy arithmetic; MLX/direct CUDA also check taps
uv run python -m tests.check_prism_readouts # native logit parity, RMSNorm and session restore
uv run python tests/check_llama_vision.py  # native image generation and final-position readouts
```

The native RMSNorm check reads the original MLX norm weights, so `tests.check_prism_readouts` requires the full pack from `-Backend both`. To validate the **already-running** app and NPZ download, use `uv run python tests/check_readout_api.py`; this check does not load a second model.

Compare implementations sequentially with identical prompts and greedy sampling; comparing both requires the `-Backend both` setup. Use a new output directory to retain the recorded measurements:

```powershell
uv run compare_backends.py --contexts 128 512 --decode 32 --reps 3 --profile --output-dir benchmarks/windows-cuda/recheck
uv run compare_backends.py --backends llama --readouts --contexts 128 512 --decode 32 --reps 3 --output-dir benchmarks/windows-cuda/readouts-recheck
```

Results include generated token IDs, prompt hashes and timings. Without `--readouts`, native generation disables hidden capture; direct CUDA still computes readouts. The model containers have different packing overhead, so this compares complete implementations. Arithmetic, kernel and tap checks establish limited numerical correctness, not broad model-quality equivalence or long-context memory limits.

Sources: [original MLX model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit), [official GGUF model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), and [Prism's supported runtime setup](https://github.com/PrismML-Eng/Bonsai-demo).

## macOS setup (Apple Silicon)

```bash
uv sync
uvx --from huggingface_hub hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --local-dir models/bonsai2-27b-mlx
(cd web && npm install && npm run build)
uv run selfcheck.py      # verifies taps match the stock logits path; greedy "2+2" -> 4
```

## 1. Terminal chat

```bash
uv run chat.py
```

Supports multi-turn chat with KV/SSM cache reuse. Commands: `/image <path>`, `/think`, `/temp <t>`, `/system <text>`, `/reset`, `/quit`.

## 2. Single-step readout web app

```bash
uv run server.py          # http://localhost:8000 (serves web/dist)
# for UI development with hot reload:
cd web && npm run dev     # http://localhost:5173, proxies /api to :8000
```

Each **Step** feeds the prompt or the previously chosen token, then shows the following for the last position. Long prefills may be split into batches; only one next token is selected per Step.

| Readout | What it is | Shape |
|---|---|---|
| **After unembedding** | `logits = lm_head(h)`: top-50 with probs, entropy, histogram of all logits | 248,320 |
| **Before unembedding: post-norm** | `h = final_RMSNorm(residual)`, the exact input of `lm_head` | 5,120 |
| **Before unembedding: pre-norm** | residual stream after block 64, before the final norm | 5,120 |

- Clicking a top-k row replaces the token chosen at the latest step. This is free, because a token is only fed to the model on the next step.
- Clicking a generated token shows that step's readouts again.
- `GET /api/last.npz` returns the full float32 `logits`, `pre_norm` and `post_norm` arrays for the latest step.

### Image tokens

All readout-capable backends support final-position readouts after an image prompt. **Individual image-token inspection requires direct CUDA or MLX**; native Prism displays an image chip with that inspector disabled.

With direct CUDA/MLX, each `<|vision_start|> <|image_pad|>×N <|vision_end|>` span is collapsed into one 🖼 chip. Clicking it opens an inspector that shows:

- the image as the model saw it, with its token grid overlaid
- per token:
  - sequence position, grid cell and pixel box
  - M-RoPE (t, h, w) position
  - norm of its vision embedding
  - an *input lens*: `lm_head(norm(vision embedding))`
  - the *final-layer* readout at that position: `lm_head(h)`, captured during the prefill step
- overlays: embedding norm, lens entropy, final-layer top-1 probability

How images are encoded:

1. The image is resized to multiples of 32 px and cut into 16×16 patches of 1536 values each (3 channels × 2 frames × 16²).
2. A 27-block ViT processes the patches. Its merger joins each 2×2 patch block and projects it to 5120 dims, giving one token per 32×32 px.
3. Those vectors replace the embeddings at the `<|image_pad|>` slots. The placeholder id carries no information.
4. M-RoPE gives the image tokens 2D positions, so the text after an image resumes from `max(h, w) + 1`.

Endpoints: `GET /api/image/{n}/info` and `GET /api/image/{n}.png`.

## 3. Chat tab (same web app)

ChatGPT-style chat at `http://localhost:8000/#chat`:

- replies stream token by token over SSE (`POST /api/chat/send`) and render as markdown, with GFM tables and highlighted code
- thinking mode shows the reasoning in a collapsible block, with a reasoning-effort selector
- Stop button, image attachments, and a system prompt (applies to a new chat)
- **max reply** caps each reply's output (thinking + answer). MLX/direct CUDA initially use 32k; the default 8k native context selects a 4k reply cap. Prompt and reply share the native context, so long prompts leave less room. A reply cut off by the cap or by Stop shows **Continue generating**, which resumes the cached session (`POST /api/chat/continue`) while context remains available.
- conversation logic lives in `bonsai_lens/conversation.py` and is shared with `chat.py`
- while a chat reply is streaming, Lens steps are refused with a 409, since both tabs share the one model

### Basis caveat

The pack stores weights with a blockwise Hadamard rotation folded in:

- Every `Packed` linear rotates its own input. That includes `lm_head`.
- The embedding applies the inverse rotation.

As a result, the residual stream (and both hidden readouts) is in the model's **natural, unrotated** basis. The output-head rotation is applied inside `lm_head`. Prism's native captures expose these same stages of its own forward pass; floating-point and activation-quantization differences across implementations can change logits.

## 4. Code tab: a coding agent in a Docker sandbox

The third tab is an agentic coding chat docked on the right. The left side shows the agent's workspace:

| Pane | What it shows |
|---|---|
| **Preview** | The app on `http://localhost:5180`, with Vite hot reload |
| **Files** | The file tree and a highlighted viewer. A `write_file` call streams into it live as the model types it |
| **Plan** | The agent's checklist |
| **Terminal** | Every command and its output |
| **Raw context** | The agent's whole context exactly as the model sees it |

The chat lists every tool call as an expandable card, with its arguments, the result and the raw XML the model generated.

**Requirements and lifecycle**

- Needs Docker Desktop running.
- On first use the server builds `sandbox/Dockerfile` (`bonsai-code-sandbox`: Node 22, npm, Python + uv, git, ripgrep, and a ready Vite React-TS app at `/opt/template` with `node_modules`). This takes a few minutes. After that, opening the tab starts a container `bonsai-code`.
- Files live only inside the container (`/workspace`). **Reset** recreates the container and clears the conversation. ⤓ downloads the workspace as a `.tar.gz`.
- The container is removed when the server stops or restarts. The port is published on 127.0.0.1 only, and nothing from the host is mounted.

**How it works** ([agent package](bonsai_lens/agent/), [sandbox](bonsai_lens/sandbox.py))

- **Tool calls:** the model calls tools in its native Qwen3-coder XML format. The tools are `read_file`, `write_file`, `edit`, `run_shell_command`, `read_process`, `kill_process`, `glob`, `grep_search` and `update_plan`. Sandbox commands run through `docker exec`; long commands return process IDs that the agent can inspect or stop.
- **KV cache:** it is reused across steps. Each step appends only the tool results. `tests/test_agent.py` checks that this hand-built continuation matches the chat template byte for byte.
- **Server-side runs:** the run doesn't depend on the browser. Events go into a log (`/api/code/events`) that the UI replays and follows, so a reload or a dropped connection loses nothing.
- **Checks and settings:** environment and project `.bonsai/settings.json` files configure `SessionStart`, `PostToolUse` and `Stop` hooks, preview startup and limits. The starter React project enables import/type checks and starts its dev server when `package.json` exists. Environment hooks check Python and preview pages. Declared success criteria are verified before finishing.
- **Context:** older turns are replaced by an action summary while preserving user requests and recent work, then the compacted prompt is re-prefilled. MLX/direct CUDA have a 56k agent context ceiling and compact around 32k. Native context is bounded by `BONSAI_CTX`, with 2k reserved for a response; at the default 8192, the agent context ceiling is 6144 and compaction starts around 4608 tokens.
- **Limits:** configurable through settings. Built-in defaults are an 800-token thinking budget (doubled on the first step or after failure), no fixed step cap, and a 1440-minute time cap. Project/environment settings can override them.

See [coding-agent architecture](docs/coding-agent-architecture.md) and [research](docs/coding-agent-research.md) for the harness design.

Tests that need no model or Docker: `uv run python tests/test_agent.py`.

## Layout

- `bonsai_lens/engine.py`: backend selection (`auto`, `mlx`, `cuda`, `llama`)
- `bonsai_lens/prism_engine.py`: fast native Prism sessions, final-layer readouts and sampling
- `native/prism_readouts.cpp` and `native/build-prism-readouts.ps1`: small DLL bridge and build against pinned Prism headers/binaries
- `bonsai_lens/llama_engine.py`: native bridge selection and original HTTP server fallback
- `bonsai_lens/cuda_engine.py`, `cuda_kernels.py`, `cuda_gdn.py`, `cuda_recurrent.py`, `cuda_graphs.py`: direct packed CUDA loader and kernels
- `bonsai_lens/mlx_engine.py`: original Apple Silicon model loader and norm taps
- `bonsai_lens/common.py`: shared prompting, sessions and readout serialization
- `bonsai_lens/conversation.py`: multi-turn state + token streaming, shared by the TUI and the Chat tab
- `bonsai_lens/agent/`: coding-agent loop, tools, checks, settings and context compaction
- `bonsai_lens/sandbox.py`: Docker sandbox driven through the docker CLI; `sandbox/Dockerfile` defines its image
- `chat.py`: TUI
- `server.py`: FastAPI and a single inference worker shared by Lens, Chat and Code
- `web/`: React + Vite UI
- `compare_backends.py`: sequential real-model performance comparison, optionally with native readouts
- `benchmarks/windows-cuda/`: measurements, raw token IDs and performance research
