#!/usr/bin/env python3
"""Generation latency benchmark across context sizes.

    uv run bench.py --label baseline                    # default suite
    uv run bench.py --label fastpath --fastpath         # with bonsai_lens.fastpath applied
    uv run bench.py --contexts 128 4096 --decode 10     # custom

Per context size it measures:
  TTFT          tokenize + prefill + sample the first token (what a user waits for after Enter)
  inter-token   latency of each of the next N tokens (one forward + sample), min / avg / max
  prefill tok/s and peak Metal memory

Appends one JSON record per run to benchmarks/results.jsonl and writes benchmarks/<label>.md.
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path


from bonsai_lens.engine import Engine

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "benchmarks"
FILLER_SRC = ROOT / "bonsai_lens" / "common.py"
QUESTION = "\n\nSummarize what the code above does in a few sentences."


def chip() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor()


def build_prompt(e: Engine, n: int, filler_ids: list[int]) -> list[int]:
    """Chat-templated prompt of exactly n tokens: filler document, then a question."""
    head = e.tokenizer.encode(e.render([{"role": "user", "content": "X"}]).split("X")[0], add_special_tokens=False)
    tail = e.tokenizer.encode(QUESTION + e.render([{"role": "user", "content": "X"}]).split("X")[1], add_special_tokens=False)
    body = n - len(head) - len(tail)
    if body < 0:
        raise ValueError(f"context {n} too small for template")
    reps = filler_ids * (body // len(filler_ids) + 1)
    return head + reps[:body] + tail


def run_once(e: Engine, ids: list[int] | None, decode: int, image: str | None = None, text: str | None = None,
             sync: bool = False):
    if getattr(e, "backend", "mlx") == "cuda":
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        import mlx.core as mx
        mx.clear_cache()
        mx.reset_peak_memory()
    t0 = time.perf_counter()
    s = e.new_session(text, images=[image]) if image else e.new_session("")
    if not image:
        s.tokens = list(ids)
    n_prompt = len(s.tokens)
    stamps = []
    if sync:  # step-by-step: forward, read the token back, repeat (the original engine loop)
        r = e.forward(s)
        tok = e.sample(r.logits, 0)
        stamps.append(time.perf_counter())
        for _ in range(decode):
            s.tokens.append(tok)
            r = e.forward(s)
            tok = e.sample(r.logits, 0)
            stamps.append(time.perf_counter())
    else:  # pipelined generator used by Chat/TUI
        for tok in e.generate(s, temperature=0.0):
            stamps.append(time.perf_counter())
            if len(stamps) > decode:
                break
    ttft = stamps[0] - t0
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return {
        "prompt_tokens": n_prompt,
        "ttft_s": ttft,
        "prefill_tok_s": n_prompt / ttft,
        "itl_ms": [g * 1000 for g in gaps],
        "peak_gb": (torch.cuda.max_memory_allocated() if getattr(e, "backend", "mlx") == "cuda" else mx.get_peak_memory()) / 2**30,
    }


def summarize(runs: list[dict]) -> dict:
    ttfts = [r["ttft_s"] for r in runs]
    itl = [x for r in runs for x in r["itl_ms"]]
    first = [r["itl_ms"][0] for r in runs]  # first decode step right after prefill
    steady = [x for r in runs for x in r["itl_ms"][1:]]
    return {
        "prompt_tokens": runs[0]["prompt_tokens"],
        "reps": len(runs),
        "ttft_s": statistics.median(ttfts),
        "ttft_min_s": min(ttfts),
        "ttft_max_s": max(ttfts),
        "prefill_tok_s": runs[0]["prompt_tokens"] / statistics.median(ttfts),
        "itl_min_ms": min(itl),
        "itl_avg_ms": statistics.mean(itl),
        "itl_max_ms": max(itl),
        "decode_tok_s": 1000 / statistics.mean(itl),
        "first_decode_ms": statistics.median(first),
        "steady_min_ms": min(steady),
        "steady_avg_ms": statistics.mean(steady),
        "steady_max_ms": max(steady),
        "steady_tok_s": 1000 / statistics.mean(steady),
        "peak_gb": max(r["peak_gb"] for r in runs),
        "decode_tokens": len(runs[0]["itl_ms"]),
        "itl_ms_raw": [[round(x, 2) for x in r["itl_ms"]] for r in runs],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--backend", choices=("mlx", "cuda"), default="mlx" if platform.system() == "Darwin" else "cuda")
    ap.add_argument("--contexts", type=int, nargs="+", default=[128, 1024, 4096, 16384, 32768, 65536])
    ap.add_argument("--decode", type=int, default=10, help="tokens to time after the first one")
    ap.add_argument("--reps", type=int, default=3, help="repetitions for contexts <= 16k (longer: 1)")
    ap.add_argument("--image", default=str(ROOT / "benchmarks" / "bench_image.png"))
    ap.add_argument("--fastpath", action="store_true")
    ap.add_argument("--sync", action="store_true", help="time the old step-by-step loop instead of the pipelined generator")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    t = time.perf_counter()
    e = Engine(backend=args.backend)
    load_s = time.perf_counter() - t
    if args.fastpath:
        if args.backend != "mlx": ap.error("--fastpath is MLX-only; CUDA already uses packed kernels")
        from bonsai_lens import fastpath

        fastpath.apply(e)
    filler = e.tokenizer.encode(FILLER_SRC.read_text(encoding="utf-8"), add_special_tokens=False)

    print("warm-up…", flush=True)
    run_once(e, build_prompt(e, 512, filler), 4, sync=args.sync)

    results = []
    for n in args.contexts:
        reps = args.reps if n <= 16384 else 1
        ids = build_prompt(e, n, filler)
        runs = [run_once(e, ids, args.decode, sync=args.sync) for _ in range(reps)]
        row = {"case": f"text {n}", **summarize(runs)}
        results.append(row)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}), flush=True)

    if args.image:
        img = Path(args.image)
        if not img.is_file():
            make_bench_image(img)
        from PIL import Image

        w, h = Image.open(img).size
        text = e.render([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}])
        runs = [run_once(e, None, args.decode, image=str(img), text=text, sync=args.sync) for _ in range(args.reps)]
        row = {"case": f"image {w}x{h}", **summarize(runs)}
        results.append(row)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}), flush=True)

    if args.backend == "cuda":
        import torch
        device_name = torch.cuda.get_device_name()
        device_ram = round(torch.cuda.get_device_properties(0).total_memory / 2**30)
    else:
        import mlx.core as mx
        device_name, device_ram = chip(), round(mx.device_info()["memory_size"] / 2**30)
    def installed(name):
        from importlib.metadata import PackageNotFoundError
        try: return version(name)
        except PackageNotFoundError: return "not installed"
    meta = {
        "label": args.label,
        "date": datetime.now().isoformat(timespec="seconds"),
        "chip": device_name,
        "backend": args.backend,
        "ram_gb": device_ram,
        "mlx": installed("mlx"),
        "torch": installed("torch"),
        "mlx_vlm": installed("mlx-vlm"),
        "fastpath": args.fastpath,
        "decode_loop": "sync" if args.sync else "pipelined",
        "model_load_s": load_s,
        "sampling": "greedy",
        "prefill_chunking": "budgeted (4 GB attention scores per chunk)",
    }
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps({**meta, "results": results}) + "\n")
    (OUT / f"{args.label}.md").write_text(render_md(meta, results), encoding="utf-8")
    print(f"\nwrote {OUT / (args.label + '.md')}")


def make_bench_image(path: Path):
    """Deterministic 1024x768 synthetic scene (≈768 image tokens)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1024, 768), "#87ceeb")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 480, 1024, 768), fill="#3a9d3a")
    d.ellipse((800, 60, 960, 220), fill="#ffd700")
    d.rectangle((150, 300, 450, 520), fill="#b22222")
    d.polygon([(130, 300), (300, 170), (470, 300)], fill="#5a2d0c")
    d.text((560, 600), "BENCHMARK", fill="black")
    img.save(path)


def render_md(meta: dict, rows: list[dict]) -> str:
    lines = [
        f"# Generation performance: `{meta['label']}`",
        "",
        f"- Date: {meta['date']}",
        f"- Machine: {meta['chip']}, {meta['ram_gb']} GB",
        f"- Backend: {meta.get('backend', 'mlx')}; Software: mlx {meta['mlx']}, mlx-vlm {meta['mlx_vlm']}, fastpath={'on' if meta['fastpath'] else 'off'}, decode loop: {meta.get('decode_loop', 'sync')}",
        "- Model: prism-ml/Ternary-Bonsai-2-27B-mlx-2bit (Hadamard-packed 2-bit, vision tower loaded)",
        f"- Model load: {meta['model_load_s']:.1f} s; sampling: {meta['sampling']}; prefill: {meta['prefill_chunking']}",
        "",
        "## Results",
        "",
        "| case | prompt tok | TTFT (median) | TTFT range | prefill tok/s | 1st decode token | next-token min | avg | max | steady tok/s | all 10: avg | peak mem |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['prompt_tokens']:,} | {fmt_s(r['ttft_s'])} | {fmt_s(r['ttft_min_s'])}–{fmt_s(r['ttft_max_s'])} "
            f"| {r['prefill_tok_s']:.0f} | {fmt_ms(r['first_decode_ms'])} | {r['steady_min_ms']:.1f} ms | {r['steady_avg_ms']:.1f} ms "
            f"| {r['steady_max_ms']:.1f} ms | {r['steady_tok_s']:.1f} | {fmt_ms(r['itl_avg_ms'])} ({r['decode_tok_s']:.1f} tok/s) | {r['peak_gb']:.1f} GB |"
        )
    lines += [
        "",
        "## Method",
        "",
        "- **TTFT** is measured from the start of tokenization to the first sampled token: tokenize, then a chunked prefill forward pass, then greedy sampling.",
        f"- **Next-token latency** is the time for each of the following {rows[0]['decode_tokens']} tokens: one forward pass plus sampling, with the real generated token fed back.",
        "  - **1st decode token** is the first of those, measured right after prefill (median over repetitions). It is reported on its own because it can stall; see the findings below.",
        f"  - **Next-token min/avg/max** and **steady tok/s** cover the remaining {rows[0]['decode_tokens'] - 1} tokens across all repetitions.",
        f"  - **All 10: avg** averages over all {rows[0]['decode_tokens']} tokens, including the first.",
        "- **Prompts** are built by `bench.py`:",
        "  - Text prompts are chat-templated, with thinking off. Their body is filler source code (mlx-vlm `language.py`, repeated), trimmed so the prompt has exactly *N* tokens, and ends with a question.",
        "  - The image case is a synthetic 1024×768 scene with a one-line prompt.",
        "- **Repetitions and warm-up:** cases up to 16k run 3 times (the median TTFT is reported); 32k and 65k run once. A warm-up run comes first, so Metal kernel compilation is excluded.",
        "- **Isolation:** nothing else used the GPU during the run (the web server was stopped).",
        "",
        "Raw records: `benchmarks/results.jsonl`.",
        "",
    ]
    return "\n".join(lines)


def fmt_ms(ms: float) -> str:
    return f"{ms:.1f} ms" if ms < 1000 else f"{ms / 1000:.2f} s"


def fmt_s(s: float) -> str:
    return f"{s * 1000:.0f} ms" if s < 1 else f"{s:.2f} s" if s < 60 else f"{s / 60:.1f} min"


if __name__ == "__main__":
    main()
