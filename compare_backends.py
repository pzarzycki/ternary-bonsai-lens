"""Compare ternary CUDA and official llama.cpp CUDA, sequentially on one GPU.

uv run compare_backends.py --backends cuda llama --contexts 128 512 --decode 32 --reps 2
"""

from __future__ import annotations
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "benchmarks" / "windows-cuda"


def readout_generate(engine, session):
    while True:
        readout = engine.forward(session)
        token = engine.sample(readout.logits, 0)
        session.tokens.append(token)
        yield token
        if token in engine.eos_ids:
            return


def one_backend(args):
    from bonsai_lens.engine import Engine

    start = time.perf_counter()
    e = Engine(backend=args.backend)
    load = time.perf_counter() - start
    if args.backend == "llama":
        e.cache_prompt = False
    # Correctness and warmup use the same ternary model and greedy settings.
    warm = e.new_session(
        e.render(
            [
                {
                    "role": "user",
                    "content": "What is 2 plus 2? Answer with the number only.",
                }
            ]
        )
    )
    generated = []
    iterator = e.generate(warm, temperature=0)
    try:
        for token in iterator:
            if token in e.eos_ids:
                break
            generated.append(token)
            if len(generated) >= 8:
                break
    finally:
        iterator.close()
    answer = e.decode(generated).strip()
    if answer != "4":
        raise AssertionError(f"{args.backend}: arithmetic check failed: {answer!r}")
    # Warmup must not retain a second recurrent cache on an 8 GB GPU.
    del warm
    cases = []
    for n in args.contexts:
        template = e.render([{"role": "user", "content": "MARKER"}])
        before, after = template.split("MARKER")
        head = e.tokenizer.encode(before, add_special_tokens=False)
        tail = e.tokenizer.encode(
            "\nExplain the design of a robot controller in detail, including sensors, feedback, safety and testing."
            + after,
            add_special_tokens=False,
        )
        filler = e.tokenizer.encode(
            "A robot has motors and sensors. ", add_special_tokens=False
        )
        body = n - len(head) - len(tail)
        if body < 0:
            raise ValueError("context too short")
        ids = head + (filler * (body // len(filler) + 1))[:body] + tail
        trials = []
        for rep in range(args.reps):
            session = e.new_session("")
            session.tokens = list(ids)
            if args.backend == "llama":
                e.max_generate = args.decode
            t0 = time.perf_counter()
            stamps = []
            tokens = []
            iterator = (
                readout_generate(e, session)
                if args.readouts
                else e.generate(session, temperature=0, top_p=1, top_k=0, min_p=0)
            )
            try:
                for token in iterator:
                    if token in e.eos_ids:
                        break
                    stamps.append(time.perf_counter())
                    tokens.append(token)
                    if len(tokens) >= args.decode:
                        break
            finally:
                iterator.close()
            if len(tokens) < 2:
                raise AssertionError("benchmark reply too short")
            intervals = [b - a for a, b in zip(stamps, stamps[1:])]
            row = {
                "ttft_s": stamps[0] - t0,
                "decode_tok_s": 1 / statistics.mean(intervals),
                "tokens": tokens,
                "text": e.decode(tokens),
                "output_tokens": len(tokens),
            }
            trials.append(row)
            print(
                args.backend,
                "context",
                n,
                "rep",
                rep + 1,
                "TTFT",
                round(row["ttft_s"], 3),
                "s,",
                round(row["decode_tok_s"], 2),
                "tok/s",
                flush=True,
            )
        cases.append(
            {
                "prompt_tokens": n,
                "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "ttft_s": statistics.median(t["ttft_s"] for t in trials),
                "decode_tok_s": statistics.median(t["decode_tok_s"] for t in trials),
                "trials": trials,
            }
        )
    result = {
        "backend": args.backend,
        "model": "Ternary-Bonsai-2-27B",
        "platform": platform.platform(),
        "runtime": e.runtime_info,
        "load_s": load,
        "validation_answer": answer,
        "greedy": True,
        "prompt_cache": False,
        "requested_decode_tokens": args.decode,
        "repetitions": args.reps,
        "readouts_each_token": args.readouts or args.backend == "cuda",
        "cases": cases,
    }
    if platform.system() == "Windows":
        result["power_plan"] = subprocess.check_output(
            ["powercfg", "/getactivescheme"], text=True
        ).strip()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / (args.backend + ".json")).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if args.profile and args.backend == "cuda":
        import torch

        session.tokens.append(tokens[-1]) if session.fed == len(
            session.tokens
        ) else None
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            e.forward(session)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(output / "cuda-profile.json"))
        (output / "cuda-profile.txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=35),
            encoding="utf-8",
        )
    if hasattr(e, "close"):
        e.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cuda", "llama"])
    ap.add_argument(
        "--backends", nargs="+", default=["cuda", "llama"], choices=["cuda", "llama"]
    )
    ap.add_argument("--contexts", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--output-dir", default=str(OUT))
    ap.add_argument("--profile", action="store_true")
    ap.add_argument(
        "--readouts",
        action="store_true",
        help="Capture real layer vectors and full logits on every token",
    )
    args = ap.parse_args()
    if args.backend:
        return one_backend(args)
    for backend in args.backends:
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--backend",
                backend,
                "--contexts",
                *[str(x) for x in args.contexts],
                "--decode",
                str(args.decode),
                "--reps",
                str(args.reps),
                "--output-dir",
                args.output_dir,
                *(["--profile"] if args.profile else []),
                *(["--readouts"] if args.readouts else []),
            ],
            check=True,
            cwd=ROOT,
        )
    results = [
        json.loads((Path(args.output_dir) / (b + ".json")).read_text(encoding="utf-8"))
        for b in args.backends
    ]
    lines = [
        "# Windows CUDA comparison",
        "",
        "Same ternary model, greedy sampling, identical prompt token IDs, fresh prompt caches. Backends run sequentially; one warmup precedes measurements. CUDA uses the MLX 2-bit container; llama.cpp uses native PTQ1_0. This compares the complete implementations, not just kernels.",
        "",
        "| Backend | Prompt tokens | First token (s) | Decode (tokens/s) |",
        "|---|---:|---:|---:|",
    ]
    for result in results:
        for case in result["cases"]:
            lines.append(
                f"| {result['backend']} | {case['prompt_tokens']} | {case['ttft_s']:.3f} | {case['decode_tok_s']:.2f} |"
            )
    for cases in zip(*(r["cases"] for r in results)):
        assert len({c["prompt_sha256"] for c in cases}) == 1, (
            "Prompts differ across backends"
        )
    lines += [
        "",
        "Arithmetic checks returned `4`. Floating-point differences across kernels may change generated sequences; raw tokens and timings are in the adjacent JSON files.",
        "",
        "macOS MLX is preserved but was not executed on this Windows host.",
    ]
    (Path(args.output_dir) / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
