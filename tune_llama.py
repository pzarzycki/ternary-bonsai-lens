"""Isolated native sweep: packing, CPU threads, microbatch, KV types and graphs.

Run after stopping all model processes: uv run tune_llama.py
"""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "benchmarks" / "windows-cuda" / "native-sweep"
OUT.mkdir(parents=True, exist_ok=True)
BIN = ROOT / "runtime" / "llama-cuda" / "llama-bench.exe"
MODEL = ROOT / "models" / "bonsai2-27b-gguf"
env = os.environ.copy()
env["PATH"] = str(BIN.parent) + os.pathsep + env.get("PATH", "")
base = [
    str(BIN),
    "-ngl",
    "99",
    "-fa",
    "on",
    "-b",
    "512",
    "-p",
    "512",
    "-n",
    "64",
    "-r",
    "2",
    "-o",
    "json",
]
tests = [
    ("ptq-threads-batch", "PTQ1_0", ["-ub", "128,512", "-t", "4,14"], {}),
    ("ptq-q8", "PTQ1_0", ["-ub", "512", "-t", "4", "-ctk", "q8_0", "-ctv", "q8_0"], {}),
    (
        "ptq-no-graphs",
        "PTQ1_0",
        ["-ub", "512", "-t", "4"],
        {"GGML_CUDA_DISABLE_GRAPHS": "1"},
    ),
    ("pq2", "PQ2_0", ["-ub", "512", "-t", "4", "-ctk", "q8_0", "-ctv", "q8_0"], {}),
]
summary = []
for label, packing, flags, settings in tests:
    path = MODEL / f"Ternary-Bonsai-2-27B-{packing}.gguf"
    if not path.is_file():
        if packing == "PQ2_0":
            print("Skipping optional PQ2_0 packing: GGUF not downloaded", flush=True)
            continue
        raise FileNotFoundError(f"Required ternary model missing: {path}")
    command = base + ["-m", str(path)] + flags
    print("Measuring", label, flush=True)
    with (OUT / (label + ".log")).open("w", encoding="utf-8") as errors:
        result = subprocess.run(
            command,
            env={**env, **settings},
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
        )
    (OUT / (label + ".json")).write_text(result.stdout, encoding="utf-8")
    entry = {
        "label": label,
        "packing": packing,
        "command": command,
        "environment": settings,
        "returncode": result.returncode,
    }
    if result.returncode == 0:
        entry["results"] = json.loads(result.stdout)
        for row in entry["results"]:
            print(
                label,
                "pp",
                row.get("n_prompt"),
                "tg",
                row.get("n_gen"),
                "ub",
                row.get("n_ubatch"),
                "threads",
                row.get("n_threads"),
                "tok/s",
                row.get("avg_ts"),
                flush=True,
            )
    else:
        print("Failed", label, "see log", flush=True)
    summary.append(entry)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
