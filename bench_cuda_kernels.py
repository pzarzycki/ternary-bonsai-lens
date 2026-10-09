"""Isolated packed-matrix kernel timings, with fixed ternary data and CUDA events."""

import json
import os
from pathlib import Path
import torch
from bonsai_lens.cuda_kernels import matmul

rows_out = []
with torch.inference_mode():
    for width, rows in ((5120, 17408), (17408, 5120), (5120, 10240)):
        weight = torch.full(
            (rows, width // 16), 0x24924924, device="cuda", dtype=torch.int32
        )
        scales = (
            torch.rand(rows, width // 128, device="cuda", dtype=torch.float16) * 0.03
        )
        for tokens in (1, 16, 128):
            x = torch.randn(1, tokens, width, device="cuda", dtype=torch.float16)
            for label, legacy, padded in (
                ("legacy", True, False),
                ("optimized", False, False),
                ("padded", False, True),
            ):
                if padded and tokens == 1:
                    continue
                os.environ["BONSAI_CUDA_LEGACY_MM"] = "1" if legacy else "0"
                os.environ["BONSAI_CUDA_MMA_PADDING"] = "1" if padded else "0"
                for _ in range(3):
                    matmul(x, weight, scales, None)
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(10):
                    matmul(x, weight, scales, None)
                end.record()
                end.synchronize()
                row = {
                    "width": width,
                    "rows": rows,
                    "tokens": tokens,
                    "kernel": label,
                    "ms": start.elapsed_time(end) / 10,
                }
                rows_out.append(row)
                print(row, flush=True)
out = Path("benchmarks/windows-cuda/kernel-timings.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows_out, indent=2), encoding="utf-8")
