"""Select MLX on Apple Silicon or the packed CUDA engine on Windows/Linux.

BONSAI_BACKEND=auto|mlx|cuda|llama, BONSAI_PACK=/path/to/model.
"""

from __future__ import annotations
import os
import platform
from pathlib import Path
from .common import Readout, Session, summarize, load_json

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACK = ROOT / "models" / "bonsai2-27b-mlx"


def Engine(pack=None, *, backend=None):
    selected = backend or os.environ.get("BONSAI_BACKEND", "auto")
    if selected == "auto":
        native_readouts = Path(
            os.environ.get(
                "BONSAI_READOUT_DLL",
                str(ROOT / "runtime" / "llama-cuda" / "bonsai-readouts.dll"),
            )
        )
        selected = (
            "mlx"
            if platform.system() == "Darwin" and platform.machine() == "arm64"
            else "llama"
            if os.name == "nt" and native_readouts.is_file()
            else "cuda"
        )
    pack = pack or os.environ.get("BONSAI_PACK") or DEFAULT_PACK
    if selected == "mlx":
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError(
                "MLX requires Apple Silicon macOS; use BONSAI_BACKEND=cuda here"
            )
        from .mlx_engine import Engine as Backend
    elif selected == "cuda":
        from .cuda_engine import Engine as Backend
    elif selected == "llama":
        from .llama_engine import Engine as Backend
    else:
        raise ValueError(f"Unknown backend {selected!r}; use auto, mlx, cuda, or llama")
    return Backend(pack)
