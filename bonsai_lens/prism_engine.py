"""Fast Prism CUDA kernels with native final-layer readouts through its DLL API."""

from __future__ import annotations
import ctypes as C
import json
import os
import weakref
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from .common import PromptMixin, Readout, Session

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "runtime" / "llama-cuda" / "bonsai-readouts.dll"


class Engine(PromptMixin):
    backend = "llama"

    def __init__(self, pack):
        self.pack = Path(pack)
        self.config = json.loads(
            (self.pack / "config.json").read_text(encoding="utf-8")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.pack))
        self.hidden_size = self.config["text_config"]["hidden_size"]
        self.vocab_size = self.config["text_config"]["vocab_size"]
        self.image_token_id = self.config["image_token_id"]
        eos = json.loads((self.pack / "generation_config.json").read_text())[
            "eos_token_id"
        ]
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self._template = None
        self._active = None
        self._handle = None
        bridge = Path(os.environ.get("BONSAI_READOUT_DLL", str(BRIDGE)))
        self._dll_dir = os.add_dll_directory(str(bridge.parent.resolve()))
        self.lib = C.CDLL(str(bridge.resolve()))
        ptr, ints, floats, bytes_ = (
            C.c_void_p,
            C.POINTER(C.c_int32),
            C.POINTER(C.c_float),
            C.POINTER(C.c_uint8),
        )
        signatures = {
            "error": ([], C.c_char_p),
            "open": ([C.c_char_p] * 3 + [C.c_int] * 5 + [C.c_char_p, C.c_bool], ptr),
            "close": ([ptr], None),
            "reset": ([ptr], None),
            "position": ([ptr], C.c_int),
            "decode": ([ptr, ints, C.c_int, C.c_bool], C.c_int),
            "image_prompt": (
                [ptr, C.c_char_p, C.POINTER(C.c_char_p), C.c_int, C.c_bool],
                C.c_int,
            ),
            "logits": ([ptr], floats),
            "pre_norm": ([ptr], floats),
            "post_norm": ([ptr], floats),
            "state_size": ([ptr], C.c_size_t),
            "state_get": ([ptr, bytes_, C.c_size_t], C.c_size_t),
            "state_set": ([ptr, bytes_, C.c_size_t, C.c_int], C.c_size_t),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, "bonsai_" + name)
            fn.argtypes, fn.restype = args, result
        self.context_limit = int(os.environ.get("BONSAI_CTX", "8192"))
        model = Path(
            os.environ.get(
                "BONSAI_GGUF",
                str(ROOT / "models/bonsai2-27b-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf"),
            )
        )
        projector = model.parent / "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
        if not model.is_file():
            raise RuntimeError(
                "Official ternary GGUF missing; run setup-windows.ps1 -Backend llama"
            )
        self._handle = self.lib.bonsai_open(
            str(bridge.parent.resolve()).encode(),
            str(model.resolve()).encode(),
            str(projector.resolve()).encode() if projector.is_file() else b"",
            self.context_limit,
            int(os.environ.get("BONSAI_THREADS", "14")),
            int(os.environ.get("BONSAI_BATCH", "512")),
            int(os.environ.get("BONSAI_UBATCH", "512")),
            int(os.environ.get("BONSAI_NGL", "99")),
            os.environ.get("BONSAI_KV_TYPE", "f16").encode(),
            os.environ.get("BONSAI_VISION_GPU", "0") == "1",
        )
        if not self._handle:
            raise RuntimeError(self.lib.bonsai_error().decode())
        self.runtime_info = {
            "backend": "llama",
            "implementation": "Prism DLL + native capture API",
            "weight_format": "PTQ1_0" if "PTQ1_0" in model.name else "PQ2_0",
            "model_path": str(model),
            "context_limit": self.context_limit,
            "vision": projector.is_file(),
            "vision_device": "cuda"
            if os.environ.get("BONSAI_VISION_GPU", "0") == "1"
            else "cpu",
            "lens_readouts": True,
            "native_build": "10754-2459f68",
            "capture_layer": 63,
            "image_token_readouts": False,
        }

    def close(self):
        if self._handle:
            self.lib.bonsai_close(self._handle)
            self._handle = None

    def new_session(self, text, images=None):
        return Session(
            tokens=list(self.tokenizer.encode(text, add_special_tokens=False)),
            images=list(images or []),
        )

    def _activate(self, session):
        old = self._active() if self._active else None
        if old is session:
            return
        if old is not None and old.fed:
            size = self.lib.bonsai_state_size(self._handle)
            data = np.empty(size, dtype=np.uint8)
            written = self.lib.bonsai_state_get(
                self._handle, data.ctypes.data_as(C.POINTER(C.c_uint8)), size
            )
            if written != size:
                raise RuntimeError("Native sequence snapshot failed")
            old.cache = (data, self.lib.bonsai_position(self._handle))
        self.lib.bonsai_reset(self._handle)
        if session.fed:
            if session.cache is None:
                raise RuntimeError("Native session has no saved state")
            data, position = session.cache
            restored = self.lib.bonsai_state_set(
                self._handle,
                data.ctypes.data_as(C.POINTER(C.c_uint8)),
                data.size,
                position,
            )
            if restored != data.size:
                raise RuntimeError("Native sequence restore failed")
            session.cache = None
        self._active = weakref.ref(session)

    def _eval(self, session, capture):
        self._activate(session)
        new = session.tokens[session.fed :]
        if not new:
            raise ValueError("nothing to feed: append a token first")
        if not session.fed and session.images:
            prompt = self.decode(session.tokens).replace(
                "<|vision_start|><|image_pad|><|vision_end|>", "<__media__>"
            )
            paths = (C.c_char_p * len(session.images))(
                *(str(Path(p).resolve()).encode() for p in session.images)
            )
            status = self.lib.bonsai_image_prompt(
                self._handle, prompt.encode(), paths, len(paths), capture
            )
        else:
            ids = np.asarray(new, dtype=np.int32)
            status = self.lib.bonsai_decode(
                self._handle,
                ids.ctypes.data_as(C.POINTER(C.c_int32)),
                ids.size,
                capture,
            )
        if status:
            raise RuntimeError(self.lib.bonsai_error().decode())
        session.fed = len(session.tokens)
        logits = self._copy("logits", self.vocab_size)
        session.next_logits = logits[None, :]
        return logits

    def _copy(self, name, count):
        ptr = getattr(self.lib, "bonsai_" + name)(self._handle)
        if not ptr:
            raise RuntimeError(f"Prism did not produce {name}")
        return np.ctypeslib.as_array(ptr, shape=(count,)).copy()

    def forward(self, session):
        logits = self._eval(session, True)
        return Readout(
            logits,
            self._copy("pre_norm", self.hidden_size),
            self._copy("post_norm", self.hidden_size),
        )

    def generate(self, session, temperature=0.0, top_p=0.95, top_k=20, min_p=0.0):
        limit = getattr(self, "max_generate", -1)
        generated = 0
        while limit < 0 or generated < limit:
            logits = (
                session.next_logits[0]
                if session.fed == len(session.tokens)
                and session.next_logits is not None
                else self._eval(session, False)
            )
            token = self.sample(logits, temperature, top_p, top_k, min_p)
            session.tokens.append(token)
            generated += 1
            yield token
            if token in self.eos_ids:
                return

    @staticmethod
    def sample(logits, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0):
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        if temperature <= 0:
            return int(np.argmax(logits))
        if top_k > 0 and top_k < logits.size:
            order = np.argpartition(-logits, top_k - 1)[:top_k]
            order = order[np.argsort(logits[order])[::-1]]
        else:
            order = np.argsort(logits)[::-1]
        values = logits[order] / temperature
        probs = np.exp(values - values[0])
        if top_k > 0:
            probs[top_k:] = 0
        if min_p > 0:
            probs[probs < probs.max() * min_p] = 0
        if top_p < 1:
            normalized = probs / probs.sum()
            probs[np.cumsum(normalized) - normalized >= top_p] = 0
        probs /= probs.sum()
        return int(np.random.choice(order, p=probs))

    def image_details(self, session):
        raise NotImplementedError(
            "Native final-layer readouts are available; per-image-token inspection still requires the direct CUDA or MLX backend."
        )
