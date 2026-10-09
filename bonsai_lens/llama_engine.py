"""Official Prism llama.cpp CUDA backend, accessed over its local native API.

Uses the native DLL bridge for real taps, with the original server as a fallback.
"""

from __future__ import annotations
import atexit
import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from transformers import AutoTokenizer
from .common import PromptMixin, Session

ROOT = Path(__file__).resolve().parents[1]


def Engine(pack):
    from .prism_engine import BRIDGE

    if (
        os.name == "nt"
        and ("BONSAI_READOUT_DLL" in os.environ or BRIDGE.is_file())
        and "BONSAI_LLAMA_URL" not in os.environ
        and "BONSAI_LLAMA_BIN" not in os.environ
        and os.environ.get("BONSAI_LLAMA_API", "dll") != "server"
    ):
        from .prism_engine import Engine as NativeEngine

        return NativeEngine(pack)
    return ServerEngine(pack)


class ServerEngine(PromptMixin):
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
        eos = json.loads(
            (self.pack / "generation_config.json").read_text(encoding="utf-8")
        )["eos_token_id"]
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self._template = None
        self.process = None
        self._log = None
        self.last_timings = None
        self.url = os.environ.get("BONSAI_LLAMA_URL", "http://127.0.0.1:8081").rstrip(
            "/"
        )
        if not self._healthy():
            self._start()
        props = self.request("/props")
        meta = self.request("/v1/models")["data"][0].get("meta") or {}
        if (
            meta.get("n_vocab") != self.vocab_size
            or meta.get("n_embd") != self.hidden_size
        ):
            self.close()
            raise RuntimeError(
                "Native model dimensions do not match the Bonsai tokenizer/pack"
            )
        self.context_limit = props["default_generation_settings"]["n_ctx"]
        self.media_marker = props["media_marker"]
        self.vision_enabled = bool(props.get("modalities", {}).get("vision"))
        model_path = props["model_path"]
        packing = next(
            (name for name in ("PTQ1_0", "PQ2_0") if name in model_path), "custom GGUF"
        )
        self.runtime_info = {
            "backend": "llama",
            "weight_format": packing,
            "model_path": model_path,
            "context_limit": self.context_limit,
            "vision": self.vision_enabled,
            "lens_readouts": False,
            "url": self.url,
            "launch_args": getattr(self, "launch_args", None),
        }

    def _healthy(self):
        try:
            return self.request("/health")["status"] == "ok"
        except (OSError, ValueError, RuntimeError):
            return False

    def _start(self):
        if "BONSAI_LLAMA_URL" in os.environ:
            raise RuntimeError(f"llama.cpp server is not ready at {self.url}")
        binaries = list(
            (ROOT / "runtime" / "llama-cuda").rglob(
                "llama-server.exe" if os.name == "nt" else "llama-server"
            )
        )
        binary = (
            Path(os.environ["BONSAI_LLAMA_BIN"])
            if "BONSAI_LLAMA_BIN" in os.environ
            else (binaries[0] if binaries else None)
        )
        model = Path(
            os.environ.get(
                "BONSAI_GGUF",
                str(
                    ROOT
                    / "models"
                    / "bonsai2-27b-gguf"
                    / "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
                ),
            )
        )
        if binary is None or not model.is_file():
            raise RuntimeError(
                "Official Prism CUDA binary/model missing. Run .\\setup-windows.ps1 -Backend llama (see README)."
            )
        args = [
            str(binary),
            "-m",
            str(model),
            "-ngl",
            os.environ.get("BONSAI_NGL", "99"),
            "-c",
            os.environ.get("BONSAI_CTX", "8192"),
            "-np",
            "1",
            "-b",
            os.environ.get("BONSAI_BATCH", "512"),
            "-ub",
            os.environ.get("BONSAI_UBATCH", "512"),
            "-fa",
            "on",
            "--host",
            "127.0.0.1",
            "--port",
            "8081",
            "--no-webui",
            "--alias",
            "Ternary-Bonsai-2-27B",
            "--jinja",
            "--no-context-shift",
        ]
        if "BONSAI_THREADS" in os.environ:
            args += ["-t", os.environ["BONSAI_THREADS"]]
        if "BONSAI_KV_TYPE" in os.environ:
            args += [
                "-ctk",
                os.environ["BONSAI_KV_TYPE"],
                "-ctv",
                os.environ["BONSAI_KV_TYPE"],
            ]
        projector = model.parent / "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
        if projector.is_file():
            args += ["--mmproj", str(projector)]
            if os.environ.get("BONSAI_VISION_GPU", "0") == "0":
                args += ["--no-mmproj-offload"]
        self.launch_args = args
        self._log = open(ROOT / "llama-server.log", "a", encoding="utf-8")
        env = os.environ.copy()
        if os.name == "nt":
            cuda = (
                Path(
                    env.get(
                        "CUDA_PATH_V12_8",
                        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
                    )
                )
                / "bin"
            )
            env["PATH"] = (
                str(binary.parent)
                + os.pathsep
                + str(cuda)
                + os.pathsep
                + env.get("PATH", "")
            )
        self.process = subprocess.Popen(
            args,
            env=env,
            stdout=self._log,
            stderr=self._log,
            cwd=str(binary.parent),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        atexit.register(self.close)
        deadline = time.monotonic() + 180
        while not self._healthy():
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama.cpp exited ({self.process.returncode}); see llama-server.log"
                )
            if time.monotonic() > deadline:
                self.close()
                raise RuntimeError("llama.cpp startup timed out; see llama-server.log")
            time.sleep(0.25)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self._log is not None:
            self._log.close()
            self._log = None

    def request(self, path, body=None, stream=False):
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=180)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"llama.cpp {exc.code}: {exc.read().decode()}") from exc
        if stream:
            return response
        with response:
            return json.load(response)

    def new_session(self, text, images=None):
        return Session(
            tokens=list(self.tokenizer.encode(text, add_special_tokens=False)),
            images=list(images or []),
        )

    def generate(self, session, temperature=0.0, top_p=0.95, top_k=20, min_p=0.0):
        prompt = session.tokens
        if session.images:
            if not self.vision_enabled:
                raise RuntimeError(
                    "Native image input requires the GGUF vision projector; rerun setup-windows.ps1 -Backend llama"
                )
            text = self.decode(prompt).replace(
                "<|vision_start|><|image_pad|><|vision_end|>", self.media_marker
            )
            prompt = {
                "prompt_string": text,
                "multimodal_data": [
                    base64.b64encode(Path(p).read_bytes()).decode()
                    for p in session.images
                ],
            }
        body = {
            "prompt": prompt,
            "n_predict": getattr(self, "max_generate", -1),
            "stream": True,
            "return_tokens": True,
            "cache_prompt": getattr(self, "cache_prompt", True),
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "seed": 1234,
        }
        with self.request("/completion", body, stream=True) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                for token in event.get("tokens", []):
                    session.fed = len(session.tokens)
                    session.tokens.append(int(token))
                    yield int(token)
                if event.get("stop"):
                    self.last_timings = event.get("timings")
                    if event.get("stop_type") == "eos":
                        token = min(self.eos_ids)
                        session.tokens.append(token)
                        yield token
                    return

    def forward(self, session):
        raise NotImplementedError(
            "The native llama.cpp server does not expose pre/post-norm Lens taps. Use BONSAI_BACKEND=cuda or mlx for Lens."
        )

    def image_details(self, session):
        raise NotImplementedError(
            "Image-token inspection requires the cuda or mlx Lens backend."
        )
