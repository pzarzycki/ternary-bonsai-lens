"""Validate the tuned native server still handles images with the vision tower on CPU."""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from bonsai_lens.engine import Engine

engine = Engine(backend="llama")
try:
    with tempfile.TemporaryDirectory(prefix="bonsai-vision-check-") as folder:
        path = Path(folder) / "red.png"
        Image.new("RGB", (96, 96), (255, 0, 0)).save(path)
        text = engine.render(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": "What color is this image? Answer with one word.",
                        },
                    ],
                }
            ]
        )
        session = engine.new_session(text, [str(path)])
        engine.max_generate = 8
        start = time.perf_counter()
        result = []
        iterator = engine.generate(session, temperature=0)
        try:
            for token in iterator:
                if token in engine.eos_ids:
                    break
                result.append(token)
                if len(result) >= 8:
                    break
        finally:
            iterator.close()
        answer = engine.decode(result).strip()
        print(
            "Image answer:",
            repr(answer),
            "seconds:",
            round(time.perf_counter() - start, 2),
            flush=True,
        )
        assert answer.lower() == "red", answer
        if engine.runtime_info.get("lens_readouts"):
            import numpy as np

            tapped = engine.new_session(text, [str(path)])
            readout = engine.forward(tapped)
            assert readout.pre_norm.shape == readout.post_norm.shape == (5120,)
            assert readout.logits.shape == (248320,)
            assert all(
                np.isfinite(v).all()
                for v in (readout.logits, readout.pre_norm, readout.post_norm)
            )
            assert (
                engine.decode([engine.sample(readout.logits, 0)]).strip().lower()
                == "red"
            )
            print("PASS: native final-layer readouts after an image prompt", flush=True)
        print(
            "PASS: CPU vision keeps native CUDA text generation and image input working",
            flush=True,
        )
finally:
    engine.close()
