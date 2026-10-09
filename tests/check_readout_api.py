"""Smoke-test real direct readouts from the running local app."""

import io
import json
import time
import urllib.request

import numpy as np

BASE = "http://127.0.0.1:8000"


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


for _ in range(180):
    info = json.loads(request("/api/info"))
    if info.get("error"):
        raise RuntimeError(info["error"])
    if info.get("ready"):
        break
    time.sleep(2)
else:
    raise TimeoutError("Model did not become ready")

assert info["backend"] in {"cuda", "llama", "mlx"} and info["lens_readouts"], info
reset = json.loads(
    request(
        "/api/reset",
        {"prompt": "What is 2 plus 2? Answer with the number only.", "thinking": False},
    )
)
first = json.loads(request("/api/step", {"strategy": "greedy", "k": 5}))
assert first["chosen"]["piece"].strip() == "4", first["chosen"]
for index in range(2):
    step = (
        first
        if index == 0
        else json.loads(request("/api/step", {"strategy": "greedy", "k": 5}))
    )
    assert len(step["readout"]["pre_norm"]) == 5120
    assert len(step["readout"]["post_norm"]) == 5120
    assert np.isfinite(step["readout"]["entropy"])
    archive = np.load(io.BytesIO(request("/api/last.npz")))
    for key, shape in (
        ("logits", (248320,)),
        ("pre_norm", (5120,)),
        ("post_norm", (5120,)),
    ):
        assert archive[key].shape == shape, (key, archive[key].shape)
        assert np.isfinite(archive[key]).all(), key
        if key != "logits":
            np.testing.assert_array_equal(archive[key], step["readout"][key])
    print(
        f"Step {step['step']}: {step['forward_ms']:.1f} ms, {step['chosen']}",
        flush=True,
    )
print(
    f"PASS: {info['backend']} readouts, finite full logits, hidden states and downloadable NPZ",
    flush=True,
)
