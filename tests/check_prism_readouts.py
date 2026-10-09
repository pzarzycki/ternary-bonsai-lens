"""Real-model checks: native tap identity, logits parity and session restoration."""

import numpy as np
from bonsai_lens.engine import Engine
from bonsai_lens.packed_file import PackedFile

e = Engine(backend="llama")
assert e.runtime_info["lens_readouts"], e.runtime_info
prompt = e.render(
    [{"role": "user", "content": "What is 2 plus 2? Answer with the number only."}]
)
with PackedFile(e.pack / "model.safetensors") as weights:
    scale = (
        weights.get_tensor("language_model.model.norm.weight")
        .numpy()
        .copy()
        .astype(np.float32)
    )
eps = e.config["text_config"]["rms_norm_eps"]
plain = e.new_session(prompt)
plain_logits, tokens = [], []
for _ in range(4):
    logits = e._eval(plain, False)
    plain_logits.append(logits)
    token = e.sample(logits, 0)
    tokens.append(token)
    plain.tokens.append(token)
    if token in e.eos_ids:
        break
assert e.decode(tokens[:1]).strip() == "4", e.decode(tokens)
tapped = e.new_session(prompt)
for expected, token in zip(plain_logits, tokens):
    readout = e.forward(tapped)
    assert readout.logits.shape == (248320,)
    assert readout.pre_norm.shape == readout.post_norm.shape == (5120,)
    for array in (readout.logits, readout.pre_norm, readout.post_norm):
        assert np.isfinite(array).all()
    # Masked native exports must leave every logit unchanged.
    delta = np.abs(readout.logits - expected)
    print(
        f"Capture logit difference: max={delta.max():.6f}, rms={np.sqrt(np.mean(delta**2)):.6f}",
        flush=True,
    )
    np.testing.assert_array_equal(readout.logits, expected)
    norm = readout.pre_norm / np.sqrt(np.mean(readout.pre_norm**2) + eps) * scale
    print(
        f"RMSNorm max error: {np.max(np.abs(readout.post_norm - norm)):.8f}", flush=True
    )
    np.testing.assert_allclose(readout.post_norm, norm, atol=3e-5, rtol=3e-5)
    assert e.sample(readout.logits, 0) == token
    tapped.tokens.append(token)
# Compare an uninterrupted continuation with one restored after another session.
expected = e._eval(tapped, False)
other = e.new_session(e.render([{"role": "user", "content": "Say hello."}]))
e.forward(other)
actual = e._eval(plain, False)
np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-5)
for length in (128, 512):
    ids = e.tokenizer.encode(
        "A robot has motors and sensors. " * 100, add_special_tokens=False
    )[:length]
    plain = e.new_session("")
    plain.tokens = list(ids)
    expected = e._eval(plain, False)
    tapped = e.new_session("")
    tapped.tokens = list(ids)
    readout = e.forward(tapped)
    np.testing.assert_array_equal(readout.logits, expected)
    norm = readout.pre_norm / np.sqrt(np.mean(readout.pre_norm**2) + eps) * scale
    np.testing.assert_allclose(readout.post_norm, norm, atol=3e-5, rtol=3e-5)
    print(
        f"PASS: {length}-token native prompt capture and exact logits parity",
        flush=True,
    )
print(
    "PASS: native pre/post-norm identity, finite full logits, capture-on/off parity, greedy 4, and recurrent session restore",
    flush=True,
)
e.close()
