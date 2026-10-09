"""CUDA packed-kernel parity against explicit PyTorch dequantization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from bonsai_lens.cuda_engine import Packed, dequantize
from bonsai_lens.cuda_kernels import rotate, matmul


def reference_rotate(x, block, signs, inverse=False):
    y = x.float() if inverse else x.float() * signs.float()
    shape = y.shape
    y = y.reshape(-1, block)
    stride = 1
    while stride < block:
        a = y.reshape(-1, block // (2 * stride), 2, stride)
        y = torch.stack(
            (a[:, :, 0] + a[:, :, 1], a[:, :, 0] - a[:, :, 1]), dim=2
        ).reshape(-1, block)
        stride *= 2
    y = y.reshape(shape) / (block**0.5)
    return (y * signs if inverse else y).half()


torch.manual_seed(12)
device = "cuda"
for width, block in ((512, 512), (1024, 1024), (5120, 1024), (6144, 1024)):
    x = torch.randn(2, 3, width, device=device, dtype=torch.float16)
    signs = (torch.randint(0, 2, (width,), device=device) * 2 - 1).half()
    for inverse in (False, True):
        torch.testing.assert_close(
            rotate(x, block, signs, inverse),
            reference_rotate(x, block, signs, inverse),
            atol=0,
            rtol=0,
        )
    w = torch.randint(
        -(2**31), 2**31 - 1, (33, width // 16), device=device, dtype=torch.int32
    )
    s = torch.rand(33, width // 128, device=device).half() * 0.03
    b = -s
    dense = dequantize(w, s, b)
    torch.testing.assert_close(
        matmul(x, w, s, b), torch.nn.functional.linear(x, dense), atol=0.002, rtol=0.003
    )
    layer = Packed((w, s, b), block, signs, False, torch.device(device))
    torch.testing.assert_close(
        layer(x),
        torch.nn.functional.linear(reference_rotate(x, block, signs), dense),
        atol=0.002,
        rtol=0.003,
    )
    emb = Packed((w, s, b), block, signs, True, torch.device(device))
    ids = torch.tensor([[0, 3, 32]], device=device)
    torch.testing.assert_close(
        emb(ids), reference_rotate(dense[ids], block, signs, True), atol=0, rtol=0
    )
    # Actual ternary container: bias is implicit and code 3 is forbidden.
    codes = torch.randint(0, 3, (33, width), device=device, dtype=torch.int64)
    shifts = torch.arange(0, 32, 2, device=device, dtype=torch.int64)
    trits = (codes.reshape(33, -1, 16) << shifts).sum(-1).to(torch.int32)
    ternary = (
        (codes.reshape(33, -1, 128) * s[..., None] - s[..., None])
        .reshape(33, width)
        .half()
    )
    torch.testing.assert_close(
        matmul(x, trits, s, None),
        torch.nn.functional.linear(x, ternary),
        atol=0.002,
        rtol=0.003,
    )
print(
    "PASS: packed CUDA linear, embedding, forward/inverse Hadamard, 512/1024 blocks, batched inputs"
)

# Tensor-core path: odd token/row tails and real projection widths.
for width in (512, 5120, 17408):
    rows = 129
    codes = torch.randint(0, 3, (rows, width), device=device, dtype=torch.int64)
    shifts = torch.arange(0, 32, 2, device=device, dtype=torch.int64)
    trits = (codes.reshape(rows, -1, 16) << shifts).sum(-1).to(torch.int32)
    scales = torch.rand(rows, width // 128, device=device).half() * 0.03
    dense = dequantize(trits, scales, None)
    for tokens in (1, 7, 8, 17, 65, 128):
        x = torch.randn(1, tokens, width, device=device, dtype=torch.float16)
        torch.testing.assert_close(
            matmul(x, trits, scales, None),
            torch.nn.functional.linear(x, dense),
            atol=0.003,
            rtol=0.004,
        )
print(
    "PASS: warp matvec and shared-memory ternary tensor-core tiles, padded rows/tokens"
)
