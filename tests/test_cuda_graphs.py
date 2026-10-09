"""Captured packed CUDA MLPs must match ordinary execution for changing inputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from bonsai_lens.cuda_engine import Packed
from bonsai_lens.cuda_graphs import GraphedMLP

torch.manual_seed(37)


def projection():
    weights = torch.full((512, 32), 0x24924924, device="cuda", dtype=torch.int32)
    scales = torch.rand(512, 4, device="cuda", dtype=torch.float16) * 0.03
    signs = (torch.randint(0, 2, (512,), device="cuda") * 2 - 1).half()
    return Packed((weights, scales, None), 512, signs, False, torch.device("cuda"))


inner = nn.Sequential(projection(), nn.SiLU(), projection())
graph = GraphedMLP(inner)
with torch.inference_mode():
    for tokens in (1, 1, 17, 1, 1):
        x = torch.randn(1, tokens, 512, device="cuda", dtype=torch.float16)
        expected = inner(x)
        actual = graph(x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
assert graph.graph is not None and not graph.disabled
print(
    "PASS: CUDA graph replay matches ordinary packed MLP, changing inputs and prefill"
)
