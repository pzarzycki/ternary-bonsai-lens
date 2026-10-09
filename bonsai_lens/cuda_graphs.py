"""Replay stateless single-token MLPs with CUDA graphs; retain ordinary prefill."""

import torch
from torch import nn


class GraphedMLP(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.graph = None
        self.disabled = False

    def forward(self, x):
        if x.shape[:-1] != (1, 1) or self.disabled:
            return self.inner(x)
        if self.graph is None:
            try:
                self.static_input = torch.empty_like(x)
                self.static_input.copy_(x)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.inner(self.static_input)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    self.static_output = self.inner(self.static_input)
                self.graph = graph
            except (RuntimeError, BufferError) as exc:
                self.disabled = True
                print(f"MLP graph capture unavailable: {exc}", flush=True)
                return self.inner(x)
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output
