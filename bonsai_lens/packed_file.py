"""Read-only, zero-copy safetensors access for memory-constrained Windows hosts."""

import json
import struct
import warnings
from pathlib import Path
import numpy as np
import torch


class PackedFile:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(size))
        self.offset = 8 + size

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def keys(self):
        return [k for k in self.header if k != "__metadata__"]

    def get_tensor(self, key):
        spec = self.header[key]
        dtype = {
            "F16": "<f2",
            "F32": "<f4",
            "BF16": "<u2",
            "U32": "<u4",
            "I32": "<i4",
            "I64": "<i8",
        }[spec["dtype"]]
        data = np.memmap(
            self.path,
            mode="r",
            dtype=dtype,
            offset=self.offset + spec["data_offsets"][0],
            shape=tuple(spec["shape"]),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given NumPy array is not writable"
            )
            tensor = torch.from_numpy(data)
        return tensor.view(torch.bfloat16) if spec["dtype"] == "BF16" else tensor
