# SPDX-License-Identifier: Apache-2.0
"""Compares UAIRT's ExecuTorch backend with ExecuTorch's own Python runtime on the same .pte file.

Exports a small convolution network to a .pte with the XNNPACK delegate (or none, with `portable`), runs one random input
through both runtimes and compares every output. Needs a Python with torch, numpy and the `executorch` package.

    python tests/compare_with_executorch.py build/run_model build/libuairt_backend_executorch.so [xnnpack|portable]
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 4, 3, stride=2, padding=1)
        self.fc = torch.nn.Linear(4 * 8 * 8, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.fc(x.flatten(1)), x.mean(dim=(2, 3))


def export(path, mode):
    program = torch.export.export(Net().eval(), (torch.zeros(1, 3, 16, 16),))
    partitioner = []
    if mode == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        partitioner = [XnnpackPartitioner()]
    path.write_bytes(to_edge_transform_and_lower(program, partitioner=partitioner).to_executorch().buffer)


def main(run_model, plugin, mode="xnnpack"):
    x = np.random.default_rng(0).random((1, 3, 16, 16), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = tmp / "net.pte"
        export(model, mode)

        method = Runtime.get().load_program(str(model)).load_method("forward")
        expected = [np.array(t) for t in method.execute([torch.from_numpy(x)])]

        x.tofile(tmp / "in0.bin")
        subprocess.run([run_model, plugin, "executorch", str(model), str(tmp / "out"), str(tmp / "in0.bin")], check=True)
        informative = 0
        for i, want in enumerate(expected):
            got = np.fromfile(tmp / f"out{i}.bin", dtype=np.float32).reshape(want.shape)
            np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
            informative += np.unique(want).size > 1
            print(f"output {i}: {want.shape} matches ExecuTorch's Python runtime "
                  f"(max abs diff {np.abs(got - want).max():.3g})")
    assert informative, "every output is constant; the comparison proves nothing"
    print(f"UAIRT ExecuTorch backend ({mode}) matches ExecuTorch")


if __name__ == "__main__":
    main(*sys.argv[1:4])
