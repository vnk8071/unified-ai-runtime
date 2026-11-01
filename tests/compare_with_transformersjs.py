# SPDX-License-Identifier: Apache-2.0
"""Compares UAIRT's transformersjs backend with Transformers.js called directly from Node on the same model and inputs.

The reference is the vendor's own API (`AutoModel.from_pretrained(...)(inputs)`), run in a separate Node process on the same device
string. The inputs are random but fixed; the check fails when the output is constant. Needs Node, numpy and
@huggingface/transformers installed in <package_dir>.

    python tests/compare_with_transformersjs.py build/run_model build/libuairt_backend_transformersjs.so <package_dir> <model_dir> \
        input_ids:int64:1x16:1000,attention_mask:int64:1x16,token_type_ids:int64:1x16 [cpu|webgpu] [atol]

An integer input is `name:dtype:shape[:exclusive maximum]`, random in [0, maximum), 2 by default so masks and segment ids stay valid.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REFERENCE = r"""
import { createRequire } from 'node:module';
import { readFileSync, writeFileSync } from 'node:fs';
const [packageDir, modelDir, device, spec, outPath] = process.argv.slice(2);
const require = createRequire(packageDir.replace(/\/?$/, '/'));
const tf = await import(require.resolve('@huggingface/transformers'));
tf.env.allowRemoteModels = false;
const model = await tf.AutoModel.from_pretrained(modelDir, { device, dtype: 'fp32' });
const T = { float32: Float32Array, int64: BigInt64Array, int32: Int32Array, uint8: Uint8Array };
const feeds = {};
for (const item of JSON.parse(spec)) {
  const bytes = readFileSync(item.file);
  const data = new T[item.dtype](bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
  feeds[item.name] = new tf.Tensor(item.dtype, data, item.dims);
}
const result = await model(feeds);
const names = Object.keys(model.sessions)[0];
const parts = model.sessions[names].outputNames.map((n) => Buffer.from(result[n].data.buffer, result[n].data.byteOffset, result[n].data.byteLength));
writeFileSync(outPath, Buffer.concat(parts));
"""

DTYPES = {"float32": np.float32, "int64": np.int64, "int32": np.int32, "uint8": np.uint8}


def main(run_model, plugin, package_dir, model_dir, inputs, device="cpu", atol="1e-3"):
    rng = np.random.default_rng(0)
    specs = []
    arrays = []
    for entry in inputs.split(","):
        name, dtype, shape, *limit = entry.split(":")
        dims = [int(d) for d in shape.split("x")]
        if dtype == "float32":
            array = rng.random(dims, dtype=np.float32)
        else:
            array = rng.integers(0, int(limit[0]) if limit else 2, size=dims).astype(DTYPES[dtype])
        specs.append((name, dtype, dims))
        arrays.append(array)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        files = []
        for (name, _, _), array in zip(specs, arrays):
            path = tmp / f"{name}.bin"
            array.tofile(path)
            files.append(str(path))
        (tmp / "reference.mjs").write_text(REFERENCE)
        feeds = json.dumps([{"name": n, "dtype": d, "dims": dims, "file": f} for (n, d, dims), f in zip(specs, files)])
        subprocess.run(["node", str(tmp / "reference.mjs"), package_dir, model_dir, device, feeds, str(tmp / "ref.bin")], check=True)
        expected = np.fromfile(tmp / "ref.bin", dtype=np.float32)

        shapes = ";".join(f"{n}:{'x'.join(map(str, dims))}" for n, _, dims in specs)
        command = [run_model, "--option", f"package_dir={package_dir}", "--option", f"device={device}", "--option",
                   f"input_shapes={shapes}", plugin, "transformersjs", model_dir, str(tmp / "out")] + files
        subprocess.run(command, check=True, stderr=subprocess.DEVNULL)
        outputs = sorted(tmp.glob("out*.bin"))
        actual = np.concatenate([np.fromfile(p, dtype=np.float32) for p in outputs])

    assert expected.size == actual.size, (expected.size, actual.size)
    assert expected.std() > 1e-6, "the reference output is constant; the comparison proves nothing"
    assert actual.std() > 1e-6, "the UAIRT output is constant"
    diff = float(np.abs(expected - actual).max())
    print(f"{device}: {actual.size} values, std {expected.std():.3g}, max abs difference {diff:.3g}")
    if diff > float(atol):
        sys.exit(f"difference {diff} is above {atol}")


if __name__ == "__main__":
    main(*sys.argv[1:])
