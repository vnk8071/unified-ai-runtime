# SPDX-License-Identifier: Apache-2.0
"""quickstart.c in Python: the built-in "reference" backend copies one float32 [1, 4] input to the output.

    PYTHONPATH=bindings/python UAIRT_LIBRARY=build-shared/libuairt.dylib python examples/python/quickstart.py

Needs a shared build of the core (cmake -DBUILD_SHARED_LIBS=ON) and NumPy.
"""
import numpy as np

import uairt


def main():
    print("UAIRT", uairt.version(), "backends:", uairt.backends())
    with uairt.Engine("reference") as engine, engine.load_model("unused") as model:
        info = model.inputs[0]
        print(f"input {info.name}: {info.dtype} {info.shape}")
        x = np.array([[1, 2, 3, 4]], dtype=info.dtype).reshape(info.shape)
        (y,) = model.run(x)
    print("output:", y.ravel().tolist())
    return 0 if y.ravel()[3] == 4.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
