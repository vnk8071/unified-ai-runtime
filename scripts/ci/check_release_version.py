# SPDX-License-Identifier: Apache-2.0
"""Fails unless pyproject.toml, include/uairt/uairt.h and (optionally) a release tag name the same version.

    python scripts/ci/check_release_version.py [v0.1.0]

A tag or pyproject version may carry a PEP 440 pre-release suffix (0.1.0rc1); the numbers must match the header.
"""
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
pyproject = re.search(r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(), re.M).group(1)
header = (root / "include" / "uairt" / "uairt.h").read_text()
core = ".".join(re.search(rf"UAIRT_VERSION_{part} (\d+)", header).group(1) for part in ("MAJOR", "MINOR", "PATCH"))

versions = {"pyproject.toml": pyproject, "include/uairt/uairt.h": core}
if len(sys.argv) > 1:
    versions["tag"] = sys.argv[1].removeprefix("v")

numeric = {name: re.match(r"\d+\.\d+\.\d+", value).group() for name, value in versions.items()}
for name, value in versions.items():
    print(f"{name:24} {value}")
if len(set(numeric.values())) != 1:
    sys.exit("the versions disagree: bump pyproject.toml and include/uairt/uairt.h together, and tag vX.Y.Z")
if sys.argv[1:] and versions["tag"] != pyproject:
    sys.exit(f"the tag says {versions['tag']} but pyproject.toml says {pyproject}")
print("versions agree")
