# SPDX-License-Identifier: Apache-2.0
from importlib import metadata

import pytest

import uairt


def test_the_package_version_matches_the_native_core():
    try:
        installed = metadata.version("uairt")
    except metadata.PackageNotFoundError:
        pytest.skip("uairt is not installed as a package")
    assert installed == uairt.version(), "pyproject.toml and include/uairt/uairt.h disagree about the version"
