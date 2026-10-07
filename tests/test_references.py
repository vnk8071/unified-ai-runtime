#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for scripts/check_references.py (no network)."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


refs = load("check_references")


class ExtractTest(unittest.TestCase):
    def test_finds_markdown_angle_and_bare_urls(self):
        text = ("See [docs](https://example.net/a/b.html) and <https://github.com/org/repo>, or https://huggingface.co/x/y.\n"
                "Code: `https://onnxruntime.ai/docs/`")
        urls, _ = refs.extract(text)
        self.assertEqual(urls, {"https://example.net/a/b.html", "https://github.com/org/repo",
                                "https://huggingface.co/x/y", "https://onnxruntime.ai/docs/"})

    def test_skips_placeholders_and_local_hosts(self):
        urls, _ = refs.extract("http://localhost:8080 https://example.com/x https://host/{name} https://a.b/$VAR/c "
                               "https://github.com/*/releases")
        self.assertEqual(urls, set())

    def test_local_links(self):
        _, local = refs.extract("[a](other.md) [b](../x/y.md#sec) [c](#top) [d](https://a.io) [e](mailto:x@y.io)")
        self.assertEqual(local, {"other.md", "../x/y.md"})

    def test_local_link_check_and_repo_has_none_broken(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "here.md").write_text("x")
            self.assertTrue(refs.check_local(Path(tmp) / "doc.md", "here.md"))
            self.assertFalse(refs.check_local(Path(tmp) / "doc.md", "gone.md"))
        self.assertEqual(refs.main(["--offline"]), 0)

    def test_default_targets_exist(self):
        self.assertGreater(len(refs.target_files(refs.DEFAULT_TARGETS)), 10)


if __name__ == "__main__":
    unittest.main()
