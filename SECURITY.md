<!-- SPDX-License-Identifier: Apache-2.0 -->
# Security

UAIRT parses model files and passes them to vendor runtimes. A malformed model can crash
a backend, and this project cannot make a vendor runtime safe against hostile input.
Only load models you trust, and run untrusted ones in a sandboxed process.

To report a vulnerability in UAIRT itself, use the repository's private vulnerability
reporting (GitHub "Security" tab) instead of a public issue. Please include the
backend, platform, SDK version and a minimal reproduction.

Vulnerabilities in a vendor runtime (QAIRT, ONNX Runtime, CoreML, TensorRT) should be
reported to that vendor.
