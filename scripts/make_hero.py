#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Writes docs/images/hero-light.svg and hero-dark.svg (for the README) and social-preview.svg (1280x640, upload it in the repository settings as a PNG) from docs/images/hero-data.json.

The bars are measured numbers (scripts/bench_model.py), not illustrations: change the data file, run this script, commit both.
    python3 scripts/make_hero.py
"""
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "docs" / "images"
THEMES = {
    "light": dict(bg="#ffffff", panel="#f6f8fa", border="#d0d7de", text="#1f2328", muted="#59636e", code="#0550ae", exact="#1a7f37",
                  reduced="#bf8700", bar_bg="#eaeef2"),
    "dark": dict(bg="#0d1117", panel="#161b22", border="#30363d", text="#e6edf3", muted="#9198a1", code="#79c0ff", exact="#3fb950",
                 reduced="#d29922", bar_bg="#21262d"),
}
BACKENDS = "ONNX Runtime  ·  QNN  ·  TensorRT  ·  CoreML  ·  TFLite  ·  OpenVINO  ·  NCNN  ·  MNN  ·  Paddle Lite  ·  ExecuTorch  ·  llama.cpp  ·  whisper.cpp"
SANS = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"


def t(x, y, text, size, fill, weight=400, family=SANS, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" xml:space="preserve">{escape(text)}</text>')


def chart(data, c, x, y, width):
    rows = data["rows"]
    top = max(r["ms"] for r in rows)
    label_w, value_w = 290, 88
    bar_w = width - label_w - value_w
    out = [t(x, y, "Same model, six ways to run it on one Mac", 15, c["text"], 600),
           t(x, y + 22, "median ms per run, lower is faster; each result checked against PyTorch", 12.5, c["muted"])]
    y += 52
    for r in rows:
        color = c["exact"] if r["precision"] == "exact" else c["reduced"]
        length = max(4, round(bar_w * r["ms"] / top))
        out.append(t(x, y + 14, r["label"], 13, c["text"]))
        out.append(f'<rect x="{x + label_w}" y="{y}" width="{bar_w}" height="20" rx="4" fill="{c["bar_bg"]}"/>')
        out.append(f'<rect x="{x + label_w}" y="{y}" width="{length}" height="20" rx="4" fill="{color}"/>')
        out.append(t(x + label_w + bar_w + 10, y + 15, f'{r["ms"]:.2f} ms', 13, c["text"], 600, MONO))
        y += 34
    y += 6
    out.append(f'<rect x="{x}" y="{y - 11}" width="12" height="12" rx="3" fill="{c["exact"]}"/>')
    out.append(t(x + 18, y, "exact (float32)", 12.5, c["muted"]))
    out.append(f'<rect x="{x + 140}" y="{y - 11}" width="12" height="12" rx="3" fill="{c["reduced"]}"/>')
    out.append(t(x + 158, y, "reduced (half precision, GPU / Neural Engine)", 12.5, c["muted"]))
    return out


def code_panel(c, x, y, w, h):
    lines = [("with uairt.Engine(", None), ('"coreml"', "accent"), (") as engine,", None)]
    body = [
        [("# the backend is a name:", "muted")],
        [("#   ", "muted"), ('"qnn"', "accent"), (", ", "muted"), ('"tensorrt"', "accent"), (", ", "muted"), ('"onnxruntime"', "accent"), (",", "muted")],
        [("#   ", "muted"), ('"openvino"', "accent"), (", ", "muted"), ('"mnn"', "accent"), (", ", "muted"), ('"llamacpp"', "accent"), (" ...", "muted")],
    ]
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{c["panel"]}" stroke="{c["border"]}"/>']
    def line(row, ly):
        spans = ""
        for text, kind in row:
            fill = c["code"] if kind == "accent" else (c["muted"] if kind == "muted" else c["text"])
            spans += f'<tspan fill="{fill}">{escape(text)}</tspan>'
        return (f'<text x="{x + 20}" y="{ly}" font-family="{MONO}" font-size="14.5" xml:space="preserve">{spans}</text>')
    ly = y + 34
    out.append(line([("with uairt.Engine(", None), ('"coreml"', "accent"), (") as engine, \\", None)], ly)); ly += 24
    out.append(line([("     engine.load_model(path) as model:", None)], ly)); ly += 24
    out.append(line([("    outputs = model.run(image)", None)], ly)); ly += 36
    for row in body:
        out.append(line(row, ly)); ly += 22
    return out


def render(theme, width, height, social=False, title=True):
    """title=False drops the big title (the README already has an H1 above the image) and the height shrinks to match."""
    c = THEMES[theme]
    data = json.loads((IMAGES / "hero-data.json").read_text())
    pad = 44
    shift = 0 if title else -80
    height = height + shift
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" '
         f'aria-label="Unified AI Runtime: one API over every inference runtime, with measured latencies on an Apple M5">',
         f'<rect width="{width}" height="{height}" rx="14" fill="{c["bg"]}"/>']
    if title:
        s += [t(pad, 74, "Unified AI Runtime", 46, c["text"], 700),
              t(pad, 110, "One API for every inference runtime. Choose the backend by name.", 20, c["muted"])]
    else:
        s.append(t(pad, 44, "One API for every inference runtime. Choose the backend by name.", 20, c["muted"]))
    split = 520 if not social else 560
    s.append(f'<g transform="translate(0,{shift})">')
    s += code_panel(c, pad, 150, split - pad - 20, 178)
    s += [t(pad, 372, "Write the application once. Run it on a Qualcomm NPU,", 15, c["text"]),
          t(pad, 394, "an NVIDIA GPU, an Apple Neural Engine or a CPU.", 15, c["text"])]
    s += chart(data, c, split + 20, 164, width - split - 20 - pad)
    s.append("</g>")
    s += [f'<line x1="{pad}" y1="{height - 74}" x2="{width - pad}" y2="{height - 74}" stroke="{c["border"]}"/>',
          t(width / 2, height - 46, BACKENDS, 13.5, c["muted"], 500, anchor="middle"),
          t(width / 2, height - 22, "C11 API  ·  C++17 core with no C++ runtime  ·  plugins  ·  Apache-2.0  ·  github.com/vnk8071/unified-ai-runtime", 12.5, c["muted"], anchor="middle"),
          "</svg>"]
    return "\n".join(s) + "\n"


def main():
    (IMAGES / "hero-light.svg").write_text(render("light", 1200, 520, title=False))
    (IMAGES / "hero-dark.svg").write_text(render("dark", 1200, 520, title=False))
    (IMAGES / "social-preview.svg").write_text(render("dark", 1280, 640, social=True))
    print("wrote", ", ".join(p.name for p in sorted(IMAGES.glob("*.svg"))))


if __name__ == "__main__":
    main()
