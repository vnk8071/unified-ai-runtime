# SPDX-License-Identifier: Apache-2.0
"""Compares UAIRT's llama.cpp backend with llama.cpp itself: greedy decoding of one prompt must give the same text.

    python tests/compare_with_llamacpp.py build/llm_generate build/libuairt_backend_llamacpp.so \
        build/llama.cpp/bin/llama-completion model.gguf [--layers 0 99]

llm_generate runs a fixed number of tokens, so the prompt must not reach an end-of-generation token within them.
"""
import argparse
import subprocess
import sys

PROMPT = "The capital of France is"
TOKENS = 30


def run(command):
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except subprocess.CalledProcessError as error:
        print(f"command failed with status {error.returncode}: {' '.join(command)}", file=sys.stderr)
        print(error.stderr, file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("llm_generate")
    parser.add_argument("plugin")
    parser.add_argument("llama_completion")
    parser.add_argument("model")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 99])
    args = parser.parse_args()

    failed = False
    for layers in args.layers:
        ours = run([args.llm_generate, args.plugin, args.model, PROMPT, str(TOKENS), str(layers)])
        theirs = run([args.llama_completion, "-m", args.model, "-p", PROMPT, "-n", str(TOKENS), "--temp", "0",
                      "-ngl", str(layers), "-no-cnv"])
        same = ours == theirs
        print(f"n_gpu_layers={layers}: {'identical' if same else 'DIFFERENT'}")
        print(f"  uairt:     {ours!r}")
        print(f"  llama.cpp: {theirs!r}")
        failed = failed or not same
    if failed:
        sys.exit(1)
    print("UAIRT llamacpp backend matches llama.cpp")


if __name__ == "__main__":
    main()
