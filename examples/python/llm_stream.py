# SPDX-License-Identifier: Apache-2.0
"""Streams text from a GGUF model through the llamacpp backend, with sampling.

    python examples/python/llm_stream.py model.gguf "The capital of France is" --max-tokens 40 --temperature 0.8 --top-k 40 --seed 7

Needs the llamacpp plugin (cmake -DUAIRT_BUILD_LLAMACPP=ON -DBUILD_SHARED_LIBS=ON) on UAIRT_PLUGIN_PATH or next to libuairt.
Sampling happens in Python here (uairt.generate); the C library with the same options is uairt_llm_generate (examples/llm_stream.c).
"""
import argparse
import sys

import uairt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("prompt")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with uairt.AutoModel.from_file(args.model, device="gpu") as model:
        sys.stdout.write(args.prompt)
        for piece in model.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temperature,
                                    top_k=args.top_k, top_p=args.top_p, seed=args.seed):
            sys.stdout.write(piece)
            sys.stdout.flush()
        print()


if __name__ == "__main__":
    main()
