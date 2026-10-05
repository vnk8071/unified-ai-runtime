# SPDX-License-Identifier: Apache-2.0
"""Text generation from a GGUF model.

    python llm_generate.py model.gguf "The capital of France is" --device gpu --max-tokens 40

Needs the llamacpp plugin on UAIRT_PLUGIN_PATH (or load it with uairt.load_backend_library).
"""
import argparse
import sys

import uairt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("prompt")
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with uairt.AutoModel.from_file(args.model, device=args.device) as model:
        print(f"backend={model.backend} options={model.options}", file=sys.stderr)
        sys.stdout.write(args.prompt)
        for piece in model.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temperature,
                                    top_k=args.top_k, top_p=args.top_p, seed=args.seed):
            sys.stdout.write(piece)
            sys.stdout.flush()
        print()


if __name__ == "__main__":
    main()
