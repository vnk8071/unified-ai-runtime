# SPDX-License-Identifier: Apache-2.0
"""Your own sampling loop over a session: the logits are NumPy arrays, so anything goes (here: a repetition penalty).

    python examples/python/session_loop.py model.gguf "Once upon a time" --max-tokens 40
"""
import argparse
import sys

import numpy as np

import uairt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("prompt")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--penalty", type=float, default=1.3)
    args = parser.parse_args()

    with uairt.AutoModel.from_file(args.model, device="gpu") as model, model.session() as session:
        seen = set()
        tokens = model.tokenize(args.prompt)
        session.append(tokens)
        generated = []
        sys.stdout.write(args.prompt)
        for _ in range(args.max_tokens):
            logits = session.logits().astype(np.float64)
            for token in seen:                      # push down tokens already used
                logits[token] = logits[token] / args.penalty if logits[token] > 0 else logits[token] * args.penalty
            token = int(np.argmax(logits))
            seen.add(token)
            generated.append(token)
            sys.stdout.write(model.detokenize(generated)[len(model.detokenize(generated[:-1])):])
            sys.stdout.flush()
            session.append([token])
        print()


if __name__ == "__main__":
    main()
