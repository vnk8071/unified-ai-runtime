# SPDX-License-Identifier: Apache-2.0
"""Text generation over a session. Sampling happens here, over the logits, so no sampler is part of the C ABI."""
import codecs
import math
from typing import Iterable, Iterator, Optional

import numpy as np

from ._core import InvalidArgument


def sample(logits, temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0,
           rng: Optional[np.random.Generator] = None) -> int:
    """One token id from `logits`. Greedy when temperature is 0; otherwise temperature, then top-k, then top-p."""
    if not math.isfinite(temperature) or temperature < 0:
        raise InvalidArgument(1, "temperature must be finite and >= 0")
    if top_k < 0:
        raise InvalidArgument(1, "top_k must be >= 0 (0 means no limit)")
    if not 0.0 < top_p <= 1.0:
        raise InvalidArgument(1, "top_p must be in (0, 1]")
    logits = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(logits).all():
        raise InvalidArgument(1, "logits contain NaN or infinity")
    if temperature == 0.0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    order = np.argsort(-scaled)
    if top_k:
        order = order[:top_k]
    probs = np.exp(scaled[order] - scaled[order[0]])
    probs /= probs.sum()
    if top_p < 1.0:
        keep = int(np.searchsorted(np.cumsum(probs), top_p)) + 1
        order, probs = order[:keep], probs[:keep]
        probs = probs / probs.sum()
    rng = rng if rng is not None else np.random.default_rng()
    return int(order[rng.choice(len(order), p=probs)])


def generate(model, prompt: str, *, max_tokens: int = 64, temperature: float = 0.0, top_k: int = 0,
             top_p: float = 1.0, seed: Optional[int] = None, stop_tokens: Iterable[int] = (),
             session=None) -> Iterator[str]:
    """Yields text pieces for `prompt`. Stops after `max_tokens` or at a token in `stop_tokens`.

    `model` is a Model or an AutoModel. A session passed in is left open (and keeps what was generated); otherwise
    one is created and closed here. Pieces are always valid text: bytes of a character split across tokens are held
    back until the character is complete. The prompt is tokenized with special-token parsing where the backend does
    that (llamacpp does): text such as `<|eot_id|>` becomes a control token, so do not pass untrusted text if that matters.
    """
    if max_tokens < 0:
        raise InvalidArgument(1, "max_tokens must be >= 0")
    if not math.isfinite(temperature) or temperature < 0 or top_k < 0 or not 0.0 < top_p <= 1.0:
        sample(np.zeros(1), temperature, top_k, top_p)  # raises the specific error
    rng = np.random.default_rng(seed)
    stop = {int(token) for token in stop_tokens}
    owns = session is None
    session = model.session() if owns else session
    try:
        session.append(model.tokenize(prompt))
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        generated, emitted = [], 0
        for _ in range(max_tokens):
            token = sample(session.logits(), temperature, top_k, top_p, rng)
            if token in stop:
                break
            generated.append(token)
            data = model.detokenize_bytes(generated)
            piece = decoder.decode(data[emitted:])
            emitted = len(data)
            if piece:
                yield piece
            if len(generated) < max_tokens:
                session.append([token])
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail
    finally:
        if owns:
            session.close()
