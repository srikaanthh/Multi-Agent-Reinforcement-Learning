from __future__ import annotations

from typing import Iterable
import hashlib
import math
import re

import numpy as np


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _stable_hash(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def embed_text(text: str, dim: int = 32) -> np.ndarray:
    """Cheap deterministic text embedding for attention and contextual-bandit state features."""
    if dim <= 0:
        raise ValueError("dim must be positive.")

    tokens = [token.lower() for token in _TOKEN_RE.findall(text or "")]
    vector = np.zeros(dim, dtype=float)
    if not tokens:
        return vector

    for token in tokens:
        index = _stable_hash(token) % dim
        vector[index] += 1.0

    # Add simple style features so terse vs verbose responses are distinguishable.
    vector[0] += min(len(tokens) / 50.0, 2.0)
    vector[1] += min(len(text) / 400.0, 2.0)
    vector[2] += sum(token.isdigit() for token in tokens) / max(1, len(tokens))

    norm = math.sqrt(float(np.dot(vector, vector)))
    if norm > 0:
        vector /= norm
    return vector


def summarize_texts(texts: Iterable[str], *, dim: int = 32) -> np.ndarray:
    vectors = [embed_text(text, dim=dim) for text in texts]
    if not vectors:
        return np.zeros(dim, dtype=float)
    stacked = np.vstack(vectors)
    mean = stacked.mean(axis=0)
    norm = math.sqrt(float(np.dot(mean, mean)))
    if norm > 0:
        mean /= norm
    return mean
