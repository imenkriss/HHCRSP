"""Optional, offline text embeddings for the hybrid retrieval path.

The default RAG path does not use embeddings. ``HashingEmbedder`` is a
dependency-free bag-of-words hashing embedding so the hybrid signal remains
deterministic and works without any network or model download.
"""

from __future__ import annotations

import hashlib
import math
import re

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic hashing embedding (no external dependency)."""

    def __init__(self, dimensions: int = 256):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_PATTERN.findall((text or "").lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector
