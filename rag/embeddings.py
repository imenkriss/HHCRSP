"""Optional, offline text embeddings for the hybrid retrieval path.

The default RAG path does not use embeddings. ``HashingEmbedder`` is a
dependency-free bag-of-words hashing embedding so the hybrid signal remains
deterministic and works without any network or model download.

``EmbeddingCache`` / ``CachedEmbedder`` persist embeddings to a JSON file so a
re-run only embeds documents that are new or changed (R1: incremental, cached
embedding generation; no re-embedding of an unchanged knowledge base).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
logger = logging.getLogger(__name__)


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


class EmbeddingCache:
    """JSON-backed embedding cache keyed by a content hash.

    The file is written atomically through :func:`data.integrity.save_json_atomic`,
    so an interrupted run never leaves a truncated cache. A corrupt cache is
    ignored (and rebuilt) instead of crashing the RAG stage.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._vectors: dict[str, list[float]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            logger.warning("Ignoring unreadable embedding cache %s (%s)", self.path, error)
            return
        vectors = payload.get("vectors") if isinstance(payload, dict) else None
        if not isinstance(vectors, dict):
            logger.warning("Ignoring malformed embedding cache %s", self.path)
            return
        self._vectors = {
            str(key): [float(value) for value in vector]
            for key, vector in vectors.items()
            if isinstance(vector, list)
        }

    def get(self, key: str) -> list[float] | None:
        return self._vectors.get(key)

    def put(self, key: str, vector: list[float]) -> None:
        self._vectors[key] = list(vector)
        self._dirty = True

    def flush(self) -> Path | None:
        if not self._dirty:
            return self.path if self.path.exists() else None
        from data.integrity import save_json_atomic

        save_json_atomic(
            {"version": 1, "vectors": self._vectors},
            self.path,
        )
        self._dirty = False
        return self.path

    def __len__(self) -> int:
        return len(self._vectors)


class CachedEmbedder:
    """Wrap an embedder with a persistent, incremental cache.

    ``cache_dir`` defaults to ``data/generated/embeddings``; nothing is written
    under ``data/`` root. ``stats`` reports cache hits/misses so the number of
    new embeddings is observable and testable.
    """

    def __init__(
        self,
        embedder: HashingEmbedder | None = None,
        cache_dir: str | Path | None = None,
    ):
        from data.integrity import GENERATED_DIR

        self.embedder = embedder or HashingEmbedder()
        self.dimensions = getattr(self.embedder, "dimensions", None)
        name = f"{type(self.embedder).__name__.lower()}-{self.dimensions or 'na'}"
        directory = Path(cache_dir) if cache_dir is not None else GENERATED_DIR / "embeddings"
        self.cache = EmbeddingCache(directory / f"{name}.json")
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        namespace = f"{type(self.embedder).__name__}:{self.dimensions}"
        return hashlib.sha256(f"{namespace}\u0000{text}".encode("utf-8")).hexdigest()

    def embed(self, text: str) -> list[float]:
        key = self._key(text or "")
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return list(cached)
        self.misses += 1
        vector = self.embedder.embed(text)
        self.cache.put(key, vector)
        return vector

    def flush(self) -> Path | None:
        return self.cache.flush()

    @property
    def stats(self) -> dict[str, int | str]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "cached_vectors": len(self.cache),
            "cache_path": str(self.cache.path),
        }
