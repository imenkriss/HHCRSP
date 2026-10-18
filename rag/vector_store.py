"""In-memory vector store used by the optional hybrid retrieval path (R1).

Vectors are expected to be L2-normalised, so cosine similarity reduces to a dot
product. The store is pluggable: any object exposing ``add`` and ``search`` with
the same shape can replace it.
"""

from __future__ import annotations


class InMemoryVectorStore:

    def __init__(self):
        self._vectors: dict[str, list[float]] = {}

    def add(self, key: str, vector: list[float]) -> None:
        self._vectors[key] = list(vector)

    def search(self, vector: list[float], top_k: int | None = None) -> dict[str, float]:
        scores = {
            key: sum(a * b for a, b in zip(query, vector))
            for key, query in self._vectors.items()
        }
        if top_k is not None:
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            return dict(ranked[: max(0, top_k)])
        return scores

    def __len__(self) -> int:
        return len(self._vectors)
