"""K-Means clustering for the D3 pipeline.

Two variants:

* SVM pipeline: cluster on ``[care_type, X, Y]``.
* Baseline: cluster on ``[X, Y]`` only (no SVM care type).

ASSUMPTION (see docs/ASSUMPTIONS.md, D3): features are standardised before
K-Means so that the binary ``care_type`` column carries comparable weight to
the coordinates (raw coordinates span up to hundreds of units and would
otherwise drown the 0/1 care type). This is the documented simplest reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

Point = Sequence[float]


@dataclass
class Clustering:
    labels: np.ndarray
    clusters: dict[int, list[int]]
    k: int
    use_care_type: bool

    def sizes(self) -> dict[int, int]:
        return {cid: len(members) for cid, members in self.clusters.items()}


def build_feature_matrix(
    points: Sequence[Point],
    care_codes: Sequence[int] | None = None,
    use_care_type: bool = True,
) -> np.ndarray:
    """Assemble ``[care_type, X, Y]`` (or ``[X, Y]``) as float features."""
    coords = np.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("points must be a sequence of (x, y) pairs")
    if not use_care_type:
        return coords
    if care_codes is None:
        raise ValueError("care_codes are required when use_care_type is True")
    care = np.asarray(care_codes, dtype=float).reshape(-1, 1)
    if care.shape[0] != coords.shape[0]:
        raise ValueError("care_codes and points must have the same length")
    return np.hstack([care, coords])


def kmeans_clusters(
    features: np.ndarray, k: int = 3, seed: int = 42, standardize: bool = True
) -> np.ndarray:
    """Return a K-Means label per row (deterministic for a given seed)."""
    data = np.asarray(features, dtype=float)
    n = data.shape[0]
    effective_k = max(1, min(int(k), n))
    if standardize and n > 0:
        data = StandardScaler().fit_transform(data)
    model = KMeans(n_clusters=effective_k, n_init=10, random_state=seed)
    return model.fit_predict(data)


def cluster_points(
    points: Sequence[Point],
    care_codes: Sequence[int] | None = None,
    k: int = 3,
    seed: int = 42,
    use_care_type: bool = True,
    standardize: bool = True,
) -> Clustering:
    """Cluster ``points`` (optionally with care types) into ``k`` groups."""
    features = build_feature_matrix(points, care_codes, use_care_type)
    labels = kmeans_clusters(features, k=k, seed=seed, standardize=standardize)
    clusters: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(index)
    return Clustering(
        labels=labels,
        clusters=clusters,
        k=len(clusters),
        use_care_type=use_care_type,
    )
