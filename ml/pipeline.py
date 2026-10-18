"""D3 pipeline: K-Means clustering + Tabu Search routing per cluster.

``run_pipeline(..., use_svm=True)`` clusters on ``[care_type, X, Y]`` (SVM
pipeline); ``use_svm=False`` is the ``K-Means + Tabu Search`` baseline on
``[X, Y]`` only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from .clustering import cluster_points
from .tabu import tabu_search

Point = Sequence[float]
DEFAULT_TABU = {"iterations": 100, "tabu_tenure": 10}


@dataclass
class ClusterRoute:
    cluster_id: int
    members: list[int]
    order: list[int]
    cost: float
    start_cost: float

    def as_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "members": list(self.members),
            "order": list(self.order),
            "cost": self.cost,
            "start_cost": self.start_cost,
        }


@dataclass
class RoutingResult:
    method: str
    n: int
    k: int
    use_care_type: bool
    clusters: list[ClusterRoute] = field(default_factory=list)
    total_cost: float = 0.0
    total_start_cost: float = 0.0
    runtime_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "n": self.n,
            "k": self.k,
            "use_care_type": self.use_care_type,
            "total_cost": self.total_cost,
            "total_start_cost": self.total_start_cost,
            "runtime_seconds": self.runtime_seconds,
            "clusters": [cluster.as_dict() for cluster in self.clusters],
        }


def _route_clusters(
    points: Sequence[Point],
    depot: Point,
    clustering,
    method: str,
    seed: int,
    tabu_kwargs: dict | None,
) -> RoutingResult:
    pts = list(points)
    options = dict(DEFAULT_TABU)
    if tabu_kwargs:
        options.update(tabu_kwargs)

    started = time.perf_counter()
    routes: list[ClusterRoute] = []
    for cluster_id, members in sorted(clustering.clusters.items()):
        member_points = [pts[index] for index in members]
        solution = tabu_search(
            member_points, depot=depot, seed=seed + cluster_id, **options
        )
        # ``order`` indexes into member_points; map back to global indices.
        global_order = [members[local] for local in solution["order"]]
        routes.append(
            ClusterRoute(
                cluster_id=cluster_id,
                members=list(members),
                order=global_order,
                cost=float(solution["cost"]),
                start_cost=float(solution["start_cost"]),
            )
        )
    total = float(sum(route.cost for route in routes))
    total_start = float(sum(route.start_cost for route in routes))
    return RoutingResult(
        method=method,
        n=len(pts),
        k=clustering.k,
        use_care_type=clustering.use_care_type,
        clusters=routes,
        total_cost=total,
        total_start_cost=total_start,
        runtime_seconds=time.perf_counter() - started,
    )


def run_pipeline(
    points: Sequence[Point],
    depot: Point = (0.0, 0.0),
    care_codes: Sequence[int] | None = None,
    k: int = 3,
    seed: int = 42,
    use_svm: bool = True,
    tabu_kwargs: dict | None = None,
    standardize: bool = True,
) -> RoutingResult:
    """Cluster ``points`` and route each cluster with Tabu Search."""
    clustering = cluster_points(
        points,
        care_codes=care_codes,
        k=k,
        seed=seed,
        use_care_type=use_svm,
        standardize=standardize,
    )
    method = "svm+kmeans+tabu" if use_svm else "kmeans+tabu"
    return _route_clusters(points, depot, clustering, method, seed, tabu_kwargs)


def compare_svm_vs_baseline(
    points: Sequence[Point],
    depot: Point = (0.0, 0.0),
    care_codes: Sequence[int] | None = None,
    k: int = 3,
    seed: int = 42,
    tabu_kwargs: dict | None = None,
) -> dict:
    """Run both pipelines on identical points and report the measured gap."""
    svm_result = run_pipeline(
        points, depot, care_codes, k=k, seed=seed, use_svm=True, tabu_kwargs=tabu_kwargs
    )
    baseline_result = run_pipeline(
        points,
        depot,
        care_codes,
        k=k,
        seed=seed,
        use_svm=False,
        tabu_kwargs=tabu_kwargs,
    )
    baseline = baseline_result.total_cost
    improvement = (
        (baseline - svm_result.total_cost) / baseline * 100.0 if baseline else 0.0
    )
    return {
        "n": svm_result.n,
        "k": svm_result.k,
        "svm": svm_result,
        "baseline": baseline_result,
        "svm_total_cost": svm_result.total_cost,
        "baseline_total_cost": baseline_result.total_cost,
        "improvement_percent": improvement,
    }
