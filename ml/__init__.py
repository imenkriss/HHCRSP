"""D3 machine-learning pipeline (ICCCI 2025 paper).

SVM (One-vs-Rest) care-type classifier -> K-Means (k=3) on
``[care_type, X, Y]`` -> Tabu Search router per cluster, plus the
``K-Means + Tabu Search`` baseline without SVM. Benchmarks: Solomon
(25/50/100 customers) and Gehring-Homberger subsets.

Everything here is measured on the local data; no paper number is copied.
"""

from .care_type import CareTypeClassifier, train_care_type_classifier
from .clustering import cluster_points, kmeans_clusters
from .mas_integration import cluster_patients, patient_cluster_map
from .pipeline import (
    ClusterRoute,
    RoutingResult,
    compare_svm_vs_baseline,
    run_pipeline,
)
from .tabu import tabu_search, tour_cost

__all__ = [
    "CareTypeClassifier",
    "ClusterRoute",
    "RoutingResult",
    "cluster_patients",
    "cluster_points",
    "compare_svm_vs_baseline",
    "kmeans_clusters",
    "patient_cluster_map",
    "run_pipeline",
    "tabu_search",
    "tour_cost",
    "train_care_type_classifier",
]
