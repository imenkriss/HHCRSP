"""Optional pre-clustering step for the multi-agent system (A1/A3).

Patients are grouped (K-Means on ``[care_type, X, Y]``) before the
orchestrator assigns them to caregivers. Grouping is a hint: the orchestrator
still enforces skills/availability, so a cluster never overrides feasibility.
"""

from __future__ import annotations

from typing import Sequence

from .care_type import LABEL_TO_CODE
from .clustering import cluster_points


def _patient_id(patient: dict, index: int) -> str:
    return str(patient.get("id", index))


def _care_code(patient: dict) -> int:
    if "care_type_code" in patient and patient["care_type_code"] is not None:
        return int(patient["care_type_code"])
    label = patient.get("care_type")
    return LABEL_TO_CODE.get(label, 0)


def cluster_patients(
    patients: Sequence[dict],
    k: int = 3,
    seed: int = 42,
    use_care_type: bool = True,
) -> dict[int, list[str]]:
    """Return ``cluster_id -> [patient_id, ...]`` for pre-assignment grouping."""
    records = list(patients)
    if not records:
        return {}
    points = [(float(p.get("x", 0.0)), float(p.get("y", 0.0))) for p in records]
    care = [_care_code(p) for p in records]
    clustering = cluster_points(
        points, care_codes=care, k=k, seed=seed, use_care_type=use_care_type
    )
    grouped: dict[int, list[str]] = {}
    for index, label in enumerate(clustering.labels):
        grouped.setdefault(int(label), []).append(_patient_id(records[index], index))
    return grouped


def patient_cluster_map(
    patients: Sequence[dict],
    k: int = 3,
    seed: int = 42,
    use_care_type: bool = True,
) -> dict[str, int]:
    """Return ``patient_id -> cluster_id`` (inverse of :func:`cluster_patients`)."""
    grouped = cluster_patients(patients, k=k, seed=seed, use_care_type=use_care_type)
    return {
        patient_id: cluster_id
        for cluster_id, members in grouped.items()
        for patient_id in members
    }
