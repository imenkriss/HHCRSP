import pytest

from agents.agentOrchestrateur import Orchestrator
from ml.mas_integration import cluster_patients, patient_cluster_map


@pytest.fixture
def patients():
    return [
        {"id": f"P{i}", "x": float(i % 4) * 10.0, "y": float(i // 4) * 10.0,
         "care_type": "type_0" if i % 2 == 0 else "type_1"}
        for i in range(12)
    ]


def test_cluster_patients_covers_every_patient(patients):
    grouped = cluster_patients(patients, k=3, seed=11)

    ids = sorted(pid for members in grouped.values() for pid in members)
    assert ids == sorted(p["id"] for p in patients)
    assert 1 <= len(grouped) <= 3


def test_patient_cluster_map_is_inverse(patients):
    grouped = cluster_patients(patients, k=3, seed=11)
    mapping = patient_cluster_map(patients, k=3, seed=11)

    for cluster_id, members in grouped.items():
        for patient_id in members:
            assert mapping[patient_id] == cluster_id


def test_orchestrator_accepts_preclusters(patients):
    grouped = cluster_patients(patients, k=3, seed=11)
    caregivers = [
        {"id": "S1", "skill": "Cardio", "skill_level": 3, "available": True,
         "current_workload": 0, "max_work_hours": 8, "delay": 0}
    ]
    orchestrator = Orchestrator(patients, caregivers, grouped)

    assert orchestrator.patient_clusters == grouped
    grouped_patients = orchestrator.grouped_patients()
    assert set(grouped_patients) == set(grouped)
    for cluster_id in grouped:
        assert [p["id"] for p in grouped_patients[cluster_id]] == grouped[cluster_id]


def test_set_patient_clusters_updates_map(patients):
    caregivers = [
        {"id": "S1", "skill": "Cardio", "skill_level": 3, "available": True,
         "current_workload": 0, "max_work_hours": 8, "delay": 0}
    ]
    orchestrator = Orchestrator(patients, caregivers)
    assert orchestrator.patient_clusters == {}

    orchestrator.set_patient_clusters({0: ["P0", "P1"], 1: ["P2"]})
    assert orchestrator.patient_cluster_of == {"P0": 0, "P1": 0, "P2": 1}
    assert [p["id"] for p in orchestrator.patients_in_cluster(0)] == ["P0", "P1"]
