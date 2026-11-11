"""Test B — the comparison table must support multiple rows and stay exportable."""

from __future__ import annotations

import json

import pytest

from experiments import comparison


def _legacy_candidate(caregiver_id: str, cost: float, satisfaction: float,
                      variance: float, unassigned: int) -> dict:
    return {
        "solution": [
            {"patient_id": "P1", "caregiver_id": caregiver_id, "service_hours": 0.5}
        ],
        "objectives": [cost, unassigned, -satisfaction, variance],
        "summary": {
            "total_cost": cost,
            "service_cost": cost * 0.4,
            "waiting_cost": cost * 0.1,
            "overtime_cost": 0.0,
            "travel_cost": cost * 0.3,
            "fixed_cost": cost * 0.2,
            "delay_penalty": 0.0,
            "served_patients": 1 - unassigned,
            "unassigned_patients": unassigned,
            "satisfaction_percent": satisfaction,
            "workload_variance": variance,
        },
        "routes": {},
    }


def test_three_test_records_become_three_rows():
    front = [
        _legacy_candidate("S1", 100.0, 80.0, 1.0, 0),
        _legacy_candidate("S2", 120.0, 90.0, 0.5, 0),
        _legacy_candidate("S3", 90.0, 60.0, 2.0, 2),
    ]

    rows = comparison.solution_table(front)

    assert len(rows) == 3
    assert [row["item_id"] for row in rows] == ["S01", "S02", "S03"]
    assert {row["item_id"] for row in rows} == {"S01", "S02", "S03"}
    # Coverage first, then satisfaction.
    assert rows[0]["unassigned_patients"] == 0
    assert rows[0]["z2_satisfaction"] == 90.0
    assert rows[-1]["unassigned_patients"] == 2
    # Consistent columns and types across rows.
    keys = set(rows[0])
    assert all(set(row) == keys for row in rows)
    assert all(isinstance(row["z1_total_cost"], float) for row in rows)
    assert all(isinstance(row["feasible"], bool) for row in rows)


def test_empty_front_is_empty_not_single_row():
    assert comparison.solution_table([]) == []
    assert comparison.solution_table(None) == []


def test_configurations_table_keeps_distinct_runs():
    def result(instance: str, algorithm: str, cost: float, patients: int) -> dict:
        return {
            "instance_info": {"name": instance},
            "patients": [{"id": f"P{i}"} for i in range(patients)],
            "soignants": [{"id": "S1"}],
            "optimization_analysis": {
                "algorithm": algorithm, "population_size": 6, "generations": 1
            },
            "selected_optimization": {"objectives": {"Z1": cost, "Z2": 10.0, "Z3": 0.1}},
            "llm_model": "qwen3:4b",
            "llm_output": {"reasoning_source": "fallback"},
            "retrieved_docs": [{"id": "R001"}],
            "final_decision": {"status": "assigned", "method": "weighted_sum"},
            "optimization": [1, 2, 3],
            "complexity": {
                "Total Pipeline": {"execution_time_sec": 1.5},
                "resources": {"peak_python_mb": 12.0, "peak_rss_mb": 90.0},
            },
            "dataset_info": {"source": "generator"},
        }

    rows = comparison.configurations_table([
        result("A_2_1", "NSGA-II", 100.0, 10),
        result("B_5_1", "NSGA-II", 200.0, 30),
        result("A_2_1", "MTCMOLite", 110.0, 10),
    ])

    assert len(rows) == 3
    assert len({row["item_id"] for row in rows}) == 3
    assert [row["instance"] for row in rows] == ["A_2_1", "B_5_1", "A_2_1"]
    assert all(set(row) == set(rows[0]) for row in rows)


def test_measured_algorithm_comparison_has_two_rows():
    outcome = comparison.algorithm_comparison(
        "A", 2, 1, seed=1, population_size=8, generations=3, runs=1
    )

    assert outcome["instance"]["name"] == "A_2_1"
    assert [row["algorithm"] for row in outcome["rows"]] == ["NSGA-II", "MTCMOLite"]
    assert len(outcome["runs"]) == 2
    assert all(row["instance"] == "A_2_1" for row in outcome["rows"])
    assert all(row["runs"] == 1 for row in outcome["rows"])
    # No fabricated numbers: each row carries a measured feasible count.
    assert all(isinstance(row["feasible_runs"], int) for row in outcome["rows"])


def test_invalid_algorithm_is_rejected():
    with pytest.raises(ValueError):
        comparison.algorithm_comparison("A", 2, 1, algorithms=("NOT-AN-ALGORITHM",))


def test_ml_baseline_reports_missing_file(tmp_path):
    status = comparison.ml_baseline_table(tmp_path / "absent.csv")["status"]
    assert status["available"] is False
    assert "ml.evaluation" in status["message"]


def test_sort_filter_and_paginate():
    rows = comparison.solution_table([
        _legacy_candidate("S1", 100.0, 80.0, 1.0, 0),
        _legacy_candidate("S2", 120.0, 90.0, 0.5, 0),
        _legacy_candidate("S3", 90.0, 60.0, 2.0, 2),
    ])
    ascending = comparison.sort_rows(rows, "z1_total_cost", ascending=True)
    assert [row["z1_total_cost"] for row in ascending] == [90.0, 100.0, 120.0]
    descending = comparison.sort_rows(rows, "z1_total_cost", ascending=False)
    assert [row["z1_total_cost"] for row in descending] == [120.0, 100.0, 90.0]

    filtered = comparison.filter_rows(rows, "s02")
    assert [row["item_id"] for row in filtered] == ["S02"]

    first_page = comparison.paginate(rows, page=1, page_size=2)
    second_page = comparison.paginate(rows, page=2, page_size=2)
    assert first_page["total"] == 3 and first_page["pages"] == 2
    assert len(first_page["rows"]) == 2 and len(second_page["rows"]) == 1
    assert (first_page["start"], first_page["end"]) == (1, 2)


def test_export_csv_and_json_keep_all_rows():
    rows = comparison.solution_table([
        _legacy_candidate("S1", 100.0, 80.0, 1.0, 0),
        _legacy_candidate("S2", 120.0, 90.0, 0.5, 0),
        _legacy_candidate("S3", 90.0, 60.0, 2.0, 2),
    ])
    csv_text = comparison.export_csv(rows)
    json_rows = json.loads(comparison.export_json(rows))
    assert len(csv_text.strip().splitlines()) == 4  # header + 3 rows
    assert len(json_rows) == 3
    assert json_rows[0]["item_id"] == "S01"
