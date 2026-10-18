"""Stress-test raw optimizer assignments before any repair.

NSGA2 currently has create_solution() only: there is no permutation chromosome,
crossover, mutation, or repair operator. Routes below are decoded only through
assignment_to_routes so that the validator receives the candidate's explicit
route representation; this conversion does not repair assignments.
"""

from __future__ import annotations

import random
from collections import Counter

from optimisation.nsga2 import NSGA2
from optimisation.validators import CHECKS, assignment_to_routes, validate_solution


SEEDS = (11, 22, 33)


def _instance(seed: int, patient_count: int = 10, caregiver_count: int = 2) -> dict:
    rng = random.Random(seed)
    patients = [
        {
            "id": f"P{i}",
            "care_type": "General",
            "required_skill_level": rng.choice((1, 2)),
            "service_duration": rng.choice((30, 60)),
            "time_windows": [(0, 360)],
            "priority": rng.choice(("Critical", "High", "Medium")),
        }
        for i in range(1, patient_count + 1)
    ]
    caregivers = [
        {
            "id": f"K{k}",
            "skill": "General",
            "skill_level": 2,
            "available": True,
            "delay": 0,
            "current_workload": 0,
            "max_work_hours": 8,
        }
        for k in range(caregiver_count)
    ]
    return {
        "patients": patients,
        "caregivers": caregivers,
        "depot_id": 0,
        "laboratory_id": patient_count + 1,
    }


def stress_seed(seed: int, candidates_per_seed: int = 20) -> dict:
    instance = _instance(seed)
    optimizer = NSGA2(
        instance["patients"],
        instance["caregivers"],
        seed=seed,
        population_size=candidates_per_seed,
        generations=0,
    )
    counts = {constraint_id: Counter() for constraint_id, _ in CHECKS}
    unassigned_counts = []
    for _ in range(candidates_per_seed):
        raw_assignment = optimizer.create_solution()
        routes, unassigned = assignment_to_routes(raw_assignment, instance)
        candidate = {
            "solution": raw_assignment,
            "routes": routes,
            "unassigned_patients": unassigned,
        }
        unassigned_counts.append(len(unassigned))
        statuses = validate_solution(candidate, instance)
        for constraint_id, status, _ in statuses:
            counts[constraint_id][status] += 1
    return {
        "seed": seed,
        "candidates": candidates_per_seed,
        "counts": counts,
        "unassigned_counts": unassigned_counts,
    }


def stress_report(seeds: tuple[int, ...] = SEEDS, candidates_per_seed: int = 20) -> list[dict]:
    return [stress_seed(seed, candidates_per_seed) for seed in seeds]


def test_raw_assignments_are_reported_without_repair():
    reports = stress_report(candidates_per_seed=3)
    assert len(reports) == 3
    for report in reports:
        assert len(report["unassigned_counts"]) == report["candidates"]
        assert all(count >= 0 for count in report["unassigned_counts"])
        for counts in report["counts"].values():
            assert sum(counts.values()) == report["candidates"]


if __name__ == "__main__":
    reports = stress_report()
    print("check | seed | SATISFIED | VIOLATED | UNVERIFIABLE")
    for report in reports:
        for check_id, _ in CHECKS:
            counts = report["counts"][check_id]
            print(
                f"{check_id} | {report['seed']} | "
                f"{counts['SATISFIED']} | {counts['VIOLATED']} | "
                f"{counts['UNVERIFIABLE']}"
            )
        print(
            f"seed {report['seed']} per-candidate unassigned counts: "
            f"{report['unassigned_counts']}"
        )
