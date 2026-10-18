from __future__ import annotations

from copy import deepcopy
import random

import pytest

from optimisation.validators import (
    SATISFIED, UNVERIFIABLE, VIOLATED, assignment_to_routes, validate_solution,
)


CONSTRAINT_IDS = (
    "(5)", "(6)", "(7)", "(8)", "(9)", "(11)", "(12)", "(13)",
    "(14)", "(15)", "(16)", "(17)", "(18)", "(19)", "(20)", "(21)",
    "overtime_240", "time_window_once",
)


def _seeded_valid_case(seed: int = 2026):
    rng = random.Random(seed)
    service = rng.choice([20, 30])
    patient = {
        "id": 1,
        "care_type": "General",
        "required_skill_level": 1,
        "service_minutes": service,
        "time_windows": [(0, 180)],
    }
    caregiver = {
        "id": "K0",
        "skill": "General",
        "available": True,
        "delay": 0,
        "current_workload": 0,
        "skill_level": rng.choice([1, 2]),
        "max_work_hours": 8,
    }
    instance = {
        "patients": [patient],
        "caregivers": [caregiver],
        "depot_id": 0,
        "laboratory_id": 2,
        "travel_times": {(0, 1): 10, (1, 2): 15},
        "lab_closing_time": 480,
        "T": 480,
        "time_unit": "minutes",
    }
    start = 25
    returned = start + service + 15
    solution = {
        "assignments": [{
            "patient_id": 1,
            "caregiver_id": "K0",
            "service_start": start,
            "service_minutes": service,
            "window_index": 0,
            "time_window": (0, 180),
        }],
        "routes": {"K0": [0, 1, 2]},
        "departure_times": {"K0": 0},
        "return_times": {"K0": returned},
        "variable_domains": {
            17: [{"kind": "binary", "values": [1]}],
            18: [{"kind": "binary", "values": [1]}],
            19: [{"kind": "binary", "values": [1]}],
            20: [{"kind": "binary", "values": [0]}],
            21: [{"kind": "integer", "values": [2]}],
        },
    }
    return instance, solution


def _break(constraint_id, instance, solution):
    if constraint_id == "(5)":
        solution["routes"]["K0"] = [0, 1, 1, 2]
    elif constraint_id == "(6)":
        solution["routes"]["K0"] = [0, 2]
    elif constraint_id == "(7)":
        solution["routes"]["K0"][0] = 1
    elif constraint_id == "(8)":
        solution["routes"]["K0"][-1] = 1
    elif constraint_id == "(9)":
        solution["assignments"][0]["service_start"] = 5
    elif constraint_id == "(11)":
        solution["routes"]["K0"] = [0, 1, 1, 2]
    elif constraint_id == "(12)":
        solution["assignments"][0]["service_start"] = 181
    elif constraint_id == "(13)":
        instance["T"] = 30
    elif constraint_id == "(14)":
        instance["lab_closing_time"] = 20
    elif constraint_id == "(15)":
        instance["caregivers"][0]["skill_level"] = 0
    elif constraint_id == "(16)":
        solution["departure_times"]["K0"] = -1
    elif constraint_id == "(17)":
        solution["variable_domains"][17][0]["values"] = [2]
    elif constraint_id == "(18)":
        solution["variable_domains"][18][0]["values"] = [0.5]
    elif constraint_id == "(19)":
        solution["variable_domains"][19][0]["values"] = [2]
    elif constraint_id == "(20)":
        solution["variable_domains"][20][0]["values"] = [0.5]
    elif constraint_id == "(21)":
        solution["variable_domains"][21][0]["values"] = [1.5]
    elif constraint_id == "overtime_240":
        solution["return_times"]["K0"] = 800
    elif constraint_id == "time_window_once":
        second = deepcopy(solution["assignments"][0])
        second["window_index"] = 0
        solution["assignments"].append(second)
    else:
        raise AssertionError(constraint_id)


@pytest.mark.parametrize("constraint_id", CONSTRAINT_IDS)
def test_each_constraint_accepts_seeded_valid_case_and_rejects_broken_case(constraint_id):
    instance, solution = _seeded_valid_case()
    assert all(status == SATISFIED for _, status, _ in validate_solution(solution, instance))

    broken_instance, broken_solution = deepcopy(instance), deepcopy(solution)
    _break(constraint_id, broken_instance, broken_solution)
    results = {item_id: status for item_id, status, _ in validate_solution(broken_solution, broken_instance)}
    assert results[constraint_id] == VIOLATED


def test_seeded_candidate_routes_match_assignment_fallback_without_a_chromosome():
    """Candidate route metadata matches the current no-permutation fallback."""
    from optimisation.nsga2 import NSGA2

    instance, _ = _seeded_valid_case()
    optimizer = NSGA2(instance["patients"], instance["caregivers"], seed=2026)
    candidates = optimizer.optimize()
    assert candidates
    for candidate in candidates:
        assert isinstance(candidate.get("routes"), dict)
        assert isinstance(candidate.get("unassigned_patients"), list)
        assert "chromosome" not in candidate
        expected_routes, expected_unassigned = assignment_to_routes(candidate["solution"], instance)
        assert candidate["routes"] == expected_routes
        assert candidate["unassigned_patients"] == expected_unassigned
        statuses = {item_id: status for item_id, status, _ in validate_solution(candidate, instance)}
        assert all(statuses[check_id] == SATISFIED for check_id in ("(5)", "(6)", "(7)", "(8)", "(11)"))
        assert all(status in {SATISFIED, VIOLATED, UNVERIFIABLE} for status in statuses.values())


def _status(rows, check_id):
    return next(status for item_id, status, _ in rows if item_id == check_id)


def test_missing_route_is_unverifiable_not_violated():
    instance, solution = _seeded_valid_case()
    del solution["routes"]
    assert _status(validate_solution(solution, instance), "(7)") == UNVERIFIABLE


def test_broken_route_is_violated():
    instance, solution = _seeded_valid_case()
    solution["routes"]["K0"][0] = 1
    assert _status(validate_solution(solution, instance), "(7)") == VIOLATED


def test_valid_route_is_satisfied():
    instance, solution = _seeded_valid_case()
    assert _status(validate_solution(solution, instance), "(7)") == SATISFIED


@pytest.mark.parametrize("check_id", ["(6)", "(7)", "(8)", "(11)"])
def test_valid_routes_satisfy_route_constraints(check_id):
    instance, solution = _seeded_valid_case()
    status = _status(validate_solution(solution, instance), check_id)
    assert status == SATISFIED


@pytest.mark.parametrize("check_id", ["(5)", "(6)", "(7)", "(8)", "(11)"])
def test_absent_routes_are_unverifiable_for_route_constraints(check_id):
    instance, solution = _seeded_valid_case()
    del solution["routes"]
    status = _status(validate_solution(solution, instance), check_id)
    assert status == UNVERIFIABLE


def test_route_visiting_patient_twice_violates_constraint_5():
    instance, solution = _seeded_valid_case()
    solution["routes"]["K0"] = [0, 1, 1, 2]
    assert _status(validate_solution(solution, instance), "(5)") == VIOLATED


def test_assignment_to_routes_orders_windows_and_reports_unassigned():
    instance, _ = _seeded_valid_case()
    instance["patients"] = [
        {"id": 1, "time_windows": [(60, 100)]},
        {"id": 2, "time_windows": [(15, 50)]},
        {"id": 3},
        {"id": 4, "time_windows": [(0, 30)]},
    ]
    assignment = [
        {"patient_id": 1, "caregiver_id": "K0"},
        {"patient_id": 2, "caregiver_id": "K0"},
        {"patient_id": 3, "caregiver_id": "K0"},
        {"patient_id": 4, "caregiver_id": None},
    ]
    routes, unassigned = assignment_to_routes(assignment, instance)
    assert routes == {"K0": [2, 1, 3]}
    assert unassigned == [4]


def test_assignment_to_routes_preserves_existing_order_without_windows():
    instance, _ = _seeded_valid_case()
    instance["patients"] = [{"id": 1}, {"id": 2}, {"id": 3}]
    assignment = [
        {"patient_id": 3, "caregiver_id": "K0"},
        {"patient_id": 1, "caregiver_id": "K0"},
        {"patient_id": 2, "caregiver_id": "K0"},
    ]
    routes, unassigned = assignment_to_routes(assignment, instance)
    assert routes == {"K0": [3, 1, 2]}
    assert unassigned == []
