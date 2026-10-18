import math

import pytest

from optimisation.mtwmo.domain import Caregiver, Instance, Patient
from optimisation.mtwmo.model import (
    CostParameters,
    budget_sum,
    choose_window,
    decode,
    evaluate,
)


def toy_instance() -> Instance:
    """4 patients, 2 caregivers, unit travel speed. Z1/Z2/Z3 hand-computable."""
    patients = [
        Patient(id=1, x=1.0, y=0.0, service_time=30.0, demand_level=1, time_windows=[(10.0, 1000.0)], preferences={"K1": 2}),
        Patient(id=2, x=2.0, y=0.0, service_time=30.0, demand_level=1, time_windows=[(0.0, 1000.0)]),
        Patient(id=3, x=3.0, y=0.0, service_time=30.0, demand_level=1, time_windows=[(0.0, 1000.0)]),
        Patient(id=4, x=4.0, y=0.0, service_time=30.0, demand_level=1, time_windows=[(0.0, 5.0)]),
    ]
    caregivers = [Caregiver(id="K1", skill_level=1), Caregiver(id="K2", skill_level=1)]
    return Instance(
        depot={"id": "depot", "x": 0.0, "y": 0.0},
        laboratory={"id": "laboratory", "x": 10.0, "y": 0.0},
        caregivers=caregivers,
        patients=patients,
        horizon=(0.0, 1000.0),
        travel_speed=1.0,
        name="toy",
    )


def _full_uncertainty(instance: Instance, alpha: float = 1.0) -> dict[tuple, float]:
    nodes = ["depot", *[patient.id for patient in instance.patients], "laboratory"]
    return {(a, b): alpha for a in nodes for b in nodes if a != b}


# --- budget_sum (P5) -------------------------------------------------------

def test_budget_sum_bertsimas_sim():
    deviations = [2.0, 1.0, 0.5, -3.0]
    assert budget_sum(deviations, 0.0) == 0.0
    assert budget_sum(deviations, 1.0) == pytest.approx(2.0)
    assert budget_sum(deviations, 1.5) == pytest.approx(2.5)
    assert budget_sum(deviations, 3.0) == pytest.approx(3.5)
    assert budget_sum(deviations, 5.0) == pytest.approx(3.5)
    assert budget_sum([], 2.0) == 0.0


# --- decoding (P8) ---------------------------------------------------------

PERMUTATION = [5, 1, 2, 6, 3, 4]


def test_decode_splits_routes_at_separators():
    instance = toy_instance()

    solution = decode(PERMUTATION, instance)

    assert [route.caregiver.id for route in solution.routes] == ["K1", "K2"]
    assert [patient.id for patient in solution.routes[0].patients] == [1, 2]
    assert [patient.id for patient in solution.routes[1].patients] == [3, 4]
    assert solution.unassigned == []


def test_decode_reuses_one_route_per_caregiver():
    instance = toy_instance()

    solution = decode([5, 1, 5, 2, 6, 3, 4], instance)

    assert len(solution.routes) == 2
    assert [patient.id for patient in solution.routes[0].patients] == [1, 2]


def test_decode_leading_patients_are_unassigned():
    instance = toy_instance()

    solution = decode([1, 5, 2, 6, 3, 4], instance)

    assert [patient.id for patient in solution.unassigned] == [1]
    assert [patient.id for patient in solution.routes[0].patients] == [2]


def test_decode_missing_patients_are_unassigned():
    instance = toy_instance()

    solution = decode([5, 1, 6, 3], instance)

    assert {patient.id for patient in solution.unassigned} == {2, 4}


def test_decode_rejects_unknown_gene():
    instance = toy_instance()

    with pytest.raises(ValueError):
        decode([7, 1], instance)


# --- soft windows (P3) -----------------------------------------------------

def test_choose_window_prefers_tardiness_then_waiting():
    window, waiting, tardiness = choose_window(5.0, [(0.0, 10.0)])
    assert window == (0.0, 10.0)
    assert waiting == 0.0 and tardiness == 0.0

    window, waiting, tardiness = choose_window(1.0, [(10.0, 1000.0)])
    assert waiting == 9.0 and tardiness == 0.0

    window, waiting, tardiness = choose_window(40.0, [(0.0, 5.0), (45.0, 90.0)])
    assert window == (45.0, 90.0)
    assert waiting == 5.0 and tardiness == 0.0


def test_choose_window_none_when_no_windows():
    assert choose_window(3.0, []) == (None, 0.0, 0.0)


# --- hand-computed objectives (P6) ----------------------------------------

def test_hand_computed_objectives():
    instance = toy_instance()
    costs = CostParameters(fixed_rate=5.0)  # 2 caregivers used -> fixed = 10

    result = evaluate(PERMUTATION, instance, costs=costs)

    breakdown = result.costs
    assert breakdown.service == pytest.approx(120.0)
    assert breakdown.waiting == pytest.approx(9.0)
    assert breakdown.overtime == pytest.approx(0.0)
    assert breakdown.travel == pytest.approx(20.0)
    assert breakdown.fixed == pytest.approx(10.0)
    assert breakdown.tardiness == pytest.approx(29.0)
    assert breakdown.total == pytest.approx(188.0)

    assert result.z2 == pytest.approx(6.0)
    assert result.z3 == pytest.approx(0.005625)
    assert result.violations.total == 0.0
    assert result.feasible

    per_route = {detail.caregiver.id: detail for detail in result.routes}
    assert per_route["K1"].return_time == pytest.approx(79.0)
    assert per_route["K1"].working_minutes == pytest.approx(79.0)
    assert per_route["K2"].return_time == pytest.approx(70.0)
    assert per_route["K2"].working_minutes == pytest.approx(70.0)


def test_evaluation_vector_and_accepts_decoded_solution():
    instance = toy_instance()
    decoded = decode(PERMUTATION, instance)

    from_permutation = evaluate(PERMUTATION, instance)
    from_solution = evaluate(decoded, instance)

    assert from_permutation.costs == from_solution.costs
    assert from_permutation.objectives == [from_permutation.z1, -from_permutation.z2, from_permutation.z3]


# --- robust budgets (P5) ---------------------------------------------------

def test_gamma_zero_equals_deterministic():
    instance = toy_instance()
    instance.travel_uncertainty = _full_uncertainty(instance)
    for caregiver in instance.caregivers:
        caregiver.departure_uncertainty = 1.0
    for patient in instance.patients:
        patient.demand_uncertainty = 1.0

    deterministic = evaluate(PERMUTATION, instance)
    robust_zero = evaluate(PERMUTATION, instance, gamma=0.0)
    robust_mapping_zero = evaluate(PERMUTATION, instance, gamma={"t": 0.0, "s": 0.0, "p": 0.0})

    assert robust_zero.costs == deterministic.costs
    assert robust_mapping_zero.costs == deterministic.costs
    assert robust_zero.z2 == deterministic.z2
    assert robust_zero.z3 == deterministic.z3


def test_travel_budget_adds_worst_case_distance_margin():
    instance = toy_instance()
    instance.travel_uncertainty = _full_uncertainty(instance)
    travel_only = {"t": 1.0, "s": 0.0, "p": 0.0}

    deterministic = evaluate(PERMUTATION, instance).costs.travel
    robust = evaluate(PERMUTATION, instance, gamma=travel_only).costs.travel
    larger = evaluate(PERMUTATION, instance, gamma={"t": 4.0, "s": 0.0, "p": 0.0}).costs.travel

    assert deterministic == pytest.approx(20.0)
    assert robust == pytest.approx(22.8)
    assert larger == pytest.approx(24.0)
    assert deterministic <= robust <= larger


def test_departure_budget_increases_working_time():
    instance = toy_instance()
    for caregiver in instance.caregivers:
        caregiver.departure_uncertainty = 1.0

    deterministic = evaluate(PERMUTATION, instance)
    robust = evaluate(PERMUTATION, instance, gamma={"t": 0.0, "s": 1.0, "p": 0.0})

    assert robust.costs.waiting <= deterministic.costs.waiting
    assert robust.costs.tardiness >= deterministic.costs.tardiness
    assert robust.routes[0].working_minutes == pytest.approx(70.0)
    assert robust.routes[1].working_minutes == pytest.approx(70.0)
    # A departure delay shifts the whole day, so working hours fall to the pure
    # travel+service time; both routes equalise and the balance cannot worsen.
    assert robust.z3 <= deterministic.z3


# --- constraint violations (P7) -------------------------------------------

def test_duplicate_patient_is_single_visit_violation():
    instance = toy_instance()

    result = evaluate([5, 1, 1, 6, 3, 4], instance)

    assert result.violations.single_visit == pytest.approx(1.0)
    assert not result.feasible


def test_skill_violation_when_caregiver_underqualified():
    patient = Patient(id=1, x=1.0, y=0.0, service_time=30.0, demand_level=3, time_windows=[(0.0, 1000.0)])
    caregiver = Caregiver(id="K1", skill_level=1)
    instance = Instance(
        depot={"id": "depot", "x": 0.0, "y": 0.0},
        laboratory={"id": "laboratory", "x": 5.0, "y": 0.0},
        caregivers=[caregiver],
        patients=[patient],
        horizon=(0.0, 1000.0),
        travel_speed=1.0,
    )

    result = evaluate([2, 1], instance)

    assert result.violations.skill == pytest.approx(1.0)
    assert result.routes[0].service_minutes[0] == pytest.approx(90.0)


def test_rejection_causes_incompatibility_violation():
    instance = toy_instance()
    instance.patients[0].rejections = {"K1": True}

    result = evaluate(PERMUTATION, instance)

    assert result.violations.incompatibility == pytest.approx(1.0)


def test_overwork_beyond_hard_limit_is_flagged():
    instance = toy_instance()
    instance.caregivers[0].rated_minutes = 50
    instance.caregivers[0].max_overtime_minutes = 10
    instance.caregivers[1].rated_minutes = 50
    instance.caregivers[1].max_overtime_minutes = 10

    result = evaluate(PERMUTATION, instance)

    # K1 works 79 min (limit 60) and K2 70 min -> 19 + 10 = 29.
    assert result.violations.workload == pytest.approx(29.0)


def test_lab_closing_violation():
    instance = toy_instance()
    instance.lab_closing_time = 60.0

    result = evaluate(PERMUTATION, instance)

    assert result.violations.lab_closing == pytest.approx(29.0)


def test_missing_window_is_counted_as_violation():
    instance = toy_instance()
    instance.patients[0].time_windows = []

    result = evaluate(PERMUTATION, instance)

    assert result.violations.window == pytest.approx(1.0)
