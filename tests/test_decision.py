import pytest

from décision.decision import DEFAULT_WEIGHTS, DecisionEngine
from optimisation.mtwmo.algorithms import NSGAII
from optimisation.mtwmo.generator import generate_instance


@pytest.fixture(scope="module")
def pareto_set():
    instance = generate_instance("A", 2, 1, seed=1)
    return NSGAII(population_size=20, generations=5, local_search=False).run(
        instance, seed=0, gamma=0.0
    )


def test_weights_are_normalised():
    engine = DecisionEngine(weights={"cost": 2, "satisfaction": 1, "balance": 1})

    assert sum(engine.weights.values()) == pytest.approx(1.0)
    assert engine.weights["cost"] == pytest.approx(0.5)


def test_invalid_method_rejected():
    with pytest.raises(ValueError):
        DecisionEngine(method="nope")


def test_select_returns_plan_and_real_justification(pareto_set):
    selection = DecisionEngine().select_from_pareto_set(pareto_set)

    assert selection is not None
    assert selection["status"] == "ok"
    assert selection["front_size"] == pareto_set.size
    assert selection["objectives"]["Z1"] >= 0.0
    assert "Z1=" in selection["justification"]
    assert "non-dominated solutions" in selection["justification"]
    # visit plan rows are consistent with the routed assignments
    assert len(selection["visit_plan"]) == len(selection["assignments"])
    assert selection["candidate"] in pareto_set.solutions


def test_weighted_sum_and_ideal_point_both_select_a_front_member(pareto_set):
    for method in ("weighted_sum", "ideal_point"):
        selection = DecisionEngine(method=method).select_from_pareto_set(pareto_set)
        assert selection["candidate"] in pareto_set.solutions
        for key in ("cost", "satisfaction", "balance"):
            assert 0.0 <= selection["normalised_scores"][key] <= 1.0


def test_selection_is_deterministic(pareto_set):
    engine = DecisionEngine()

    first = engine.select_from_pareto_set(pareto_set)
    second = engine.select_from_pareto_set(pareto_set)

    assert first["index"] == second["index"]


def test_visit_plan_is_time_ordered(pareto_set):
    selection = DecisionEngine().select_from_pareto_set(pareto_set)

    starts = [row["service_start"] for row in selection["visit_plan"]]
    assert starts == sorted(starts)


def test_empty_pareto_set_returns_none():
    assert DecisionEngine().select_from_pareto_set([]) is None


def test_legacy_dict_front_uses_documented_rule():
    front = [
        {"summary": {"unassigned_patients": 1, "satisfaction_percent": 10, "total_cost": 5, "workload_variance": 1}},
        {"summary": {"unassigned_patients": 0, "satisfaction_percent": 20, "total_cost": 9, "workload_variance": 2}},
    ]

    chosen = DecisionEngine().choose_pareto_solution(front)

    assert chosen is front[1]  # fewer unassigned patients wins first


def test_optimizer_dict_solution_is_returned_as_assignments_and_visit_plan():
    candidate = {
        "summary": {
            "unassigned_patients": 0,
            "satisfaction_percent": 80,
            "total_cost": 10,
            "workload_variance": 2,
        },
        "solution": [
            {"patient_id": "P1", "caregiver_id": "C1", "service_start": 30},
        ],
    }

    selection = DecisionEngine().select_from_pareto_set([candidate])

    assert selection["assignments"] == [{"patient_id": "P1", "caregiver_id": "C1"}]
    assert selection["visit_plan"] == [
        {"patient_id": "P1", "caregiver_id": "C1", "service_start": 30.0}
    ]


def test_default_weights_cover_three_objectives():
    assert set(DEFAULT_WEIGHTS) == {"cost", "satisfaction", "balance"}
