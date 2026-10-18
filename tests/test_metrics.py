from types import SimpleNamespace

import pytest

from complexity.metrics import ComplexityMetrics, hypervolume, nadir_reference


# --- hypervolume (exact, deterministic) ----------------------------------

def test_hypervolume_2d_staircase():
    assert hypervolume([(1, 3), (2, 2), (3, 1)], (4, 4)) == pytest.approx(6.0)


def test_hypervolume_3d_unit_cube():
    assert hypervolume([(1, 1, 1)], (2, 2, 2)) == pytest.approx(1.0)


def test_hypervolume_3d_box():
    assert hypervolume([(1, 1, 1)], (3, 3, 3)) == pytest.approx(8.0)


def test_hypervolume_empty_and_outside_reference():
    assert hypervolume([], (1, 1, 1)) == 0.0
    assert hypervolume([(5, 5)], (4, 4)) == 0.0


def test_hypervolume_union_of_two_boxes():
    # boxes [1,4]x[3,4] and [3,4]x[1,4] against ref (4,4): 3 + 3 - 1 = 5
    assert hypervolume([(1, 3), (3, 1)], (4, 4)) == pytest.approx(5.0)


def test_nadir_reference_adds_margin():
    assert nadir_reference([(1, 5), (3, 2)], margin=1.0) == [4.0, 6.0]


# --- stage timers (no global state) --------------------------------------

def test_stage_context_manager_records_time():
    metrics = ComplexityMetrics()

    with metrics.stage("RAG"):
        pass

    recorded = metrics.get_metrics()
    assert "RAG" in recorded
    assert recorded["RAG"]["execution_time_sec"] >= 0.0


def test_instances_do_not_share_state():
    first, second = ComplexityMetrics(), ComplexityMetrics()

    with first.stage("X"):
        pass

    assert "X" in first.get_metrics()
    assert "X" not in second.get_metrics()


def test_stage_is_recorded_even_when_body_raises():
    metrics = ComplexityMetrics()

    with pytest.raises(ValueError):
        with metrics.stage("boom"):
            raise ValueError("fail")

    assert "boom" in metrics.get_metrics()


def test_total_time_sums_stages():
    metrics = ComplexityMetrics()
    metrics.add_stage_time("A", 0.5)
    metrics.add_stage_time("B", 1.25)

    assert metrics.total_time() == pytest.approx(1.75)


# --- problem size / agents / delays / quality ----------------------------

def test_problem_size_and_observed_interactions():
    metrics = ComplexityMetrics()
    patients = [{"id": "P1"}, {"id": "P2"}]
    caregivers = [{"id": "S1"}]

    metrics.problem_size(patients, caregivers)
    metrics.agent_interactions(patients, caregivers, observed_count=4)

    recorded = metrics.get_metrics()
    assert recorded["problem_size"]["possible_assignments"] == 2
    assert recorded["agent_interactions"]["observed_messages"] == 4


def test_delays_and_overtime_uses_route_details():
    metrics = ComplexityMetrics()
    route = SimpleNamespace(
        working_minutes=520.0,
        caregiver=SimpleNamespace(rated_minutes=480),
        tardiness_total=7.0,
        waiting_total=3.0,
    )
    caregivers = [{"id": "S1", "delay": 15}]

    metrics.delays_and_overtime([route], caregivers)

    recorded = metrics.get_metrics()["delays_and_overtime"]
    assert recorded["overtime_minutes"] == pytest.approx(40.0)
    assert recorded["tardiness_minutes"] == pytest.approx(7.0)
    assert recorded["caregiver_delay_total_minutes"] == pytest.approx(15.0)


def test_pareto_quality_reports_size_and_hypervolume():
    metrics = ComplexityMetrics()

    metrics.pareto_quality([(1, 1), (2, 2)], reference=(4, 4), feasible_count=2)

    recorded = metrics.get_metrics()["pareto_quality"]
    assert recorded["non_dominated_solutions"] == 2
    assert recorded["feasible_solutions"] == 2
    assert recorded["hypervolume"] == pytest.approx(9.0)


def test_balance_and_satisfaction_from_schedule():
    metrics = ComplexityMetrics()

    metrics.balance_from_schedule(0.5, [8.0, 6.0])
    metrics.satisfaction_from_schedule(12.0, served=3, total=4)

    recorded = metrics.get_metrics()
    assert recorded["workload_balance"]["workload_variance_hours"] == pytest.approx(0.5)
    assert recorded["patient_satisfaction"]["served_patients"] == 3
    assert recorded["patient_satisfaction"]["percentage"] == pytest.approx(75.0)
