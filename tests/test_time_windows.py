import pytest

from optimisation.nsga2 import NSGA2


@pytest.mark.parametrize(
    "windows",
    [
        [(10, 80)],
        [(10, 80), (100, 180)],
        [(10, 80), (100, 180), (200, 280)],
    ],
    ids=["one-window", "two-windows", "three-windows"],
)
def test_patient_window_count_is_preserved(windows):
    patient = {"time_windows": windows}

    assert NSGA2._time_windows(patient) == windows


def test_patients_may_have_different_window_counts():
    patients = [
        {"time_windows": [(10, 80)]},
        {"time_windows": [(10, 80), (100, 180)]},
        {"time_windows": [(10, 80), (100, 180), (200, 280)]},
    ]

    assert [len(NSGA2._time_windows(patient)) for patient in patients] == [1, 2, 3]


def _optimizer_for_window_costs():
    patient = {
        "id": "P1",
        "care_type": "General",
        "required_skill_level": 1,
        "service_hours": 1,
        "service_hours_min": 1,
        "service_hours_max": 1,
        "travel_time": 0,
        "travel_time_min": 0,
        "travel_time_max": 0,
        "time_windows": [(10, 30)],
        "waiting_cost_rate": 2,
        "delay_penalty_rate": 3,
    }
    caregiver = {
        "id": "K1",
        "skill": "General",
        "skill_level": 1,
        "available": True,
        "delay": 0,
        "current_workload": 0,
        "max_work_hours": 8,
    }
    return NSGA2([patient], [caregiver]), patient, caregiver


def _costs_for_arrival(arrival_minute):
    optimizer, patient, caregiver = _optimizer_for_window_costs()
    assignment = {
        "patient_id": patient["id"],
        "caregiver_id": caregiver["id"],
        "service_hours": 1,
        "travel_time": 0,
        "time_window": patient["time_windows"][0],
        "arrival_time": arrival_minute,
    }
    return optimizer._summary([assignment])


def test_arrival_before_window_start_has_waiting_cost_only():
    costs = _costs_for_arrival(5)

    assert costs["waiting_cost"] == pytest.approx(5 * 2)
    assert costs["delay_penalty"] == pytest.approx(0)


def test_arrival_after_window_end_has_tardiness_penalty():
    costs = _costs_for_arrival(35)

    assert costs["waiting_cost"] == pytest.approx(0)
    assert costs["delay_penalty"] == pytest.approx(5 * 3)


def test_arrival_inside_window_has_no_waiting_or_tardiness_cost():
    costs = _costs_for_arrival(20)

    assert costs["waiting_cost"] == pytest.approx(0)
    assert costs["delay_penalty"] == pytest.approx(0)


def test_one_window_is_selected_and_patient_is_assigned_at_most_once():
    windows = [(10, 80), (100, 180), (200, 280)]
    patient = {
        "id": "P1",
        "care_type": "General",
        "required_skill_level": 1,
        "service_hours": 1,
        "time_windows": windows,
    }
    caregiver = {
        "id": "K1",
        "skill": "General",
        "skill_level": 1,
        "available": True,
        "delay": 0,
        "current_workload": 0,
        "max_work_hours": 8,
    }
    solution = NSGA2([patient], [caregiver], seed=1).create_solution()
    patient_assignments = [item for item in solution if item["patient_id"] == "P1"]

    assert len(patient_assignments) == 1
    assert patient_assignments[0]["time_window"] in windows
