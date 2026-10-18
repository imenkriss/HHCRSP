from pathlib import Path

import pytest

from optimisation.mtwmo.domain import Caregiver, evaluate_window_arrival
from optimisation.mtwmo.loaders import load_benchmark

ROOT = Path(__file__).resolve().parents[1]


def test_caregiver_phase_zero_working_time_limits():
    caregiver = Caregiver(id="K1")

    assert caregiver.rated_minutes == 480
    assert caregiver.max_overtime_minutes == 240
    assert caregiver.rated_minutes + caregiver.max_overtime_minutes == 720
    assert caregiver.hard_daily_limit_minutes == 720


@pytest.mark.parametrize(
    ("arrival", "expected_waiting", "expected_tardiness"),
    [
        (85, 15, 0),   # before [100, 130]
        (115, 0, 0),   # inside [100, 130]
        (142, 0, 12),  # after [100, 130]
    ],
)
def test_soft_time_window_components(arrival, expected_waiting, expected_tardiness):
    timing = evaluate_window_arrival(arrival, (100, 130))

    assert timing.waiting_time == expected_waiting
    assert timing.tardiness == expected_tardiness


def test_benchmark_patients_and_caregivers_have_placeholder_levels():
    instance = load_benchmark(ROOT / "data" / "Solomon" / "In" / "c101.txt")

    assert instance.patients
    assert instance.caregivers
    assert all(patient.demand_level == 1 for patient in instance.patients)
    assert all(caregiver.skill_level == 1 for caregiver in instance.caregivers)
