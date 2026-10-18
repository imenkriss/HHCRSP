from collections import Counter
import math
import os
from pathlib import Path
import warnings

import pytest

from optimisation.mtwmo.generator import (
    DATASET_TABLE,
    HORIZON_END,
    LOCATION_SIDE,
    choose_window_count,
    generate_instance,
)


def signature(instance):
    return (
        instance.name,
        instance.seed,
        instance.depot,
        instance.laboratory,
        tuple((p.id, p.x, p.y, p.service_time, p.demand_level,
               tuple(p.time_windows), tuple(p.sampled_window_widths),
               tuple(sorted(p.preferences.items())), tuple(sorted(p.rejections.items())))
              for p in instance.patients),
        tuple((c.id, c.skill_level, c.start_time) for c in instance.caregivers),
    )


def test_generator_is_seeded_and_deterministic():
    first = generate_instance("A", 2, 1, seed=103)
    same = generate_instance("A", 2, 1, seed=103)
    other = generate_instance("A", 2, 1, seed=104)

    assert signature(first) == signature(same)
    assert signature(first) != signature(other)


@pytest.mark.parametrize("dataset,caregivers,location", [
    ("A", 2, 1),
    ("C", 12, 2),
    ("F", 58, 3),
])
def test_table_1_counts_and_names(dataset, caregivers, location):
    instance = generate_instance(dataset, caregivers, location, seed=7)
    expected_patients = DATASET_TABLE[dataset][0]

    assert len(instance.patients) == expected_patients
    assert len(instance.caregivers) == caregivers
    assert instance.name == f"{dataset}_{caregivers}_{location}"


def test_generated_values_and_clipped_window_widths_are_reported():
    instance = generate_instance("A", 2, 1, seed=71)
    side = LOCATION_SIDE[1]
    assert instance.horizon == (0.0, HORIZON_END)
    assert instance.travel_speed == 1.0
    points = [(instance.depot["x"], instance.depot["y"]),
              (instance.laboratory["x"], instance.laboratory["y"])]
    points.extend((p.x, p.y) for p in instance.patients)
    assert all(0 <= x <= side and 0 <= y <= side for x, y in points)

    clipping_changes = []
    for patient in instance.patients:
        assert 30 <= patient.service_time <= 90
        assert len(patient.time_windows) in (1, 2, 3)
        assert len(patient.sampled_window_widths) == len(patient.time_windows)
        assert len(patient.clipped_window_widths) == len(patient.time_windows)
        for (start, end), sampled_width, clipped in zip(
            patient.time_windows, patient.sampled_window_widths, patient.clipped_window_widths
        ):
            assert 0 <= start <= end <= HORIZON_END
            assert 60 <= sampled_width <= 120
            actual_width = end - start
            assert 0 < actual_width <= sampled_width + 1e-9
            assert clipped == (actual_width != pytest.approx(sampled_width))
            if clipped:
                clipping_changes.append((patient.id, sampled_width, actual_width))
    if clipping_changes:
        warnings.warn(f"Clipping changed sampled window widths: {clipping_changes}", UserWarning)


def test_demand_and_window_count_shares_over_200_seeded_instances():
    window_counts = Counter()
    demands = Counter()
    samples = 0
    for seed in range(200):
        instance = generate_instance("A", 2, 1, seed=seed)
        samples += len(instance.patients)
        window_counts.update(len(patient.time_windows) for patient in instance.patients)
        demands.update(patient.demand_level for patient in instance.patients)

    expected_windows = {1: 0.50, 2: 0.30, 3: 0.20}
    expected_demand = {1: 0.50, 2: 0.35, 3: 0.15}
    for value, expected in expected_windows.items():
        assert abs(window_counts[value] / samples - expected) <= 0.05
    for value, expected in expected_demand.items():
        assert abs(demands[value] / samples - expected) <= 0.05


def test_pair_distance_and_travel_time_match_fixed_seed():
    instance = generate_instance("A", 2, 1, seed=29)
    patient_ids = [patient.id for patient in instance.patients]
    pairs = [
        ("depot", "laboratory"),
        ("depot", patient_ids[0]),
        ("laboratory", patient_ids[0]),
        (patient_ids[0], patient_ids[1]),
        (patient_ids[0], patient_ids[0]),
    ]

    for origin, destination in pairs:
        if origin == "depot":
            origin_xy = (instance.depot["x"], instance.depot["y"])
        elif origin == "laboratory":
            origin_xy = (instance.laboratory["x"], instance.laboratory["y"])
        else:
            patient = next(p for p in instance.patients if p.id == origin)
            origin_xy = (patient.x, patient.y)
        if destination == "depot":
            destination_xy = (instance.depot["x"], instance.depot["y"])
        elif destination == "laboratory":
            destination_xy = (instance.laboratory["x"], instance.laboratory["y"])
        else:
            patient = next(p for p in instance.patients if p.id == destination)
            destination_xy = (patient.x, patient.y)

        expected = math.dist(origin_xy, destination_xy)
        assert instance.distance(origin, destination) == pytest.approx(expected)
        assert instance.travel_time(origin, destination) == pytest.approx(expected)
    assert instance.distance(patient_ids[0], patient_ids[0]) == 0


def test_table_lists_explicit_supported_combinations():
    assert DATASET_TABLE == {
        "A": (10, (2, 3, 4), (1, 2)),
        "B": (30, (5, 8), (1, 2)),
        "C": (50, (10, 12), (1, 2)),
        "D": (75, (15, 18), (1, 2)),
        "E": (100, (20, 25), (1, 2)),
        "F": (200, (45, 58), (2, 3)),
    }


@pytest.mark.slow
def test_large_F_instances_have_nonempty_bounded_windows_and_report_downgrades():
    downgraded_total = 0
    slow_seeds = int(os.environ.get("GENERATOR_SLOW_SEEDS", "50"))
    for caregivers, location in ((58, 3), (45, 2)):
        for seed in range(slow_seeds):
            instance = generate_instance("F", caregivers, location, seed=seed)
            for patient in instance.patients:
                assert patient.used_window_count == len(patient.time_windows)
                assert patient.requested_window_count in (1, 2, 3)
                assert patient.used_window_count in (1, 2, 3)
                downgraded_total += patient.used_window_count < patient.requested_window_count
                for earliest, latest in patient.time_windows:
                    assert earliest < latest
                    assert 0 <= earliest <= HORIZON_END
                    assert 0 <= latest <= HORIZON_END

    message = f"Total downgraded patients across F_58_3 and F_45_2, seeds 0..{slow_seeds - 1}: {downgraded_total}"
    print(message)
    warnings.warn(message, UserWarning)


def test_reversed_first_three_window_interval_causes_downgrade():
    assert choose_window_count(t0=250, theta=30, requested=3) <= 2


def test_requested_window_count_distribution_over_200_seeds():
    requested_counts = Counter()
    patients_seen = 0
    for seed in range(200):
        instance = generate_instance("A", 2, 1, seed=seed)
        requested_counts.update(patient.requested_window_count for patient in instance.patients)
        patients_seen += len(instance.patients)

    expected_shares = {1: 0.50, 2: 0.30, 3: 0.20}
    for count, expected_share in expected_shares.items():
        observed_share = requested_counts[count] / patients_seen
        assert abs(observed_share - expected_share) <= 0.05
