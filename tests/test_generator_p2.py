from collections import Counter

import pytest

from data import loader
from optimisation.mtwmo.generator import (
    DATASET_TABLE,
    generate_instance,
    service_time_for,
)


def test_dataset_names_and_counts():
    for dataset, (patient_count, caregiver_options, location_options) in DATASET_TABLE.items():
        caregivers = caregiver_options[0]
        location = location_options[0]
        instance = generate_instance(dataset, caregivers, location, seed=1)

        assert instance.name == f"{dataset}_{caregivers}_{location}"
        assert len(instance.patients) == patient_count
        assert len(instance.caregivers) == caregivers
        # P11: patients ~ 3-5x caregivers (dataset A with K=4 is the documented 2.5x).
        ratio = patient_count / caregivers
        assert 2.4 <= ratio <= 6.1


def test_seed_is_reproducible_including_uncertainty():
    first = generate_instance("A", 2, 1, seed=99)
    same = generate_instance("A", 2, 1, seed=99)
    other = generate_instance("A", 2, 1, seed=100)

    assert [p.time_windows for p in first.patients] == [p.time_windows for p in same.patients]
    assert first.travel_uncertainty == same.travel_uncertainty
    assert first.travel_uncertainty != other.travel_uncertainty


def test_window_and_demand_proportions_within_tolerance():
    window_counts: Counter[int] = Counter()
    demand_counts: Counter[int] = Counter()
    patients_seen = 0
    for seed in range(200):
        instance = generate_instance("A", 2, 1, seed=seed)
        patients_seen += len(instance.patients)
        window_counts.update(len(patient.time_windows) for patient in instance.patients)
        demand_counts.update(patient.demand_level for patient in instance.patients)

    for value, expected in {1: 0.50, 2: 0.30, 3: 0.20}.items():
        assert abs(window_counts[value] / patients_seen - expected) <= 0.06
    for value, expected in {1: 0.50, 2: 0.35, 3: 0.15}.items():
        assert abs(demand_counts[value] / patients_seen - expected) <= 0.06


def test_windows_are_valid_and_service_time_range():
    instance = generate_instance("A", 2, 1, seed=7)

    for patient in instance.patients:
        assert 30.0 <= patient.service_time <= 90.0
        assert 1 <= len(patient.time_windows) <= 3
        for earliest, latest in patient.time_windows:
            assert earliest < latest
            assert 0.0 <= earliest <= 720.0
            assert 0.0 <= latest <= 720.0


def test_skill_rule_and_service_time_formula():
    instance = generate_instance("A", 3, 1, seed=5)

    for patient in instance.patients:
        assert patient.demand_level in (1, 2, 3)
        for caregiver in instance.caregivers:
            assert caregiver.skill_level in (1, 2, 3)
            if caregiver.skill_level >= patient.demand_level:
                expected = patient.service_time * patient.demand_level / caregiver.skill_level
                assert service_time_for(patient, caregiver) == pytest.approx(expected)
            else:
                with pytest.raises(ValueError):
                    service_time_for(patient, caregiver)


def test_preferences_rejections_departure_and_uncertainty_levels():
    instance = generate_instance("A", 2, 1, seed=3)

    for patient in instance.patients:
        assert 0.0 <= patient.demand_uncertainty <= 1.0
        for caregiver in instance.caregivers:
            assert patient.preferences[caregiver.id] in (0, 1, 2, 3)
            assert isinstance(patient.rejections[caregiver.id], bool)

    for caregiver in instance.caregivers:
        assert caregiver.start_time == 0
        assert 0.0 <= caregiver.departure_uncertainty <= 1.0

    assert instance.travel_uncertainty
    for alpha in instance.travel_uncertainty.values():
        assert 0.0 <= alpha <= 1.0


def test_csv_sourced_patients_use_supplied_coordinates_and_care_type():
    df, _ = loader.load_predictions_dataset()
    supplied = loader.patients_from_dataframe(df, count=10, area=100, seed=1)

    instance = generate_instance("A", 2, 1, seed=1, patients=supplied)

    supplied_xy = {(round(item["x"], 6), round(item["y"], 6)) for item in supplied}
    for patient in instance.patients:
        assert (round(patient.x, 6), round(patient.y, 6)) in supplied_xy
        assert patient.care_type in {"type_0", "type_1"}

    specialties = {caregiver.specialty for caregiver in instance.caregivers}
    assert specialties <= {"type_0", "type_1"}
    assert specialties == {"type_0", "type_1"}


def test_supplied_patients_must_cover_the_required_count():
    df, _ = loader.load_predictions_dataset()
    too_few = loader.patients_from_dataframe(df, count=2, area=50, seed=1)

    with pytest.raises(ValueError):
        generate_instance("A", 2, 1, seed=1, patients=too_few)


def test_generate_instance_from_csv_bridge():
    df, _ = loader.load_predictions_dataset()

    instance = loader.generate_instance_from_csv(df, "A", 2, 1, seed=3)

    assert instance.name == "A_2_1"
    assert len(instance.patients) == 10
    assert all(0.0 <= patient.x <= 50.0 for patient in instance.patients)
    assert all(patient.care_type in {"type_0", "type_1"} for patient in instance.patients)
