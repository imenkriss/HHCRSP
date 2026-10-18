"""Seeded MTWMO instance generator following Section 5.1 and Table 1.

Travel speed, demand/skill interpretation, rejection probability, P, and the
window-width-to-interval mapping are explicit project decisions/placeholders.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

from .domain import Caregiver, Instance, Patient

DATASET_TABLE: dict[str, tuple[int, tuple[int, ...], tuple[int, ...]]] = {
    "A": (10, (2, 3, 4), (1, 2)),
    "B": (30, (5, 8), (1, 2)),
    "C": (50, (10, 12), (1, 2)),
    "D": (75, (15, 18), (1, 2)),
    "E": (100, (20, 25), (1, 2)),
    "F": (200, (45, 58), (2, 3)),
}
LOCATION_SIDE = {1: 50.0, 2: 100.0, 3: 200.0}
HORIZON_END = 720.0
TRAVEL_SPEED = 1.0  # decision: one coordinate unit per minute
REJECTION_PROBABILITY = 0.05  # placeholder
WINDOW_COUNT_WEIGHTS = ((1, 0.50), (2, 0.30), (3, 0.20))
DEMAND_WEIGHTS = ((1, 0.50), (2, 0.35), (3, 0.15))



def _weighted_choice(rng: random.Random, values_and_weights: tuple[tuple[int, float], ...]) -> int:
    draw = rng.random()
    cumulative = 0.0
    for value, weight in values_and_weights:
        cumulative += weight
        if draw < cumulative:
            return value
    return values_and_weights[-1][0]



def choose_window_count(t0: float, theta: float, requested: int) -> int | None:
    """Choose the largest feasible window count no greater than requested.

    Returns None only when even the one-window center interval is empty. No
    random state is consumed, so the helper is deterministic for its inputs.
    """
    if requested not in (1, 2, 3):
        raise ValueError("requested window count must be 1, 2, or 3")
    bounds_by_count = {
        1: [(t0, HORIZON_END - t0 - theta)],
        2: [(t0, HORIZON_END / 2),
            (HORIZON_END / 2, HORIZON_END - t0 - theta)],
        3: [(t0, HORIZON_END / 3),
            (HORIZON_END / 3, 2 * HORIZON_END / 3),
            (2 * HORIZON_END / 3, HORIZON_END - t0 - theta)],
    }
    for count in range(requested, 0, -1):
        if all(lower <= upper for lower, upper in bounds_by_count[count]):
            return count
    return None


def generate_instance(
    dataset: str,
    caregivers: int,
    location: int,
    seed: int,
    patients: Sequence[Mapping[str, object]] | None = None,
) -> Instance:
    """Generate one deterministic instance for a Table 1 dataset triple.

    Names follow <dataset>_<caregivers>_<location>. Coordinates and random
    quantities use a private RNG seeded solely by the explicit seed argument.

    When ``patients`` is supplied (e.g. built by ``data.loader`` from the teacher
    CSV) the first ``patient_count`` records drawn with the seeded RNG provide
    each patient's coordinates and ``care_type``; otherwise coordinates are
    random in the 50/100/200 location square. ``care_type`` only feeds caregiver
    specialty compatibility / the MAS (ASSUMPTION in docs/ASSUMPTIONS.md); the
    demand level p_i still comes from the paper's 50/35/15 generator.
    """
    dataset = dataset.upper()
    if dataset not in DATASET_TABLE:
        raise ValueError(f"Unknown dataset {dataset!r}; expected A-F")
    patient_count, caregiver_options, location_options = DATASET_TABLE[dataset]
    if caregivers not in caregiver_options:
        raise ValueError(f"Dataset {dataset} caregiver count must be one of {caregiver_options}")
    if location not in location_options:
        raise ValueError(f"Dataset {dataset} location code must be one of {location_options}")

    rng = random.Random(seed)
    side = LOCATION_SIDE[location]

    def point() -> tuple[float, float]:
        return rng.uniform(0.0, side), rng.uniform(0.0, side)

    depot_xy = point()
    lab_xy = point()

    if patients is None:
        patients_xy = [point() for _ in range(patient_count)]
        patients_meta: list[dict[str, object]] = [{} for _ in range(patient_count)]
    else:
        available = list(patients)
        if len(available) < patient_count:
            raise ValueError(
                f"Need at least {patient_count} supplied patients, got {len(available)}"
            )
        chosen = rng.sample(range(len(available)), patient_count)
        patients_xy = []
        patients_meta = []
        for chosen_index in chosen:
            record = available[chosen_index]
            if "x" not in record or "y" not in record:
                raise ValueError("Supplied patients must provide 'x' and 'y' coordinates")
            patients_xy.append((float(record["x"]), float(record["y"])))
            patients_meta.append({
                "care_type": record.get("care_type"),
                "priority": record.get("priority", "Medium"),
            })

    depot = {"id": "depot", "x": depot_xy[0], "y": depot_xy[1]}
    laboratory = {"id": "laboratory", "x": lab_xy[0], "y": lab_xy[1]}

    instance_name = f"{dataset}_{caregivers}_{location}"
    patient_objects: list[Patient] = []
    for index, ((x, y), meta) in enumerate(zip(patients_xy, patients_meta), start=1):
        service_time = rng.uniform(30.0, 90.0)
        demand = _weighted_choice(rng, DEMAND_WEIGHTS)
        window_count = _weighted_choice(rng, WINDOW_COUNT_WEIGHTS)
        t0 = math.dist(depot_xy, (x, y)) / TRAVEL_SPEED
        used_window_count = choose_window_count(t0, service_time, window_count)
        if used_window_count is None:
            raise ValueError(
                f"No non-empty time-window center intervals for instance "
                f"{instance_name}, patient {index}"
            )
        if used_window_count == 1:
            center_bounds = [(t0, HORIZON_END - t0 - service_time)]
        elif used_window_count == 2:
            center_bounds = [
                (t0, HORIZON_END / 2),
                (HORIZON_END / 2, HORIZON_END - t0 - service_time),
            ]
        else:
            center_bounds = [
                (t0, HORIZON_END / 3),
                (HORIZON_END / 3, 2 * HORIZON_END / 3),
                (2 * HORIZON_END / 3, HORIZON_END - t0 - service_time),
            ]
        windows: list[tuple[float, float]] = []
        sampled_widths: list[float] = []
        clipped_widths: list[bool] = []
        for lower_center, upper_center in center_bounds:
            center = rng.uniform(lower_center, upper_center)
            width = rng.uniform(60.0, 120.0)
            raw_start, raw_end = center - width / 2.0, center + width / 2.0
            start = min(HORIZON_END, max(0.0, raw_start))
            end = min(HORIZON_END, max(0.0, raw_end))
            windows.append((start, end))
            sampled_widths.append(width)
            clipped_widths.append(not math.isclose(end - start, width, abs_tol=1e-12))
        patient_objects.append(Patient(
            id=index,
            x=x,
            y=y,
            service_time=service_time,
            demand_level=demand,
            time_windows=windows,
            preferences={},
            rejections={},
            sampled_window_widths=sampled_widths,
            clipped_window_widths=clipped_widths,
            requested_window_count=window_count,
            used_window_count=used_window_count,
            care_type=meta.get("care_type"),
            priority=str(meta.get("priority", "Medium")),
        ))

    caregiver_objects = [
        Caregiver(id=f"K{k}", skill_level=rng.choice((1, 2, 3)), start_time=0)
        for k in range(1, caregivers + 1)
    ]
    for patient in patient_objects:
        for caregiver in caregiver_objects:
            patient.preferences[caregiver.id] = rng.randint(0, 3)
            patient.rejections[caregiver.id] = rng.random() < REJECTION_PROBABILITY

    # Caregiver specialty follows the CSV care types (round-robin), only when
    # patients were supplied with a care_type. This never changes p_i / g_k.
    specialties = sorted({
        str(patient.care_type) for patient in patient_objects if patient.care_type
    })
    if specialties:
        for k, caregiver in enumerate(caregiver_objects):
            caregiver.specialty = specialties[k % len(specialties)]

    # Uncertainty levels (P5): sigma_i, beta_k and alpha_ij in [0, 1]. Drawn
    # after the historical quantities so the pre-existing random stream (and the
    # existing generator tests) is unchanged.
    for patient in patient_objects:
        patient.demand_uncertainty = rng.uniform(0.0, 1.0)
    for caregiver in caregiver_objects:
        caregiver.departure_uncertainty = rng.uniform(0.0, 1.0)
    node_ids = ["depot", "laboratory"] + [patient.id for patient in patient_objects]
    travel_uncertainty = {
        (origin, destination): rng.uniform(0.0, 1.0)
        for origin in node_ids
        for destination in node_ids
        if origin != destination
    }

    max_patients = math.ceil(patient_count / caregivers) + 2
    return Instance(
        depot=depot,
        laboratory=laboratory,
        caregivers=caregiver_objects,
        patients=patient_objects,
        horizon=(0.0, HORIZON_END),
        travel_speed=TRAVEL_SPEED,
        name=instance_name,
        max_patients_per_caregiver=max_patients,
        seed=seed,
        travel_uncertainty=travel_uncertainty,
    )


def service_time_for(patient: Patient, caregiver: Caregiver) -> float:
    """Deterministic service time theta_ik = theta_i * p_i / g_k (P4).

    The skill rule g_k >= p_i must hold; otherwise the caregiver cannot serve
    the patient and a ValueError is raised.
    """
    if caregiver.skill_level < patient.demand_level:
        raise ValueError(
            f"Caregiver {caregiver.id} (g={caregiver.skill_level}) cannot serve "
            f"patient {patient.id} (p={patient.demand_level})"
        )
    return patient.service_time * patient.demand_level / caregiver.skill_level
