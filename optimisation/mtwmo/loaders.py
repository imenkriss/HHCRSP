"""Readers for Solomon and Gehring-Homberger benchmark files."""

from pathlib import Path

from data.dataset_loader import _parse_instance
from .domain import Caregiver, Instance, Patient


def load_benchmark(path: str | Path) -> Instance:
    """Load a Solomon-format benchmark.

    The first numeric row in the customer section is treated as the depot;
    each remaining row becomes one patient with one [ready, due] window.
    Vehicle count/capacity parsing is delegated to dataset_loader's existing
    Solomon parser so both benchmark families use the same convention.

    Clinical demand level, caregiver skill, preferences, travel speed, and a
    distinct laboratory are not encoded in these files. Demand and skill use
    level-1 placeholders pending enrichment; the laboratory is absent.
    """
    source = Path(path)
    parsed = _parse_instance(source)
    raw_rows = [parsed["depot"], *parsed["customers"]]
    if not raw_rows:
        raise ValueError(f"Benchmark has no customer rows: {source}")

    # The source convention places the depot first (currently customer id 0).
    depot_row = raw_rows[0]
    if depot_row["customer_id"] != 0:
        raise ValueError(f"First customer row is not the depot (id 0): {source}")
    patients = [
        Patient(
            id=row["customer_id"],
            x=float(row["x"]),
            y=float(row["y"]),
            service_time=float(row["service_duration"]),
            demand_level=1,
            time_windows=[(float(row["ready_time"]), float(row["due_date"]))],
        )
        for row in parsed["customers"]
    ]
    caregivers = [
        Caregiver(id=f"K{index + 1}", capacity=float(parsed["vehicle_capacity"]))
        for index in range(parsed["vehicle_count"])
    ]
    all_times = [value for patient in patients for window in patient.time_windows for value in window]
    depot_times = (float(depot_row["ready_time"]), float(depot_row["due_date"]))
    times = all_times + list(depot_times)
    horizon = (min(times), max(times)) if times else depot_times
    return Instance(
        depot={"id": float(depot_row["customer_id"]), "x": float(depot_row["x"]), "y": float(depot_row["y"])},
        laboratory=None,
        caregivers=caregivers,
        patients=patients,
        horizon=horizon,
        travel_speed=None,
        vehicle_capacity=float(parsed["vehicle_capacity"]),
        name=source.stem,
    )
