"""Combine dementia profiles with Solomon and Gehring-Homberger instances."""

from __future__ import annotations

import csv
import math
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
BENCHMARK_DIRS = (DATA_DIR / "Gehing&Homberger", DATA_DIR / "Solomon")
CLINICAL_FILE = DATA_DIR / "dementia" / "dementia_patients_health_data.csv"
CARE_TYPES = ("Cardio", "Diabetes", "General")


def _numbers(line: str) -> list[float] | None:
    try:
        return [float(value) for value in line.split()]
    except ValueError:
        return None


def _parse_instance(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    vehicle_count = None
    vehicle_capacity = None
    customers = []
    in_customers = False

    for index, line in enumerate(lines):
        heading = line.strip().upper()
        if heading.startswith("NUMBER") and "CAPACITY" in heading:
            for candidate in lines[index + 1:]:
                row = _numbers(candidate)
                if row and len(row) >= 2:
                    vehicle_count, vehicle_capacity = int(row[0]), row[1]
                    break
        if heading.startswith("CUSTOMER"):
            in_customers = True
            continue
        if not in_customers:
            continue
        row = _numbers(line)
        if row and len(row) >= 7:
            customers.append({
                "customer_id": int(row[0]),
                "x": row[1],
                "y": row[2],
                "demand": row[3],
                "ready_time": row[4],
                "due_date": row[5],
                "service_duration": row[6],
            })

    depot = next((row for row in customers if row["customer_id"] == 0), None)
    if vehicle_count is None or vehicle_capacity is None or depot is None:
        raise ValueError(f"Invalid Solomon-format benchmark: {path}")
    return {
        "instance": path.stem,
        "vehicle_count": vehicle_count,
        "vehicle_capacity": vehicle_capacity,
        "depot": depot,
        "customers": [row for row in customers if row["customer_id"] != 0],
    }


def _load_instances() -> list[dict]:
    instances = []
    for directory in BENCHMARK_DIRS:
        paths = sorted(
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() == ".txt"
        ) if directory.exists() else []
        for path in paths:
            try:
                instances.append(_parse_instance(path))
            except (OSError, UnicodeError, ValueError):
                continue
    if not instances:
        raise FileNotFoundError(
            "No benchmark .txt files found in "
            + ", ".join(str(path) for path in BENCHMARK_DIRS)
        )
    return instances


def _load_clinical_records() -> list[dict]:
    if not CLINICAL_FILE.exists():
        raise FileNotFoundError(f"Dementia dataset not found: {CLINICAL_FILE}")
    with CLINICAL_FILE.open(newline="", encoding="utf-8-sig") as file:
        records = []
        for row in csv.DictReader(file):
            record = {}
            for key, value in row.items():
                value = (value or "").strip()
                try:
                    number = float(value)
                    record[key.strip()] = int(number) if number.is_integer() else number
                except ValueError:
                    record[key.strip()] = value
            records.append(record)
    return records


def _care_type(record: dict) -> str:
    conditions = str(record.get("Chronic_Health_Conditions", "")).lower()
    if record.get("Diabetic") == 1 or "diabet" in conditions:
        return "Diabetes"
    if any(term in conditions for term in ("heart", "cardiac", "cardio")):
        return "Cardio"
    return "General"


def _priority(record: dict) -> tuple[str, int]:
    oxygen = float(record.get("BloodOxygenLevel", 100) or 100)
    heart_rate = float(record.get("HeartRate", 75) or 75)
    temperature = float(record.get("BodyTemperature", 37) or 37)
    if oxygen < 85 or heart_rate > 130 or heart_rate < 40 or temperature >= 39.5 or temperature < 35:
        return "Critical", 5
    condition = str(record.get("Chronic_Health_Conditions", "")).strip().lower()
    if record.get("Dementia") == 1 or condition not in ("", "none", "no", "nan"):
        return "High", 3
    return "Medium", 1


def _geographic_rows(instances: list[dict]) -> list[dict]:
    rows = []
    max_rows = max(len(instance["customers"]) for instance in instances)
    for customer_index in range(max_rows):
        for instance in instances:
            if customer_index >= len(instance["customers"]):
                continue
            row = instance["customers"][customer_index].copy()
            row["source_instance"] = instance["instance"]
            row["depot"] = instance["depot"]
            rows.append(row)
    return rows


def load_combined_dataset() -> tuple[list[dict], list[dict]]:
    """Return clinical patients linked to benchmark locations and caregivers."""
    instances = _load_instances()
    geographic_rows = _geographic_rows(instances)
    clinical_records = _load_clinical_records()
    patients = []
    for index, clinical in enumerate(clinical_records):
        location = geographic_rows[index % len(geographic_rows)]
        priority, required_skill_level = _priority(clinical)
        depot = location["depot"]
        patients.append({
            **clinical,
            "id": f"D{index + 1:05d}",
            "care_type": _care_type(clinical),
            "priority": priority,
            "required_skill_level": required_skill_level,
            "preferred_caregiver": "",
            "x": location["x"],
            "y": location["y"],
            "demand": location["demand"],
            "ready_time": location["ready_time"],
            "due_date": location["due_date"],
            "time_windows": clinical.get("time_windows") or [
                (location["ready_time"], location["due_date"])
            ],
            "service_duration": location["service_duration"],
            "service_hours": max(0.25, location["service_duration"] / 60),
            "travel_time": math.dist((location["x"], location["y"]), (depot["x"], depot["y"])),
            "source_instance": location["source_instance"],
            "source_customer_id": location["customer_id"],
        })

    benchmark = instances[0]
    caregivers = [
        {
            "id": f"S{index + 1}",
            "skill": CARE_TYPES[index % len(CARE_TYPES)],
            "max_work_hours": 8,
            "current_workload": 0,
            "skill_level": 3 + index % 3,
            "available": True,
            "delay": 0,
            "x": benchmark["depot"]["x"],
            "y": benchmark["depot"]["y"],
            "vehicle_capacity": benchmark["vehicle_capacity"],
            "source_instance": benchmark["instance"],
        }
        for index in range(benchmark["vehicle_count"])
    ]
    return patients, caregivers