"""Dataset loading for the HHCOP / HHCRSP pipeline.

This module is **read-only**: it never writes, truncates or rewrites a file
under ``data/``. Generated artefacts (embedding caches, exports) must go to
``data/generated/`` or ``outputs/`` via :mod:`data.integrity`.

Source of truth
---------------
``data/dementia/new_dataset_with_predictions.csv`` — 1000 rows, 26 columns,
all clinical features min-max normalised to ``[0, 1]`` (D1), with
``XCOORD.``/``YCOORD.`` normalised coordinates and ``Predictions`` = the
care-type label (300 filled, 700 NaN).

``data/dementia_patients_health_data.csv`` (one record) and the empty
``data/admin_caregivers.csv`` are *not* used as patient/caregiver sources any
more: they were placeholders created by the old ``ensure_default_csv_files``
writer and are what made the whole pipeline run on a single patient. They are
left untouched for traceability.

ASSUMPTION (documented in docs/ASSUMPTIONS.md):
* ``service_hours`` is 0.5 h (30 min) because the teacher CSV has no service
  duration; P4 generated instances keep the paper's U[30, 90] service time.
* ``required_skill_level`` is 1 for teacher-CSV patients because the CSV has no
  clinical demand level; the paper's 50/35/15 demand generator is applied to
  generated instances.
* one wide window ``(0, 480)`` is used for teacher-CSV patients, matching the
  previous loader behaviour.
* missing ``Predictions`` (700 rows) is not filled here: ``care_type`` becomes
  ``"General"`` so every caregiver stays compatible. The SVM completion remains
  available through :func:`data.loader.load_predictions_dataset` (D1/D3) but is
  deliberately not run on every pipeline execution.
"""

from __future__ import annotations

import csv
import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
TEACHER_CSV = DATA_DIR / "dementia" / "new_dataset_with_predictions.csv"
CAREGIVER_CSV = DATA_DIR / "admin_caregivers.csv"

#: Coordinate side of the P11 location codes (1 -> 50x50, 2 -> 100x100, 3 -> 200x200).
LOCATION_AREA: dict[int, float] = {1: 50.0, 2: 100.0, 3: 200.0}

#: Working-set default. The full 1000-row file stays on disk untouched; this is
#: only how many rows enter one optimisation run (override with ``limit=None``).
DEFAULT_PATIENT_LIMIT = 100
DEFAULT_SERVICE_HOURS = 0.5
DEFAULT_WINDOW = (0.0, 480.0)
DEFAULT_CAREGIVERS_RATIO = 4  # P11: patients are 3-5x caregivers


class DatasetSourceError(RuntimeError):
    """Raised when a source dataset is missing, empty or malformed."""


# --------------------------------------------------------------------------- #
# Benchmark parser (compatibility shim)
# --------------------------------------------------------------------------- #
def _parse_instance(path: str | Path) -> dict[str, Any]:
    """Parse a Solomon / Gehring-Homberger benchmark file.

    Kept as the historical entry point used by
    :func:`optimisation.mtwmo.loaders.load_benchmark`; the implementation lives
    once in :func:`data.loader._parse_benchmark` so both paths share a single
    convention (depot = customer 0, customers = the remaining rows).
    """
    from data.loader import _parse_benchmark

    return _parse_benchmark(Path(path))


# --------------------------------------------------------------------------- #
# Teacher CSV (patients)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2)
def _teacher_frame_cached(path: str, modified_ns: int, size: int) -> pd.DataFrame:
    """Cached read keyed by path + mtime + size so edits invalidate the cache."""
    del modified_ns, size  # only used as cache-key components
    return pd.read_csv(path)


def load_teacher_frame(path: str | Path | None = None) -> pd.DataFrame:
    """Return a copy of the teacher CSV (never the shared cached frame)."""
    source = Path(path) if path is not None else TEACHER_CSV
    if not source.exists():
        raise DatasetSourceError(f"Teacher dataset not found: {source}")
    stat = source.stat()
    frame = _teacher_frame_cached(str(source), stat.st_mtime_ns, stat.st_size)
    return frame.copy()


def _validate_patient_frame(frame: pd.DataFrame, source: Path) -> None:
    if frame.empty:
        raise DatasetSourceError(f"Teacher dataset is empty: {source}")
    missing = [name for name in ("XCOORD.", "YCOORD.") if name not in frame.columns]
    if missing:
        raise DatasetSourceError(
            f"Teacher dataset {source} is missing required columns: {missing}"
        )
    if frame[["XCOORD.", "YCOORD."]].isna().all(axis=None):
        raise DatasetSourceError(f"Teacher dataset has no usable coordinates: {source}")


def _as_int(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return int(round(number))


def build_patient_records(
    frame: pd.DataFrame,
    *,
    limit: int | None = DEFAULT_PATIENT_LIMIT,
    area: float = 50.0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Turn teacher rows into optimiser/agent patient records.

    ``limit=None`` keeps all rows; otherwise the first ``limit`` rows are used.
    A contiguous prefix (rather than a random sample) keeps identifiers stable
    (``P<row>``), keeps ``P1`` available to the default UI query, and makes the
    working set directly checkable against the source file. ``seed`` is kept
    for API compatibility and for the caregiver scaffold.
    """
    if area is None or float(area) <= 0:
        raise ValueError(f"area must be a positive number, got {area!r}")
    if limit is not None and int(limit) <= 0:
        raise ValueError(f"limit must be positive or None, got {limit!r}")

    _validate_patient_frame(frame, TEACHER_CSV)

    total = len(frame)
    del seed  # selection is a deterministic prefix; see docstring
    positions = list(range(min(total, int(limit) if limit is not None else total)))

    side = float(area)
    records: list[dict[str, Any]] = []
    for position in positions:
        row = frame.iloc[position]
        x = float(row["XCOORD."]) * side
        y = float(row["YCOORD."]) * side
        if not math.isfinite(x) or not math.isfinite(y):
            raise DatasetSourceError(
                f"Non-finite coordinates at row {position + 1} of the teacher dataset"
            )
        diabetic = _as_int(row.get("Diabetic")) or 0
        prediction = _as_int(row.get("Predictions"))
        care_type = (
            "type_1" if prediction == 1
            else "type_0" if prediction == 0
            else "General"
        )
        records.append({
            "id": f"P{position + 1}",
            "source_row": position + 1,
            "source": "teacher_csv",
            "care_type": care_type,
            "priority": "High" if diabetic == 1 else "Medium",
            "required_skill_level": 1,
            "service_hours": DEFAULT_SERVICE_HOURS,
            "x": x,
            "y": y,
            "time_windows": [DEFAULT_WINDOW],
            "diabetic": diabetic,
            "prediction": prediction,
        })
    return records


def load_hhcop_patients(
    limit: int | None = DEFAULT_PATIENT_LIMIT,
    area: float = 50.0,
    seed: int = 42,
    source: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load validated patient records from the teacher CSV (read-only)."""
    frame = load_teacher_frame(source)
    return build_patient_records(frame, limit=limit, area=area, seed=seed)


# --------------------------------------------------------------------------- #
# Caregivers
# --------------------------------------------------------------------------- #
def caregiver_count_for(patient_count: int) -> int:
    """Default caregiver count keeping the P11 3-5x patient/caregiver ratio."""
    if patient_count <= 0:
        raise ValueError("patient_count must be positive")
    return max(1, math.ceil(patient_count / DEFAULT_CAREGIVERS_RATIO))


_INT_CAREGIVER_FIELDS = ("max_work_hours", "current_workload", "delay", "skill_level",
                         "max_overtime_hours")


def _coerce_caregiver_row(row: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for raw_key, raw_value in row.items():
        if raw_key is None:
            continue
        key = str(raw_key).strip()
        value = "" if raw_value is None else str(raw_value).strip()
        if key in _INT_CAREGIVER_FIELDS:
            try:
                record[key] = int(float(value)) if value else 0
            except ValueError:
                record[key] = 0
        elif key == "available":
            record[key] = value.lower() in {"true", "1", "yes", "oui"}
        else:
            try:
                record[key] = float(value) if "." in value else (
                    int(value) if value.isdigit() else value
                )
            except ValueError:
                record[key] = value
    return record


def _read_caregiver_rows(source: Path) -> list[dict[str, Any]]:
    with source.open(newline="", encoding="utf-8-sig") as handle:
        return [_coerce_caregiver_row(row) for row in csv.DictReader(handle)]


def _normalize_caregiver(record: dict[str, Any], index: int) -> dict[str, Any]:
    caregiver = dict(record)
    caregiver.setdefault("id", f"S{index + 1}")
    caregiver.setdefault("skill", "General")
    caregiver.setdefault("skill_level", 1)
    caregiver.setdefault("available", True)
    caregiver.setdefault("max_work_hours", 8)
    caregiver.setdefault("max_overtime_hours", 2)
    caregiver.setdefault("current_workload", 0)
    caregiver.setdefault("delay", 0)
    caregiver.setdefault("x", 0.0)
    caregiver.setdefault("y", 0.0)
    caregiver.setdefault("time_windows", [DEFAULT_WINDOW])
    caregiver.setdefault("source", "admin_csv")
    caregiver["available"] = bool(caregiver["available"])
    for key in ("skill_level", "max_work_hours", "current_workload", "delay",
                "max_overtime_hours"):
        try:
            caregiver[key] = int(float(caregiver[key]))
        except (TypeError, ValueError):
            caregiver[key] = 0
    for key in ("x", "y"):
        try:
            caregiver[key] = float(caregiver[key])
        except (TypeError, ValueError):
            caregiver[key] = 0.0
    return caregiver


def generate_caregivers(
    count: int,
    *,
    area: float = 50.0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Deterministic scaffold caregivers used when the admin CSV is empty.

    These are *configuration inputs*, not measured results: skill levels rotate
    over {1, 2, 3}, all are available with zero delay and the same 8 h shift.
    """
    if count <= 0:
        raise ValueError(f"caregiver count must be positive, got {count!r}")
    if area is None or float(area) <= 0:
        raise ValueError(f"area must be a positive number, got {area!r}")
    rng = random.Random(seed)
    side = float(area)
    return [
        {
            "id": f"S{index + 1}",
            "skill": "General",
            "skill_level": 1 + (index % 3),
            "available": True,
            "max_work_hours": 8,
            "max_overtime_hours": 2,
            "current_workload": 0,
            "delay": 0,
            "x": rng.uniform(0.0, side),
            "y": rng.uniform(0.0, side),
            "time_windows": [DEFAULT_WINDOW],
            "source": "generated",
        }
        for index in range(count)
    ]


def load_hhcop_caregivers(
    count: int | None = None,
    *,
    patient_count: int | None = None,
    area: float = 50.0,
    seed: int = 42,
    source: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load caregivers from ``admin_caregivers.csv`` or build the scaffold.

    An absent or empty (0-byte) admin CSV yields ``count`` generated caregivers
    (default: ``ceil(patient_count / 4)``). Nothing is written to ``data/``.
    """
    path = Path(source) if source is not None else CAREGIVER_CSV
    rows: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        rows = _read_caregiver_rows(path)
    if rows:
        caregivers = [_normalize_caregiver(row, index) for index, row in enumerate(rows)]
        if count is not None:
            if int(count) <= 0:
                raise ValueError(f"caregiver count must be positive, got {count!r}")
            caregivers = caregivers[: int(count)]
        return caregivers

    if count is None:
        if patient_count is None:
            raise ValueError(
                "provide count or patient_count when no admin caregiver file is available"
            )
        count = caregiver_count_for(int(patient_count))
    return generate_caregivers(int(count), area=area, seed=seed)


def generate_random_instance(
    dataset: str,
    caregivers: int,
    location: int,
    seed: int = 0,
    **kwargs: Any,
):
    """P11/Table 1 instance generator, exposed from the data layer.

    Thin delegate to :func:`optimisation.mtwmo.generator.generate_instance` so
    callers that look for the generator in ``data.dataset_loader`` (and the
    Table 1 conformance test) use the single canonical implementation.
    """
    from optimisation.mtwmo.generator import generate_instance

    return generate_instance(dataset, caregivers, location, seed, **kwargs)


# --------------------------------------------------------------------------- #
# Convenience wrapper kept for backward compatibility
# --------------------------------------------------------------------------- #
def load_combined_dataset(
    *, limit: int | None = DEFAULT_PATIENT_LIMIT, area: float = 50.0, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load patients + caregivers with the canonical sources (read-only)."""
    patients = load_hhcop_patients(limit=limit, area=area, seed=seed)
    caregivers = load_hhcop_caregivers(
        patient_count=len(patients), area=area, seed=seed
    )
    return patients, caregivers
