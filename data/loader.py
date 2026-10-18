"""Teacher-CSV loader, coordinate helpers, benchmark parsers and D4 check.

This module is read-only: it never edits the raw files under ``data/``. It
implements the D1–D4 items of ``AGENTS.md``:

* D1 — load and validate ``new_dataset_with_predictions.csv`` (1000 rows, 26
  columns), expose ``care_type`` (derived from ``Predictions``) plus a
  ``care_type_source`` column in ``{"given", "svm"}``.
* SVM fill for the 700 unlabelled rows: One-vs-Rest on the 300 labelled rows,
  excluding ``Predictions``, ``XCOORD.``, ``YCOORD.`` from the features, with
  accuracy / precision / F1 reported by cross-validation.
* Coordinates — rescale ``XCOORD.``/``YCOORD.`` to 50x50 / 100x100 / 200x200 or
  the Gehring-Homberger range.
* D4 — compare the CSV coordinates with a Gehring-Homberger 1000-customer file.

The leaky nature of the label (``Predictions == 1 - Diabetic`` on the 300
labelled rows, only 2 classes) is recorded in ``docs/ASSUMPTIONS.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_DIR = Path(__file__).resolve().parent
PREDICTIONS_CSV = DATA_DIR / "dementia" / "new_dataset_with_predictions.csv"
GEHRING_HOMBERGER_DIR = DATA_DIR / "Gehing&Homberger"
SOLOMON_DIR = DATA_DIR / "Solomon"

EXPECTED_ROWS = 1000
EXCLUDE_COLUMNS = ("Predictions", "XCOORD.", "YCOORD.")
CARE_TYPE_LABELS = {0: "type_0", 1: "type_1"}
LOCATION_SIDE = {1: 50.0, 2: 100.0, 3: 200.0}


# --------------------------------------------------------------------------- #
# D1 — teacher CSV loading, validation and SVM completion
# --------------------------------------------------------------------------- #
def read_predictions_csv(path: str | Path | None = None) -> pd.DataFrame:
    """Read the teacher CSV with an explicit pathlib path (D1)."""
    source = Path(path) if path is not None else PREDICTIONS_CSV
    return pd.read_csv(source)


def validate_predictions_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Validate 1000 rows / 26 columns and return an information mapping (D1)."""
    if df.shape[0] != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {df.shape[0]}")
    if df.shape[1] != 26:
        raise ValueError(f"Expected 26 columns, found {df.shape[1]}")
    labelled = int(df["Predictions"].notna().sum())
    info: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "labelled": labelled,
        "unlabelled": int(df.shape[0] - labelled),
        "classes": sorted({int(value) for value in df["Predictions"].dropna().unique()}),
        "notes": [
            "On the 300 labelled rows Predictions == 1 - Diabetic exactly.",
            "The file has only 2 classes, not the 4 departments of the paper.",
            "The OvR SVM therefore effectively learns the deterministic 1 - Diabetic rule.",
        ],
    }
    return info


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in EXCLUDE_COLUMNS]


def build_care_type_svm() -> Pipeline:
    """One-vs-Rest SVM pipeline over the imputed numerics (D1)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ovr", OneVsRestClassifier(SVC(kernel="linear", C=1.0))),
    ])


def train_care_type_svm(
    df: pd.DataFrame, cv_folds: int = 5
) -> tuple[Pipeline, dict[str, float]]:
    """Train the OvR SVM on the labelled rows and report cross-validated metrics."""
    features = df[_feature_columns(df)]
    labelled = df["Predictions"].notna()
    x_labelled = features[labelled]
    y_labelled = df.loc[labelled, "Predictions"].astype(int)

    pipeline = build_care_type_svm()
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline,
        x_labelled,
        y_labelled,
        cv=splitter,
        scoring=["accuracy", "precision", "f1"],
    )
    metrics = {
        "accuracy": float(np.mean(scores["test_accuracy"])),
        "precision": float(np.mean(scores["test_precision"])),
        "f1": float(np.mean(scores["test_f1"])),
        "folds": int(cv_folds),
    }
    pipeline.fit(x_labelled, y_labelled)
    return pipeline, metrics


def assign_care_type(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Add ``care_type`` and ``care_type_source`` (given/svm) columns (D1)."""
    result = df.copy()
    labelled = df["Predictions"].notna()
    result["care_type"] = pd.NA
    result["care_type_source"] = pd.NA

    result.loc[labelled, "care_type"] = (
        result.loc[labelled, "Predictions"].astype(int).map(CARE_TYPE_LABELS)
    )
    result.loc[labelled, "care_type_source"] = "given"

    unlabelled = ~labelled
    predictions = model.predict(df.loc[unlabelled, _feature_columns(df)]).astype(int)
    result.loc[unlabelled, "care_type"] = [
        CARE_TYPE_LABELS[int(value)] for value in predictions
    ]
    result.loc[unlabelled, "care_type_source"] = "svm"
    return result


def load_predictions_dataset(
    path: str | Path | None = None, cv_folds: int = 5
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, validate and SVM-complete the teacher CSV.

    Returns ``(dataframe, info)``. The dataframe keeps the 26 original columns
    and adds ``care_type`` and ``care_type_source``. ``info`` carries the
    validation counts, SVM metrics and the documented assumptions.
    """
    raw = read_predictions_csv(path)
    info = validate_predictions_dataset(raw)
    model, metrics = train_care_type_svm(raw, cv_folds=cv_folds)
    completed = assign_care_type(raw, model)
    info["svm_metrics"] = metrics
    info["given"] = int((completed["care_type_source"] == "given").sum())
    info["svm_filled"] = int((completed["care_type_source"] == "svm").sum())
    completed.attrs["info"] = info
    return completed, info


# --------------------------------------------------------------------------- #
# Coordinates — rescaling and the D4 hypothesis check
# --------------------------------------------------------------------------- #
def _resolve_area(
    area: int | float | str | Sequence[float],
    gh_path: str | Path | None,
) -> tuple[float, float, float, float]:
    """Return ``(xmin, xmax, ymin, ymax)`` for the requested target area."""
    if isinstance(area, str):
        if area.lower() not in ("gh", "gehring", "gehring_homberger", "homberger"):
            raise ValueError(f"Unknown area keyword: {area!r}")
        return gehring_homberger_coordinate_range(gh_path)
    if isinstance(area, (tuple, list)):
        width, height = float(area[0]), float(area[1])
        return 0.0, width, 0.0, height
    size = float(area)
    return 0.0, size, 0.0, size


def rescale_coordinates(
    df: pd.DataFrame,
    area: int | float | str | Sequence[float],
    gh_path: str | Path | None = None,
) -> pd.DataFrame:
    """Rescale normalised ``XCOORD.``/``YCOORD.`` onto a target area (D1).

    ``area`` may be ``50``/``100``/``200``, a ``(width, height)`` pair, or the
    keyword ``"gh"`` (the Gehring-Homberger coordinate range). The original
    coordinates are assumed min-max normalised to ``[0, 1]``.
    """
    xmin, xmax, ymin, ymax = _resolve_area(area, gh_path)
    rescaled = df.copy()
    rescaled["XCOORD."] = xmin + df["XCOORD."] * (xmax - xmin)
    rescaled["YCOORD."] = ymin + df["YCOORD."] * (ymax - ymin)
    return rescaled


def gehring_homberger_coordinate_range(
    path: str | Path | None = None,
) -> tuple[float, float, float, float]:
    """Return ``(xmin, xmax, ymin, ymax)`` from a Gehring-Homberger file."""
    instance = parse_gehring_homberger_file(path)
    coordinates = [instance["depot"], *instance["customers"]]
    xs = [row["x"] for row in coordinates]
    ys = [row["y"] for row in coordinates]
    return min(xs), max(xs), min(ys), max(ys)


def _default_gh_file() -> Path:
    paths = sorted(
        p for p in (GEHRING_HOMBERGER_DIR / "1000-customers").glob("*.TXT")
    )
    if not paths:
        raise FileNotFoundError(
            f"No Gehring-Homberger 1000-customer files under {GEHRING_HOMBERGER_DIR}"
        )
    return paths[0]


def check_d4_hypothesis(
    df: pd.DataFrame,
    gh_path: str | Path | None = None,
    tolerance: float = 0.01,
) -> dict[str, float | int]:
    """Compare the CSV coordinates with a Gehring-Homberger instance (D4).

    Both coordinate sets are min-max normalised to ``[0, 1]`` before the
    comparison. The function reports the row-order Pearson correlation, the
    correlation after sorting both sets, and the nearest-neighbour match rate
    within ``tolerance``. It *prints* the result and returns it; the hypothesis
    is not assumed to hold.
    """
    instance = parse_gehring_homberger_file(gh_path)
    gh_customers = instance["customers"]
    gh_x = np.array([row["x"] for row in gh_customers], dtype=float)
    gh_y = np.array([row["y"] for row in gh_customers], dtype=float)

    def _normalize(values: np.ndarray) -> np.ndarray:
        span = values.max() - values.min()
        return (values - values.min()) / span if span else np.zeros_like(values)

    gh_x_norm, gh_y_norm = _normalize(gh_x), _normalize(gh_y)
    csv_x = df["XCOORD."].to_numpy(dtype=float)
    csv_y = df["YCOORD."].to_numpy(dtype=float)

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2 or a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    common = min(len(csv_x), len(gh_x_norm))
    x_order = _pearson(csv_x[:common], gh_x_norm[:common])
    y_order = _pearson(csv_y[:common], gh_y_norm[:common])

    csv_sorted = np.sort(csv_x)[:common]
    gh_sorted = np.sort(gh_x_norm)[:common]
    x_sorted = _pearson(csv_sorted, gh_sorted)

    gh_points = np.column_stack([gh_x_norm, gh_y_norm])
    matches = 0
    for x_value, y_value in zip(csv_x, csv_y):
        if len(gh_points) == 0:
            break
        distances = np.hypot(gh_points[:, 0] - x_value, gh_points[:, 1] - y_value)
        if float(distances.min()) <= tolerance:
            matches += 1
    match_rate = matches / len(csv_x) if len(csv_x) else 0.0

    result: dict[str, float | int] = {
        "x_correlation": round(x_order, 4),
        "y_correlation": round(y_order, 4),
        "x_correlation_sorted": round(x_sorted, 4),
        "match_rate": round(match_rate, 4),
        "tolerance": tolerance,
        "csv_points": int(len(csv_x)),
        "gh_points": int(len(gh_x_norm)),
    }
    print(
        "[D4 hypothesis] CSV vs {}".format(
            (Path(gh_path) if gh_path else _default_gh_file()).name
        )
    )
    print(
        f"  x_corr(order)={result['x_correlation']} "
        f"y_corr(order)={result['y_correlation']} "
        f"x_corr(sorted)={result['x_correlation_sorted']}"
    )
    print(
        f"  match_rate(<= {tolerance}) = {result['match_rate']} "
        f"({matches}/{result['csv_points']})"
    )
    return result


# --------------------------------------------------------------------------- #
# Benchmark parsers (Solomon and Gehring-Homberger, already under data/)
# --------------------------------------------------------------------------- #
def _numbers(line: str) -> list[float] | None:
    try:
        return [float(value) for value in line.split()]
    except ValueError:
        return None


def _parse_benchmark(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    vehicle_count: int | None = None
    vehicle_capacity: float | None = None
    customers: list[dict[str, float]] = []
    in_customers = False

    for index, line in enumerate(lines):
        heading = line.strip().upper()
        if heading.startswith("NUMBER") and "CAPACITY" in heading:
            for candidate in lines[index + 1:]:
                row = _numbers(candidate)
                if row and len(row) >= 2:
                    vehicle_count, vehicle_capacity = int(row[0]), float(row[1])
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
        raise ValueError(f"Invalid benchmark file: {source}")
    patients = [row for row in customers if row["customer_id"] != 0]
    xs = [row["x"] for row in customers]
    ys = [row["y"] for row in customers]
    return {
        "name": source.stem,
        "vehicle_count": vehicle_count,
        "vehicle_capacity": vehicle_capacity,
        "depot": depot,
        "customers": patients,
        "coordinate_range": (min(xs), max(xs), min(ys), max(ys)),
    }


def parse_solomon_file(path: str | Path) -> dict[str, Any]:
    """Parse a Solomon instance file already stored under ``data/``."""
    instance = _parse_benchmark(path)
    instance["family"] = "Solomon"
    return instance


def parse_gehring_homberger_file(path: str | Path | None = None) -> dict[str, Any]:
    """Parse a Gehring-Homberger instance file already stored under ``data/``."""
    instance = _parse_benchmark(path if path is not None else _default_gh_file())
    instance["family"] = "Gehring-Homberger"
    return instance


# --------------------------------------------------------------------------- #
# Bridge to the P2/P4/P11 instance generator
# --------------------------------------------------------------------------- #
def patients_from_dataframe(
    df: pd.DataFrame,
    count: int,
    area: int | float | str | Sequence[float],
    seed: int = 42,
    gh_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build generator patient records (coordinates + care_type) from the CSV.

    Coordinates are rescaled to ``area`` and ``count`` rows are drawn with the
    seeded RNG. ``priority`` defaults to ``"Medium"`` because the teacher CSV
    has no priority column.
    """
    rescaled = rescale_coordinates(df, area, gh_path=gh_path)
    if count > len(rescaled):
        raise ValueError(f"Cannot draw {count} patients from {len(rescaled)} rows")
    import random

    chosen = random.Random(seed).sample(range(len(rescaled)), count)
    has_care_type = "care_type" in rescaled.columns
    records: list[dict[str, Any]] = []
    for index in chosen:
        row = rescaled.iloc[index]
        records.append({
            "x": float(row["XCOORD."]),
            "y": float(row["YCOORD."]),
            "care_type": str(row["care_type"]) if has_care_type else None,
            "priority": "Medium",
        })
    return records


def generate_instance_from_csv(
    df: pd.DataFrame,
    dataset: str,
    caregivers: int,
    location: int,
    seed: int,
    **kwargs: Any,
):
    """Generate a P2/P4/P11 instance using CSV coordinates and care types."""
    from optimisation.mtwmo.generator import DATASET_TABLE, generate_instance

    patient_count = DATASET_TABLE[dataset.upper()][0]
    supplied = patients_from_dataframe(
        df, patient_count, LOCATION_SIDE[location], seed=seed, **kwargs
    )
    return generate_instance(dataset, caregivers, location, seed, patients=supplied)
