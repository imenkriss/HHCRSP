"""Dataset integrity checks and atomic output writing.

The pipeline must never mutate its source data. This module provides:

* :class:`DatasetSnapshot` / :func:`snapshot_csv` — a fingerprint (sha256), row
  count, columns, duplicate-id count and missing-value count for one source
  file;
* :func:`assert_unchanged` — raises :class:`DatasetIntegrityError` when a file
  changed between two snapshots;
* :func:`check_records` — structural validation of the in-memory patient or
  caregiver records (count, unique ids, required fields, missing values);
* atomic writers (:func:`save_json_atomic`, :func:`save_csv_atomic`,
  :func:`save_dataframe_atomic`) that write to a temporary file in the target
  directory and ``os.replace`` it, so an interrupted run cannot leave a
  half-written export;
* :func:`ensure_output_dir` — creates ``data/generated`` / ``outputs`` style
  directories on demand (never ``data/`` itself).

Only stdlib + pandas are used here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = PROJECT_ROOT / "data" / "generated"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


class DatasetIntegrityError(RuntimeError):
    """Raised when source data is missing, malformed or unexpectedly modified."""


# --------------------------------------------------------------------------- #
# File fingerprints
# --------------------------------------------------------------------------- #
def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the sha256 hex digest of ``path`` (empty string if absent)."""
    source = Path(path)
    if not source.exists():
        return ""
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetSnapshot:
    """Immutable description of one source dataset file."""

    path: str
    exists: bool
    size_bytes: int
    sha256: str
    rows: int | None = None
    columns: tuple[str, ...] = ()
    duplicate_ids: int = 0
    missing_values: int = 0

    @property
    def fingerprint(self) -> tuple[Any, ...]:
        """Value compared by :func:`assert_unchanged`."""
        return (self.exists, self.size_bytes, self.sha256, self.rows, self.columns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "rows": self.rows,
            "columns": list(self.columns),
            "duplicate_ids": self.duplicate_ids,
            "missing_values": self.missing_values,
        }


def snapshot_csv(
    path: str | Path,
    *,
    id_column: str | None = None,
) -> DatasetSnapshot:
    """Fingerprint a CSV file without modifying it.

    ``rows``/``columns``/quality counters are ``None``/empty for a missing or
    0-byte file; that is recorded instead of raising so callers can report
    "source missing" themselves.
    """
    source = Path(path)
    if not source.exists():
        return DatasetSnapshot(str(source), False, 0, "")
    size = source.stat().st_size
    digest = file_sha256(source)
    if size == 0:
        return DatasetSnapshot(str(source), True, 0, digest, 0, ())
    try:
        frame = pd.read_csv(source)
    except Exception as error:  # pragma: no cover - depends on pandas version text
        raise DatasetIntegrityError(f"Cannot parse CSV {source}: {error}") from error
    duplicate_ids = 0
    if id_column and id_column in frame.columns:
        duplicate_ids = int(frame[id_column].duplicated().sum())
    return DatasetSnapshot(
        path=str(source),
        exists=True,
        size_bytes=size,
        sha256=digest,
        rows=int(len(frame)),
        columns=tuple(str(column) for column in frame.columns),
        duplicate_ids=duplicate_ids,
        missing_values=int(frame.isna().sum().sum()),
    )


def snapshot_files(
    paths: Iterable[str | Path], *, id_columns: Mapping[str, str] | None = None
) -> dict[str, DatasetSnapshot]:
    """Snapshot several files, keyed by their string path."""
    id_columns = id_columns or {}
    snapshots: dict[str, DatasetSnapshot] = {}
    for path in paths:
        key = str(Path(path))
        snapshots[key] = snapshot_csv(key, id_column=id_columns.get(key))
    return snapshots


def assert_unchanged(
    before: Mapping[str, DatasetSnapshot],
    after: Mapping[str, DatasetSnapshot],
    *,
    label: str = "dataset",
) -> None:
    """Raise if any source file changed (content, size, rows or columns)."""
    problems: list[str] = []
    for key, original in before.items():
        current = after.get(key) or snapshot_csv(key)
        if original.fingerprint != current.fingerprint:
            problems.append(
                f"{key}: rows {original.rows}->{current.rows}, "
                f"size {original.size_bytes}->{current.size_bytes}, "
                f"sha256 {original.sha256[:12]}->{current.sha256[:12]}"
            )
    if problems:
        raise DatasetIntegrityError(
            f"{label} was modified during execution (source data must stay intact):\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )


# --------------------------------------------------------------------------- #
# In-memory record validation
# --------------------------------------------------------------------------- #
DEFAULT_PATIENT_FIELDS = ("id", "x", "y", "time_windows")
DEFAULT_CAREGIVER_FIELDS = ("id", "max_work_hours", "available")


@dataclass
class RecordReport:
    """Structural validation result for a list of in-memory records."""

    label: str
    count: int
    unique_ids: int
    duplicate_ids: list[Any] = field(default_factory=list)
    missing_fields: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "count": self.count,
            "unique_ids": self.unique_ids,
            "duplicate_ids": list(self.duplicate_ids),
            "missing_fields": dict(self.missing_fields),
            "problems": list(self.problems),
            "ok": self.ok,
        }


def check_records(
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
    required: Sequence[str] = DEFAULT_PATIENT_FIELDS,
    minimum: int = 1,
) -> RecordReport:
    """Validate count, unique ids and required (non-empty) fields."""
    count = len(records)
    report = RecordReport(label=label, count=count, unique_ids=0)
    if count < minimum:
        report.problems.append(f"{label}: expected at least {minimum} records, got {count}")

    ids: list[Any] = []
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            report.problems.append(f"{label}: record {position} is not a mapping")
            continue
        for name in required:
            if name not in record or record.get(name) in (None, "", []):
                report.missing_fields[name] = report.missing_fields.get(name, 0) + 1
        if "id" in record and record.get("id") is not None:
            ids.append(record["id"])

    seen: set[Any] = set()
    duplicates: set[Any] = set()
    for value in ids:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    report.unique_ids = len(seen)
    report.duplicate_ids = sorted(duplicates, key=str)
    if duplicates:
        report.problems.append(f"{label}: duplicate ids {report.duplicate_ids}")
    for name, missing in report.missing_fields.items():
        report.problems.append(f"{label}: {missing} record(s) missing '{name}'")
    return report


def validate_exports(table: Sequence[Mapping[str, Any]], *, label: str) -> None:
    """Guard a comparison table before display/export: keys must be consistent."""
    if not table:
        return
    first = tuple(table[0].keys())
    for position, row in enumerate(table):
        if tuple(row.keys()) != first:
            raise DatasetIntegrityError(
                f"{label}: row {position} has columns {tuple(row.keys())} "
                f"instead of {first}"
            )


# --------------------------------------------------------------------------- #
# Atomic output writing
# --------------------------------------------------------------------------- #
def ensure_output_dir(path: str | Path | None = None) -> Path:
    """Create an output directory (default ``outputs/``) and return it."""
    target = Path(path) if path is not None else OUTPUT_DIR
    if target.name == "data" or target.resolve() == (PROJECT_ROOT / "data").resolve():
        raise DatasetIntegrityError("refusing to use the raw data/ directory as output")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _atomic_write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return path


def save_json_atomic(payload: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Write JSON atomically (UTF-8)."""
    text = json.dumps(payload, indent=indent, ensure_ascii=False, default=str)
    return _atomic_write_bytes(Path(path), text.encode("utf-8"))


def save_csv_atomic(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write a list of mappings as CSV atomically (header only when empty)."""
    target = Path(path)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return _atomic_write_bytes(target, buffer.getvalue().encode("utf-8"))


def save_dataframe_atomic(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
) -> Path:
    """Write a DataFrame as CSV atomically."""
    target = Path(path)
    return _atomic_write_bytes(target, frame.to_csv(index=index).encode("utf-8"))


def rows_to_json_text(rows: Sequence[Mapping[str, Any]], *, indent: int = 2) -> str:
    """Serialise comparison rows to JSON text (used by the UI download button)."""
    return json.dumps(list(rows), indent=indent, ensure_ascii=False, default=str)


def rows_to_csv_text(
    rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str] | None = None
) -> str:
    """Serialise comparison rows to CSV text (used by the UI download button)."""
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return buffer.getvalue()
