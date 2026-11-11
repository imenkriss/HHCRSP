"""Tests A, C and E — dataset integrity, repeated execution, invalid inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

import main
from data import dataset_loader
from data.integrity import (
    DatasetIntegrityError,
    assert_unchanged,
    check_records,
    save_csv_atomic,
    save_json_atomic,
    snapshot_csv,
)

TEACHER = dataset_loader.TEACHER_CSV


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Test A — the pipeline must not change the source dataset
# --------------------------------------------------------------------------- #
def test_pipeline_leaves_source_dataset_unchanged():
    before = snapshot_csv(TEACHER)
    sha_before = _sha(TEACHER)

    result = main.run_hhcop_pipeline(
        "Get the info for patient P1 with high priority",
        patient_limit=25,
        population_size=6,
        generations=1,
    )

    after = snapshot_csv(TEACHER)
    assert _sha(TEACHER) == sha_before
    assert after.rows == before.rows == 1000
    assert after.columns == before.columns
    assert result["dataset_info"]["integrity"]["ok"] is True
    assert result["dataset_info"]["source_rows"] == 1000
    assert result["dataset_info"]["working_patients"] == 25

    # The UI dataset panel receives a multi-row table, not one record.
    frame = pd.DataFrame(result["patients"])
    assert len(frame) == 25
    assert frame["id"].is_unique
    assert result["complexity"]["data_integrity"]["records"]["patients"]["count"] == 25


# --------------------------------------------------------------------------- #
# Test C — repeated execution must be idempotent and not shrink/duplicate
# --------------------------------------------------------------------------- #
def test_repeated_execution_is_idempotent():
    source_rows = [snapshot_csv(TEACHER).rows for _ in range(2)]
    assert source_rows == [1000, 1000]

    kwargs = dict(patient_limit=30, population_size=6, generations=1)
    first = main.run_hhcop_pipeline("Patient P1 urgent", **kwargs)
    second = main.run_hhcop_pipeline("Patient P1 urgent", **kwargs)

    assert len(first["patients"]) == len(second["patients"]) == 30
    first_ids = [record["id"] for record in first["patients"]]
    second_ids = [record["id"] for record in second["patients"]]
    assert first_ids == second_ids
    assert len(set(first_ids)) == 30
    assert [record["x"] for record in first["patients"]] == [
        record["x"] for record in second["patients"]
    ]
    assert snapshot_csv(TEACHER).rows == 1000
    assert second["patients"] == first["patients"]


def test_embedding_cache_is_reused_across_runs():
    main.run_hhcop_pipeline("urgent patient", patient_limit=10, population_size=4, generations=0)
    second = main.run_hhcop_pipeline(
        "urgent patient", patient_limit=10, population_size=4, generations=0
    )
    stats = second["rag"]["embedding_stats"]
    assert stats is not None
    assert stats["hits"] >= 1
    assert stats["cached_vectors"] >= stats["misses"]


# --------------------------------------------------------------------------- #
# Test E — empty and invalid inputs must fail loudly, never silently
# --------------------------------------------------------------------------- #
def test_missing_teacher_dataset_raises(tmp_path):
    missing = tmp_path / "absent.csv"
    with pytest.raises(dataset_loader.DatasetSourceError):
        dataset_loader.load_hhcop_patients(source=missing)


def test_empty_teacher_dataset_raises(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("XCOORD.,YCOORD.\n", encoding="utf-8")
    with pytest.raises(dataset_loader.DatasetSourceError):
        dataset_loader.load_hhcop_patients(source=empty)


def test_missing_coordinate_columns_raise(tmp_path):
    malformed = tmp_path / "malformed.csv"
    malformed.write_text("No,Coords\n1,2\n", encoding="utf-8")
    with pytest.raises(dataset_loader.DatasetSourceError):
        dataset_loader.load_hhcop_patients(source=malformed)


@pytest.mark.parametrize("limit", [0, -3])
def test_invalid_patient_limit_raises(limit):
    with pytest.raises(ValueError):
        dataset_loader.load_hhcop_patients(limit=limit)


@pytest.mark.parametrize("area", [0, -50])
def test_invalid_area_raises(area):
    with pytest.raises(ValueError):
        dataset_loader.build_patient_records(
            dataset_loader.load_teacher_frame(), limit=5, area=area
        )


def test_duplicate_patient_ids_are_reported():
    report = check_records(
        [{"id": "P1", "x": 1, "y": 1, "time_windows": [(0, 1)]},
         {"id": "P1", "x": 2, "y": 2, "time_windows": [(0, 1)]}],
        label="patients",
    )
    assert report.ok is False
    assert report.duplicate_ids == ["P1"]


def test_missing_required_field_is_reported():
    report = check_records([{"id": "P1", "x": 1}], label="patients")
    assert report.ok is False
    assert "y" in report.missing_fields
    assert "time_windows" in report.missing_fields


def test_empty_record_list_is_reported():
    report = check_records([], label="patients", minimum=2)
    assert report.ok is False
    assert "at least 2" in report.problems[0]


def test_assert_unchanged_detects_modification(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    before = {str(path): snapshot_csv(path)}
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    after = {str(path): snapshot_csv(path)}

    with pytest.raises(DatasetIntegrityError):
        assert_unchanged(before, after)


def test_atomic_writers_create_readable_files(tmp_path):
    csv_path = tmp_path / "out.csv"
    json_path = tmp_path / "out.json"
    save_csv_atomic([{"a": 1, "b": "x"}], csv_path)
    save_json_atomic({"ok": True}, json_path)

    assert csv_path.read_text(encoding="utf-8").splitlines() == ["a,b", "1,x"]
    assert '"ok": true' in json_path.read_text(encoding="utf-8")
    # No temporary file is left behind.
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_empty_csv_writes_header_only(tmp_path):
    path = tmp_path / "empty.csv"
    save_csv_atomic([], path, fieldnames=["item_id", "value"])
    assert path.read_text(encoding="utf-8").strip() == "item_id,value"


def test_corrupt_embedding_cache_is_ignored(tmp_path):
    from rag.embeddings import CachedEmbedder, HashingEmbedder

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "hashingembedder-256.json").write_text("{not json", encoding="utf-8")
    embedder = CachedEmbedder(HashingEmbedder(), cache_dir=cache_dir)
    assert len(embedder.embed("urgent patient")) == 256
    assert embedder.flush() is not None
