import csv
from pathlib import Path

import pytest

from ml.care_type import train_care_type_classifier, load_teacher_dataframe
from ml.evaluation import (
    attach_care_codes,
    gehring_subset,
    solomon_subset,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def classifier():
    return train_care_type_classifier(load_teacher_dataframe())


def test_solomon_subset_sizes():
    depot, points = solomon_subset(ROOT / "data" / "Solomon" / "In" / "c101.txt", 25)

    assert len(points) == 25
    assert len(depot) == 2
    assert all(len(point) == 2 for point in points)


def test_gehring_subset_sizes():
    path = ROOT / "data" / "Gehing&Homberger" / "200-custmers" / "C1_2_1.TXT"
    depot, points = gehring_subset(path, 30)

    assert len(points) == 30


def test_attach_care_codes_is_cyclic():
    codes = attach_care_codes(5, [0, 1, 0])

    assert codes == [0, 1, 0, 0, 1]


def test_evaluate_writes_measured_rows(tmp_path, classifier):
    from ml.evaluation import evaluate

    specs = [
        {"family": "Solomon", "instance": "c101",
         "path": ROOT / "data" / "Solomon" / "In" / "c101.txt", "sizes": [12]}
    ]
    out = tmp_path / "ml_benchmark.csv"
    rows = evaluate(specs=specs, seed=1, tabu_kwargs={"iterations": 10},
                    out_path=out, classifier=classifier)

    assert len(rows) == 1
    row = rows[0]
    assert row["size"] == 12
    assert row["svm_total_cost"] > 0 and row["baseline_total_cost"] > 0
    assert out.exists()
    with out.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 1
    assert written[0]["instance"] == "c101"
