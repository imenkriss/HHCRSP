from pathlib import Path

import pandas as pd
import pytest

from data import loader

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset():
    return loader.load_predictions_dataset()


def test_teacher_csv_validates_1000_rows_and_26_columns(dataset):
    df, info = dataset

    assert info["rows"] == 1000
    assert info["columns"] == 26
    assert df.shape[0] == 1000
    # 26 original columns plus the two derived columns.
    assert df.shape[1] == 28


def test_invalid_shape_is_rejected():
    with pytest.raises(ValueError):
        loader.validate_predictions_dataset(pd.DataFrame({"Predictions": [0.0]}))


def test_care_type_source_is_given_or_svm(dataset):
    df, _ = dataset

    assert set(df["care_type_source"]) == {"given", "svm"}
    assert int((df["care_type_source"] == "given").sum()) == 300
    assert int((df["care_type_source"] == "svm").sum()) == 700
    assert df["care_type"].notna().all()


def test_svm_metrics_are_bounded_and_label_is_leaky(dataset):
    df, info = dataset

    metrics = info["svm_metrics"]
    for key in ("accuracy", "precision", "f1"):
        assert 0.0 <= metrics[key] <= 1.0
    assert info["classes"] == [0, 1]

    given = df[df["care_type_source"] == "given"]
    assert (given["Predictions"] == 1 - given["Diabetic"]).all()


def test_care_type_uses_only_two_opaque_classes(dataset):
    df, _ = dataset

    assert set(df["care_type"].unique()) <= {"type_0", "type_1"}


def test_rescale_coordinates_to_square_areas(dataset):
    df, _ = dataset

    for area in (50, 100, 200):
        rescaled = loader.rescale_coordinates(df, area)
        assert rescaled["XCOORD."].min() >= 0.0
        assert rescaled["XCOORD."].max() <= area
        assert rescaled["YCOORD."].max() <= area
    # The original frame must not be mutated.
    assert df["XCOORD."].max() <= 1.0


def test_rescale_coordinates_to_gehring_homberger_range(dataset):
    df, _ = dataset

    xmin, xmax, ymin, ymax = loader.gehring_homberger_coordinate_range()
    rescaled = loader.rescale_coordinates(df, "gh")

    assert rescaled["XCOORD."].min() == pytest.approx(xmin)
    assert rescaled["XCOORD."].max() == pytest.approx(xmax)
    assert rescaled["YCOORD."].max() == pytest.approx(ymax)


def test_d4_hypothesis_returns_correlations_and_match_rate(dataset):
    df, _ = dataset

    result = loader.check_d4_hypothesis(df)

    assert -1.0 <= result["x_correlation"] <= 1.0
    assert -1.0 <= result["y_correlation"] <= 1.0
    assert 0.0 <= result["match_rate"] <= 1.0
    assert result["csv_points"] == 1000
    assert result["gh_points"] == 1000


def test_parse_solomon_file():
    instance = loader.parse_solomon_file(ROOT / "data" / "Solomon" / "In" / "c101.txt")

    assert instance["family"] == "Solomon"
    assert instance["vehicle_count"] == 25
    assert instance["depot"]["customer_id"] == 0
    assert len(instance["customers"]) == 100
    assert all(row["customer_id"] != 0 for row in instance["customers"])


def test_parse_gehring_homberger_file():
    path = ROOT / "data" / "Gehing&Homberger" / "1000-customers" / "C1_10_1.TXT"
    instance = loader.parse_gehring_homberger_file(path)

    assert instance["family"] == "Gehring-Homberger"
    assert instance["vehicle_count"] == 250
    assert instance["depot"]["customer_id"] == 0
    assert len(instance["customers"]) == 1000


def test_patients_from_dataframe_uses_rescaled_coordinates_and_care_type(dataset):
    df, _ = dataset

    records = loader.patients_from_dataframe(df, count=10, area=100, seed=1)

    assert len(records) == 10
    for record in records:
        assert 0.0 <= record["x"] <= 100.0
        assert 0.0 <= record["y"] <= 100.0
        assert record["care_type"] in {"type_0", "type_1"}
