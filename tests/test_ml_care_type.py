import numpy as np
import pytest

from ml.care_type import (
    EXCLUDED_COLUMNS,
    METRIC_KEYS,
    CareTypeClassifier,
    feature_columns,
    load_teacher_dataframe,
    train_care_type_classifier,
)


@pytest.fixture(scope="module")
def teacher():
    return load_teacher_dataframe()


@pytest.fixture(scope="module")
def classifier(teacher):
    return train_care_type_classifier(teacher)


def test_feature_columns_exclude_target_and_coordinates(teacher):
    columns = feature_columns(teacher)

    assert "Predictions" not in columns
    for name in EXCLUDED_COLUMNS:
        assert name not in columns
    assert "XCOORD." not in columns and "YCOORD." not in columns


def test_metrics_are_bounded_and_present(classifier):
    for key in METRIC_KEYS:
        assert key in classifier.metrics
        assert 0.0 <= classifier.metrics[key] <= 1.0


def test_two_opaque_classes(classifier):
    assert classifier.classes == [0, 1]


def test_label_is_deterministic_function_of_diabetic(teacher, classifier):
    # D2: on labelled rows Predictions == 1 - Diabetic exactly (leaky target).
    labelled = teacher[teacher["Predictions"].notna()]
    assert (labelled["Predictions"] == 1 - labelled["Diabetic"]).all()
    # The leaky target makes the measured classifier near-perfect; assert the
    # measured value equals the measured value, not a paper number.
    assert classifier.metrics["accuracy"] == pytest.approx(1.0)


def test_predict_returns_int_codes(teacher, classifier):
    codes = classifier.predict(teacher.iloc[:25])

    assert codes.shape == (25,)
    assert set(np.unique(codes)) <= {0, 1}
    labels = classifier.predict_labels(teacher.iloc[:25])
    assert set(labels) <= {"type_0", "type_1"}


def test_complete_fills_all_rows(teacher, classifier):
    completed = classifier.complete(teacher)

    assert completed["care_type_code"].notna().all()
    assert set(completed["care_type_code"].unique()) <= {0, 1}
    assert int((completed["care_type_source"] == "given").sum()) == 300
    assert int((completed["care_type_source"] == "svm").sum()) == 700


def test_training_is_deterministic(teacher):
    first = CareTypeClassifier().fit(teacher)
    second = CareTypeClassifier().fit(teacher)

    assert first.metrics == second.metrics
    assert np.array_equal(first.predict(teacher), second.predict(teacher))
