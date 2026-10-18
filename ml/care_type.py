"""SVM (One-vs-Rest) care-type classifier (D3).

The teacher file ``data/dementia/new_dataset_with_predictions.csv`` has only
300 labelled rows (column ``Predictions``), 700 NaN. The label is a
deterministic function of ``Diabetic`` (D2), so the measured metrics are
near-perfect; this is reported honestly, not hidden.

The classifier is a sklearn ``Pipeline`` (imputer -> scaler -> OneVsRest SVM)
and exposes macro-averaged accuracy / precision / F1 computed with stratified
cross-validation on the labelled rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Columns never used as features: the target and the raw coordinates.
EXCLUDED_COLUMNS = ("Predictions", "XCOORD.", "YCOORD.")
LABEL_TARGET = "Predictions"
CARE_TYPE_LABELS = {0: "type_0", 1: "type_1"}
LABEL_TO_CODE = {name: code for code, name in CARE_TYPE_LABELS.items()}

TEACHER_CSV = Path(__file__).resolve().parents[1] / "data" / "dementia" / (
    "new_dataset_with_predictions.csv"
)

METRIC_KEYS = ("accuracy", "precision", "f1")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the numeric feature columns (target and coordinates removed)."""
    return [column for column in df.columns if column not in EXCLUDED_COLUMNS]


def load_teacher_dataframe(path: Path | str | None = None) -> pd.DataFrame:
    """Read the teacher CSV (read-only) as a DataFrame."""
    return pd.read_csv(Path(path) if path is not None else TEACHER_CSV)


def _scoring() -> dict:
    return {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average="macro", zero_division=0),
        "f1": make_scorer(f1_score, average="macro", zero_division=0),
    }


@dataclass
class CareTypeClassifier:
    """One-vs-Rest SVM that predicts the patient care type (D3)."""

    cv_folds: int = 5
    random_state: int = 42
    kernel: str = "linear"
    C: float = 1.0
    pipeline: Pipeline | None = None
    metrics: dict = field(default_factory=dict)
    classes: list = field(default_factory=list)

    def _build(self) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "svm",
                    OneVsRestClassifier(
                        SVC(kernel=self.kernel, C=self.C, random_state=self.random_state)
                    ),
                ),
            ]
        )

    def fit(self, df: pd.DataFrame, evaluate: bool = True) -> "CareTypeClassifier":
        columns = feature_columns(df)
        labelled = df[df[LABEL_TARGET].notna()].copy()
        if labelled.empty:
            raise ValueError("no labelled rows to fit the care-type classifier")

        features = labelled[columns].to_numpy(dtype=float)
        target = labelled[LABEL_TARGET].to_numpy(dtype=int)
        self.classes = sorted(set(target.tolist()))

        self.pipeline = self._build()
        self.pipeline.fit(features, target)

        if evaluate:
            self.metrics = self.cross_validate_metrics(features, target)
        else:
            self.metrics = {}
        return self

    def cross_validate_metrics(
        self, features: np.ndarray, target: np.ndarray
    ) -> dict:
        folds = max(2, min(self.cv_folds, int(np.bincount(target).min())))
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=self.random_state
        )
        scores = cross_validate(
            self._build(), features, target, scoring=_scoring(), cv=splitter
        )
        return {
            key: float(np.mean(scores[f"test_{key}"])) for key in METRIC_KEYS
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("classifier is not fitted")
        return self.pipeline.predict(df[feature_columns(df)].to_numpy(dtype=float))

    def predict_labels(self, df: pd.DataFrame) -> list[str]:
        return [CARE_TYPE_LABELS[int(code)] for code in self.predict(df)]

    def complete(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with an integer ``care_type_code`` for every row.

        Labelled rows keep their given ``Predictions`` value; unlabelled rows
        receive the SVM prediction (mirrors ``data.loader``).
        """
        result = df.copy()
        predicted = self.predict(df)
        given = result[LABEL_TARGET].to_numpy(dtype=float)
        codes = np.where(np.isnan(given), predicted, given).astype(int)
        result["care_type_code"] = codes
        result["care_type_source"] = np.where(
            np.isnan(given), "svm", "given"
        )
        return result


def train_care_type_classifier(
    df: pd.DataFrame | None = None, **kwargs
) -> CareTypeClassifier:
    """Convenience: load (if needed) and train the care-type classifier."""
    if df is None:
        df = load_teacher_dataframe()
    return CareTypeClassifier(**kwargs).fit(df)
