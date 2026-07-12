from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier

from src.models.canonical_models import (
    CANONICAL_ESTIMATOR_PATHS,
    CANONICAL_MODEL_NAMES,
    CanonicalModelError,
    CanonicalXGBClassifier,
    aligned_predict_proba,
    build_estimator,
    build_model_pipeline,
    validate_model_feature_frame,
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "numeric": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            "category": ["a", "b", None, "a", "b", "c"],
        }
    )


def _parameters(model_name: str) -> tuple[dict, dict]:
    if model_name == "logistic_regression":
        return ({"solver": "lbfgs", "max_iter": 100, "l1_ratio": 0.0}, {"C": 1.0})
    if model_name == "random_forest":
        return ({"n_estimators": 2, "n_jobs": 1}, {"max_depth": 2})
    if model_name == "lightgbm":
        return (
            {
                "n_estimators": 2,
                "learning_rate": 0.1,
                "verbosity": -1,
                "deterministic": True,
                "force_col_wise": True,
                "n_jobs": 1,
            },
            {"num_leaves": 3},
        )
    return (
        {
            "n_estimators": 2,
            "learning_rate": 0.1,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": 1,
        },
        {"max_depth": 2, "class_weight": None},
    )


def test_target_and_identifier_are_rejected_before_pipeline_construction() -> None:
    frame = _features().assign(EmpNumber=range(6), PerformanceRating=[2, 3, 4, 2, 3, 4])
    with pytest.raises(CanonicalModelError, match="Forbidden target/identifier"):
        validate_model_feature_frame(
            frame,
            forbidden_features=["EmpNumber", "PerformanceRating"],
        )


@pytest.mark.parametrize(
    "frame, message",
    [
        (pd.DataFrame({1: [1.0, 2.0]}), "must be strings"),
        (pd.DataFrame({"   ": [1.0, 2.0]}), "non-blank"),
        (pd.DataFrame({"all_null": [None, None]}), "at least one observed value"),
    ],
)
def test_feature_schema_rejects_non_string_blank_and_all_null_columns(
    frame: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(CanonicalModelError, match=message):
        validate_model_feature_frame(frame)


def test_all_four_models_share_the_same_train_fitted_preprocessing_contract() -> None:
    frame = _features()
    schemas = []
    for model_name in CANONICAL_MODEL_NAMES:
        fixed, candidate = _parameters(model_name)
        pipeline = build_model_pipeline(
            model_name,
            frame,
            fixed_parameters=fixed,
            candidate_parameters=candidate,
            random_state=42,
        )
        assert is_classifier(pipeline)
        preprocessor = pipeline.named_steps["preprocessor"]
        schemas.append(
            [
                (name, tuple(columns), tuple(transformer.named_steps))
                for name, transformer, columns in preprocessor.transformers
            ]
        )
    assert all(schema == schemas[0] for schema in schemas)
    assert schemas[0] == [
        ("numeric", ("numeric",), ("imputer", "scaler")),
        ("categorical", ("category",), ("imputer", "one_hot")),
    ]


def test_xgboost_balanced_weights_are_derived_from_current_fit_labels_only() -> None:
    observed: list[np.ndarray] = []

    def _weights(*, class_weight, y):
        assert class_weight == "balanced"
        observed.append(np.asarray(y).copy())
        return np.ones(len(y), dtype=float)

    class _FakeXGB:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y, sample_weight=None):
            self.sample_weight = np.asarray(sample_weight)
            return self

    labels = np.asarray([2, 2, 3, 4])
    estimator = CanonicalXGBClassifier(n_estimators=2, class_weight="balanced", n_jobs=1)
    with patch("src.models.canonical_models.compute_sample_weight", side_effect=_weights), patch(
        "xgboost.XGBClassifier", _FakeXGB
    ):
        estimator.fit(np.zeros((4, 2)), labels)

    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], labels)
    assert estimator.sample_weight_source_ == "current_fit_training_labels_only"


def test_probability_columns_are_reordered_to_global_label_contract() -> None:
    class _ReversedClassifier:
        classes_ = np.asarray([4, 3, 2])

        def predict_proba(self, features):
            return np.tile([0.6, 0.3, 0.1], (len(features), 1))

    aligned = aligned_predict_proba(_ReversedClassifier(), _features().iloc[:2], labels=[2, 3, 4])
    np.testing.assert_allclose(aligned, [[0.1, 0.3, 0.6], [0.1, 0.3, 0.6]])


@pytest.mark.parametrize(
    "classes, probabilities, labels, message",
    [
        ([2, 3, 4], [[-0.1, 0.5, 0.6], [-0.1, 0.5, 0.6]], [2, 3, 4], "outside"),
        ([2, 3, 4], [[0.2, 0.3, 0.5]], [2, 3, 4], "shape"),
        ([2, 2, 4], [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]], [2, 3, 4], "duplicates"),
        ([2, 3, 4], [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]], [2, 2, 4], "duplicates"),
    ],
)
def test_probability_alignment_rejects_bounds_row_count_and_duplicate_labels(
    classes,
    probabilities,
    labels,
    message,
) -> None:
    class _Classifier:
        classes_ = np.asarray(classes)

        def predict_proba(self, features):
            return np.asarray(probabilities, dtype=float)

    features = _features().iloc[:2]
    with pytest.raises(CanonicalModelError, match=message):
        aligned_predict_proba(_Classifier(), features, labels=labels)


def test_random_state_is_owned_by_protocol_and_estimator_registry_is_immutable() -> None:
    with pytest.raises(CanonicalModelError, match="owned by the outer scientific protocol"):
        build_estimator("logistic_regression", {"C": 1.0, "random_state": 99}, random_state=42)
    with pytest.raises(CanonicalModelError, match="Unsupported parameters"):
        build_estimator("lightgbm", {"unknown_scientific_knob": 1}, random_state=42)
    with pytest.raises(TypeError):
        CANONICAL_ESTIMATOR_PATHS["xgboost"] = "different.Estimator"  # type: ignore[index]


def test_estimators_reject_nested_parallel_oversubscription() -> None:
    fixed, candidate = _parameters("random_forest")
    fixed["n_jobs"] = -1
    with pytest.raises(CanonicalModelError, match="n_jobs=1"):
        build_model_pipeline(
            "random_forest",
            _features(),
            fixed_parameters=fixed,
            candidate_parameters=candidate,
            random_state=42,
        )
