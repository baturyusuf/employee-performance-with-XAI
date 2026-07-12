"""Canonical model and preprocessing factories for the manuscript benchmark.

Every benchmark family consumes the same training-fitted transformation.  The
module contains no data loading or split selection so callers cannot
accidentally fit preprocessing outside an explicitly supplied training split.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


CANONICAL_MODEL_NAMES = (
    "logistic_regression",
    "random_forest",
    "lightgbm",
    "xgboost",
)

CANONICAL_ESTIMATOR_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "logistic_regression": "sklearn.linear_model.LogisticRegression",
        "random_forest": "sklearn.ensemble.RandomForestClassifier",
        "lightgbm": "lightgbm.LGBMClassifier",
        "xgboost": "xgboost.XGBClassifier",
    }
)

CANONICAL_PARAMETER_NAMES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "logistic_regression": frozenset(
            {"solver", "max_iter", "tol", "l1_ratio", "C", "class_weight"}
        ),
        "random_forest": frozenset(
            {
                "n_estimators",
                "max_features",
                "bootstrap",
                "n_jobs",
                "max_depth",
                "min_samples_leaf",
                "class_weight",
            }
        ),
        "lightgbm": frozenset(
            {
                "n_estimators",
                "learning_rate",
                "subsample",
                "subsample_freq",
                "colsample_bytree",
                "deterministic",
                "force_col_wise",
                "verbosity",
                "n_jobs",
                "num_leaves",
                "min_child_samples",
                "class_weight",
            }
        ),
        "xgboost": frozenset(
            {
                "n_estimators",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "objective",
                "eval_metric",
                "tree_method",
                "reg_lambda",
                "n_jobs",
                "verbosity",
                "max_depth",
                "min_child_weight",
                "class_weight",
            }
        ),
    }
)


class CanonicalModelError(ValueError):
    """Raised when a model or preprocessing contract is unsafe or incomplete."""


def validate_model_feature_frame(
    frame: pd.DataFrame,
    *,
    forbidden_features: Sequence[str] = (),
) -> None:
    """Reject empty, duplicate, or prohibited model inputs before fitting."""

    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.shape[1] == 0:
        raise CanonicalModelError("Model feature input must be a non-empty DataFrame.")
    non_string = [repr(column) for column in frame.columns if not isinstance(column, str)]
    if non_string:
        raise CanonicalModelError(f"Model feature names must be strings: {non_string}.")
    blank_columns = [column for column in frame.columns if not column.strip()]
    if blank_columns:
        raise CanonicalModelError("Model feature names must be non-blank strings.")
    duplicate_columns = frame.columns[frame.columns.duplicated()].astype(str).tolist()
    if duplicate_columns:
        raise CanonicalModelError(f"Model input has duplicate columns: {duplicate_columns}.")
    forbidden = sorted(set(map(str, forbidden_features)).intersection(map(str, frame.columns)))
    if forbidden:
        raise CanonicalModelError(f"Forbidden target/identifier features entered model input: {forbidden}.")
    all_null = [str(column) for column in frame.columns if frame[column].isna().all()]
    if all_null:
        raise CanonicalModelError(
            f"Training feature columns must contain at least one observed value: {all_null}."
        )


def build_common_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    """Create the shared train-fitted dense transformation for all four models."""

    validate_model_feature_frame(frame)
    numeric_columns = [
        str(column)
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical_columns = [str(column) for column in frame.columns if str(column) not in numeric_columns]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:  # protected by the non-empty feature check
        raise CanonicalModelError("No numeric or categorical feature groups were inferred.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


class CanonicalXGBClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost classifier with sklearn tags and training-only label/weight logic."""

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        objective: str = "multi:softprob",
        eval_metric: str = "mlogloss",
        tree_method: str = "hist",
        reg_lambda: float = 1.0,
        n_jobs: int = 1,
        verbosity: int = 0,
        max_depth: int = 3,
        min_child_weight: float = 1.0,
        class_weight: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "CanonicalXGBClassifier":
        from xgboost import XGBClassifier

        if self.n_jobs != 1:
            raise CanonicalModelError("Canonical XGBoost must be single-threaded inside nested search.")
        if self.class_weight not in {None, "balanced"}:
            raise CanonicalModelError(
                "Canonical XGBoost class_weight must be null or 'balanced'."
            )
        labels = np.asarray(y)
        self.label_encoder_ = LabelEncoder().fit(labels)
        self.classes_ = self.label_encoder_.classes_
        encoded = self.label_encoder_.transform(labels)
        sample_weight = (
            compute_sample_weight(class_weight="balanced", y=labels)
            if self.class_weight == "balanced"
            else None
        )
        self.sample_weight_source_ = (
            "current_fit_training_labels_only" if sample_weight is not None else "none"
        )
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective=self.objective,
            eval_metric=self.eval_metric,
            tree_method=self.tree_method,
            reg_lambda=self.reg_lambda,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            random_state=self.random_state,
            num_class=len(self.classes_),
        )
        self.model_.fit(X, encoded, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        encoded = np.asarray(self.model_.predict(X), dtype=int)
        return self.label_encoder_.inverse_transform(encoded)

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(X), dtype=float)


def merge_model_parameters(
    fixed_parameters: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    overlap = sorted(set(fixed_parameters).intersection(candidate_parameters))
    if overlap:
        raise CanonicalModelError(
            f"Candidate parameters cannot overwrite fixed parameters: {overlap}."
        )
    return {**dict(fixed_parameters), **dict(candidate_parameters)}


def build_estimator(
    model_name: str,
    parameters: Mapping[str, Any],
    *,
    random_state: int,
) -> BaseEstimator:
    """Instantiate one of the exact four accepted single-thread estimators."""

    if model_name not in CANONICAL_MODEL_NAMES:
        raise CanonicalModelError(
            f"Unknown canonical model {model_name!r}; expected {list(CANONICAL_MODEL_NAMES)}."
        )
    params = dict(parameters)
    if "random_state" in params:
        raise CanonicalModelError(
            "random_state is owned by the outer scientific protocol and must not appear "
            "in fixed or candidate parameters."
        )
    unexpected = sorted(set(params).difference(CANONICAL_PARAMETER_NAMES[model_name]))
    if unexpected:
        raise CanonicalModelError(
            f"Unsupported parameters for canonical model {model_name!r}: {unexpected}."
        )
    params["random_state"] = int(random_state)
    if model_name == "logistic_regression":
        return LogisticRegression(**params)
    if model_name == "random_forest":
        if params.get("n_jobs", 1) != 1:
            raise CanonicalModelError("Canonical Random Forest must use n_jobs=1.")
        params["n_jobs"] = 1
        return RandomForestClassifier(**params)
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        if params.get("n_jobs", 1) != 1:
            raise CanonicalModelError("Canonical LightGBM must use n_jobs=1.")
        params["n_jobs"] = 1
        return LGBMClassifier(**params)
    if params.get("n_jobs", 1) != 1:
        raise CanonicalModelError("Canonical XGBoost must use n_jobs=1.")
    params["n_jobs"] = 1
    return CanonicalXGBClassifier(**params)


def build_model_pipeline(
    model_name: str,
    training_features: pd.DataFrame,
    *,
    fixed_parameters: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any],
    random_state: int,
    forbidden_features: Sequence[str] = (),
) -> Pipeline:
    """Build an unfitted common-preprocessing/model pipeline for one split."""

    validate_model_feature_frame(training_features, forbidden_features=forbidden_features)
    parameters = merge_model_parameters(fixed_parameters, candidate_parameters)
    estimator = build_estimator(model_name, parameters, random_state=random_state)
    return Pipeline(
        [
            ("preprocessor", build_common_preprocessor(training_features)),
            ("model", estimator),
        ]
    )


def aligned_predict_proba(
    fitted_estimator: Any,
    features: pd.DataFrame,
    *,
    labels: Sequence[int],
) -> np.ndarray:
    """Return probabilities in the declared global label order or fail closed."""

    observed_classes = np.asarray(getattr(fitted_estimator, "classes_", ()))
    expected_classes = np.asarray(list(labels))
    if observed_classes.ndim != 1 or expected_classes.ndim != 1 or len(expected_classes) == 0:
        raise CanonicalModelError("Observed and expected classifier labels must be non-empty vectors.")
    if len(np.unique(observed_classes)) != len(observed_classes):
        raise CanonicalModelError(f"Classifier classes contain duplicates: {observed_classes.tolist()}.")
    if len(np.unique(expected_classes)) != len(expected_classes):
        raise CanonicalModelError(f"Declared labels contain duplicates: {expected_classes.tolist()}.")
    if len(observed_classes) != len(expected_classes) or set(observed_classes.tolist()) != set(
        expected_classes.tolist()
    ):
        raise CanonicalModelError(
            f"Classifier classes {observed_classes.tolist()} do not match labels "
            f"{expected_classes.tolist()}."
        )
    probabilities = np.asarray(fitted_estimator.predict_proba(features), dtype=float)
    expected_shape = (len(features), len(observed_classes))
    if probabilities.ndim != 2 or probabilities.shape != expected_shape:
        raise CanonicalModelError(
            f"Invalid probability shape {probabilities.shape} for classes "
            f"{observed_classes.tolist()}; expected {expected_shape}."
        )
    positions = [int(np.where(observed_classes == label)[0][0]) for label in expected_classes]
    aligned = probabilities[:, positions]
    if not np.all(np.isfinite(aligned)):
        raise CanonicalModelError("Predicted probabilities are non-finite.")
    if np.any(aligned < -1e-12) or np.any(aligned > 1.0 + 1e-12):
        raise CanonicalModelError("Predicted probabilities fall outside [0,1].")
    if not np.allclose(aligned.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise CanonicalModelError("Predicted probabilities do not sum to one.")
    return aligned


__all__ = [
    "CANONICAL_ESTIMATOR_PATHS",
    "CANONICAL_MODEL_NAMES",
    "CANONICAL_PARAMETER_NAMES",
    "CanonicalModelError",
    "CanonicalXGBClassifier",
    "aligned_predict_proba",
    "build_common_preprocessor",
    "build_estimator",
    "build_model_pipeline",
    "merge_model_parameters",
    "validate_model_feature_frame",
]
