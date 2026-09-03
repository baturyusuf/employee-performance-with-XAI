"""Additive v3 ordinal estimators and training-only naive baselines.

These estimators are deliberately isolated from the immutable v2 model
registry.  They accept already split training data and can be placed after the
existing common preprocessing transformer inside an sklearn pipeline.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted, validate_data


V3_ORDINAL_MODEL_NAMES = (
    "proportional_odds_logistic",
    "cumulative_threshold_xgboost",
)
V3_NAIVE_BASELINE_NAMES = (
    "majority_baseline",
    "stratified_baseline",
    "ordinal_median_baseline",
)

V3_ORDINAL_ESTIMATOR_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "proportional_odds_logistic": (
            "src.models.ordinal_models_v3.ProportionalOddsClassifier"
        ),
        "cumulative_threshold_xgboost": (
            "src.models.ordinal_models_v3.CumulativeThresholdXGBClassifier"
        ),
    }
)

V3_BASELINE_STRATEGIES: Mapping[str, str] = MappingProxyType(
    {
        "majority_baseline": "majority",
        "stratified_baseline": "stratified",
        "ordinal_median_baseline": "ordinal_median",
    }
)


class OrdinalModelContractError(ValueError):
    """Raised when an ordinal estimator or baseline contract is invalid."""


def _validated_ordered_classes(y: np.ndarray) -> np.ndarray:
    if y.ndim != 1 or len(y) == 0:
        raise OrdinalModelContractError("Ordinal targets must be a non-empty vector.")
    if any(value is None for value in y.tolist()):
        raise OrdinalModelContractError("Ordinal targets cannot contain missing values.")
    try:
        classes = np.unique(y)
    except TypeError as exc:
        raise OrdinalModelContractError(
            "Ordinal labels must have one deterministic sortable type."
        ) from exc
    if len(classes) < 3:
        raise OrdinalModelContractError(
            "The v3 ordinal estimators require at least three ordered classes."
        )
    return classes


def _validate_class_weight(class_weight: str | None) -> None:
    if class_weight not in {None, "balanced"}:
        raise OrdinalModelContractError(
            "class_weight must be null or 'balanced' and is derived from training labels only."
        )


def _thresholds_from_unconstrained(raw: np.ndarray, min_gap: float) -> np.ndarray:
    thresholds = np.empty_like(raw, dtype=np.float64)
    thresholds[0] = raw[0]
    if len(raw) > 1:
        thresholds[1:] = raw[0] + np.cumsum(
            min_gap + np.logaddexp(0.0, raw[1:]), dtype=np.float64
        )
    return thresholds


def _unconstrained_from_thresholds(thresholds: np.ndarray, min_gap: float) -> np.ndarray:
    raw = np.empty_like(thresholds, dtype=np.float64)
    raw[0] = thresholds[0]
    if len(thresholds) > 1:
        adjusted = np.maximum(np.diff(thresholds) - min_gap, 1e-10)
        raw[1:] = np.log(np.expm1(adjusted))
    return raw


def _ordinal_probabilities(
    design: np.ndarray,
    coefficients: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    eta = design @ coefficients
    cumulative = expit(thresholds[np.newaxis, :] - eta[:, np.newaxis])
    probabilities = np.empty((len(design), len(thresholds) + 1), dtype=np.float64)
    probabilities[:, 0] = cumulative[:, 0]
    if len(thresholds) > 1:
        probabilities[:, 1:-1] = np.diff(cumulative, axis=1)
    probabilities[:, -1] = 1.0 - cumulative[:, -1]
    probabilities = np.clip(probabilities, 0.0, 1.0)
    row_sums = probabilities.sum(axis=1, dtype=np.float64)
    if np.any(row_sums <= 0.0) or not np.all(np.isfinite(row_sums)):
        raise OrdinalModelContractError(
            "Proportional-odds probabilities cannot be normalized."
        )
    return probabilities / row_sums[:, np.newaxis]


class ProportionalOddsClassifier(ClassifierMixin, BaseEstimator):
    """Cumulative-logit proportional-odds classifier fitted by penalized MLE."""

    def __init__(
        self,
        *,
        C: float = 1.0,
        class_weight: str | None = None,
        max_iter: int = 500,
        tol: float = 1e-8,
        min_threshold_gap: float = 1e-6,
        probability_floor: float = 1e-12,
        random_state: int | None = None,
    ) -> None:
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.min_threshold_gap = min_threshold_gap
        self.probability_floor = probability_floor
        self.random_state = random_state

    def _validate_hyperparameters(self) -> None:
        _validate_class_weight(self.class_weight)
        if not np.isfinite(self.C) or self.C <= 0:
            raise OrdinalModelContractError("C must be finite and strictly positive.")
        if isinstance(self.max_iter, bool) or int(self.max_iter) != self.max_iter or self.max_iter < 1:
            raise OrdinalModelContractError("max_iter must be a positive integer.")
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise OrdinalModelContractError("tol must be finite and strictly positive.")
        if not np.isfinite(self.min_threshold_gap) or self.min_threshold_gap <= 0:
            raise OrdinalModelContractError(
                "min_threshold_gap must be finite and strictly positive."
            )
        if (
            not np.isfinite(self.probability_floor)
            or self.probability_floor <= 0
            or self.probability_floor >= 1
        ):
            raise OrdinalModelContractError("probability_floor must fall strictly inside (0, 1).")

    def fit(self, X: Any, y: Any) -> "ProportionalOddsClassifier":
        self._validate_hyperparameters()
        design, target = validate_data(
            self,
            X,
            y,
            reset=True,
            dtype=np.float64,
            ensure_min_samples=3,
        )
        target = np.asarray(target)
        self.classes_ = _validated_ordered_classes(target)
        class_position = {label: position for position, label in enumerate(self.classes_)}
        encoded = np.asarray([class_position[label] for label in target], dtype=np.int64)
        class_counts = np.bincount(encoded, minlength=len(self.classes_))
        if np.any(class_counts == 0):  # protected by np.unique, retained as a fail-closed invariant
            raise OrdinalModelContractError("Every declared ordinal class needs training support.")

        sample_weight = (
            compute_sample_weight(class_weight="balanced", y=target).astype(np.float64)
            if self.class_weight == "balanced"
            else np.ones(len(target), dtype=np.float64)
        )
        weight_total = float(sample_weight.sum())
        if weight_total <= 0 or not np.all(np.isfinite(sample_weight)):
            raise OrdinalModelContractError("Training-derived sample weights are invalid.")

        cumulative_prevalence = np.cumsum(class_counts[:-1], dtype=np.float64) / len(target)
        cumulative_prevalence = np.clip(
            cumulative_prevalence,
            self.probability_floor,
            1.0 - self.probability_floor,
        )
        initial_thresholds = np.log(cumulative_prevalence / (1.0 - cumulative_prevalence))
        initial_raw_thresholds = _unconstrained_from_thresholds(
            initial_thresholds, float(self.min_threshold_gap)
        )
        n_features = design.shape[1]
        initial = np.concatenate(
            [np.zeros(n_features, dtype=np.float64), initial_raw_thresholds]
        )

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            coefficients = parameters[:n_features]
            raw_thresholds = parameters[n_features:]
            thresholds = _thresholds_from_unconstrained(
                raw_thresholds, float(self.min_threshold_gap)
            )
            eta = design @ coefficients
            cumulative = expit(thresholds[np.newaxis, :] - eta[:, np.newaxis])
            cumulative_derivative = cumulative * (1.0 - cumulative)

            upper = np.ones(len(target), dtype=np.float64)
            lower = np.zeros(len(target), dtype=np.float64)
            upper_derivative = np.zeros(len(target), dtype=np.float64)
            lower_derivative = np.zeros(len(target), dtype=np.float64)
            has_upper = encoded < len(self.classes_) - 1
            has_lower = encoded > 0
            upper_rows = np.flatnonzero(has_upper)
            lower_rows = np.flatnonzero(has_lower)
            upper_columns = encoded[has_upper]
            lower_columns = encoded[has_lower] - 1
            upper[has_upper] = cumulative[upper_rows, upper_columns]
            lower[has_lower] = cumulative[lower_rows, lower_columns]
            upper_derivative[has_upper] = cumulative_derivative[
                upper_rows, upper_columns
            ]
            lower_derivative[has_lower] = cumulative_derivative[
                lower_rows, lower_columns
            ]

            raw_probability = upper - lower
            probability = np.maximum(raw_probability, float(self.probability_floor))
            active_probability = raw_probability > float(self.probability_floor)
            inverse_probability = np.zeros_like(probability)
            inverse_probability[active_probability] = 1.0 / probability[active_probability]
            weighted_nll = -float(
                np.dot(sample_weight, np.log(probability)) / weight_total
            )
            penalty_scale = 1.0 / (float(self.C) * len(target))
            loss = weighted_nll + 0.5 * penalty_scale * float(
                np.dot(coefficients, coefficients)
            )

            derivative_eta = (
                upper_derivative - lower_derivative
            ) * inverse_probability
            derivative_eta *= sample_weight / weight_total
            coefficient_gradient = design.T @ derivative_eta
            coefficient_gradient += penalty_scale * coefficients

            threshold_gradient = np.zeros(len(self.classes_) - 1, dtype=np.float64)
            np.add.at(
                threshold_gradient,
                upper_columns,
                -(sample_weight[has_upper] / weight_total)
                * upper_derivative[has_upper]
                * inverse_probability[has_upper],
            )
            np.add.at(
                threshold_gradient,
                lower_columns,
                (sample_weight[has_lower] / weight_total)
                * lower_derivative[has_lower]
                * inverse_probability[has_lower],
            )
            raw_threshold_gradient = np.empty_like(threshold_gradient)
            raw_threshold_gradient[0] = threshold_gradient.sum()
            if len(raw_threshold_gradient) > 1:
                reverse_sums = np.cumsum(threshold_gradient[::-1])[::-1]
                raw_threshold_gradient[1:] = (
                    expit(raw_thresholds[1:]) * reverse_sums[1:]
                )
            gradient = np.concatenate([coefficient_gradient, raw_threshold_gradient])
            return loss, gradient

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": int(self.max_iter), "ftol": float(self.tol)},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise OrdinalModelContractError(
                "Proportional-odds optimization did not converge: "
                f"status={result.status}, message={result.message}."
            )
        self.coef_ = np.asarray(result.x[:n_features], dtype=np.float64)[np.newaxis, :]
        self.thresholds_ = _thresholds_from_unconstrained(
            np.asarray(result.x[n_features:], dtype=np.float64),
            float(self.min_threshold_gap),
        )
        if not np.all(np.diff(self.thresholds_) > 0):
            raise OrdinalModelContractError("Fitted proportional-odds thresholds are not ordered.")
        self.n_iter_ = np.asarray([int(result.nit)], dtype=np.int32)
        self.objective_value_ = float(result.fun)
        self.class_support_ = tuple(int(value) for value in class_counts)
        self.sample_weight_source_ = (
            "current_fit_training_labels_only" if self.class_weight == "balanced" else "none"
        )
        self.optimization_contract_ = (
            "scipy_lbfgsb_analytic_gradient_mean_weighted_nll_plus_l2_coefficients_only"
        )
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ("classes_", "coef_", "thresholds_"))
        design = validate_data(self, X, reset=False, dtype=np.float64)
        return _ordinal_probabilities(design, self.coef_[0], self.thresholds_)

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


def _project_nonincreasing(values: np.ndarray) -> np.ndarray:
    """Euclidean isotonic projection onto a non-increasing sequence."""

    block_means: list[float] = []
    block_weights: list[int] = []
    for value in np.asarray(values, dtype=np.float64):
        block_means.append(float(value))
        block_weights.append(1)
        while len(block_means) >= 2 and block_means[-2] < block_means[-1]:
            total_weight = block_weights[-2] + block_weights[-1]
            pooled = (
                block_means[-2] * block_weights[-2]
                + block_means[-1] * block_weights[-1]
            ) / total_weight
            block_means[-2:] = [pooled]
            block_weights[-2:] = [total_weight]
    projected = np.concatenate(
        [np.repeat(mean, weight) for mean, weight in zip(block_means, block_weights)]
    )
    return np.clip(projected, 0.0, 1.0)


class CumulativeThresholdXGBClassifier(ClassifierMixin, BaseEstimator):
    """Nonlinear ordinal model from independently fitted cumulative XGBoost tasks."""

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_child_weight: float = 1.0,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        class_weight: str | None = None,
        tree_method: str = "hist",
        n_jobs: int = 1,
        verbosity: int = 0,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.class_weight = class_weight
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.random_state = random_state

    def _validate_hyperparameters(self) -> None:
        _validate_class_weight(self.class_weight)
        if isinstance(self.n_jobs, bool) or self.n_jobs != 1:
            raise OrdinalModelContractError(
                "Cumulative-threshold XGBoost must use n_jobs=1 inside nested search."
            )
        if isinstance(self.n_estimators, bool) or int(self.n_estimators) != self.n_estimators or self.n_estimators < 1:
            raise OrdinalModelContractError("n_estimators must be a positive integer.")
        if isinstance(self.max_depth, bool) or int(self.max_depth) != self.max_depth or self.max_depth < 1:
            raise OrdinalModelContractError("max_depth must be a positive integer.")
        for name in ("learning_rate", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise OrdinalModelContractError(f"{name} must be finite and positive.")
        if self.subsample > 1 or self.colsample_bytree > 1:
            raise OrdinalModelContractError("subsample and colsample_bytree cannot exceed one.")
        if self.tree_method != "hist":
            raise OrdinalModelContractError("tree_method must remain 'hist'.")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, (int, np.integer)):
            raise OrdinalModelContractError("random_state must be an integer.")

    def fit(self, X: Any, y: Any) -> "CumulativeThresholdXGBClassifier":
        from xgboost import XGBClassifier

        self._validate_hyperparameters()
        design, target = validate_data(
            self,
            X,
            y,
            reset=True,
            dtype=np.float64,
            ensure_min_samples=3,
        )
        target = np.asarray(target)
        self.classes_ = _validated_ordered_classes(target)
        estimators = []
        threshold_support = []
        for threshold_index, threshold in enumerate(self.classes_[:-1]):
            binary_target = (target > threshold).astype(np.int8)
            support = tuple(int(value) for value in np.bincount(binary_target, minlength=2))
            if min(support) == 0:
                raise OrdinalModelContractError(
                    f"Cumulative threshold {threshold!r} lacks binary training support: {support}."
                )
            sample_weight = (
                compute_sample_weight(class_weight="balanced", y=binary_target)
                if self.class_weight == "balanced"
                else None
            )
            estimator = XGBClassifier(
                n_estimators=int(self.n_estimators),
                learning_rate=float(self.learning_rate),
                max_depth=int(self.max_depth),
                min_child_weight=float(self.min_child_weight),
                subsample=float(self.subsample),
                colsample_bytree=float(self.colsample_bytree),
                reg_lambda=float(self.reg_lambda),
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method=self.tree_method,
                n_jobs=1,
                verbosity=int(self.verbosity),
                random_state=int(self.random_state) + threshold_index,
            )
            estimator.fit(design, binary_target, sample_weight=sample_weight)
            if set(np.asarray(estimator.classes_).tolist()) != {0, 1}:
                raise OrdinalModelContractError(
                    f"Cumulative threshold {threshold!r} did not fit both binary classes."
                )
            estimators.append(estimator)
            threshold_support.append(support)
        self.estimators_ = tuple(estimators)
        self.thresholds_ = tuple(self.classes_[:-1].tolist())
        self.threshold_class_support_ = tuple(threshold_support)
        self.sample_weight_source_ = (
            "current_fit_threshold_binary_training_labels_only"
            if self.class_weight == "balanced"
            else "none"
        )
        self.cumulative_probability_contract_ = (
            "P(y>threshold)_independent_binary_xgboost_then_rowwise_nonincreasing_pava"
        )
        return self

    def raw_cumulative_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ("classes_", "estimators_", "thresholds_"))
        design = validate_data(self, X, reset=False, dtype=np.float64)
        columns = []
        for estimator in self.estimators_:
            model_classes = np.asarray(estimator.classes_)
            positive_positions = np.flatnonzero(model_classes == 1)
            if len(positive_positions) != 1:
                raise OrdinalModelContractError(
                    "A cumulative binary estimator lacks exactly one positive-class column."
                )
            probabilities = np.asarray(estimator.predict_proba(design), dtype=np.float64)
            columns.append(probabilities[:, int(positive_positions[0])])
        cumulative = np.column_stack(columns)
        if not np.all(np.isfinite(cumulative)) or np.any(cumulative < 0) or np.any(cumulative > 1):
            raise OrdinalModelContractError(
                "Raw cumulative probabilities must be finite and inside [0, 1]."
            )
        return cumulative

    def predict_proba(self, X: Any) -> np.ndarray:
        raw = self.raw_cumulative_proba(X)
        projected = np.vstack([_project_nonincreasing(row) for row in raw])
        self.last_raw_monotonic_violation_count_ = int(np.sum(np.diff(raw, axis=1) > 0.0))
        self.last_raw_monotonic_violation_row_count_ = int(
            np.sum(np.any(np.diff(raw, axis=1) > 0.0, axis=1))
        )
        probabilities = np.empty((len(projected), len(self.classes_)), dtype=np.float64)
        probabilities[:, 0] = 1.0 - projected[:, 0]
        if projected.shape[1] > 1:
            probabilities[:, 1:-1] = projected[:, :-1] - projected[:, 1:]
        probabilities[:, -1] = projected[:, -1]
        probabilities = np.clip(probabilities, 0.0, 1.0)
        row_sums = probabilities.sum(axis=1, dtype=np.float64)
        if np.any(row_sums <= 0.0) or not np.all(np.isfinite(row_sums)):
            raise OrdinalModelContractError(
                "Projected cumulative probabilities cannot be normalized."
            )
        return probabilities / row_sums[:, np.newaxis]

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class TrainingOnlyBaselineClassifier(ClassifierMixin, BaseEstimator):
    """Majority, stratified-random, or lower ordinal-median baseline."""

    def __init__(self, *, strategy: str, random_state: int = 42) -> None:
        self.strategy = strategy
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "TrainingOnlyBaselineClassifier":
        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state, (int, np.integer)
        ):
            raise OrdinalModelContractError("Baseline random_state must be an integer.")
        _, target = validate_data(self, X, y, reset=True, ensure_min_samples=1)
        target = np.asarray(target)
        self.classes_ = _validated_ordered_classes(target)
        if self.strategy not in {"majority", "stratified", "ordinal_median"}:
            raise OrdinalModelContractError(
                "Baseline strategy must be majority, stratified, or ordinal_median."
            )
        class_position = {label: position for position, label in enumerate(self.classes_)}
        encoded = np.asarray([class_position[label] for label in target], dtype=np.int64)
        counts = np.bincount(encoded, minlength=len(self.classes_)).astype(np.int64)
        self.class_support_ = tuple(int(value) for value in counts)
        self.class_prior_ = counts.astype(np.float64) / len(target)
        self.majority_position_ = int(np.argmax(counts))
        cumulative = np.cumsum(counts, dtype=np.int64)
        self.ordinal_median_position_ = int(
            np.flatnonzero(cumulative >= int(np.ceil(len(target) / 2)))[0]
        )
        self.baseline_fit_source_ = "current_outer_training_labels_only"
        self.ordinal_median_tie_rule_ = "lowest_class_with_cumulative_support_at_least_half"
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ("classes_", "class_prior_"))
        features = validate_data(self, X, reset=False)
        probabilities = np.zeros((len(features), len(self.classes_)), dtype=np.float64)
        if self.strategy == "majority":
            probabilities[:, self.majority_position_] = 1.0
        elif self.strategy == "ordinal_median":
            probabilities[:, self.ordinal_median_position_] = 1.0
        else:
            rng = np.random.default_rng(int(self.random_state))
            positions = rng.choice(
                len(self.classes_), size=len(features), replace=True, p=self.class_prior_
            )
            probabilities[np.arange(len(features)), positions] = 1.0
        return probabilities

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


def build_v3_ordinal_estimator(
    model_name: str,
    parameters: Mapping[str, Any],
    *,
    random_state: int,
) -> BaseEstimator:
    """Build one v3 ordinal estimator while keeping random-state ownership explicit."""

    if model_name not in V3_ORDINAL_MODEL_NAMES:
        raise OrdinalModelContractError(
            f"Unknown v3 ordinal model {model_name!r}; expected {list(V3_ORDINAL_MODEL_NAMES)}."
        )
    params = dict(parameters)
    if "random_state" in params:
        raise OrdinalModelContractError(
            "random_state is owned by the scientific protocol and cannot be overridden."
        )
    params["random_state"] = int(random_state)
    if model_name == "proportional_odds_logistic":
        return ProportionalOddsClassifier(**params)
    return CumulativeThresholdXGBClassifier(**params)


def build_v3_naive_baseline(
    baseline_name: str,
    *,
    random_state: int,
) -> TrainingOnlyBaselineClassifier:
    """Build one of the three predeclared training-only naive comparators."""

    try:
        strategy = V3_BASELINE_STRATEGIES[baseline_name]
    except KeyError as exc:
        raise OrdinalModelContractError(
            f"Unknown v3 baseline {baseline_name!r}; expected {list(V3_NAIVE_BASELINE_NAMES)}."
        ) from exc
    return TrainingOnlyBaselineClassifier(
        strategy=strategy,
        random_state=int(random_state),
    )


__all__ = [
    "CumulativeThresholdXGBClassifier",
    "OrdinalModelContractError",
    "ProportionalOddsClassifier",
    "TrainingOnlyBaselineClassifier",
    "V3_BASELINE_STRATEGIES",
    "V3_NAIVE_BASELINE_NAMES",
    "V3_ORDINAL_ESTIMATOR_PATHS",
    "V3_ORDINAL_MODEL_NAMES",
    "build_v3_naive_baseline",
    "build_v3_ordinal_estimator",
]
