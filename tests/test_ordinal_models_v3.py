from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone, is_classifier

from src.models.ordinal_models_v3 import (
    CumulativeThresholdXGBClassifier,
    OrdinalModelContractError,
    ProportionalOddsClassifier,
    V3_NAIVE_BASELINE_NAMES,
    V3_ORDINAL_MODEL_NAMES,
    build_v3_naive_baseline,
    build_v3_ordinal_estimator,
)


def _ordered_training_data() -> tuple[np.ndarray, np.ndarray]:
    values = np.linspace(-3.0, 3.0, 90)
    design = np.column_stack([values, np.sin(values)])
    target = np.where(values < -0.8, 2, np.where(values < 1.0, 3, 4))
    return design, target


def test_proportional_odds_fits_ordered_thresholds_and_probability_simplex() -> None:
    design, target = _ordered_training_data()
    estimator = ProportionalOddsClassifier(C=2.0, max_iter=500, tol=1e-9)
    estimator.fit(design, target)

    assert is_classifier(estimator)
    assert estimator.classes_.tolist() == [2, 3, 4]
    assert np.all(np.diff(estimator.thresholds_) > 0)
    probabilities = estimator.predict_proba(design)
    assert probabilities.shape == (90, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.mean(estimator.predict(design) == target) > 0.85
    assert estimator.optimization_contract_.startswith("scipy_lbfgsb")


def test_proportional_odds_is_cloneable_and_requires_three_classes() -> None:
    estimator = ProportionalOddsClassifier(C=0.5, random_state=19)
    copied = clone(estimator)
    assert copied.get_params() == estimator.get_params()
    with pytest.raises(OrdinalModelContractError, match="at least three"):
        estimator.fit(np.arange(12).reshape(6, 2), [2, 2, 2, 3, 3, 3])


def test_cumulative_threshold_xgboost_fits_one_binary_model_per_cutpoint() -> None:
    design, target = _ordered_training_data()
    estimator = CumulativeThresholdXGBClassifier(
        n_estimators=8,
        learning_rate=0.2,
        max_depth=2,
        subsample=1.0,
        colsample_bytree=1.0,
        n_jobs=1,
        random_state=23,
    ).fit(design, target)

    assert estimator.classes_.tolist() == [2, 3, 4]
    assert len(estimator.estimators_) == 2
    assert estimator.thresholds_ == (2, 3)
    assert all(min(support) > 0 for support in estimator.threshold_class_support_)
    probabilities = estimator.predict_proba(design[:7])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_cumulative_threshold_projection_repairs_crossing_probabilities() -> None:
    class _BinaryEstimator:
        classes_ = np.asarray([0, 1])

        def __init__(self, positive: float) -> None:
            self.positive = positive

        def predict_proba(self, design: np.ndarray) -> np.ndarray:
            return np.tile([1.0 - self.positive, self.positive], (len(design), 1))

    estimator = CumulativeThresholdXGBClassifier()
    estimator.classes_ = np.asarray([2, 3, 4])
    estimator.thresholds_ = (2, 3)
    estimator.estimators_ = (_BinaryEstimator(0.2), _BinaryEstimator(0.8))
    estimator.n_features_in_ = 1
    probabilities = estimator.predict_proba(np.zeros((3, 1)))

    np.testing.assert_allclose(probabilities, np.tile([0.5, 0.0, 0.5], (3, 1)))
    assert estimator.last_raw_monotonic_violation_count_ == 3
    assert estimator.last_raw_monotonic_violation_row_count_ == 3


@pytest.mark.parametrize("baseline_name", V3_NAIVE_BASELINE_NAMES)
def test_naive_baselines_use_training_labels_and_are_deterministic(baseline_name: str) -> None:
    design = np.zeros((8, 2))
    target = np.asarray([2, 2, 3, 3, 3, 3, 4, 4])
    estimator = build_v3_naive_baseline(baseline_name, random_state=17).fit(design, target)
    first = estimator.predict_proba(np.zeros((12, 2)))
    second = estimator.predict_proba(np.zeros((12, 2)))

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.sum(axis=1), 1.0)
    assert estimator.class_support_ == (2, 4, 2)
    assert estimator.baseline_fit_source_ == "current_outer_training_labels_only"
    if baseline_name == "majority_baseline":
        assert np.all(estimator.predict(np.zeros((3, 2))) == 3)
    if baseline_name == "ordinal_median_baseline":
        assert np.all(estimator.predict(np.zeros((3, 2))) == 3)


def test_ordinal_median_uses_declared_lower_tie_rule() -> None:
    estimator = build_v3_naive_baseline(
        "ordinal_median_baseline", random_state=7
    ).fit(np.zeros((6, 1)), [2, 2, 2, 3, 4, 4])
    assert estimator.predict(np.zeros((1, 1))).tolist() == [2]
    assert estimator.ordinal_median_tie_rule_.startswith("lowest_class")


def test_v3_factories_own_random_state_and_reject_unknown_names() -> None:
    assert V3_ORDINAL_MODEL_NAMES == (
        "proportional_odds_logistic",
        "cumulative_threshold_xgboost",
    )
    proportional = build_v3_ordinal_estimator(
        "proportional_odds_logistic", {"C": 1.0}, random_state=31
    )
    assert proportional.random_state == 31
    with pytest.raises(OrdinalModelContractError, match="owned by the scientific protocol"):
        build_v3_ordinal_estimator(
            "proportional_odds_logistic",
            {"C": 1.0, "random_state": 99},
            random_state=31,
        )
    with pytest.raises(OrdinalModelContractError, match="Unknown v3 ordinal model"):
        build_v3_ordinal_estimator("not_a_model", {}, random_state=31)
    with pytest.raises(OrdinalModelContractError, match="Unknown v3 baseline"):
        build_v3_naive_baseline("not_a_baseline", random_state=31)


def test_cumulative_threshold_rejects_parallel_nested_fit() -> None:
    design, target = _ordered_training_data()
    with pytest.raises(OrdinalModelContractError, match="n_jobs=1"):
        CumulativeThresholdXGBClassifier(n_jobs=-1).fit(design, target)
