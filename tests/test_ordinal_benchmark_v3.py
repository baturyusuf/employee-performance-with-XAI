from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.ordinal_benchmark_v3 import (
    EXTENSION_MODEL_NAMES,
    V3OrdinalBenchmarkError,
    evaluate_ordinal_extension_v3,
    exact_p3_feature_frame,
    summarize_combined_oof_v3,
)
from src.experiments.shared_folds import generate_shared_folds


def _contract() -> dict:
    return json.loads(Path("configs/ordinal_benchmark_v3.json").read_text(encoding="utf-8"))


def _synthetic_folds():
    target = np.repeat([2, 3, 4], 20)
    frame = pd.DataFrame(
        {
            "EmpNumber": np.arange(1000, 1060),
            "PerformanceRating": target,
        }
    )
    folds = generate_shared_folds(
        frame,
        target_column="PerformanceRating",
        id_column="EmpNumber",
        run_id="synthetic_v3_contract_test",
        config_hash="a" * 64,
        scientific_input_hash="b" * 64,
        dataset_key="inx_primary",
        dataset_sha256="c" * 64,
        outer_splits=10,
        inner_splits=5,
        seed=42,
        inner_seed=43,
    )
    features = pd.DataFrame(
        {
            "numeric": np.linspace(-2.0, 2.0, 60),
            "category": np.resize(np.asarray(["a", "b", "c"]), 60),
        }
    )
    return features, pd.Series(target), folds


class _FastClassifier:
    def fit(self, X, y):
        self.classes_, counts = np.unique(np.asarray(y), return_counts=True)
        self.prior_ = counts.astype(float) / counts.sum()
        return self

    def predict_proba(self, X):
        return np.tile(self.prior_, (len(X), 1))

    def predict(self, X):
        return np.repeat(self.classes_[int(np.argmax(self.prior_))], len(X))


def test_extension_uses_shared_10x5_folds_and_exactly_once_oof(monkeypatch) -> None:
    features, target, folds = _synthetic_folds()
    monkeypatch.setattr(
        "src.experiments.ordinal_benchmark_v3._v3_pipeline",
        lambda *args, **kwargs: _FastClassifier(),
    )
    monkeypatch.setattr(
        "src.experiments.ordinal_benchmark_v3._selection_scores",
        lambda *args, **kwargs: (0.5, 0.1),
    )
    monkeypatch.setattr(
        "src.experiments.ordinal_benchmark_v3._fit_or_fail",
        lambda estimator, X, y, **kwargs: estimator.fit(X, y),
    )
    result = evaluate_ordinal_extension_v3(
        features,
        target,
        folds,
        _contract(),
        run_id="synthetic_v3_contract_test",
        benchmark_contract_sha256="d" * 64,
        scientific_input_sha256="e" * 64,
    )

    assert result.evidence_status == "complete_exactly_once_oof"
    assert result.oof_predictions.shape[0] == len(EXTENSION_MODEL_NAMES) * len(features)
    assert result.oof_predictions.groupby("model")["sample_index"].nunique().eq(60).all()
    assert result.fold_metrics.shape[0] == len(EXTENSION_MODEL_NAMES) * 10
    assert result.selected_hyperparameters.shape[0] == len(EXTENSION_MODEL_NAMES) * 10
    assert result.candidate_search_results.shape[0] == (6 + 8) * 10
    assert not result.candidate_search_results["outer_test_used_for_selection"].any()
    assert result.candidate_search_results.groupby(["outer_fold", "model"])[
        "selected_by_protocol"
    ].sum().eq(1).all()


def test_diagnostic_fold_subset_is_explicitly_inadmissible(monkeypatch) -> None:
    features, target, folds = _synthetic_folds()
    monkeypatch.setattr(
        "src.experiments.ordinal_benchmark_v3._v3_pipeline",
        lambda *args, **kwargs: _FastClassifier(),
    )
    monkeypatch.setattr(
        "src.experiments.ordinal_benchmark_v3._selection_scores",
        lambda *args, **kwargs: (0.5, 0.1),
    )
    monkeypatch.setattr(
        "src.experiments.ordinal_benchmark_v3._fit_or_fail",
        lambda estimator, X, y, **kwargs: estimator.fit(X, y),
    )
    result = evaluate_ordinal_extension_v3(
        features,
        target,
        folds,
        _contract(),
        run_id="synthetic_diagnostic",
        benchmark_contract_sha256="d" * 64,
        scientific_input_sha256="e" * 64,
        outer_fold_subset=[1],
    )
    assert result.evidence_status == "diagnostic_incomplete_never_canonical"
    assert result.fold_metrics["outer_fold"].unique().tolist() == [1]


def test_exact_p3_feature_frame_follows_contract_order_and_exclusions() -> None:
    contract = json.loads(
        Path("configs/feature_availability_v3.json").read_text(encoding="utf-8")
    )
    columns = [row["feature_name"] for row in contract["features"]]
    frame = pd.DataFrame({column: [1, 2, 3] for column in columns})
    features, exclusions = exact_p3_feature_frame(frame, contract)
    assert len(features.columns) == 20
    assert set(exclusions) == set(contract["policies"][3]["excluded_features"])
    assert not set(exclusions).intersection(features.columns)
    assert features.columns.tolist() == [
        column for column in columns if column not in set(exclusions)
    ]


def test_combined_oof_summary_requires_all_nine_aligned_models() -> None:
    model_names = [
        "logistic_regression",
        "random_forest",
        "lightgbm",
        "xgboost",
        *EXTENSION_MODEL_NAMES,
    ]
    rows = []
    for model_name in model_names:
        for sample_index, label in enumerate((2, 3, 4)):
            probability = {2: 0.0, 3: 0.0, 4: 0.0}
            probability[label] = 1.0
            rows.append(
                {
                    "model": model_name,
                    "sample_index": sample_index,
                    "outer_fold": sample_index + 1,
                    "y_true": label,
                    "y_pred": label,
                    **{
                        f"prob_class_{class_label}": probability[class_label]
                        for class_label in (2, 3, 4)
                    },
                }
            )
    combined = pd.DataFrame(rows)
    aggregate, per_class, confusion = summarize_combined_oof_v3(combined)
    assert aggregate.shape[0] == 9 * 16
    assert per_class.shape[0] == 9 * 3
    assert confusion.shape[0] == 9 * 9
    assert aggregate.groupby("model_name")["metric"].nunique().eq(16).all()

    with pytest.raises(V3OrdinalBenchmarkError, match="exactly nine"):
        summarize_combined_oof_v3(combined[combined["model"] != "xgboost"])


def test_combined_oof_summary_rejects_label_probability_disagreement() -> None:
    model_names = [
        "logistic_regression",
        "random_forest",
        "lightgbm",
        "xgboost",
        *EXTENSION_MODEL_NAMES,
    ]
    rows = []
    for model_name in model_names:
        for sample_index, label in enumerate((2, 3, 4)):
            rows.append(
                {
                    "model": model_name,
                    "sample_index": sample_index,
                    "outer_fold": sample_index + 1,
                    "y_true": label,
                    "y_pred": label,
                    "prob_class_2": float(label == 2),
                    "prob_class_3": float(label == 3),
                    "prob_class_4": float(label == 4),
                }
            )
    next(
        row
        for row in rows
        if row["model"] == "cumulative_threshold_xgboost" and row["sample_index"] == 0
    )["y_pred"] = 4
    with pytest.raises(V3OrdinalBenchmarkError, match="probability argmax"):
        summarize_combined_oof_v3(pd.DataFrame(rows))
