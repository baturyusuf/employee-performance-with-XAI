from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.hrdataset_sensitivity_v3 import (
    EXPECTED_LOCAL_FILES,
    SYSTEMS,
    HRDatasetSensitivityV3Error,
    _baseline_comparisons,
    _calibrator_parameter_rows,
    _canonical_v2_class_results,
    _formulation_map,
    _protocol_comparison,
    _summarize_oof,
    _target_series,
    preflight_hrdataset_sensitivity_v3,
)
from src.experiments.manuscript_calibration import (
    apply_sigmoid_calibrator,
    calibrator_from_parameter_rows,
    fit_sigmoid_calibrator,
)
from src.governance.hrdataset_sensitivity_contract_v3 import METRICS, PRIORITY_METRICS


def _contract() -> dict:
    return json.loads(Path("configs/hrdataset_sensitivity_v3.json").read_text(encoding="utf-8"))


def _synthetic_oof() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    target_by_formulation = {
        "primary_three_class": np.asarray([2, 3, 4, 3, 2, 4]),
        "raw_order_four_class": np.asarray([1, 2, 3, 4, 3, 2, 1, 4]),
    }
    labels_by_formulation = {
        "primary_three_class": (2, 3, 4),
        "raw_order_four_class": (1, 2, 3, 4),
    }
    for formulation_id, target in target_by_formulation.items():
        labels = labels_by_formulation[formulation_id]
        for repetition in range(1, 6):
            for system_position, system in enumerate(SYSTEMS):
                prediction = target.copy()
                if system_position > 1:
                    prediction[0] = labels[min(1, len(labels) - 1)]
                for sample_index, (truth, predicted) in enumerate(zip(target, prediction)):
                    probability = np.full(len(labels), 0.03, dtype=float)
                    probability[labels.index(int(predicted))] = 1.0 - 0.03 * (len(labels) - 1)
                    row: dict[str, object] = {
                        "run_id": "synthetic",
                        "contract_sha256": "a" * 64,
                        "scientific_input_sha256": "b" * 64,
                        "formulation_id": formulation_id,
                        "repetition": repetition,
                        "system": system,
                        "sample_index": sample_index,
                        "outer_fold": sample_index % 5 + 1,
                        "y_true": int(truth),
                        "y_pred": int(predicted),
                    }
                    row.update({f"prob_class_{label}": probability[column] for column, label in enumerate(labels)})
                    rows.append(row)
    return pd.DataFrame(rows)


def test_synthetic_repeated_oof_produces_complete_metric_class_and_confusion_grids() -> None:
    formulations = _formulation_map(_contract())
    metrics, variability, per_class, confusion = _summarize_oof(
        _synthetic_oof(), formulations, full_run=True
    )
    assert len(metrics) == 2 * 5 * len(SYSTEMS) * len(METRICS)
    assert len(variability) == 2 * len(SYSTEMS) * len(METRICS)
    assert variability["repetition_count"].eq(5).all()
    assert len(per_class) == 5 * len(SYSTEMS) * (3 + 4)
    assert len(confusion) == 5 * len(SYSTEMS) * (3**2 + 4**2)
    assert np.isfinite(metrics["value"].to_numpy(float)).all()


def test_baseline_comparisons_are_within_formulation_and_repetition_only() -> None:
    formulations = _formulation_map(_contract())
    metrics, _, _, _ = _summarize_oof(_synthetic_oof(), formulations, full_run=True)
    comparisons = _baseline_comparisons(metrics)
    assert len(comparisons) == 2 * 5 * len(PRIORITY_METRICS) * 3
    assert comparisons["comparison"].str.startswith("xgboost_raw_minus_").all()
    assert set(comparisons["inference"]) == {
        "descriptive_matched_repetition_no_confidence_interval"
    }


def test_repetition_summary_rejects_an_incomplete_system_grid() -> None:
    frame = _synthetic_oof()
    frame = frame.loc[
        ~(
            (frame["formulation_id"] == "primary_three_class")
            & (frame["repetition"] == 1)
            & (frame["system"] == "majority_baseline")
        )
    ]
    with pytest.raises(HRDatasetSensitivityV3Error, match="metric grid is incomplete"):
        _summarize_oof(frame, _formulation_map(_contract()), full_run=True)


def test_canonical_v2_class_results_are_recomputed_from_exact_oof() -> None:
    per_class, confusion, metrics = _canonical_v2_class_results(_contract())
    assert len(per_class) == 2 * 3
    assert len(confusion) == 2 * 3 * 3
    assert set(metrics) == {"xgboost_raw", "xgboost_sigmoid"}
    assert all(set(values) == set(METRICS) for values in metrics.values())
    assert per_class.groupby("system")["support"].sum().eq(311).all()
    assert confusion.groupby("system")["count"].sum().eq(311).all()


def test_protocol_comparison_covers_all_requested_components_and_rejects_equivalence() -> None:
    comparison = _protocol_comparison()
    assert comparison["component"].tolist() == [
        "folds", "tuning", "calibration", "SHAP", "subgroup", "proxy", "target", "features"
    ]
    assert comparison.loc[comparison["component"] == "target", "same_semantics"].item() is False
    assert "locked" not in " ".join(comparison["notes"].str.lower())


def test_target_formulations_retain_observed_raw_order(monkeypatch) -> None:
    class Dataset:
        raw = pd.DataFrame(
            {"PerformanceScore": ["PIP", "Needs Improvement", "Fully Meets", "Exceeds"]}
        )
        canonical = pd.DataFrame(index=pd.RangeIndex(4))

    formulations = _formulation_map(_contract())
    assert _target_series(Dataset(), formulations["primary_three_class"]).tolist() == [2, 2, 3, 4]
    assert _target_series(Dataset(), formulations["raw_order_four_class"]).tolist() == [1, 2, 3, 4]


def test_persisted_calibrator_rows_are_replay_complete() -> None:
    probabilities = np.asarray(
        [
            [0.80, 0.15, 0.05],
            [0.70, 0.20, 0.10],
            [0.15, 0.75, 0.10],
            [0.10, 0.80, 0.10],
            [0.05, 0.20, 0.75],
            [0.10, 0.25, 0.65],
        ]
    )
    target = np.asarray([2, 2, 3, 3, 4, 4])
    calibrator = fit_sigmoid_calibrator(probabilities, target, (2, 3, 4), seed=4501)
    rows = _calibrator_parameter_rows(
        calibrator,
        identity={"run_id": "synthetic", "repetition": 1},
        formulation_id="primary_three_class",
        outer_fold=1,
        selected_candidate_index=0,
    )
    replay = calibrator_from_parameter_rows(pd.DataFrame(rows))
    np.testing.assert_allclose(
        apply_sigmoid_calibrator(replay, probabilities),
        apply_sigmoid_calibrator(calibrator, probabilities),
        rtol=0.0,
        atol=1e-15,
    )
    assert "calibration_training_oof.csv" in EXPECTED_LOCAL_FILES


@pytest.mark.skipif(
    not Path("data/external/hrdataset_v14/raw.csv").is_file(),
    reason="ignored local HRDataset_v14 input is unavailable",
)
def test_real_preflight_is_fit_free_and_generates_ten_distinct_fold_contracts(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.experiments.hrdataset_sensitivity_v3._fit_or_fail",
        lambda *args, **kwargs: pytest.fail("preflight attempted a fit"),
    )
    receipt = preflight_hrdataset_sensitivity_v3()
    assert receipt["status"] == "passed"
    assert receipt["sample_count"] == 311
    assert receipt["feature_count"] == 7
    assert receipt["distinct_fold_contracts"] == 10
    assert receipt["distinct_outer_assignments"] == 10
    assert receipt["model_fit_count"] == 0
    assert receipt["network_calls"] == receipt["paid_api_calls"] == 0
