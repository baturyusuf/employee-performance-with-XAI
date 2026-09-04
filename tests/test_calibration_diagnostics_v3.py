from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments import calibration_diagnostics_v3 as diagnostics
from src.experiments.calibration_diagnostics_v3 import (
    _load_json,
    _plot_reliability,
    _validated_predictions,
    calibration_intercept_slope_v3,
    evaluate_calibration_diagnostics_v3,
    expected_calibration_error_v3,
    preflight_calibration_diagnostics_v3,
    reliability_bin_rows_v3,
)
from src.governance.calibration_diagnostics_contract_v3 import (
    DEFAULT_CALIBRATION_DIAGNOSTICS_CONTRACT,
)


@pytest.fixture(scope="module")
def real_result():
    contract = _load_json(DEFAULT_CALIBRATION_DIAGNOSTICS_CONTRACT)
    predictions = _validated_predictions(contract)
    return evaluate_calibration_diagnostics_v3(contract, predictions)


def test_reliability_bins_preserve_empty_bins_and_boundary_rule() -> None:
    outcomes = np.asarray([0, 1, 1, 0])
    scores = np.asarray([0.0, 0.1, 0.10001, 1.0])
    bins = reliability_bin_rows_v3(
        outcomes,
        scores,
        method="test",
        reliability_scope="test",
        target_id="test",
    )
    assert len(bins) == 10
    assert bins["n_samples"].tolist() == [2, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    assert bins.loc[bins["n_samples"] == 0, "bin_status"].eq("empty").all()
    assert bins.loc[bins["n_samples"] == 0, "observed_frequency"].isna().all()


def test_ece_is_support_weighted_absolute_gap() -> None:
    bins = reliability_bin_rows_v3(
        [0, 1],
        [0.2, 0.8],
        method="test",
        reliability_scope="test",
        target_id="test",
    )
    assert expected_calibration_error_v3(bins) == pytest.approx(0.2)


def test_calibration_intercept_slope_is_finite_and_deterministic() -> None:
    outcomes = np.asarray([0, 0, 0, 1, 0, 1, 1, 1])
    scores = np.asarray([0.05, 0.15, 0.25, 0.35, 0.45, 0.65, 0.75, 0.9])
    first = calibration_intercept_slope_v3(outcomes, scores)
    second = calibration_intercept_slope_v3(outcomes, scores)
    assert first == second
    assert first["regression_converged"] is True
    assert np.isfinite(first["calibration_intercept"])
    assert np.isfinite(first["calibration_slope"])
    assert first["regression_iterations"] <= 100


def test_real_phase2b_evaluation_has_complete_prespecified_grids(real_result) -> None:
    assert len(real_result.metric_summary) == 2
    assert len(real_result.classwise_metrics) == 6
    assert len(real_result.cumulative_metrics) == 4
    assert len(real_result.reliability_bins) == 120
    assert len(real_result.method_comparison) == 6
    assert set(real_result.reliability_bins["reliability_scope"]) == {
        "top_label",
        "one_vs_rest_class",
        "cumulative_threshold",
    }
    assert set(real_result.reliability_bins["bin"].astype(int)) == set(range(1, 11))


def test_real_phase2b_replays_legacy_metrics_and_rps_identity(real_result) -> None:
    observed = real_result.metric_summary.set_index("method")
    source = pd.read_csv(
        "reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/"
        "core/sigmoid_calibration/calibration_method_comparison.csv"
    ).set_index("method")
    for method in ("raw", "sigmoid"):
        assert observed.loc[method, "nll_log_loss"] == pytest.approx(
            source.loc[method, "nll_log_loss_oof"], abs=1e-14
        )
        assert observed.loc[method, "multiclass_brier"] == pytest.approx(
            source.loc[method, "multiclass_brier_oof"], abs=1e-14
        )
        assert observed.loc[method, "top_label_ece"] == pytest.approx(
            source.loc[method, "ece_confidence_oof"], abs=1e-14
        )
        assert observed.loc[method, "ranked_probability_score"] == pytest.approx(
            observed.loc[method, "mean_cumulative_binary_brier"], abs=1e-15
        )


def test_phase2b_classwise_bins_replay_canonical_bin_evidence(real_result) -> None:
    observed = real_result.reliability_bins[
        real_result.reliability_bins["reliability_scope"]
        == "one_vs_rest_class"
    ].copy()
    observed["class_label"] = observed["target_id"].str.removeprefix("class_").astype(int)
    source = pd.read_csv(
        "reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/"
        "core/sigmoid_calibration/calibration_bins.csv"
    )
    keys = ["method", "class_label", "bin"]
    columns = [
        "n_samples",
        "n_positive",
        "mean_predicted_probability",
        "observed_frequency",
        "absolute_gap",
    ]
    merged = observed.merge(source, on=keys, suffixes=("_v3", "_v2"), validate="one_to_one")
    assert len(merged) == 60
    for column in columns:
        pd.testing.assert_series_equal(
            merged[f"{column}_v3"],
            merged[f"{column}_v2"],
            check_names=False,
            check_dtype=False,
            rtol=0.0,
            atol=1e-15,
        )


def test_real_phase2b_retains_mixed_calibration_result(real_result) -> None:
    comparison = real_result.method_comparison.set_index("metric")
    assert comparison.loc["nll_log_loss", "direction_aligned_improvement"] > 0
    assert comparison.loc["multiclass_brier", "direction_aligned_improvement"] > 0
    assert comparison.loc["ranked_probability_score", "direction_aligned_improvement"] > 0
    assert comparison.loc["top_label_ece", "direction_aligned_improvement"] < 0
    assert not comparison["test_set_method_selection_performed"].any()
    assert not comparison["all_metrics_improved_claim_allowed"].any()


def test_reliability_figures_render_from_aggregate_bins(
    real_result, tmp_path: Path
) -> None:
    png, svg = _plot_reliability(
        real_result.reliability_bins,
        scopes=[
            ("cumulative_threshold", "Y_le_2", "Y≤2"),
            ("cumulative_threshold", "Y_le_3", "Y≤3"),
        ],
        title="Test cumulative reliability",
        output_stem=tmp_path / "figure",
    )
    assert png.is_file() and png.stat().st_size > 10_000
    assert svg.is_file() and svg.stat().st_size > 10_000
    assert "Date" not in svg.read_text(encoding="utf-8")


def test_phase2b_preflight_is_fit_free() -> None:
    receipt = preflight_calibration_diagnostics_v3()
    assert receipt["status"] == "preflight_passed"
    assert receipt["prediction_rows"] == 2400
    assert receipt["planned_new_model_fit_calls"] == 0
    assert receipt["planned_new_calibrator_fit_calls"] == 0
    assert receipt["planned_diagnostic_regression_fit_calls"] == 10
    assert receipt["network_calls"] == receipt["paid_api_calls"] == 0


def test_phase2b_runner_atomically_publishes_complete_local_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = {"commit": "a" * 40, "branch": "test"}
    monkeypatch.setattr(diagnostics, "_clean_git_identity", lambda: identity)
    monkeypatch.setattr(diagnostics, "source_tree_hash", lambda _root: "b" * 64)
    output = tmp_path / "calibration_diagnostics"
    receipt = diagnostics.run_calibration_diagnostics_v3(
        contract_path=DEFAULT_CALIBRATION_DIAGNOSTICS_CONTRACT,
        output_dir=output,
        run_id="diagnostic_incomplete_never_canonical",
    )
    assert receipt["status"] == "complete"
    assert set(path.name for path in output.iterdir()) == diagnostics.EXPECTED_LOCAL_FILES
    metadata = json.loads((output / "stage_metadata.json").read_text(encoding="utf-8"))
    assert metadata["prediction_row_count"] == 2400
    assert metadata["reliability_bin_row_count"] == 120
    assert metadata["new_model_fit_calls"] == 0
    assert metadata["new_calibrator_fit_calls"] == 0
    assert metadata["network_calls"] == metadata["paid_api_calls"] == 0
    assert not any(path.name.endswith("predictions.csv") for path in output.iterdir())
