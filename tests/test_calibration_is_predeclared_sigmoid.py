from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from src.experiments import manuscript_calibration as calibration
from src.governance.manuscript_contract import (
    load_manuscript_config,
    manuscript_settings,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "manuscript_final.yaml"


def _settings() -> dict:
    return deepcopy(manuscript_settings(load_manuscript_config(CONFIG_PATH)))


def _identity() -> dict[str, str]:
    values = {field: "a" * 64 for field in calibration.IDENTITY_FIELDS}
    values["run_id"] = "calibration-unit-test"
    return values


def _outcome_frames_where_raw_is_better() -> tuple[pd.DataFrame, pd.DataFrame]:
    higher_is_better = {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "quadratic_weighted_kappa",
    }
    fold_rows: list[dict] = []
    interval_rows: list[dict] = []
    identity = _identity()
    for method in calibration.SYSTEM_ORDER:
        for outer_fold in range(1, 11):
            row = {
                **identity,
                "method": method,
                "outer_fold": outer_fold,
                "n_outer_test": 12,
            }
            for metric in calibration.METRICS:
                raw_value = 0.90 if metric in higher_is_better else 0.05
                sigmoid_value = 0.10 if metric in higher_is_better else 0.80
                row[metric] = raw_value if method == "raw" else sigmoid_value
            fold_rows.append(row)
        for metric in calibration.METRICS:
            raw_value = 0.90 if metric in higher_is_better else 0.05
            sigmoid_value = 0.10 if metric in higher_is_better else 0.80
            point = raw_value if method == "raw" else sigmoid_value
            interval_rows.append(
                {
                    "system_id": method,
                    "metric": metric,
                    "point_estimate": point,
                    "ci_low": max(0.0, point - 0.01),
                    "ci_high": min(1.0, point + 0.01),
                    "bootstrap_std": 0.005,
                }
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(interval_rows)


def test_repository_protocol_predeclares_sigmoid_without_method_selection() -> None:
    protocol = calibration.validate_calibration_protocol(_settings())

    assert calibration.PRIMARY_METHOD == "sigmoid"
    assert calibration.SYSTEM_ORDER == ("raw", "sigmoid")
    assert protocol["primary_method"] == "sigmoid"
    assert protocol["selection_performed"] is False
    assert protocol["method_selection"] == "predeclared_not_outer_test_selected"
    assert protocol["outer_test_usage"] == "evaluation_only"
    assert protocol[
        "outer_test_used_for_tuning_fitting_selection_or_thresholds"
    ] is False
    assert "isotonic" not in protocol["comparison_systems"]


def test_primary_method_cannot_change_when_raw_outer_test_results_are_better() -> None:
    fold_metrics, intervals = _outcome_frames_where_raw_is_better()

    summary = calibration.summarize_calibration_methods(fold_metrics, intervals)
    primary = summary.loc[summary["primary_method"]]
    raw = summary.loc[summary["method"] == "raw"].iloc[0]
    sigmoid = summary.loc[summary["method"] == "sigmoid"].iloc[0]

    assert primary["method"].tolist() == ["sigmoid"]
    assert raw["macro_f1_oof"] > sigmoid["macro_f1_oof"]
    assert raw["nll_log_loss_oof"] < sigmoid["nll_log_loss_oof"]
    assert (summary["selection_source"] == "predeclared_config").all()
    assert not summary["selection_performed"].any()
    assert not summary["outer_test_used_for_selection"].any()
    assert "selected" not in summary.columns
    assert not any(column.endswith("_fold_ci_low") for column in summary.columns)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("primary_method", "raw", "must remain sigmoid"),
        ("selection_performed", True, "protocol drifted"),
        (
            "outer_test_used_for_tuning_fitting_selection_or_thresholds",
            True,
            "protocol drifted",
        ),
    ],
)
def test_protocol_drift_toward_outcome_selection_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    settings = _settings()
    settings["calibration"][field] = value

    with pytest.raises(calibration.CalibrationContractError, match=message):
        calibration.validate_calibration_protocol(settings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confidence_level", 0.90),
        ("stratify_by", ["y_true"]),
        ("method", "unpaired_percentile"),
        ("quantile_method", "nearest"),
    ],
)
def test_calibration_bootstrap_contract_rejects_uncertainty_drift(
    field: str,
    value: object,
) -> None:
    settings = _settings()
    settings["evaluation"]["bootstrap"][field] = value

    with pytest.raises(
        calibration.CalibrationContractError,
        match="frozen paired stratified 95%",
    ):
        calibration._bootstrap_protocol(settings)
