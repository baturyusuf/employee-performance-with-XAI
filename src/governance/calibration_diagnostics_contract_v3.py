"""Fail-closed validation for the v3 extended calibration-diagnostics contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.canonical_loader import sha256_file
from src.utils.config_loader import PROJECT_ROOT


DEFAULT_CALIBRATION_DIAGNOSTICS_CONTRACT = Path(
    "configs/calibration_diagnostics_v3.json"
)
EXPECTED_TOP_LEVEL = frozenset(
    {
        "schema_version",
        "contract_id",
        "dataset_key",
        "target",
        "ordered_labels",
        "policy_id",
        "policy_name",
        "model",
        "purpose",
        "source_contracts",
        "canonical_identity",
        "source_calibration",
        "reliability",
        "ordinal_probability_diagnostics",
        "calibration_regression",
        "comparison",
        "publication",
    }
)
EXPECTED_SOURCE_NAMES = frozenset(
    {
        "canonical_v2_receipt",
        "calibration_predictions",
        "calibration_bins",
        "calibration_method_comparison",
        "calibration_metric_intervals",
        "calibration_paired_differences",
        "calibration_protocol",
        "calibration_metadata",
        "calibration_validation",
    }
)
IDENTITY_FIELDS = (
    "run_id",
    "config_hash",
    "scientific_input_hash",
    "fold_contract_hash",
    "xgboost_model_set_sha256",
    "dataset_sha256",
    "calibration_protocol_sha256",
)


class CalibrationDiagnosticsContractV3Error(RuntimeError):
    """Raised when the Phase 2B design or its exact source evidence drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationDiagnosticsContractV3Error(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationDiagnosticsContractV3Error(
            f"Could not read {path.as_posix()}: {exc}"
        ) from exc
    _require(isinstance(payload, dict), f"{path.name} must contain a JSON object.")
    return payload


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _exact(mapping: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key, value in expected.items():
        _require(mapping.get(key) == value, f"{label} drifted for {key}.")


def _bool_values(series: pd.Series, *, context: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
        1: True,
        0: False,
        "1": True,
        "0": False,
    }
    values = series.map(mapping)
    _require(values.notna().all(), f"{context} contains invalid Boolean values.")
    return values.astype(bool)


def _validate_sources(contract: Mapping[str, Any]) -> dict[str, Any]:
    sources = contract["source_contracts"]
    _require(set(sources) == EXPECTED_SOURCE_NAMES, "Calibration source inventory drifted.")
    for name, record in sources.items():
        _require(set(record) == {"path", "sha256"}, f"Source schema drifted for {name}.")
        _require(_digest(record["sha256"]), f"Source digest is invalid for {name}.")
        path = PROJECT_ROOT / str(record["path"])
        _require(path.is_file(), f"Required Phase 2B source is absent: {name}.")
        _require(sha256_file(path) == record["sha256"], f"Phase 2B source hash drifted: {name}.")

    identity = contract["canonical_identity"]
    metadata = _load_json(
        PROJECT_ROOT / str(sources["calibration_metadata"]["path"])
    )
    validation = _load_json(
        PROJECT_ROOT / str(sources["calibration_validation"]["path"])
    )
    protocol = _load_json(
        PROJECT_ROOT / str(sources["calibration_protocol"]["path"])
    )
    for field in IDENTITY_FIELDS:
        expected = identity[
            "calibration_protocol_sha256" if field == "calibration_protocol_sha256" else field
        ]
        for payload, context in (
            (metadata, "metadata"),
            (validation, "validation"),
            (protocol, "protocol"),
        ):
            _require(payload.get(field) == expected, f"Canonical calibration {context} identity drifted for {field}.")
    _require(metadata.get("status") == "complete", "Canonical calibration metadata is incomplete.")
    _require(validation.get("status") == "validated_complete", "Canonical calibration validation is incomplete.")
    _require(protocol.get("status") == "complete", "Canonical calibration protocol is incomplete.")
    _require(metadata.get("primary_method") == "sigmoid", "Canonical primary calibration method drifted.")
    for payload, context in ((metadata, "metadata"), (validation, "validation")):
        _require(payload.get("outer_test_used_for_tuning_fitting_selection_or_thresholds", payload.get("outer_test_used_for_fit_or_selection")) is False, f"Canonical calibration {context} permits outer-test leakage.")
    _require(metadata.get("selection_performed") is False, "Canonical calibration metadata records method selection.")
    _require(metadata.get("outer_model_refit_in_calibration_stage") is False, "Canonical calibration unexpectedly refit outer models.")

    predictions = pd.read_csv(
        PROJECT_ROOT / str(sources["calibration_predictions"]["path"])
    )
    _require(len(predictions) == 2400, "Canonical calibration prediction count drifted.")
    _require(set(predictions["method"].astype(str)) == {"raw", "sigmoid"}, "Calibration method set drifted.")
    probability_columns = ["prob_class_2", "prob_class_3", "prob_class_4"]
    for method, rows in predictions.groupby("method", sort=True):
        rows = rows.sort_values("sample_index")
        _require(len(rows) == 1200 and rows["sample_index"].nunique() == 1200, f"{method} OOF coverage drifted.")
        _require(set(rows["sample_index"].astype(int)) == set(range(1200)), f"{method} sample set drifted.")
        _require(set(rows["outer_fold"].astype(int)) == set(range(1, 11)), f"{method} fold coverage drifted.")
        probabilities = rows[probability_columns].to_numpy(float)
        _require(np.isfinite(probabilities).all(), f"{method} probabilities are non-finite.")
        _require(np.all((0.0 <= probabilities) & (probabilities <= 1.0)), f"{method} probabilities escaped [0,1].")
        _require(np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-9), f"{method} probability simplex drifted.")
        expected_prediction = np.asarray([2, 3, 4])[np.argmax(probabilities, axis=1)]
        _require(np.array_equal(expected_prediction, rows["y_pred"].to_numpy(int)), f"{method} argmax labels drifted.")
    raw = predictions[predictions["method"] == "raw"].sort_values("sample_index")
    sigmoid = predictions[predictions["method"] == "sigmoid"].sort_values("sample_index")
    for column in ("sample_index", "outer_fold", "y_true"):
        _require(np.array_equal(raw[column].to_numpy(int), sigmoid[column].to_numpy(int)), f"Raw/sigmoid alignment drifted for {column}.")
    _require(not _bool_values(raw["primary_method"], context="raw primary flag").any(), "Raw comparator is incorrectly primary.")
    _require(_bool_values(sigmoid["primary_method"], context="sigmoid primary flag").all(), "Sigmoid primary flag drifted.")

    bins = pd.read_csv(PROJECT_ROOT / str(sources["calibration_bins"]["path"]))
    _require(len(bins) == 60, "Canonical classwise reliability grid drifted.")
    expected_grid = {
        (method, label, bin_index)
        for method in ("raw", "sigmoid")
        for label in (2, 3, 4)
        for bin_index in range(1, 11)
    }
    observed_grid = set(
        bins[["method", "class_label", "bin"]].itertuples(index=False, name=None)
    )
    _require(observed_grid == expected_grid, "Canonical classwise reliability key grid drifted.")
    for (method, label), rows in bins.groupby(["method", "class_label"], sort=True):
        _require(int(rows["n_samples"].sum()) == 1200, f"Reliability denominator drifted for {method}/class {label}.")
        empty = rows["n_samples"].astype(int) == 0
        _require((rows.loc[empty, "bin_status"] == "empty").all(), "Empty-bin status drifted.")
        _require(rows.loc[empty, ["mean_predicted_probability", "observed_frequency", "absolute_gap"]].isna().all().all(), "Empty-bin values must remain missing.")

    comparison = pd.read_csv(
        PROJECT_ROOT / str(sources["calibration_method_comparison"]["path"])
    )
    _require(len(comparison) == 2 and set(comparison["method"]) == {"raw", "sigmoid"}, "Canonical calibration summary drifted.")
    paired = pd.read_csv(
        PROJECT_ROOT / str(sources["calibration_paired_differences"]["path"])
    )
    _require(len(paired) == 9 and set(paired["comparison_id"]) == {"sigmoid_minus_raw"}, "Canonical paired comparison grid drifted.")
    _require(not _bool_values(paired["gate_eligible"], context="paired gate").any(), "Calibration comparison is incorrectly gate-eligible.")
    return {
        "validated": True,
        "prediction_rows": len(predictions),
        "samples_per_method": 1200,
        "classwise_bin_rows": len(bins),
        "paired_metric_rows": len(paired),
    }


def validate_calibration_diagnostics_contract_v3(
    contract_path: Path | str = DEFAULT_CALIBRATION_DIAGNOSTICS_CONTRACT,
) -> dict[str, Any]:
    """Validate the exact Phase 2B design without fitting any model or calibrator."""

    path = Path(contract_path)
    contract = _load_json(PROJECT_ROOT / path)
    _require(set(contract) == EXPECTED_TOP_LEVEL, "Phase 2B top-level inventory drifted.")
    _exact(
        contract,
        {
            "schema_version": 1,
            "contract_id": "calibration_diagnostics_v3",
            "dataset_key": "inx_primary",
            "target": "PerformanceRating",
            "ordered_labels": [2, 3, 4],
            "policy_id": "P3",
            "policy_name": "PRIMARY_LEAKAGE_AWARE",
            "model": "xgboost",
            "purpose": "extend_exact_oof_raw_and_predeclared_sigmoid_evidence_with_explicit_multiclass_and_ordinal_calibration_diagnostics",
        },
        "Phase 2B identity",
    )
    identity = contract["canonical_identity"]
    _exact(
        identity,
        {
            "run_id": "canonical_v2_20260714T221501Z_483f96f",
            "generation_commit": "483f96fdbaab16cb0f32d03d9dbe676a759af44a",
            "config_hash": "51415c2ce68c89d9ce2b042b0a7a811fe3e98180b72460006f9d0465f6bf49b7",
            "scientific_input_hash": "06c507bee525ea1daca43b61249764007d4d8baaa05c9333f23446ea723ce160",
            "dataset_sha256": "b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a",
            "fold_contract_hash": "c1300316fe5baec24e789c06aec35dd4f283fa4843b71c7aab1edbf4818f8e91",
            "xgboost_model_set_sha256": "d483d8ddb93a47a99a2eab54fe3d138d8f614798dd418bca860ea3f20c813f51",
            "calibration_protocol_sha256": "2c9293055f64114c63c8f85e652d94e838c84501a718eea6c3852de78ce4808b",
            "sample_count": 1200,
            "outer_folds": 10,
        },
        "Canonical calibration identity",
    )
    source = contract["source_calibration"]
    _exact(
        source,
        {
            "systems": ["raw", "sigmoid"],
            "primary_method": "sigmoid",
            "raw_role": "unaltered_exact_canonical_xgboost_oof_comparator",
            "sigmoid_algorithm": "one_vs_rest_platt_on_class_probability_logit_then_row_renormalize",
            "classwise_calibrators": True,
            "probability_clip": 0.000001,
            "renormalization": "divide_each_positive_classwise_sigmoid_output_by_row_sum",
            "simplex_tolerance": 0.000000001,
            "calibrator_training": "five_inner_fold_cross_fitted_probabilities_within_each_outer_training_partition_only",
            "outer_test_role": "evaluation_only",
            "method_selection": "predeclared_sigmoid_no_test_selected_comparison",
            "method_selection_performed": False,
            "outer_test_used_for_calibrator_fit_or_selection": False,
            "new_model_fit_calls": 0,
            "new_calibrator_fit_calls": 0,
        },
        "Source calibration",
    )
    reliability = contract["reliability"]
    _exact(
        reliability,
        {
            "bin_count": 10,
            "binning": "fixed_equal_width_over_zero_to_one",
            "boundary_rule": "first_bin_closed_both_sides_then_left_open_right_closed",
            "empty_bins": "persist_with_zero_support_and_missing_means",
            "top_label_ece": "support_weighted_absolute_gap_between_max_probability_and_argmax_accuracy",
            "classwise_ece": "one_vs_rest_support_weighted_absolute_gap_for_each_class_probability",
            "cumulative_thresholds": ["Y<=2", "Y<=3"],
            "cumulative_ece": "support_weighted_absolute_gap_for_cumulative_probability_and_binary_threshold_event",
            "classwise_reliability_diagrams": True,
            "cumulative_reliability_diagrams": True,
        },
        "Reliability",
    )
    ordinal = contract["ordinal_probability_diagnostics"]
    _require(ordinal.get("rps_normalized_by_threshold_count") is True, "RPS normalization drifted.")
    _require(ordinal.get("cumulative_binary_brier_reported") is True, "Cumulative Brier reporting drifted.")
    regression = contract["calibration_regression"]
    _exact(
        regression,
        {
            "targets": ["one_vs_rest_class_2", "one_vs_rest_class_3", "one_vs_rest_class_4", "cumulative_Y<=2", "cumulative_Y<=3"],
            "predictor": "logit_of_probability_clipped_to_1e-6_and_1_minus_1e-6",
            "model": "unpenalized_binary_logistic_intercept_and_slope_joint_fit",
            "optimizer": "deterministic_newton_raphson",
            "initial_intercept": 0.0,
            "initial_slope": 1.0,
            "maximum_iterations": 100,
            "step_tolerance": 1e-10,
            "hessian_ridge_for_numerical_solution": 1e-12,
            "reporting_scope": "pooled_exactly_once_oof_descriptive_diagnostic",
            "confidence_interval_applicable": False,
            "future_calibration_validation_claim_allowed": False,
        },
        "Calibration regression",
    )
    comparison = contract["comparison"]
    _require(comparison.get("contrast") == "sigmoid_minus_raw", "Calibration contrast drifted.")
    _require(comparison.get("test_set_method_selection_performed") is False, "Test-selected calibration is incorrectly permitted.")
    _require(comparison.get("inferential_claim_for_new_diagnostics_allowed") is False, "New diagnostic inference is incorrectly permitted.")
    _require(comparison.get("all_metrics_improved_claim_allowed") is False, "Universal calibration improvement is incorrectly permitted.")
    publication = contract["publication"]
    _require(publication.get("publish_oof_prediction_rows") is False, "OOF-row publication is incorrectly permitted.")
    _require(publication.get("publish_fitted_calibration_regressions") is False, "Fitted diagnostic object publication is incorrectly permitted.")
    _require(publication.get("publish_calibrator_parameters") is False, "Calibrator-parameter publication is incorrectly permitted.")
    _require(
        set(publication.get("prohibited_claims", []))
        == {
            "calibration_improved_everything",
            "test_selected_calibration_method",
            "future_calibration_proven",
            "deployment_ready_probability",
            "autonomous_hr_threshold_validated",
        },
        "Calibration prohibited-claim registry drifted.",
    )
    local = _validate_sources(contract)
    return {
        "status": "passed",
        "contract_sha256": sha256_file(PROJECT_ROOT / path),
        "sample_count": 1200,
        "methods": ["raw", "sigmoid"],
        "classwise_targets": 3,
        "cumulative_targets": 2,
        "bin_count": 10,
        "planned_new_model_fit_calls": 0,
        "planned_new_calibrator_fit_calls": 0,
        "source_evidence": local,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def main() -> int:
    print(
        json.dumps(
            validate_calibration_diagnostics_contract_v3(), indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
