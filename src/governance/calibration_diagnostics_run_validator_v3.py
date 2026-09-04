"""Independent closed-world validator for the complete v3 Phase 2B run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.canonical_loader import sha256_file
from src.governance.calibration_diagnostics_contract_v3 import (
    DEFAULT_CALIBRATION_DIAGNOSTICS_CONTRACT,
    validate_calibration_diagnostics_contract_v3,
)
from src.governance.offline_runtime import validate_policy_receipt
from src.utils.config_loader import PROJECT_ROOT


DEFAULT_CALIBRATION_DIAGNOSTICS_RUN = Path(
    "reports/major_revision_v3_runs/"
    "phase2b_v3_20260904T120838Z_21d1aec/calibration_diagnostics"
)
EXPECTED_GENERATION_COMMIT = "21d1aecb6e61511e95aee498ab81c54fe6e5a6ab"
LABELS = (2, 3, 4)
METHODS = ("raw", "sigmoid")
PROBABILITY_COLUMNS = ("prob_class_2", "prob_class_3", "prob_class_4")
EXPECTED_FILES = frozenset(
    {
        "calibration_metric_summary.csv",
        "classwise_calibration_metrics.csv",
        "classwise_reliability.png",
        "classwise_reliability.svg",
        "cumulative_calibration_metrics.csv",
        "cumulative_reliability.png",
        "cumulative_reliability.svg",
        "diagnostic_receipt.json",
        "extended_reliability_bins.csv",
        "method_comparison.csv",
        "stage_metadata.json",
    }
)
OUTPUT_HASH_FILES = EXPECTED_FILES - {"stage_metadata.json"}
EXPECTED_IMPLEMENTATIONS = frozenset(
    {
        "src/experiments/calibration_diagnostics_v3.py",
        "src/governance/calibration_diagnostics_contract_v3.py",
        "src/models/ordinal_evaluation_v3.py",
    }
)
EXPECTED_ROW_COUNTS = {
    "calibration_metric_summary.csv": 2,
    "classwise_calibration_metrics.csv": 6,
    "cumulative_calibration_metrics.csv": 4,
    "extended_reliability_bins.csv": 120,
    "method_comparison.csv": 6,
}


class V3CalibrationDiagnosticsRunValidationError(RuntimeError):
    """Raised when persisted Phase 2B evidence is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V3CalibrationDiagnosticsRunValidationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V3CalibrationDiagnosticsRunValidationError(
            f"Could not read {path.as_posix()}: {exc}"
        ) from exc
    _require(isinstance(payload, dict), f"{path.name} must contain a JSON object.")
    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise V3CalibrationDiagnosticsRunValidationError(
            f"Could not parse {path.name}: {exc}"
        ) from exc


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_blob(commit: str, relative_path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise V3CalibrationDiagnosticsRunValidationError(
            f"Could not resolve generation blob {commit}:{relative_path}: {exc}"
        ) from exc


def _assert_frame_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_columns: Sequence[str],
    context: str,
    tolerance: float = 1e-12,
) -> None:
    _require(
        set(observed.columns) == set(expected.columns),
        f"{context} schema drifted.",
    )
    columns = list(expected.columns)
    try:
        pd.testing.assert_frame_equal(
            observed.loc[:, columns]
            .sort_values(list(sort_columns))
            .reset_index(drop=True),
            expected.loc[:, columns]
            .sort_values(list(sort_columns))
            .reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=tolerance,
        )
    except AssertionError as exc:
        raise V3CalibrationDiagnosticsRunValidationError(
            f"{context} does not match independent recomputation: {exc}"
        ) from exc


def _bool_values(series: pd.Series, *, context: str) -> pd.Series:
    values = series.map(
        {
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
    )
    _require(values.notna().all(), f"{context} contains an invalid Boolean.")
    return values.astype(bool)


def _bin_rows(
    outcomes: np.ndarray,
    scores: np.ndarray,
    *,
    method: str,
    scope: str,
    target_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, 11)
    for position in range(1, 11):
        low, high = edges[position - 1], edges[position]
        selected = (
            (scores >= low) & (scores <= high)
            if position == 1
            else (scores > low) & (scores <= high)
        )
        count = int(selected.sum())
        positives = int(outcomes[selected].sum()) if count else 0
        mean_score = float(scores[selected].mean()) if count else np.nan
        frequency = float(positives / count) if count else np.nan
        rows.append(
            {
                "method": method,
                "reliability_scope": scope,
                "target_id": target_id,
                "bin": position,
                "bin_low": float(low),
                "bin_high": float(high),
                "n_samples": count,
                "n_positive": positives,
                "bin_status": "observed" if count else "empty",
                "mean_predicted_probability": mean_score,
                "observed_frequency": frequency,
                "absolute_gap": abs(mean_score - frequency) if count else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    _require(int(frame["n_samples"].sum()) == len(outcomes), "Independent bins lost samples.")
    return frame


def _ece(bins: pd.DataFrame) -> float:
    observed = bins[bins["n_samples"].astype(int) > 0]
    total = float(observed["n_samples"].sum())
    return float(
        np.sum(
            observed["n_samples"].to_numpy(float)
            / total
            * observed["absolute_gap"].to_numpy(float)
        )
    )


def _calibration_regression(outcomes: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(scores.astype(float), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    design = np.column_stack((np.ones(len(logits)), logits))
    beta = np.asarray([0.0, 1.0], dtype=float)
    converged = False
    maximum_step = float("inf")
    condition_number = float("nan")
    for iteration in range(1, 101):
        linear = design @ beta
        fitted = np.empty_like(linear)
        nonnegative = linear >= 0.0
        fitted[nonnegative] = 1.0 / (1.0 + np.exp(-linear[nonnegative]))
        exponential = np.exp(linear[~nonnegative])
        fitted[~nonnegative] = exponential / (1.0 + exponential)
        weights = np.maximum(fitted * (1.0 - fitted), np.finfo(float).eps)
        information = design.T @ (weights[:, np.newaxis] * design)
        condition_number = float(np.linalg.cond(information))
        gradient = design.T @ (outcomes.astype(float) - fitted)
        step = np.linalg.solve(information + 1e-12 * np.eye(2), gradient)
        beta += step
        maximum_step = float(np.max(np.abs(step)))
        if maximum_step <= 1e-10:
            converged = True
            break
    _require(converged, "Independent calibration regression did not converge.")
    return {
        "calibration_intercept": float(beta[0]),
        "calibration_slope": float(beta[1]),
        "regression_converged": True,
        "regression_iterations": iteration,
        "maximum_final_step": maximum_step,
        "information_condition_number": condition_number,
    }


def _binary_log_loss(outcomes: np.ndarray, scores: np.ndarray) -> float:
    clipped = np.clip(scores.astype(float), 1e-15, 1.0 - 1e-15)
    return float(
        -np.mean(
            outcomes * np.log(clipped)
            + (1.0 - outcomes) * np.log1p(-clipped)
        )
    )


def _binary_row(
    outcomes: np.ndarray,
    scores: np.ndarray,
    *,
    method: str,
    target_type: str,
    target_id: str,
    ece: float,
) -> dict[str, Any]:
    return {
        "method": method,
        "target_type": target_type,
        "target_id": target_id,
        "n_samples": len(outcomes),
        "n_positive": int(outcomes.sum()),
        "n_negative": int(len(outcomes) - outcomes.sum()),
        "prevalence": float(outcomes.mean()),
        "expected_calibration_error": ece,
        "binary_brier": float(np.mean((scores - outcomes) ** 2)),
        "binary_log_loss": _binary_log_loss(outcomes, scores),
        **_calibration_regression(outcomes, scores),
        "regression_scope": "pooled_exactly_once_oof_descriptive_diagnostic",
        "confidence_interval_applicable": False,
    }


def _independent_recomputation(
    contract: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    predictions_path = PROJECT_ROOT / str(
        contract["source_contracts"]["calibration_predictions"]["path"]
    )
    predictions = _read_csv(predictions_path)
    all_bins: list[pd.DataFrame] = []
    class_rows: list[dict[str, Any]] = []
    cumulative_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method in METHODS:
        rows = predictions[predictions["method"] == method].sort_values("sample_index")
        _require(len(rows) == 1200, f"Independent source coverage drifted for {method}.")
        y_true = rows["y_true"].to_numpy(int)
        y_pred = rows["y_pred"].to_numpy(int)
        probability = rows.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
        top_bins = _bin_rows(
            (y_true == y_pred).astype(int),
            probability.max(axis=1),
            method=method,
            scope="top_label",
            target_id="argmax_correct",
        )
        all_bins.append(top_bins)
        class_eces: list[float] = []
        cumulative_eces: list[float] = []
        cumulative_briers: list[float] = []
        for position, label in enumerate(LABELS):
            outcomes = (y_true == label).astype(int)
            scores = probability[:, position]
            bins = _bin_rows(
                outcomes,
                scores,
                method=method,
                scope="one_vs_rest_class",
                target_id=f"class_{label}",
            )
            all_bins.append(bins)
            value = _ece(bins)
            class_eces.append(value)
            class_rows.append(
                {
                    "class_label": label,
                    **_binary_row(
                        outcomes,
                        scores,
                        method=method,
                        target_type="one_vs_rest_class",
                        target_id=f"class_{label}",
                        ece=value,
                    ),
                }
            )
        for position, threshold in enumerate(LABELS[:-1]):
            outcomes = (y_true <= threshold).astype(int)
            scores = probability[:, : position + 1].sum(axis=1)
            bins = _bin_rows(
                outcomes,
                scores,
                method=method,
                scope="cumulative_threshold",
                target_id=f"Y_le_{threshold}",
            )
            all_bins.append(bins)
            value = _ece(bins)
            diagnostic = _binary_row(
                outcomes,
                scores,
                method=method,
                target_type="cumulative_threshold",
                target_id=f"Y_le_{threshold}",
                ece=value,
            )
            cumulative_rows.append(
                {
                    "threshold_label": threshold,
                    "cumulative_event": f"Y<={threshold}",
                    **diagnostic,
                }
            )
            cumulative_eces.append(value)
            cumulative_briers.append(float(diagnostic["binary_brier"]))
        encoded = np.zeros_like(probability)
        positions = {label: index for index, label in enumerate(LABELS)}
        encoded[np.arange(len(y_true)), [positions[value] for value in y_true]] = 1.0
        true_probability = probability[
            np.arange(len(y_true)), [positions[value] for value in y_true]
        ]
        cumulative_observed = np.column_stack(
            [(y_true <= threshold).astype(float) for threshold in LABELS[:-1]]
        )
        cumulative_probability = np.column_stack(
            [probability[:, : position + 1].sum(axis=1) for position in range(2)]
        )
        rps = float(np.mean((cumulative_probability - cumulative_observed) ** 2))
        method_bins = pd.concat(all_bins, ignore_index=True)
        method_bins = method_bins[method_bins["method"] == method]
        summary_rows.append(
            {
                "method": method,
                "primary_method": method == "sigmoid",
                "n_samples": 1200,
                "nll_log_loss": float(-np.mean(np.log(np.clip(true_probability, 1e-15, 1.0)))),
                "multiclass_brier": float(np.mean(np.sum((probability - encoded) ** 2, axis=1))),
                "top_label_ece": _ece(top_bins),
                "macro_classwise_ece": float(np.mean(class_eces)),
                "mean_cumulative_ece": float(np.mean(cumulative_eces)),
                "ranked_probability_score": rps,
                "mean_cumulative_binary_brier": float(np.mean(cumulative_briers)),
                "empty_reliability_bin_count": int((method_bins["n_samples"] == 0).sum()),
                "calibration_method_selection_performed": False,
            }
        )
    summary = pd.DataFrame(summary_rows)
    classwise = pd.DataFrame(class_rows).sort_values(["method", "class_label"]).reset_index(drop=True)
    cumulative = pd.DataFrame(cumulative_rows).sort_values(["method", "threshold_label"]).reset_index(drop=True)
    reliability = pd.concat(all_bins, ignore_index=True).sort_values(
        ["method", "reliability_scope", "target_id", "bin"]
    ).reset_index(drop=True)
    paired_path = PROJECT_ROOT / str(
        contract["source_contracts"]["calibration_paired_differences"]["path"]
    )
    paired = _read_csv(paired_path)
    legacy_names = {
        "nll_log_loss": "nll_log_loss",
        "multiclass_brier": "multiclass_brier",
        "top_label_ece": "ece_confidence",
    }
    lookup = summary.set_index("method")
    comparison_rows: list[dict[str, Any]] = []
    for metric in contract["comparison"]["metrics"]:
        raw_value = float(lookup.loc["raw", metric])
        sigmoid_value = float(lookup.loc["sigmoid", metric])
        legacy_name = legacy_names.get(metric)
        source = paired[paired["metric"] == legacy_name] if legacy_name else pd.DataFrame()
        interval = len(source) == 1
        comparison_rows.append(
            {
                "comparison_id": "sigmoid_minus_raw",
                "metric": metric,
                "better_direction": "lower",
                "raw_value": raw_value,
                "sigmoid_value": sigmoid_value,
                "raw_difference_sigmoid_minus_raw": sigmoid_value - raw_value,
                "direction_aligned_improvement": raw_value - sigmoid_value,
                "bootstrap_interval_available": interval,
                "raw_difference_ci_low": float(source.iloc[0]["raw_difference_ci_low"]) if interval else np.nan,
                "raw_difference_ci_high": float(source.iloc[0]["raw_difference_ci_high"]) if interval else np.nan,
                "improvement_ci_low": float(source.iloc[0]["improvement_ci_low"]) if interval else np.nan,
                "improvement_ci_high": float(source.iloc[0]["improvement_ci_high"]) if interval else np.nan,
                "interval_source": "canonical_v2_paired_stratified_sample_bootstrap" if interval else "not_estimated_for_new_diagnostic",
                "test_set_method_selection_performed": False,
                "all_metrics_improved_claim_allowed": False,
            }
        )
    return {
        "calibration_metric_summary.csv": summary,
        "classwise_calibration_metrics.csv": classwise,
        "cumulative_calibration_metrics.csv": cumulative,
        "extended_reliability_bins.csv": reliability,
        "method_comparison.csv": pd.DataFrame(comparison_rows),
    }


def _validate_figures(root: Path) -> None:
    expected_png_dimensions = {
        "classwise_reliability.png": (4680, 1440),
        "cumulative_reliability.png": (3120, 1440),
    }
    for filename, (expected_width, expected_height) in expected_png_dimensions.items():
        data = (root / filename).read_bytes()
        _require(data[:8] == b"\x89PNG\r\n\x1a\n", f"{filename} is not a PNG.")
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        _require((width, height) == (expected_width, expected_height), f"{filename} dimensions drifted.")
    required_svg = {
        "classwise_reliability.svg": (
            "Classwise OOF reliability",
            "Class 2 one-vs-rest",
            "Class 3 one-vs-rest",
            "Class 4 one-vs-rest",
        ),
        "cumulative_reliability.svg": (
            "Ordinal cumulative OOF reliability",
            "Cumulative event Y≤2",
            "Cumulative event Y≤3",
        ),
    }
    for filename, labels in required_svg.items():
        text = (root / filename).read_text(encoding="utf-8")
        _require("<svg" in text and all(label in text for label in labels), f"{filename} labels drifted.")
        _require("<dc:date>" not in text, f"{filename} contains nondeterministic date metadata.")


def validate_calibration_diagnostics_run_v3(
    run_dir: Path | str = DEFAULT_CALIBRATION_DIAGNOSTICS_RUN,
) -> dict[str, Any]:
    """Validate a complete local Phase 2B run without trusting its calculations."""

    root = Path(run_dir)
    _require(root.is_dir(), f"Phase 2B run directory is absent: {root.as_posix()}.")
    inventory = {path.name for path in root.iterdir() if path.is_file()}
    _require(inventory == EXPECTED_FILES, f"Phase 2B closed-world inventory drifted: {sorted(inventory ^ EXPECTED_FILES)}.")
    _require(not any(path.is_dir() for path in root.iterdir()), "Phase 2B run contains an unexpected directory.")
    metadata = _load_json(root / "stage_metadata.json")
    _require(metadata.get("status") == "complete", "Phase 2B metadata is incomplete.")
    _require(metadata.get("stage") == "calibration_diagnostics_v3", "Phase 2B stage drifted.")
    _require(metadata.get("run_id") == root.parent.name, "Phase 2B run-id/path drifted.")
    generation_commit = str(metadata.get("git_identity", {}).get("commit"))
    _require(generation_commit == EXPECTED_GENERATION_COMMIT, "Phase 2B generation commit drifted.")
    _require(str(metadata["run_id"]).endswith("_21d1aec"), "Phase 2B run-id suffix drifted.")
    expected_metadata = {
        "prediction_row_count": 2400,
        "sample_count_per_method": 1200,
        "method_count": 2,
        "new_model_fit_calls": 0,
        "new_calibrator_fit_calls": 0,
        "diagnostic_calibration_regression_fit_count": 10,
        "metric_summary_row_count": 2,
        "classwise_metric_row_count": 6,
        "cumulative_metric_row_count": 4,
        "reliability_bin_row_count": 120,
        "method_comparison_row_count": 6,
        "same_dataset_oof_diagnostic_not_future_validation": True,
        "test_set_method_selection_performed": False,
        "all_metrics_improved_claim_allowed": False,
        "network_calls": 0,
        "paid_api_calls": 0,
    }
    for field, expected in expected_metadata.items():
        _require(metadata.get(field) == expected, f"Phase 2B metadata {field} drifted.")
    validate_policy_receipt(metadata["runtime_policy"])
    output_hashes = metadata.get("output_hashes")
    _require(isinstance(output_hashes, Mapping) and set(output_hashes) == OUTPUT_HASH_FILES, "Phase 2B output-hash inventory drifted.")
    for filename, expected_hash in output_hashes.items():
        _require(sha256_file(root / filename) == expected_hash, f"Phase 2B output hash drifted for {filename}.")

    contract_receipt = validate_calibration_diagnostics_contract_v3()
    _require(contract_receipt["contract_sha256"] == metadata["contract_sha256"], "Phase 2B contract hash drifted.")
    contract = _load_json(PROJECT_ROOT / DEFAULT_CALIBRATION_DIAGNOSTICS_CONTRACT)
    scientific_inputs = metadata.get("scientific_inputs")
    _require(isinstance(scientific_inputs, Mapping), "Phase 2B scientific inputs are absent.")
    _require(_canonical_json_sha256(scientific_inputs) == metadata["scientific_input_sha256"], "Phase 2B scientific-input digest drifted.")
    _require(scientific_inputs.get("git_identity") == metadata.get("git_identity"), "Phase 2B Git identity drifted.")
    _require(scientific_inputs.get("contract_sha256") == metadata.get("contract_sha256"), "Phase 2B contract receipt drifted.")
    bound_sources = scientific_inputs.get("bound_source_hashes")
    _require(isinstance(bound_sources, Mapping) and set(bound_sources) == set(contract["source_contracts"]), "Phase 2B bound-source inventory drifted.")
    for name, record in contract["source_contracts"].items():
        _require(bound_sources[name] == record["sha256"], f"Phase 2B source receipt drifted for {name}.")
        _require(sha256_file(PROJECT_ROOT / str(record["path"])) == record["sha256"], f"Phase 2B source bytes drifted for {name}.")
    implementations = scientific_inputs.get("implementation_hashes")
    _require(isinstance(implementations, Mapping) and set(implementations) == EXPECTED_IMPLEMENTATIONS, "Phase 2B implementation inventory drifted.")
    for relative_path, expected_hash in implementations.items():
        _require(_sha256_bytes(_git_blob(generation_commit, relative_path)) == expected_hash, f"Phase 2B generation blob drifted for {relative_path}.")
    _require(
        _sha256_bytes(_git_blob(generation_commit, DEFAULT_CALIBRATION_DIAGNOSTICS_CONTRACT.as_posix()))
        == metadata["contract_sha256"],
        "Generation Phase 2B contract blob drifted.",
    )

    expected_frames = _independent_recomputation(contract)
    observed_frames = {
        filename: _read_csv(root / filename) for filename in EXPECTED_ROW_COUNTS
    }
    sort_columns = {
        "calibration_metric_summary.csv": ["method"],
        "classwise_calibration_metrics.csv": ["method", "class_label"],
        "cumulative_calibration_metrics.csv": ["method", "threshold_label"],
        "extended_reliability_bins.csv": ["method", "reliability_scope", "target_id", "bin"],
        "method_comparison.csv": ["metric"],
    }
    for filename, expected_count in EXPECTED_ROW_COUNTS.items():
        _require(len(observed_frames[filename]) == expected_count, f"{filename} row count drifted.")
        _assert_frame_equal(
            observed_frames[filename],
            expected_frames[filename],
            sort_columns=sort_columns[filename],
            context=filename,
        )
    bins = observed_frames["extended_reliability_bins.csv"]
    _require(set(bins["bin"].astype(int)) == set(range(1, 11)), "Phase 2B bin grid drifted.")
    empty = bins["n_samples"].astype(int) == 0
    _require((bins.loc[empty, "bin_status"] == "empty").all(), "Phase 2B empty-bin labels drifted.")
    _require(bins.loc[empty, ["mean_predicted_probability", "observed_frequency", "absolute_gap"]].isna().all().all(), "Phase 2B empty-bin values drifted.")
    comparison = observed_frames["method_comparison.csv"].set_index("metric")
    _require(float(comparison.loc["top_label_ece", "direction_aligned_improvement"]) < 0.0, "Adverse top-label ECE finding was lost.")
    _require(not _bool_values(comparison["test_set_method_selection_performed"], context="method selection").any(), "Test-selected calibration is incorrectly recorded.")
    _require(not _bool_values(comparison["all_metrics_improved_claim_allowed"], context="universal improvement").any(), "Universal improvement is incorrectly permitted.")
    _validate_figures(root)

    diagnostic = _load_json(root / "diagnostic_receipt.json")
    for field, expected in {
        "run_id": metadata["run_id"],
        "contract_sha256": metadata["contract_sha256"],
        "scientific_input_sha256": metadata["scientific_input_sha256"],
        "sample_count_per_method": 1200,
        "new_model_fit_calls": 0,
        "new_calibrator_fit_calls": 0,
        "diagnostic_calibration_regression_fit_count": 10,
        "reliability_bin_count": 10,
        "empty_bins_persisted": True,
        "rps_normalized_by_threshold_count": True,
        "rps_equals_mean_cumulative_binary_brier": True,
        "future_calibration_validation_claim_allowed": False,
        "all_metrics_improved_claim_allowed": False,
    }.items():
        _require(diagnostic.get(field) == expected, f"Diagnostic receipt {field} drifted.")
    summary = observed_frames["calibration_metric_summary.csv"].set_index("method")
    classwise = observed_frames["classwise_calibration_metrics.csv"]
    return {
        "status": "passed",
        "run_id": metadata["run_id"],
        "generation_commit": generation_commit,
        "contract_sha256": metadata["contract_sha256"],
        "scientific_input_sha256": metadata["scientific_input_sha256"],
        "file_count": len(EXPECTED_FILES),
        "reliability_bin_row_count": len(bins),
        "empty_reliability_bin_count": int(empty.sum()),
        "maximum_regression_condition_number": float(classwise["information_condition_number"].max()),
        "raw_metrics": {
            metric: float(summary.loc["raw", metric])
            for metric in (
                "nll_log_loss",
                "multiclass_brier",
                "top_label_ece",
                "macro_classwise_ece",
                "mean_cumulative_ece",
                "ranked_probability_score",
            )
        },
        "sigmoid_metrics": {
            metric: float(summary.loc["sigmoid", metric])
            for metric in (
                "nll_log_loss",
                "multiclass_brier",
                "top_label_ece",
                "macro_classwise_ece",
                "mean_cumulative_ece",
                "ranked_probability_score",
            )
        },
        "new_model_fit_calls": 0,
        "new_calibrator_fit_calls": 0,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_CALIBRATION_DIAGNOSTICS_RUN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    print(json.dumps(validate_calibration_diagnostics_run_v3(args.run_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
