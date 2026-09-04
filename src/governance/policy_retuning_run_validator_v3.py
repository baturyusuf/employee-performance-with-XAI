"""Independent closed-world validator for a complete v3 policy-retuning run."""

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
from src.experiments.manuscript_model_benchmark import (
    select_candidate_index,
    validate_benchmark_config,
)
from src.experiments.policy_retuning_v3 import FIXED_ESTIMAND, RETUNED_ESTIMAND
from src.experiments.shared_folds import read_shared_folds
from src.governance.offline_runtime import validate_policy_receipt
from src.governance.policy_retuning_contract_v3 import (
    CANONICAL_V2_ROOT,
    DEFAULT_POLICY_RETUNING_CONTRACT_PATH,
    HEADLINE_METRICS,
    POLICY_FEATURE_COUNTS,
    POLICY_IDS,
    POLICY_NAMES,
    validate_policy_retuning_contract_v3,
)
from src.models.ordinal_evaluation_v3 import ordinal_evaluation_bundle_v3
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_POLICY_RETUNING_RUN = Path(
    "reports/major_revision_v3_runs/"
    "phase1d_v3_20260904T063324Z_823c848/policy_retuning"
)
EXPECTED_FILES = frozenset(
    {
        "aggregate_metrics.csv",
        "candidate_search_results.csv",
        "combined_oof_predictions.csv",
        "fixed_oof_predictions.csv",
        "fold_metrics.csv",
        "headline_policy_comparison.csv",
        "metric_comparison.csv",
        "policy_feature_contract.csv",
        "retuned_oof_predictions.csv",
        "selected_candidate_frequency.csv",
        "selected_hyperparameters.csv",
        "stage_metadata.json",
    }
)
OUTPUT_HASH_FILES = EXPECTED_FILES - {"stage_metadata.json"}
EXPECTED_IMPLEMENTATIONS = frozenset(
    {
        "src/experiments/policy_retuning_v3.py",
        "src/governance/policy_retuning_contract_v3.py",
        "src/experiments/shared_folds.py",
        "src/models/canonical_models.py",
        "src/models/ordinal_evaluation_v3.py",
    }
)
IDENTITY_COLUMNS = (
    "run_id",
    "policy_contract_sha256",
    "scientific_input_sha256",
    "fold_contract_hash",
)
PROBABILITY_COLUMNS = ("prob_class_2", "prob_class_3", "prob_class_4")
FIXED_SOURCE_POLICY = {
    "P0": "full_feature_upper_bound",
    "P1": "no_salary_hike_no_attrition_sensitive_retaining_audit",
    "P2": "no_salary_hike_no_attrition",
    "P3": "no_salary_hike_no_attrition_no_department",
}
HIGHER_IS_BETTER = frozenset(
    {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "weighted_precision",
        "macro_recall",
        "weighted_recall",
        "quadratic_weighted_kappa",
        "adjacent_accuracy",
    }
)


class V3PolicyRetuningRunValidationError(RuntimeError):
    """Raised when persisted Phase 1D evidence is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V3PolicyRetuningRunValidationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V3PolicyRetuningRunValidationError(
            f"Could not read {path.as_posix()}: {exc}"
        ) from exc
    _require(isinstance(payload, dict), f"{path.name} must contain a JSON object.")
    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise V3PolicyRetuningRunValidationError(
            f"Could not parse {path.name}: {exc}"
        ) from exc


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
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
        raise V3PolicyRetuningRunValidationError(
            f"Could not resolve generation blob {commit}:{relative_path}: {exc}"
        ) from exc


def _assert_frame_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_columns: Sequence[str],
    context: str,
    tolerance: float = 1e-14,
) -> None:
    _require(
        set(observed.columns) == set(expected.columns),
        f"{context} schema drifted: observed={sorted(observed.columns)}, "
        f"expected={sorted(expected.columns)}.",
    )
    columns = list(expected.columns)
    try:
        pd.testing.assert_frame_equal(
            observed.loc[:, columns].sort_values(list(sort_columns)).reset_index(drop=True),
            expected.loc[:, columns].sort_values(list(sort_columns)).reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=tolerance,
        )
    except AssertionError as exc:
        raise V3PolicyRetuningRunValidationError(
            f"{context} does not match independent recomputation: {exc}"
        ) from exc


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
    converted = series.map(mapping)
    _require(converted.notna().all(), f"{context} contains an invalid Boolean value.")
    return converted.astype(bool)


def _json_object(value: Any, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise V3PolicyRetuningRunValidationError(
            f"{context} is not valid JSON: {exc}"
        ) from exc
    _require(isinstance(payload, dict), f"{context} must contain a JSON object.")
    return payload


def _json_scores(value: Any, *, context: str) -> list[float]:
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise V3PolicyRetuningRunValidationError(
            f"{context} is not valid JSON: {exc}"
        ) from exc
    _require(isinstance(payload, list) and len(payload) == 5, f"{context} must contain five values.")
    scores = [float(item) for item in payload]
    _require(all(math.isfinite(item) for item in scores), f"{context} contains non-finite values.")
    return scores


def _validate_identity(frame: pd.DataFrame, metadata: Mapping[str, Any], *, context: str) -> None:
    _require(set(IDENTITY_COLUMNS).issubset(frame.columns), f"{context} identity schema drifted.")
    expected = {
        "run_id": metadata["run_id"],
        "policy_contract_sha256": metadata["policy_contract_sha256"],
        "scientific_input_sha256": metadata["scientific_input_sha256"],
        "fold_contract_hash": metadata["fold_contract_hash"],
    }
    for column, value in expected.items():
        _require(set(frame[column].astype(str)) == {str(value)}, f"{context} identity drifted for {column}.")


def _expected_policy_features() -> pd.DataFrame:
    contract = _load_json(Path("configs/feature_availability_v3.json"))
    all_features = [str(record["feature_name"]) for record in contract["features"]]
    policies = {str(record["policy_id"]): record for record in contract["policies"]}
    rows: list[dict[str, Any]] = []
    for order, (policy_id, policy_name) in enumerate(zip(POLICY_IDS, POLICY_NAMES)):
        record = policies[policy_id]
        excluded = list(map(str, record["excluded_features"]))
        retained = [feature for feature in all_features if feature not in set(excluded)]
        rows.append(
            {
                "policy_id": policy_id,
                "policy_order": order,
                "policy_name": policy_name,
                "policy_role": str(record["role"]),
                "n_features": len(retained),
                "retained_features_json": json.dumps(retained, separators=(",", ":")),
                "excluded_features_json": json.dumps(excluded, separators=(",", ":")),
            }
        )
    return pd.DataFrame(rows)


def _validate_oof(
    fixed: pd.DataFrame,
    retuned: pd.DataFrame,
    combined: pd.DataFrame,
    *,
    metadata: Mapping[str, Any],
    selected: pd.DataFrame,
    source_fixed_oof: pd.DataFrame,
) -> None:
    for frame, estimand, context in (
        (fixed, FIXED_ESTIMAND, "fixed OOF"),
        (retuned, RETUNED_ESTIMAND, "retuned OOF"),
    ):
        _validate_identity(frame, metadata, context=context)
        _require(len(frame) == 7200, f"{context} row count drifted.")
        _require(set(frame["estimand"].astype(str)) == {estimand}, f"{context} estimand drifted.")
        _require(set(frame["policy_id"].astype(str)) == set(POLICY_IDS), f"{context} policy set drifted.")
        _require(set(frame["model"].astype(str)) == {"xgboost"}, f"{context} model drifted.")
        for policy_id, rows in frame.groupby("policy_id", sort=True):
            _require(len(rows) == 1200, f"{context}/{policy_id} row count drifted.")
            _require(rows["sample_index"].nunique() == 1200, f"{context}/{policy_id} is not exactly-once OOF.")
            _require(set(rows["sample_index"].astype(int)) == set(range(1200)), f"{context}/{policy_id} sample coverage drifted.")
        probability = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
        _require(np.all(np.isfinite(probability)), f"{context} contains non-finite probabilities.")
        _require(np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-9), f"{context} probability simplex drifted.")
        predicted = np.asarray([2, 3, 4])[np.argmax(probability, axis=1)]
        _require(np.array_equal(predicted, frame["y_pred"].to_numpy(int)), f"{context} label/probability mismatch.")

    folds = read_shared_folds(CANONICAL_V2_ROOT / "core/shared_folds")
    reference = folds.outer_assignments.set_index("sample_index")[["outer_fold", "y_true"]].astype(int).sort_index()
    for context, frame in (("fixed OOF", fixed), ("retuned OOF", retuned)):
        for policy_id, rows in frame.groupby("policy_id", sort=True):
            observed = rows.set_index("sample_index")[["outer_fold", "y_true"]].astype(int).sort_index()
            _require(observed.equals(reference), f"{context}/{policy_id} fold or target lineage drifted.")

    expected_combined = pd.concat([fixed, retuned], ignore_index=True)
    _assert_frame_equal(
        combined,
        expected_combined,
        sort_columns=["estimand", "policy_id", "sample_index"],
        context="combined OOF composition",
    )
    _validate_identity(combined, metadata, context="combined OOF")

    selected_lookup = selected.set_index(["policy_id", "outer_fold"])["selected_candidate_index"].astype(int)
    schedule = pd.read_csv(
        CANONICAL_V2_ROOT / "core/model_benchmarks/selected_hyperparameters.csv"
    )
    primary_schedule = schedule[schedule["model"] == "xgboost"].set_index("outer_fold")["selected_candidate_index"].astype(int)
    for policy_id, rows in fixed.groupby("policy_id", sort=True):
        observed = rows.groupby("outer_fold")["selected_candidate_index"].first().astype(int).sort_index()
        _require(observed.equals(primary_schedule.sort_index()), f"Fixed schedule drifted for {policy_id}.")
    for (policy_id, outer_fold), rows in retuned.groupby(["policy_id", "outer_fold"], sort=True):
        expected_index = int(selected_lookup.loc[(policy_id, int(outer_fold))])
        _require(set(rows["selected_candidate_index"].astype(int)) == {expected_index}, f"Retuned candidate lineage drifted for {policy_id}/fold {outer_fold}.")

    for policy_id, source_policy in FIXED_SOURCE_POLICY.items():
        observed = fixed[fixed["policy_id"] == policy_id].sort_values("sample_index")
        source = source_fixed_oof[
            (source_fixed_oof["system_id"] == source_policy)
            & (source_fixed_oof["model"] == "xgboost")
        ].sort_values("sample_index")
        _require(len(source) == 1200, f"Canonical fixed source count drifted for {policy_id}.")
        for column in ("sample_index", "outer_fold", "y_true", "y_pred", "selected_candidate_index"):
            _require(
                np.array_equal(observed[column].to_numpy(int), source[column].to_numpy(int)),
                f"Canonical fixed reuse drifted for {policy_id}/{column}.",
            )
        _require(
            np.allclose(
                observed.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
                source.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
                rtol=0.0,
                atol=1e-15,
            ),
            f"Canonical fixed probability reuse drifted for {policy_id}.",
        )
        _require(set(observed["source_policy"].astype(str)) == {source_policy}, f"Fixed source-policy label drifted for {policy_id}.")
        _require(
            set(observed["evidence_source"].astype(str))
            == {"canonical_v2_fixed_policy_oof_exact_feature_set_reuse"},
            f"Fixed evidence-source label drifted for {policy_id}.",
        )
    new_fixed = fixed[fixed["policy_id"].isin(["P4", "P5"])]
    _require(new_fixed["source_policy"].isna().all(), "P4/P5 fixed source_policy must be empty.")
    _require(
        set(new_fixed["evidence_source"].astype(str))
        == {"v3_new_outer_train_fit_with_primary_P3_schedule"},
        "P4/P5 fixed evidence-source label drifted.",
    )
    _require(retuned["source_policy"].isna().all(), "Retuned source_policy must be empty.")
    _require(
        set(retuned["evidence_source"].astype(str))
        == {"v3_policy_specific_inner_selection_and_outer_refit"},
        "Retuned evidence-source label drifted.",
    )

    fixed_p3 = fixed[fixed["policy_id"] == "P3"].sort_values("sample_index")
    retuned_p3 = retuned[retuned["policy_id"] == "P3"].sort_values("sample_index")
    for column in ("y_pred", "selected_candidate_index"):
        _require(np.array_equal(fixed_p3[column].to_numpy(int), retuned_p3[column].to_numpy(int)), f"P3 replay drifted for {column}.")
    _require(
        np.allclose(
            fixed_p3.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
            retuned_p3.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        ),
        "P3 probability replay exceeded tolerance.",
    )


def _validate_candidate_selection(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    frequency: pd.DataFrame,
    *,
    metadata: Mapping[str, Any],
) -> None:
    _validate_identity(candidates, metadata, context="candidate search")
    _validate_identity(selected, metadata, context="selected hyperparameters")
    _require(len(candidates) == 480, "Candidate-search row count drifted.")
    _require(len(selected) == 60, "Selected-hyperparameter row count drifted.")
    for frame, context in ((candidates, "candidate search"), (selected, "selected hyperparameters")):
        _require(set(frame["estimand"].astype(str)) == {RETUNED_ESTIMAND}, f"{context} estimand drifted.")
        _require(set(frame["model"].astype(str)) == {"xgboost"}, f"{context} model drifted.")
        _require(not _bool_values(frame["outer_test_used_for_selection"], context=context).any(), f"Outer test entered {context}.")
    _require(set(candidates["candidate_status"].astype(str)) == {"complete"}, "Candidate status drifted.")
    _require(set(candidates["n_inner_folds"].astype(int)) == {5}, "Inner-fold count drifted.")

    model_definition = validate_benchmark_config(load_config("configs/model_grid.yaml"))["models"]["xgboost"]
    registry = [dict(record) for record in model_definition["candidates"]]
    fixed_parameters = dict(model_definition["fixed_params"])
    for (policy_id, outer_fold), rows in candidates.groupby(["policy_id", "outer_fold"], sort=True):
        rows = rows.sort_values("candidate_index")
        _require(rows["candidate_index"].astype(int).tolist() == list(range(8)), f"Candidate grid drifted for {policy_id}/fold {outer_fold}.")
        macro_means: list[float] = []
        qwk_means: list[float] = []
        for _, row in rows.iterrows():
            index = int(row["candidate_index"])
            _require(_json_object(row["parameters_json"], context="candidate parameters") == registry[index], f"Candidate parameters drifted at index {index}.")
            macro_scores = _json_scores(row["inner_macro_f1_scores_json"], context="inner macro-F1 scores")
            qwk_scores = _json_scores(row["inner_qwk_scores_json"], context="inner QWK scores")
            macro_mean = float(np.mean(macro_scores))
            qwk_mean = float(np.mean(qwk_scores))
            _require(math.isclose(float(row["inner_macro_f1_mean"]), macro_mean, rel_tol=0.0, abs_tol=1e-14), "Candidate macro-F1 mean drifted.")
            _require(math.isclose(float(row["inner_qwk_mean"]), qwk_mean, rel_tol=0.0, abs_tol=1e-14), "Candidate QWK mean drifted.")
            macro_means.append(macro_mean)
            qwk_means.append(qwk_mean)
        expected_index = select_candidate_index(
            macro_means,
            qwk_means,
            practical_tie_tolerance=0.001,
            better_direction="higher",
        )
        selected_flags = _bool_values(rows["selected_by_protocol"], context="candidate selection")
        _require(int(selected_flags.sum()) == 1, f"Candidate selection multiplicity drifted for {policy_id}/fold {outer_fold}.")
        _require(int(rows.loc[selected_flags, "candidate_index"].iloc[0]) == expected_index, f"Candidate selection replay drifted for {policy_id}/fold {outer_fold}.")
        selected_row = selected[
            (selected["policy_id"] == policy_id)
            & (selected["outer_fold"].astype(int) == int(outer_fold))
        ]
        _require(len(selected_row) == 1, f"Selected row multiplicity drifted for {policy_id}/fold {outer_fold}.")
        chosen = selected_row.iloc[0]
        _require(int(chosen["selected_candidate_index"]) == expected_index, "Selected candidate index drifted.")
        _require(_json_object(chosen["selected_candidate_parameters_json"], context="selected parameters") == registry[expected_index], "Selected parameters drifted.")
        _require(_json_object(chosen["fixed_parameters_json"], context="fixed parameters") == fixed_parameters, "Fixed parameters drifted.")
        _require(math.isclose(float(chosen["selected_inner_macro_f1_mean"]), macro_means[expected_index], rel_tol=0.0, abs_tol=1e-14), "Selected macro-F1 drifted.")
        _require(math.isclose(float(chosen["selected_inner_qwk_mean"]), qwk_means[expected_index], rel_tol=0.0, abs_tol=1e-14), "Selected QWK drifted.")

    expected_frequency = (
        selected.groupby(
            ["policy_id", "policy_name", "selected_candidate_index", "selected_candidate_parameters_json"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "selection_count"})
    )
    totals = selected.groupby("policy_id").size().rename("selection_opportunities")
    expected_frequency = expected_frequency.merge(totals, on="policy_id", validate="many_to_one")
    expected_frequency["selection_frequency"] = (
        expected_frequency["selection_count"] / expected_frequency["selection_opportunities"]
    )
    _assert_frame_equal(
        frequency,
        expected_frequency,
        sort_columns=["policy_id", "selected_candidate_index"],
        context="selected candidate frequency",
    )
    _require(frequency.groupby("policy_id")["selection_count"].sum().eq(10).all(), "Selection opportunities drifted.")


def _recompute_summaries(
    combined: pd.DataFrame,
    policy_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for (estimand, policy_id), rows in combined.groupby(["estimand", "policy_id"], sort=True):
        rows = rows.sort_values("sample_index")
        identity = {column: rows.iloc[0][column] for column in IDENTITY_COLUMNS}
        policy = {
            "estimand": estimand,
            "policy_id": policy_id,
            "policy_name": rows.iloc[0]["policy_name"],
            "n_features": int(rows.iloc[0]["n_features"]),
            "model_name": "xgboost",
        }
        bundle = ordinal_evaluation_bundle_v3(
            rows["y_true"].astype(int),
            rows["y_pred"].astype(int),
            rows.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
            labels=(2, 3, 4),
            dataset_key="inx_primary",
            model_name="xgboost",
        )
        aggregate_rows.extend(
            {**identity, **policy, "metric": metric, "value": float(value)}
            for metric, value in bundle["aggregate_metrics"].items()
        )
        for outer_fold, fold_rows_source in rows.groupby("outer_fold", sort=True):
            fold_bundle = ordinal_evaluation_bundle_v3(
                fold_rows_source["y_true"].astype(int),
                fold_rows_source["y_pred"].astype(int),
                fold_rows_source.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
                labels=(2, 3, 4),
                dataset_key="inx_primary",
                model_name="xgboost",
            )
            fold_rows.append(
                {
                    **identity,
                    **policy,
                    "outer_fold": int(outer_fold),
                    "n_train": 1200 - len(fold_rows_source),
                    "n_test": len(fold_rows_source),
                    **fold_bundle["aggregate_metrics"],
                }
            )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(["policy_id", "estimand", "metric"]).reset_index(drop=True)
    fold_metrics = pd.DataFrame(fold_rows).sort_values(["policy_id", "estimand", "outer_fold"]).reset_index(drop=True)
    comparison = aggregate.pivot(
        index=["policy_id", "policy_name", "n_features", "metric"],
        columns="estimand",
        values="value",
    ).reset_index()
    comparison = comparison.rename(
        columns={FIXED_ESTIMAND: "fixed_value", RETUNED_ESTIMAND: "retuned_value"}
    )
    comparison.columns.name = None
    comparison["raw_difference_retuned_minus_fixed"] = comparison["retuned_value"] - comparison["fixed_value"]
    comparison["better_direction"] = np.where(comparison["metric"].isin(HIGHER_IS_BETTER), "higher", "lower")
    comparison["direction_aligned_improvement"] = np.where(
        comparison["better_direction"] == "higher",
        comparison["raw_difference_retuned_minus_fixed"],
        -comparison["raw_difference_retuned_minus_fixed"],
    )
    comparison = comparison.sort_values(["policy_id", "metric"]).reset_index(drop=True)
    feature_lookup = policy_features.set_index("policy_id")
    headline_rows: list[dict[str, Any]] = []
    for policy_id in POLICY_IDS:
        values = comparison[comparison["policy_id"] == policy_id].set_index("metric")
        row: dict[str, Any] = {
            "policy_id": policy_id,
            "policy_name": feature_lookup.loc[policy_id, "policy_name"],
            "n_features": int(feature_lookup.loc[policy_id, "n_features"]),
        }
        for metric in HEADLINE_METRICS:
            row[f"fixed_{metric}"] = float(values.loc[metric, "fixed_value"])
            row[f"retuned_{metric}"] = float(values.loc[metric, "retuned_value"])
            row[f"raw_difference_{metric}"] = float(values.loc[metric, "raw_difference_retuned_minus_fixed"])
            row[f"direction_aligned_improvement_{metric}"] = float(values.loc[metric, "direction_aligned_improvement"])
        headline_rows.append(row)
    return fold_metrics, aggregate, comparison, pd.DataFrame(headline_rows)


def validate_policy_retuning_run_v3(
    run_dir: Path | str = DEFAULT_POLICY_RETUNING_RUN,
) -> dict[str, Any]:
    """Validate a complete local run without trusting its acceptance decision."""

    root = Path(run_dir)
    _require(root.is_dir(), f"Policy-retuning run directory is absent: {root.as_posix()}.")
    inventory = {path.name for path in root.iterdir() if path.is_file()}
    _require(inventory == EXPECTED_FILES, f"Policy-retuning closed-world inventory drifted: {sorted(inventory ^ EXPECTED_FILES)}.")
    _require(not any(path.is_dir() for path in root.iterdir()), "Policy-retuning run contains an unexpected directory.")

    metadata = _load_json(root / "stage_metadata.json")
    _require(metadata.get("status") == "complete", "Run metadata is not complete.")
    _require(metadata.get("stage") == "policy_retuning_v3", "Run stage drifted.")
    _require(metadata.get("evidence_status") == "complete_two_estimand_exactly_once_oof", "Run evidence status drifted.")
    _require(metadata.get("run_id") == root.parent.name, "Run id/path identity drifted.")
    _require(metadata.get("git_identity", {}).get("commit") == "823c84866b461266c75f3224527f679a86ab670e", "Generation commit drifted.")
    _require(str(metadata["run_id"]).endswith("_823c848"), "Run-id commit suffix drifted.")
    for field, expected in {
        "policy_count": 6,
        "sample_count": 1200,
        "outer_folds": 10,
        "inner_folds": 5,
        "candidate_count": 8,
        "planned_new_estimator_fit_calls": 2480,
        "candidate_search_row_count": 480,
        "selected_hyperparameter_row_count": 60,
        "fixed_oof_row_count": 7200,
        "retuned_oof_row_count": 7200,
        "combined_oof_row_count": 14400,
        "fold_metric_row_count": 120,
        "aggregate_metric_row_count": 192,
        "metric_comparison_row_count": 96,
        "headline_policy_row_count": 6,
        "network_calls": 0,
        "paid_api_calls": 0,
    }.items():
        _require(metadata.get(field) == expected, f"Metadata {field} drifted.")
    _require(metadata.get("outer_test_used_for_selection") is False, "Metadata permits outer-test selection.")
    _require(metadata.get("seed_or_policy_selected_from_results") is False, "Metadata permits result-selected policy/seed.")
    _require(metadata.get("employee_level_outputs_publication_authorized") is False, "Metadata authorizes row-level publication.")
    validate_policy_receipt(metadata["runtime_policy"])

    output_hashes = metadata.get("output_hashes")
    _require(isinstance(output_hashes, Mapping) and set(output_hashes) == OUTPUT_HASH_FILES, "Output-hash inventory drifted.")
    for filename, expected_hash in output_hashes.items():
        _require(sha256_file(root / filename) == expected_hash, f"Output hash drifted for {filename}.")

    contract_receipt = validate_policy_retuning_contract_v3(DEFAULT_POLICY_RETUNING_CONTRACT_PATH)
    _require(contract_receipt["contract_sha256"] == metadata["policy_contract_sha256"], "Policy-contract hash drifted.")
    scientific_inputs = metadata.get("scientific_inputs")
    _require(isinstance(scientific_inputs, Mapping), "Scientific-input receipt is absent.")
    _require(_canonical_json_sha256(scientific_inputs) == metadata["scientific_input_sha256"], "Scientific-input digest drifted.")
    for field in ("git_identity", "source_tree_hash"):
        _require(scientific_inputs.get(field) == metadata.get(field), f"Scientific-input {field} drifted.")
    for field in ("policy_contract_sha256", "dataset_sha256", "fold_contract_hash"):
        _require(scientific_inputs.get(field) == metadata.get(field), f"Scientific-input {field} drifted.")
    contract = _load_json(DEFAULT_POLICY_RETUNING_CONTRACT_PATH)
    bound_sources = scientific_inputs.get("bound_source_hashes")
    _require(isinstance(bound_sources, Mapping) and set(bound_sources) == set(contract["source_contracts"]), "Bound-source inventory drifted.")
    for name, record in contract["source_contracts"].items():
        _require(bound_sources[name] == record["sha256"], f"Bound source receipt drifted for {name}.")
        _require(sha256_file(record["path"]) == record["sha256"], f"Bound source bytes drifted for {name}.")

    implementation_hashes = scientific_inputs.get("implementation_hashes")
    _require(isinstance(implementation_hashes, Mapping) and set(implementation_hashes) == EXPECTED_IMPLEMENTATIONS, "Implementation inventory drifted.")
    generation_commit = str(metadata["git_identity"]["commit"])
    for relative_path, expected_hash in implementation_hashes.items():
        _require(_sha256_bytes(_git_blob(generation_commit, relative_path)) == expected_hash, f"Generation implementation blob drifted for {relative_path}.")
    _require(
        _sha256_bytes(_git_blob(generation_commit, "configs/policy_retuning_v3.json"))
        == metadata["policy_contract_sha256"],
        "Generation policy-contract blob drifted.",
    )

    frames = {
        filename: _read_csv(root / filename)
        for filename in EXPECTED_FILES
        if filename.endswith(".csv")
    }
    policy_features = frames["policy_feature_contract.csv"]
    _assert_frame_equal(
        policy_features,
        _expected_policy_features(),
        sort_columns=["policy_order"],
        context="policy feature contract",
    )
    _require(policy_features["n_features"].astype(int).tolist() == list(POLICY_FEATURE_COUNTS), "Policy feature counts drifted.")

    candidates = frames["candidate_search_results.csv"]
    selected = frames["selected_hyperparameters.csv"]
    frequency = frames["selected_candidate_frequency.csv"]
    _validate_candidate_selection(candidates, selected, frequency, metadata=metadata)

    fixed = frames["fixed_oof_predictions.csv"]
    retuned = frames["retuned_oof_predictions.csv"]
    combined = frames["combined_oof_predictions.csv"]
    source_fixed = pd.read_csv(CANONICAL_V2_ROOT / "core/policy_ablation/oof_predictions.csv")
    _validate_oof(
        fixed,
        retuned,
        combined,
        metadata=metadata,
        selected=selected,
        source_fixed_oof=source_fixed,
    )

    expected_fold, expected_aggregate, expected_comparison, expected_headline = _recompute_summaries(combined, policy_features)
    for filename, expected, sort_columns in (
        ("fold_metrics.csv", expected_fold, ["policy_id", "estimand", "outer_fold"]),
        ("aggregate_metrics.csv", expected_aggregate, ["policy_id", "estimand", "metric"]),
        ("metric_comparison.csv", expected_comparison, ["policy_id", "metric"]),
        ("headline_policy_comparison.csv", expected_headline, ["policy_id"]),
    ):
        _assert_frame_equal(frames[filename], expected, sort_columns=sort_columns, context=filename)

    headline = expected_headline.set_index("policy_id")
    return {
        "status": "passed",
        "run_id": metadata["run_id"],
        "generation_commit": generation_commit,
        "policy_contract_sha256": metadata["policy_contract_sha256"],
        "scientific_input_sha256": metadata["scientific_input_sha256"],
        "file_count": len(EXPECTED_FILES),
        "candidate_search_row_count": len(candidates),
        "selected_hyperparameter_row_count": len(selected),
        "combined_oof_row_count": len(combined),
        "fold_metric_row_count": len(expected_fold),
        "aggregate_metric_row_count": len(expected_aggregate),
        "metric_comparison_row_count": len(expected_comparison),
        "p3_maximum_probability_replay_error": float(
            np.max(
                np.abs(
                    fixed[fixed["policy_id"] == "P3"].sort_values("sample_index").loc[:, PROBABILITY_COLUMNS].to_numpy(float)
                    - retuned[retuned["policy_id"] == "P3"].sort_values("sample_index").loc[:, PROBABILITY_COLUMNS].to_numpy(float)
                )
            )
        ),
        "headline_macro_f1_raw_differences": {
            policy_id: float(headline.loc[policy_id, "raw_difference_macro_f1"])
            for policy_id in POLICY_IDS
        },
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_POLICY_RETUNING_RUN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    print(json.dumps(validate_policy_retuning_run_v3(args.run_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
