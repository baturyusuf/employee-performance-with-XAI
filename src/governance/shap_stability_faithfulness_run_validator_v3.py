"""Independent closed-world validator for the complete v3 Phase 2A run."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from src.data.canonical_loader import sha256_file
from src.governance.offline_runtime import validate_policy_receipt
from src.governance.shap_stability_faithfulness_contract_v3 import (
    DEFAULT_SHAP_STABILITY_FAITHFULNESS_CONTRACT,
    DELETION_COUNTS,
    TOP_K_VALUES,
    validate_shap_stability_faithfulness_contract_v3,
)
from src.utils.config_loader import PROJECT_ROOT


DEFAULT_SHAP_STABILITY_FAITHFULNESS_RUN = Path(
    "reports/major_revision_v3_runs/"
    "phase2a_v3_20260904T073008Z_6e52de7/shap_stability_faithfulness"
)
EXPECTED_GENERATION_COMMIT = "6e52de76f7e486985cbc2b32a53b2554c1c6f6c1"
EXPECTED_FILES = frozenset(
    {
        "additivity_checks.csv",
        "aggregation_receipt.json",
        "deletion_auc_contrast.csv",
        "deletion_auc_sample.csv",
        "deletion_auc_summary.csv",
        "faithfulness_contrasts.csv",
        "faithfulness_feature_frequency.csv",
        "faithfulness_sample_results.csv",
        "faithfulness_summary.csv",
        "fold_feature_importance.csv",
        "stability_pairwise.csv",
        "stability_run_rankings.csv",
        "stability_summary.csv",
        "stage_metadata.json",
    }
)
OUTPUT_HASH_FILES = EXPECTED_FILES - {"stage_metadata.json"}
EXPECTED_IMPLEMENTATIONS = frozenset(
    {
        "src/experiments/shap_stability_faithfulness_v3.py",
        "src/governance/shap_stability_faithfulness_contract_v3.py",
        "src/explainability/canonical_shap_axis.py",
        "src/experiments/benchmark_artifact_contract.py",
        "src/models/canonical_models.py",
    }
)
EXPECTED_ROW_COUNTS = {
    "fold_feature_importance.csv": 2200,
    "additivity_checks.csv": 110,
    "stability_run_rankings.csv": 220,
    "stability_pairwise.csv": 75,
    "stability_summary.csv": 9,
    "faithfulness_sample_results.csv": 75600,
    "faithfulness_summary.csv": 63,
    "faithfulness_contrasts.csv": 6,
    "deletion_auc_sample.csv": 25200,
    "deletion_auc_summary.csv": 21,
    "deletion_auc_contrast.csv": 1,
}


class V3ShapStabilityFaithfulnessRunValidationError(RuntimeError):
    """Raised when persisted Phase 2A evidence is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V3ShapStabilityFaithfulnessRunValidationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V3ShapStabilityFaithfulnessRunValidationError(
            f"Could not read {path.as_posix()}: {exc}"
        ) from exc
    _require(isinstance(payload, dict), f"{path.name} must contain a JSON object.")
    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise V3ShapStabilityFaithfulnessRunValidationError(
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
        raise V3ShapStabilityFaithfulnessRunValidationError(
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
        raise V3ShapStabilityFaithfulnessRunValidationError(
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


def _feature_universe(contract: Mapping[str, Any]) -> list[str]:
    record = contract["source_contracts"]["feature_availability"]
    payload = _load_json(PROJECT_ROOT / str(record["path"]))
    all_features = [str(item["feature_name"]) for item in payload["features"]]
    p3 = next(item for item in payload["policies"] if item["policy_id"] == "P3")
    excluded = set(map(str, p3["excluded_features"]))
    retained = [feature for feature in all_features if feature not in excluded]
    _require(len(retained) == 20, "P3 feature universe drifted.")
    return retained


def _run_registry(contract: Mapping[str, Any]) -> dict[tuple[str, str], tuple[int, float | None]]:
    registry: dict[tuple[str, str], tuple[int, float | None]] = {
        ("model_seed", "canonical_seed_42"): (42, None)
    }
    for record in contract["seed_stability"]["new_runs"]:
        registry[("model_seed", str(record["run_label"]))] = (
            int(record["model_seed"]),
            None,
        )
    for seed in contract["resampling_stability"]["subsample_seeds"]:
        registry[("outer_train_resample", f"resample_{seed}")] = (42, float(seed))
    return registry


def _expected_memberships(
    contract: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[tuple[str, str, int], tuple[int, str]]]:
    source = contract["source_contracts"]["canonical_v2_outer_assignments"]
    assignments = pd.read_csv(PROJECT_ROOT / str(source["path"]))
    assignments = assignments.loc[:, ["sample_index", "outer_fold", "y_true"]].astype(int)
    all_ids = set(assignments["sample_index"].tolist())
    expected: dict[tuple[str, str, int], tuple[int, str]] = {}
    for (stability_type, run_label), (_, subsample_seed) in _run_registry(contract).items():
        for outer_fold in range(1, 11):
            test_ids = set(
                assignments.loc[
                    assignments["outer_fold"] == outer_fold, "sample_index"
                ].tolist()
            )
            train_ids = sorted(all_ids - test_ids)
            if stability_type == "outer_train_resample":
                sampled, _ = train_test_split(
                    train_ids,
                    train_size=0.8,
                    random_state=int(subsample_seed) + outer_fold,
                    stratify=assignments.set_index("sample_index").loc[train_ids, "y_true"],
                )
                train_ids = sorted(map(int, sampled))
            expected[(stability_type, run_label, outer_fold)] = (
                len(train_ids),
                _canonical_json_sha256(train_ids),
            )
    return assignments, expected


def _rebuild_rankings(fold_importance: pd.DataFrame) -> pd.DataFrame:
    keys = ["stability_type", "run_label", "model_seed", "subsample_seed"]
    rows: list[dict[str, Any]] = []
    for key, group in fold_importance.groupby(keys, dropna=False, sort=True):
        weighted = (
            group.assign(weighted=lambda frame: frame["mean_abs_grouped_shap"] * frame["n_test"])
            .groupby("feature", as_index=False)
            .agg(weighted_sum=("weighted", "sum"), n_oof=("n_test", "sum"))
        )
        weighted["mean_abs_grouped_shap"] = weighted["weighted_sum"] / weighted["n_oof"]
        weighted = weighted.sort_values(
            ["mean_abs_grouped_shap", "feature"], ascending=[False, True]
        ).reset_index(drop=True)
        for rank, record in enumerate(weighted.itertuples(index=False), start=1):
            rows.append(
                {
                    "stability_type": key[0],
                    "run_label": key[1],
                    "model_seed": key[2],
                    "subsample_seed": key[3],
                    "feature": record.feature,
                    "mean_abs_grouped_shap": float(record.mean_abs_grouped_shap),
                    "rank": rank,
                    "n_oof": int(record.n_oof),
                }
            )
    return pd.DataFrame(rows)


def _rebuild_pairwise(rankings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stability_type, scoped in rankings.groupby("stability_type", sort=True):
        by_run = {
            str(label): group.sort_values("rank")["feature"].astype(str).tolist()
            for label, group in scoped.groupby("run_label", sort=True)
        }
        for left, right in itertools.combinations(sorted(by_run), 2):
            left_features, right_features = by_run[left], by_run[right]
            _require(set(left_features) == set(right_features), "Stability feature universe drifted.")
            right_rank = {
                feature: rank for rank, feature in enumerate(right_features, start=1)
            }
            rho = float(
                spearmanr(
                    range(1, len(left_features) + 1),
                    [right_rank[feature] for feature in left_features],
                ).statistic
            )
            for top_k in TOP_K_VALUES:
                left_set = set(left_features[:top_k])
                right_set = set(right_features[:top_k])
                rows.append(
                    {
                        "stability_type": stability_type,
                        "left_run": left,
                        "right_run": right,
                        "top_k": top_k,
                        "jaccard": len(left_set & right_set) / len(left_set | right_set),
                        "all_feature_spearman": rho,
                        "pair_independence_assumed": False,
                        "confidence_interval_applicable": False,
                    }
                )
    return pd.DataFrame(rows)


def _rebuild_stability_summary(
    pairwise: pd.DataFrame, canonical_fold_summary: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (stability_type, top_k), values in pairwise.groupby(
        ["stability_type", "top_k"], sort=True
    ):
        rows.append(
            {
                "stability_type": stability_type,
                "top_k": int(top_k),
                "n_pairs": len(values),
                "jaccard_mean": float(values["jaccard"].mean()),
                "jaccard_sample_sd": float(values["jaccard"].std(ddof=1)),
                "jaccard_median": float(values["jaccard"].median()),
                "jaccard_min": float(values["jaccard"].min()),
                "jaccard_max": float(values["jaccard"].max()),
                "spearman_mean": float(values["all_feature_spearman"].mean()),
                "spearman_sample_sd": float(values["all_feature_spearman"].std(ddof=1)),
                "spearman_median": float(values["all_feature_spearman"].median()),
                "spearman_min": float(values["all_feature_spearman"].min()),
                "spearman_max": float(values["all_feature_spearman"].max()),
                "pair_independence_assumed": False,
                "confidence_interval_applicable": False,
            }
        )
    for record in canonical_fold_summary.itertuples(index=False):
        rows.append(
            {
                "stability_type": "canonical_outer_fold_pair",
                "top_k": int(record.top_k),
                "n_pairs": int(record.n_fold_pairs),
                "jaccard_mean": float(record.jaccard_mean),
                "jaccard_sample_sd": float(record.jaccard_std),
                "jaccard_median": float(record.jaccard_median),
                "jaccard_min": float(record.jaccard_min),
                "jaccard_max": float(record.jaccard_max),
                "spearman_mean": float(record.spearman_mean),
                "spearman_sample_sd": float(record.spearman_std),
                "spearman_median": float(record.spearman_median),
                "spearman_min": float(record.spearman_min),
                "spearman_max": float(record.spearman_max),
                "pair_independence_assumed": False,
                "confidence_interval_applicable": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["stability_type", "top_k"]).reset_index(drop=True)


def _validate_stability(
    frames: Mapping[str, pd.DataFrame],
    contract: Mapping[str, Any],
    aggregation: Mapping[str, Any],
) -> None:
    features = _feature_universe(contract)
    registry = _run_registry(contract)
    assignments, memberships = _expected_memberships(contract)
    fold_importance = frames["fold_feature_importance.csv"]
    additivity = frames["additivity_checks.csv"]
    observed_registry = set(
        fold_importance[["stability_type", "run_label"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    _require(observed_registry == set(registry), "Stability run registry drifted.")
    schedule_source = contract["source_contracts"]["canonical_v2_selected_hyperparameters"]
    schedule = pd.read_csv(PROJECT_ROOT / str(schedule_source["path"]))
    schedule = schedule[schedule["model"] == "xgboost"].set_index("outer_fold")
    selected_lookup = schedule["selected_candidate_index"].astype(int).to_dict()

    for key, (model_seed, subsample_seed) in registry.items():
        for frame, context, expected_feature_rows in (
            (fold_importance, "fold feature importance", 20),
            (additivity, "additivity checks", 1),
        ):
            scoped = frame[
                (frame["stability_type"] == key[0])
                & (frame["run_label"] == key[1])
            ]
            _require(len(scoped) == 10 * expected_feature_rows, f"{context}/{key} row count drifted.")
            _require(set(scoped["outer_fold"].astype(int)) == set(range(1, 11)), f"{context}/{key} fold coverage drifted.")
            _require(set(scoped["model_seed"].astype(int)) == {model_seed}, f"{context}/{key} model seed drifted.")
            if subsample_seed is None:
                _require(scoped["subsample_seed"].isna().all(), f"{context}/{key} subsample seed must be empty.")
            else:
                _require(set(scoped["subsample_seed"].astype(int)) == {int(subsample_seed)}, f"{context}/{key} subsample seed drifted.")
            for outer_fold, rows in scoped.groupby("outer_fold", sort=True):
                expected_n_train, expected_hash = memberships[(key[0], key[1], int(outer_fold))]
                _require(set(rows["n_train"].astype(int)) == {expected_n_train}, f"{context}/{key}/fold {outer_fold} training count drifted.")
                _require(set(rows["n_test"].astype(int)) == {120}, f"{context}/{key}/fold {outer_fold} test count drifted.")
                _require(set(rows["training_membership_sha256"].astype(str)) == {expected_hash}, f"{context}/{key}/fold {outer_fold} membership hash drifted.")
                _require(set(rows["selected_candidate_index"].astype(int)) == {selected_lookup[int(outer_fold)]}, f"{context}/{key}/fold {outer_fold} candidate schedule drifted.")
        scoped_importance = fold_importance[
            (fold_importance["stability_type"] == key[0])
            & (fold_importance["run_label"] == key[1])
        ]
        for _, rows in scoped_importance.groupby("outer_fold", sort=True):
            _require(set(rows["feature"].astype(str)) == set(features), f"Fold feature coverage drifted for {key}.")
        expected_source = (
            "exact_canonical_v2_outer_model"
            if key == ("model_seed", "canonical_seed_42")
            else "new_contract_bound_refit"
        )
        _require(set(scoped_importance["model_source"].astype(str)) == {expected_source}, f"Model-source label drifted for {key}.")

    _require(np.isfinite(fold_importance["mean_abs_grouped_shap"]).all(), "SHAP importance contains non-finite values.")
    _require((fold_importance["mean_abs_grouped_shap"] >= 0).all(), "Absolute SHAP importance is negative.")
    _require(set(additivity["n_transformed_features"].astype(int)) == {46}, "Transformed feature count drifted.")
    _require(set(additivity["feature_perturbation"].astype(str)) == {"tree_path_dependent"}, "SHAP perturbation mode drifted.")
    _require(set(additivity["model_output"].astype(str)) == {"raw_margin"}, "SHAP output space drifted.")
    maximum_additivity = float(additivity["maximum_raw_margin_additivity_error"].max())
    maximum_grouping = float(additivity["maximum_grouped_sum_error"].max())
    _require(maximum_additivity <= 1e-5, "Raw-margin additivity tolerance was exceeded.")
    _require(maximum_grouping <= 1e-12, "Signed grouping sum-preservation tolerance was exceeded.")
    _require(math.isclose(float(aggregation["maximum_observed_raw_margin_additivity_error"]), maximum_additivity, rel_tol=0.0, abs_tol=1e-15), "Aggregation receipt additivity maximum drifted.")
    _require(math.isclose(float(aggregation["maximum_observed_grouped_sum_error"]), maximum_grouping, rel_tol=0.0, abs_tol=1e-15), "Aggregation receipt grouping maximum drifted.")

    expected_rankings = _rebuild_rankings(fold_importance)
    _assert_frame_equal(
        frames["stability_run_rankings.csv"],
        expected_rankings,
        sort_columns=["stability_type", "run_label", "rank"],
        context="stability rankings",
    )
    expected_pairwise = _rebuild_pairwise(expected_rankings)
    _assert_frame_equal(
        frames["stability_pairwise.csv"],
        expected_pairwise,
        sort_columns=["stability_type", "left_run", "right_run", "top_k"],
        context="pairwise stability",
    )
    canonical_source = contract["source_contracts"]["canonical_v2_shap_stability"]
    canonical_summary = pd.read_csv(PROJECT_ROOT / str(canonical_source["path"]))
    expected_summary = _rebuild_stability_summary(expected_pairwise, canonical_summary)
    _assert_frame_equal(
        frames["stability_summary.csv"],
        expected_summary,
        sort_columns=["stability_type", "top_k"],
        context="stability summary",
    )
    _require(not _bool_values(expected_pairwise["pair_independence_assumed"], context="pairwise stability").any(), "Stability pairs are incorrectly marked independent.")
    _require(not _bool_values(expected_pairwise["confidence_interval_applicable"], context="pairwise stability").any(), "Pairwise confidence intervals are incorrectly permitted.")
    _require(len(assignments) == 1200, "Canonical assignment count drifted.")


def _json_feature_list(value: Any, *, context: str) -> list[str]:
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise V3ShapStabilityFaithfulnessRunValidationError(
            f"{context} is not valid JSON: {exc}"
        ) from exc
    _require(isinstance(payload, list), f"{context} must contain a JSON list.")
    return [str(item) for item in payload]


def _rebuild_faithfulness_outputs(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        results.groupby(["method", "random_repetition", "deleted_feature_count"], as_index=False)
        .agg(
            n_samples=("sample_index", "size"),
            mean_probability_drop=("probability_drop", "mean"),
            sample_sd_probability_drop=("probability_drop", "std"),
            median_probability_drop=("probability_drop", "median"),
            mean_raw_margin_drop=("raw_margin_drop", "mean"),
            sample_sd_raw_margin_drop=("raw_margin_drop", "std"),
            median_raw_margin_drop=("raw_margin_drop", "median"),
            positive_probability_drop_fraction=(
                "probability_drop",
                lambda values: float(np.mean(np.asarray(values) > 0)),
            ),
        )
        .sort_values(["method", "random_repetition", "deleted_feature_count"])
        .reset_index(drop=True)
    )
    contrast_rows: list[dict[str, Any]] = []
    for count in DELETION_COUNTS:
        guided = summary[
            (summary["method"] == "shap_guided")
            & (summary["deleted_feature_count"] == count)
        ].iloc[0]
        random = summary[
            (summary["method"] == "random")
            & (summary["deleted_feature_count"] == count)
        ]
        for metric in ("mean_probability_drop", "mean_raw_margin_drop"):
            random_values = random[metric].to_numpy(float)
            guided_value = float(guided[metric])
            contrast_rows.append(
                {
                    "deleted_feature_count": count,
                    "metric": metric,
                    "guided_value": guided_value,
                    "random_repetition_mean": float(np.mean(random_values)),
                    "random_repetition_sample_sd": float(np.std(random_values, ddof=1)),
                    "random_repetition_min": float(np.min(random_values)),
                    "random_repetition_max": float(np.max(random_values)),
                    "guided_minus_random_mean": guided_value - float(np.mean(random_values)),
                    "guided_exceeds_random_repetition_fraction": float(np.mean(guided_value > random_values)),
                    "inferential_p_value": np.nan,
                }
            )
    contrasts = pd.DataFrame(contrast_rows)

    auc_rows: list[dict[str, Any]] = []
    x = np.asarray([0.0, 0.2, 0.6, 1.0])
    for (method, repetition, sample_index), group in results.groupby(
        ["method", "random_repetition", "sample_index"], sort=True
    ):
        values = group.set_index("deleted_feature_count")["probability_drop"]
        y = np.asarray(
            [0.0, float(values.loc[1]), float(values.loc[3]), float(values.loc[5])]
        )
        auc_rows.append(
            {
                "method": method,
                "random_repetition": int(repetition),
                "sample_index": int(sample_index),
                "deletion_auc_probability_drop": float(np.trapezoid(y, x)),
            }
        )
    auc_sample = pd.DataFrame(auc_rows)
    auc_summary = (
        auc_sample.groupby(["method", "random_repetition"], as_index=False)
        .agg(
            n_samples=("sample_index", "size"),
            mean_deletion_auc=("deletion_auc_probability_drop", "mean"),
            sample_sd_deletion_auc=("deletion_auc_probability_drop", "std"),
            median_deletion_auc=("deletion_auc_probability_drop", "median"),
        )
    )
    guided_auc = float(
        auc_summary.loc[
            auc_summary["method"] == "shap_guided", "mean_deletion_auc"
        ].iloc[0]
    )
    random_auc = auc_summary.loc[
        auc_summary["method"] == "random", "mean_deletion_auc"
    ].to_numpy(float)
    auc_contrast = pd.DataFrame(
        [
            {
                "guided_mean_deletion_auc": guided_auc,
                "random_repetition_mean": float(np.mean(random_auc)),
                "random_repetition_sample_sd": float(np.std(random_auc, ddof=1)),
                "random_repetition_min": float(np.min(random_auc)),
                "random_repetition_max": float(np.max(random_auc)),
                "guided_minus_random_mean": guided_auc - float(np.mean(random_auc)),
                "guided_exceeds_random_repetition_fraction": float(np.mean(guided_auc > random_auc)),
                "inferential_p_value": np.nan,
            }
        ]
    )
    return summary, contrasts, auc_sample, auc_summary, auc_contrast


def _validate_faithfulness(
    frames: Mapping[str, pd.DataFrame], contract: Mapping[str, Any]
) -> None:
    results = frames["faithfulness_sample_results.csv"]
    features = _feature_universe(contract)
    feature_set = set(features)
    assignment_source = contract["source_contracts"]["canonical_v2_outer_assignments"]
    assignments = pd.read_csv(PROJECT_ROOT / str(assignment_source["path"]))
    reference = assignments.set_index("sample_index")[["outer_fold", "y_true"]].astype(int)
    canonical_source = contract["source_contracts"]["canonical_v2_benchmark_oof"]
    canonical_oof = pd.read_csv(PROJECT_ROOT / str(canonical_source["path"]))
    canonical_oof = canonical_oof[canonical_oof["model"] == "xgboost"].set_index("sample_index")
    _require(len(canonical_oof) == 1200, "Canonical XGBoost OOF coverage drifted.")

    expected_methods = {("shap_guided", 0), *{("random", value) for value in range(1, 21)}}
    observed_methods = set(
        results[["method", "random_repetition"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    _require(observed_methods == expected_methods, "Faithfulness method/repetition grid drifted.")
    for method, repetition in expected_methods:
        scoped = results[
            (results["method"] == method)
            & (results["random_repetition"].astype(int) == repetition)
        ]
        _require(len(scoped) == 3600, f"Faithfulness rows drifted for {method}/{repetition}.")
        for count in DELETION_COUNTS:
            count_rows = scoped[scoped["deleted_feature_count"].astype(int) == count]
            _require(len(count_rows) == 1200, f"Faithfulness sample count drifted for {method}/{repetition}/top{count}.")
            _require(set(count_rows["sample_index"].astype(int)) == set(range(1200)), f"Faithfulness exactly-once coverage drifted for {method}/{repetition}/top{count}.")

    observed_identity = (
        results[["sample_index", "outer_fold", "y_true"]]
        .drop_duplicates()
        .set_index("sample_index")
        .sort_index()
        .astype(int)
    )
    _require(observed_identity.equals(reference.sort_index()), "Faithfulness fold/target lineage drifted.")
    expected_prediction = canonical_oof["y_pred"].astype(int).sort_index()
    observed_prediction = (
        results[["sample_index", "original_predicted_class"]]
        .drop_duplicates()
        .set_index("sample_index")["original_predicted_class"]
        .astype(int)
        .sort_index()
    )
    _require(observed_prediction.equals(expected_prediction), "Faithfulness baseline prediction lineage drifted.")

    probability_error = np.abs(
        results["baseline_probability"].to_numpy(float)
        - results["perturbed_probability"].to_numpy(float)
        - results["probability_drop"].to_numpy(float)
    )
    margin_error = np.abs(
        results["baseline_raw_margin"].to_numpy(float)
        - results["perturbed_raw_margin"].to_numpy(float)
        - results["raw_margin_drop"].to_numpy(float)
    )
    _require(float(probability_error.max()) <= 5e-15, "Faithfulness probability-drop arithmetic drifted.")
    _require(float(margin_error.max()) <= 5e-14, "Faithfulness raw-margin-drop arithmetic drifted.")
    _require(results["baseline_probability"].between(0.0, 1.0).all(), "Baseline probability is outside [0,1].")
    _require(results["perturbed_probability"].between(0.0, 1.0).all(), "Perturbed probability is outside [0,1].")

    baseline_counts = results.groupby("sample_index")[["baseline_probability", "baseline_raw_margin"]].nunique(dropna=False)
    _require((baseline_counts == 1).all().all(), "Faithfulness baseline is not invariant within sample.")
    probability_columns = ["prob_class_2", "prob_class_3", "prob_class_4"]
    expected_baselines = []
    for sample_index, predicted_class in expected_prediction.items():
        expected_baselines.append(
            float(canonical_oof.loc[sample_index, probability_columns[int(predicted_class) - 2]])
        )
    observed_baselines = (
        results[["sample_index", "baseline_probability"]]
        .drop_duplicates()
        .sort_values("sample_index")["baseline_probability"]
        .to_numpy(float)
    )
    _require(np.allclose(observed_baselines, expected_baselines, rtol=0.0, atol=1e-12), "Faithfulness baseline probabilities drifted from canonical OOF.")

    parsed: dict[tuple[str, int, int, int], list[str]] = {}
    for row in results.itertuples(index=False):
        key = (
            str(row.method),
            int(row.random_repetition),
            int(row.sample_index),
            int(row.deleted_feature_count),
        )
        selected = _json_feature_list(
            row.deleted_features_json,
            context=f"deleted features {key}",
        )
        _require(len(selected) == key[3], f"Deleted feature count drifted for {key}.")
        _require(len(set(selected)) == len(selected), f"Deleted features repeat for {key}.")
        _require(set(selected).issubset(feature_set), f"Unknown deleted feature for {key}.")
        parsed[key] = selected
    _require(len(parsed) == len(results), "Faithfulness key multiplicity drifted.")
    for method, repetition in expected_methods:
        for sample_index in range(1200):
            top1 = parsed[(method, repetition, sample_index, 1)]
            top3 = parsed[(method, repetition, sample_index, 3)]
            top5 = parsed[(method, repetition, sample_index, 5)]
            _require(top3[:1] == top1 and top5[:3] == top3, f"Deletion order is not nested for {method}/{repetition}/sample {sample_index}.")
            if method == "random":
                seed = int(contract["faithfulness"]["random_baseline_seeds"][repetition - 1])
                order = np.random.default_rng(seed + sample_index * 1009).permutation(len(features))
                expected = [features[int(index)] for index in order[:5]]
                _require(top5 == expected, f"Random baseline replay drifted for repetition {repetition}/sample {sample_index}.")

    guided = results[
        (results["method"] == "shap_guided")
        & (results["random_repetition"].astype(int) == 0)
    ]
    frequency_rows: list[dict[str, Any]] = []
    for count in DELETION_COUNTS:
        counts: dict[str, int] = {}
        for value in guided.loc[
            guided["deleted_feature_count"].astype(int) == count,
            "deleted_features_json",
        ]:
            for feature in _json_feature_list(value, context="guided feature frequency"):
                counts[feature] = counts.get(feature, 0) + 1
        for feature, selection_count in counts.items():
            frequency_rows.append(
                {
                    "deleted_feature_count": count,
                    "feature": feature,
                    "selection_count": selection_count,
                    "selection_opportunities": 1200,
                    "selection_frequency": selection_count / 1200,
                }
            )
    expected_frequency = pd.DataFrame(frequency_rows).sort_values(
        ["deleted_feature_count", "selection_count", "feature"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    _assert_frame_equal(
        frames["faithfulness_feature_frequency.csv"],
        expected_frequency,
        sort_columns=["deleted_feature_count", "feature"],
        context="faithfulness feature frequency",
    )

    summary, contrasts, auc_sample, auc_summary, auc_contrast = _rebuild_faithfulness_outputs(results)
    for filename, expected, sort_columns in (
        ("faithfulness_summary.csv", summary, ["method", "random_repetition", "deleted_feature_count"]),
        ("faithfulness_contrasts.csv", contrasts, ["deleted_feature_count", "metric"]),
        ("deletion_auc_sample.csv", auc_sample, ["method", "random_repetition", "sample_index"]),
        ("deletion_auc_summary.csv", auc_summary, ["method", "random_repetition"]),
        ("deletion_auc_contrast.csv", auc_contrast, ["guided_mean_deletion_auc"]),
    ):
        _assert_frame_equal(
            frames[filename],
            expected,
            sort_columns=sort_columns,
            context=filename,
        )


def _validate_aggregation_receipt(
    receipt: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    expected = {
        "schema_version": 1,
        "library": "shap",
        "library_version": "0.51.0",
        "explainer": "shap.TreeExplainer",
        "background_data": None,
        "feature_perturbation": "tree_path_dependent",
        "model_output": "raw_margin",
        "class_labels": [2, 3, 4],
        "normalized_axis_order": "sample_class_transformed_feature",
        "operation_order": [
            "normalize_library_output_to_sample_class_transformed_feature",
            "sum_signed_encoded_values_within_raw_feature_family",
            "take_absolute_grouped_value",
            "mean_across_classes",
            "mean_across_exactly_once_oof_samples",
        ],
        "absolute_before_transformed_grouping": False,
        "grouped_sum_preservation_required_per_sample_class": True,
        "additivity_tolerance": 1e-5,
        "probability_space_explanation": False,
        "stability_is_faithfulness": False,
        "run_id": metadata["run_id"],
        "contract_sha256": metadata["contract_sha256"],
        "scientific_input_sha256": metadata["scientific_input_sha256"],
    }
    for field, value in expected.items():
        _require(receipt.get(field) == value, f"Aggregation receipt drifted for {field}.")


def validate_shap_stability_faithfulness_run_v3(
    run_dir: Path | str = DEFAULT_SHAP_STABILITY_FAITHFULNESS_RUN,
) -> dict[str, Any]:
    """Validate a complete local Phase 2A run without trusting its acceptance decision."""

    root = Path(run_dir)
    _require(root.is_dir(), f"Phase 2A run directory is absent: {root.as_posix()}.")
    inventory = {path.name for path in root.iterdir() if path.is_file()}
    _require(inventory == EXPECTED_FILES, f"Phase 2A closed-world inventory drifted: {sorted(inventory ^ EXPECTED_FILES)}.")
    _require(not any(path.is_dir() for path in root.iterdir()), "Phase 2A run contains an unexpected directory.")

    metadata = _load_json(root / "stage_metadata.json")
    _require(metadata.get("status") == "complete", "Run metadata is not complete.")
    _require(metadata.get("stage") == "shap_stability_faithfulness_v3", "Run stage drifted.")
    _require(metadata.get("run_id") == root.parent.name, "Run id/path identity drifted.")
    generation_commit = str(metadata.get("git_identity", {}).get("commit"))
    _require(generation_commit == EXPECTED_GENERATION_COMMIT, "Generation commit drifted.")
    _require(str(metadata["run_id"]).endswith("_6e52de7"), "Run-id commit suffix drifted.")
    for field, expected in {
        "sample_count": 1200,
        "feature_count": 20,
        "outer_folds": 10,
        "seed_stability_run_count_including_reference": 6,
        "resampling_run_count": 5,
        "planned_new_estimator_fit_calls": 100,
        "fold_feature_importance_row_count": 2200,
        "additivity_check_row_count": 110,
        "stability_ranking_row_count": 220,
        "stability_pairwise_row_count": 75,
        "stability_summary_row_count": 9,
        "faithfulness_sample_row_count": 75600,
        "faithfulness_summary_row_count": 63,
        "faithfulness_contrast_row_count": 6,
        "deletion_auc_sample_row_count": 25200,
        "deletion_auc_summary_row_count": 21,
        "network_calls": 0,
        "paid_api_calls": 0,
    }.items():
        _require(metadata.get(field) == expected, f"Metadata {field} drifted.")
    _require(metadata.get("masking_out_of_distribution_risk") is True, "Masking OOD limitation is absent.")
    _require(metadata.get("human_usefulness_claim_allowed") is False, "Human-usefulness claim is incorrectly permitted.")
    validate_policy_receipt(metadata["runtime_policy"])

    output_hashes = metadata.get("output_hashes")
    _require(isinstance(output_hashes, Mapping) and set(output_hashes) == OUTPUT_HASH_FILES, "Output-hash inventory drifted.")
    for filename, expected_hash in output_hashes.items():
        _require(sha256_file(root / filename) == expected_hash, f"Output hash drifted for {filename}.")

    contract_receipt = validate_shap_stability_faithfulness_contract_v3(
        DEFAULT_SHAP_STABILITY_FAITHFULNESS_CONTRACT
    )
    _require(contract_receipt["contract_sha256"] == metadata["contract_sha256"], "Phase 2A contract hash drifted.")
    contract = _load_json(PROJECT_ROOT / DEFAULT_SHAP_STABILITY_FAITHFULNESS_CONTRACT)
    scientific_inputs = metadata.get("scientific_inputs")
    _require(isinstance(scientific_inputs, Mapping), "Scientific-input receipt is absent.")
    _require(_canonical_json_sha256(scientific_inputs) == metadata["scientific_input_sha256"], "Scientific-input digest drifted.")
    _require(scientific_inputs.get("git_identity") == metadata.get("git_identity"), "Scientific-input Git identity drifted.")
    _require(scientific_inputs.get("contract_sha256") == metadata.get("contract_sha256"), "Scientific-input contract hash drifted.")
    _require(scientific_inputs.get("dataset_sha256") == metadata.get("dataset_sha256"), "Scientific-input dataset hash drifted.")
    bound_sources = scientific_inputs.get("bound_source_hashes")
    _require(isinstance(bound_sources, Mapping) and set(bound_sources) == set(contract["source_contracts"]), "Bound-source inventory drifted.")
    for name, record in contract["source_contracts"].items():
        _require(bound_sources[name] == record["sha256"], f"Bound source receipt drifted for {name}.")
        _require(sha256_file(PROJECT_ROOT / str(record["path"])) == record["sha256"], f"Bound source bytes drifted for {name}.")

    implementation_hashes = scientific_inputs.get("implementation_hashes")
    _require(isinstance(implementation_hashes, Mapping) and set(implementation_hashes) == EXPECTED_IMPLEMENTATIONS, "Implementation inventory drifted.")
    for relative_path, expected_hash in implementation_hashes.items():
        _require(_sha256_bytes(_git_blob(generation_commit, relative_path)) == expected_hash, f"Generation implementation blob drifted for {relative_path}.")
    _require(
        _sha256_bytes(
            _git_blob(generation_commit, DEFAULT_SHAP_STABILITY_FAITHFULNESS_CONTRACT.as_posix())
        )
        == metadata["contract_sha256"],
        "Generation Phase 2A contract blob drifted.",
    )

    frames = {
        filename: _read_csv(root / filename)
        for filename in EXPECTED_FILES
        if filename.endswith(".csv")
    }
    for filename, expected in EXPECTED_ROW_COUNTS.items():
        _require(len(frames[filename]) == expected, f"{filename} row count drifted.")
    aggregation = _load_json(root / "aggregation_receipt.json")
    _validate_aggregation_receipt(aggregation, metadata)
    _validate_stability(frames, contract, aggregation)
    _validate_faithfulness(frames, contract)

    stability = frames["stability_summary.csv"].set_index(["stability_type", "top_k"])
    contrasts = frames["faithfulness_contrasts.csv"].set_index(
        ["deleted_feature_count", "metric"]
    )
    auc = frames["deletion_auc_contrast.csv"].iloc[0]
    return {
        "status": "passed",
        "run_id": metadata["run_id"],
        "generation_commit": generation_commit,
        "contract_sha256": metadata["contract_sha256"],
        "scientific_input_sha256": metadata["scientific_input_sha256"],
        "file_count": len(EXPECTED_FILES),
        "stability_pairwise_row_count": len(frames["stability_pairwise.csv"]),
        "faithfulness_sample_row_count": len(frames["faithfulness_sample_results.csv"]),
        "maximum_raw_margin_additivity_error": float(
            frames["additivity_checks.csv"]["maximum_raw_margin_additivity_error"].max()
        ),
        "seed_stability_top5_jaccard_mean": float(
            stability.loc[("model_seed", 5), "jaccard_mean"]
        ),
        "resample_stability_top5_jaccard_mean": float(
            stability.loc[("outer_train_resample", 5), "jaccard_mean"]
        ),
        "guided_minus_random_probability_drop": {
            str(count): float(
                contrasts.loc[(count, "mean_probability_drop"), "guided_minus_random_mean"]
            )
            for count in DELETION_COUNTS
        },
        "guided_minus_random_mean_deletion_auc": float(
            auc["guided_minus_random_mean"]
        ),
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_SHAP_STABILITY_FAITHFULNESS_RUN,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    print(
        json.dumps(
            validate_shap_stability_faithfulness_run_v3(args.run_dir),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
