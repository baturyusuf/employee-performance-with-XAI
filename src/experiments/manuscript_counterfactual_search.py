from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from src.core.io_utils import ensure_dir, write_json, write_jsonl
from src.data.canonical_loader import load_canonical_dataset
from src.experiments.final_evidence_common import align_proba, predict_labels_from_proba
from src.experiments.manuscript_policy_ablation import _model_parameters, exact_policy_frame, resolve_seed
from src.features.feature_sets import taxonomy_by_feature
from src.governance.manuscript_contract import (
    canonical_config_hash,
    sha256_file,
    validate_counterfactual_search_contract,
)
from src.models.canonical_models import CanonicalModelError, build_model_pipeline
from src.utils.config_loader import load_config


RELATIONAL_CONSTRAINTS = (
    ("ExperienceYearsInCurrentRole", "ExperienceYearsAtThisCompany"),
    ("YearsWithCurrManager", "ExperienceYearsAtThisCompany"),
    ("YearsSinceLastPromotion", "ExperienceYearsAtThisCompany"),
    ("ExperienceYearsAtThisCompany", "TotalWorkExperienceInYears"),
)
SUMMARY_IDENTITY_FIELDS = (
    "run_id",
    "config_hash",
    "scientific_input_hash",
    "source_tree_hash",
    "dataset_sha256",
    "fold_contract_sha256",
    "feature_policy_sha256",
    "source_oof_probability_sha256",
    "model_set_sha256",
    "policy",
    "dataset_key",
    "task_type",
    "evidence_role",
)


class CounterfactualProtocolError(RuntimeError):
    """Raised when heuristic-search evidence would violate the OOF protocol."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _index_sha256(values: Sequence[Any]) -> str:
    return _canonical_sha256([int(value) for value in values])


def _serialized_model_sha256(pipeline: Any) -> str:
    """Hash the actual fitted preprocessing/model state used for OOF search."""

    buffer = io.BytesIO()
    try:
        joblib.dump(pipeline, buffer, compress=0, protocol=4)
    except (TypeError, ValueError, pickle.PickleError) as exc:
        raise CounterfactualProtocolError(
            f"Supplementary counterfactual model serialization failed: {type(exc).__name__}: {exc}"
        ) from exc
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _fit_supplementary_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: Mapping[str, Any],
    seed: int,
    *,
    forbidden_features: Sequence[str],
) -> Any:
    """Fit the supplementary heuristic model without importing calibration internals."""

    fixed = dict(parameters)
    fixed.pop("random_state", None)
    fixed["n_jobs"] = 1
    try:
        pipeline = build_model_pipeline(
            "xgboost",
            X_train,
            fixed_parameters=fixed,
            candidate_parameters={},
            random_state=int(seed),
            forbidden_features=forbidden_features,
        )
        with threadpool_limits(limits=1):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                pipeline.fit(X_train, y_train)
    except (CanonicalModelError, TypeError, ValueError, Warning) as exc:
        raise CounterfactualProtocolError(
            f"Supplementary counterfactual model fit failed: {type(exc).__name__}: {exc}"
        ) from exc
    return pipeline


def wilson_interval(successes: int, denominator: int, confidence: float = 0.95) -> tuple[float, float]:
    if denominator <= 0:
        return math.nan, math.nan
    if successes < 0 or successes > denominator:
        raise ValueError("successes must be between zero and denominator")
    # The canonical confidence level is 95%; retaining the parameter makes the
    # output contract explicit without introducing an undeclared dependency.
    if not math.isclose(confidence, 0.95):
        raise CounterfactualProtocolError("Only the predeclared 95% Wilson interval is supported.")
    z = 1.959963984540054
    proportion = successes / denominator
    denominator_term = 1.0 + z * z / denominator
    center = (proportion + z * z / (2.0 * denominator)) / denominator_term
    half_width = z * math.sqrt(
        (proportion * (1.0 - proportion) + z * z / (4.0 * denominator)) / denominator
    ) / denominator_term
    return max(0.0, center - half_width), min(1.0, center + half_width)


def training_scales(frame: pd.DataFrame) -> Dict[str, float]:
    scales: Dict[str, float] = {}
    for feature in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[feature]):
            values = pd.to_numeric(frame[feature], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                scales[feature] = 1.0
                continue
            q25, q75 = np.quantile(values, [0.25, 0.75])
            scale = float(q75 - q25)
            if not np.isfinite(scale) or scale <= 0:
                scale = float(np.max(values) - np.min(values))
            scales[feature] = scale if np.isfinite(scale) and scale > 0 else 1.0
        else:
            scales[feature] = 1.0
    return scales


def change_cost(feature: str, old_value: Any, new_value: Any, scales: Mapping[str, float]) -> float:
    if (pd.isna(old_value) and pd.isna(new_value)) or str(old_value) == str(new_value):
        return 0.0
    try:
        return abs(float(new_value) - float(old_value)) / max(float(scales.get(feature, 1.0)), 1e-12)
    except (TypeError, ValueError):
        return 1.0


def respects_relational_constraints(row: pd.Series) -> bool:
    for lower_feature, upper_feature in RELATIONAL_CONSTRAINTS:
        if lower_feature not in row.index or upper_feature not in row.index:
            continue
        lower = pd.to_numeric(pd.Series([row[lower_feature]]), errors="coerce").iloc[0]
        upper = pd.to_numeric(pd.Series([row[upper_feature]]), errors="coerce").iloc[0]
        if pd.notna(lower) and pd.notna(upper) and float(lower) > float(upper):
            return False
    return True


def candidate_scope_features(
    scope: str,
    available_features: Iterable[str],
    feature_scopes: Mapping[str, Sequence[str]],
    taxonomy: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    if scope not in feature_scopes:
        raise CounterfactualProtocolError(f"Unknown candidate feature scope: {scope}")
    allowed_control_types = set(feature_scopes[scope])
    return [
        feature
        for feature in available_features
        if taxonomy.get(feature, {}).get("control_type", "unknown") in allowed_control_types
        and taxonomy.get(feature, {}).get("control_type") != "forbidden"
    ]


def _prototype_distance(
    sample: pd.Series,
    prototype: pd.Series,
    features: Sequence[str],
    scales: Mapping[str, float],
) -> float:
    costs = [change_cost(feature, sample[feature], prototype[feature], scales) for feature in features]
    return float(np.mean(costs)) if costs else math.inf


def _build_candidate_pool(
    sample: pd.Series,
    prototypes: pd.DataFrame,
    allowed_features: Sequence[str],
    scales: Mapping[str, float],
    *,
    maximum_features_changed: int,
    maximum_prototypes: int,
) -> tuple[
    pd.DataFrame,
    list[list[Dict[str, Any]]],
    list[Dict[str, int]],
    Dict[str, int],
]:
    if prototypes.empty or not allowed_features:
        return (
            pd.DataFrame(columns=sample.index),
            [],
            [],
            {"prototypes_considered": 0, "domain_rejections": 0},
        )
    ordered = prototypes.copy()
    ordered["__distance"] = [
        _prototype_distance(sample, row, allowed_features, scales)
        for _, row in ordered.iterrows()
    ]
    ordered = (
        ordered.sort_values(["__distance"], kind="mergesort")
        .head(maximum_prototypes)
        .drop(columns="__distance")
    )
    candidate_rows: list[pd.Series] = []
    change_sets: list[list[Dict[str, Any]]] = []
    budget_markers: list[Dict[str, int]] = []
    domain_rejections = 0
    seen: set[tuple[str, ...]] = set()
    for prototype_rank, (_, prototype) in enumerate(ordered.iterrows(), start=1):
        differences = []
        for feature in allowed_features:
            cost = change_cost(feature, sample[feature], prototype[feature], scales)
            if cost > 0:
                differences.append((feature, cost, sample[feature], prototype[feature]))
        differences.sort(key=lambda item: (item[1], item[0]))
        for count in range(1, min(maximum_features_changed, len(differences)) + 1):
            modified = sample.copy()
            changes: list[Dict[str, Any]] = []
            for feature, cost, old_value, new_value in differences[:count]:
                modified[feature] = new_value
                changes.append(
                    {
                        "feature": feature,
                        "old_value": old_value.item() if isinstance(old_value, np.generic) else old_value,
                        "new_value": new_value.item() if isinstance(new_value, np.generic) else new_value,
                        "normalized_cost": float(cost),
                    }
                )
            if not respects_relational_constraints(modified):
                domain_rejections += 1
                continue
            key = tuple(str(modified[feature]) for feature in sample.index)
            if key in seen:
                continue
            seen.add(key)
            candidate_rows.append(modified)
            change_sets.append(changes)
            budget_markers.append(
                {
                    "prototype_rank": prototype_rank,
                    "n_changed_features": count,
                }
            )
    frame = pd.DataFrame(candidate_rows, columns=sample.index) if candidate_rows else pd.DataFrame(columns=sample.index)
    return frame, change_sets, budget_markers, {
        "prototypes_considered": int(len(ordered)),
        "domain_rejections": domain_rejections,
    }


def build_candidates(
    sample: pd.Series,
    prototypes: pd.DataFrame,
    allowed_features: Sequence[str],
    scales: Mapping[str, float],
    *,
    max_features_changed: int,
    max_prototypes: int,
) -> tuple[pd.DataFrame, list[list[Dict[str, Any]]], Dict[str, int]]:
    """Return one bounded candidate pool for direct protocol-level tests."""

    frame, change_sets, _, diagnostics = _build_candidate_pool(
        sample,
        prototypes,
        allowed_features,
        scales,
        maximum_features_changed=max_features_changed,
        maximum_prototypes=max_prototypes,
    )
    return frame, change_sets, diagnostics


def find_search_scenarios(
    pipeline: Any,
    sample: pd.Series,
    prototypes: pd.DataFrame,
    allowed_features: Sequence[str],
    scales: Mapping[str, float],
    labels: Sequence[int],
    desired_class: int,
    original_probability: np.ndarray,
    taxonomy: Mapping[str, Mapping[str, Any]],
    *,
    search_budgets: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if not search_budgets:
        raise CounterfactualProtocolError("At least one explicit search budget is required.")
    maximum_features = max(int(row["max_features_changed"]) for row in search_budgets)
    maximum_prototypes = max(int(row["max_prototypes"]) for row in search_budgets)
    candidates, change_sets, markers, diagnostics = _build_candidate_pool(
        sample,
        prototypes,
        allowed_features,
        scales,
        maximum_features_changed=maximum_features,
        maximum_prototypes=maximum_prototypes,
    )
    results: Dict[str, Dict[str, Any]] = {}
    if candidates.empty:
        reason = "no_training_fold_desired_class_prototypes" if prototypes.empty else "no_domain_valid_candidate_changes"
        for budget in search_budgets:
            results[str(budget["budget_id"])] = {
                "search_success": False,
                "search_failure_reason": reason,
                "candidate_pool_size": 0,
                "candidates_within_budget": 0,
                "actual_probability_evaluations": 0,
                **diagnostics,
            }
        return results
    probabilities = align_proba(
        pipeline.predict_proba(candidates),
        pipeline.named_steps["model"].classes_,
        list(labels),
    )
    predicted = predict_labels_from_proba(probabilities, list(labels))
    desired_index = list(labels).index(desired_class)
    for budget in search_budgets:
        budget_id = str(budget["budget_id"])
        positions = [
            position
            for position, marker in enumerate(markers)
            if marker["prototype_rank"] <= int(budget["max_prototypes"])
            and marker["n_changed_features"] <= int(budget["max_features_changed"])
        ]
        successful_positions = [
            position for position in positions if int(predicted[position]) >= desired_class
        ]
        common = {
            "candidate_pool_size": int(len(candidates)),
            "candidates_within_budget": int(len(positions)),
            "actual_probability_evaluations": int(len(candidates)),
            **diagnostics,
        }
        if not successful_positions:
            results[budget_id] = {
                "search_success": False,
                "search_failure_reason": "no_candidate_reached_desired_or_higher_class",
                **common,
            }
            continue
        ranked: list[Dict[str, Any]] = []
        for position in successful_positions:
            changes = change_sets[position]
            total_cost = float(
                sum(float(change["normalized_cost"]) for change in changes)
                + 0.15 * len(changes)
            )
            desired_probability = float(probabilities[position, desired_index])
            ranked.append(
                {
                    "search_success": True,
                    "search_failure_reason": "",
                    "scenario_class": int(predicted[position]),
                    "desired_class_probability": desired_probability,
                    "probability_gain": desired_probability
                    - float(original_probability[desired_index]),
                    "normalized_search_cost": total_cost,
                    "n_changed_features": len(changes),
                    "changes": changes,
                    "changed_features": [change["feature"] for change in changes],
                    "changed_control_types": [
                        taxonomy.get(change["feature"], {}).get(
                            "control_type", "unknown"
                        )
                        for change in changes
                    ],
                    **common,
                }
            )
        results[budget_id] = min(
            ranked,
            key=lambda item: (
                item["normalized_search_cost"],
                -item["desired_class_probability"],
                item["n_changed_features"],
            ),
        )
    return results


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan, 0
    generator = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        means[index] = generator.choice(array, size=len(array), replace=True).mean()
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)), n_resamples


def summarize_search_success(
    cases: pd.DataFrame,
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[Dict[str, Any]] = []
    uncertainty_rows: list[Dict[str, Any]] = []
    grouped = cases.groupby(["candidate_feature_scope", "budget_id"], sort=False)
    for group_index, ((scope, budget_id), group) in enumerate(grouped):
        eligible = group[group["eligible_for_upward_shift"].astype(bool)]
        successful = eligible[eligible["search_success"].astype(bool)]
        denominator = int(len(eligible))
        successes = int(len(successful))
        low, high = wilson_interval(successes, denominator, confidence)
        identity = {field: group[field].iloc[0] for field in SUMMARY_IDENTITY_FIELDS}
        summary_rows.append(
            {
                **identity,
                "candidate_feature_scope": scope,
                "budget_id": budget_id,
                "budget_role": group["budget_role"].iloc[0],
                "max_prototypes": int(group["max_prototypes"].iloc[0]),
                "max_features_changed": int(group["max_features_changed"].iloc[0]),
                "n_total_oof_cases": int(len(group)),
                "n_eligible_oof_cases": denominator,
                "n_search_successes": successes,
                "heuristic_search_success_rate": (
                    successes / denominator if denominator else math.nan
                ),
                "search_success_ci_low": low,
                "search_success_ci_high": high,
                "search_success_interval_method": "wilson_95_ci",
                "candidate_count_within_budget": int(
                    pd.to_numeric(
                        eligible["candidates_within_budget"], errors="raise"
                    ).sum()
                ),
                "mean_probability_gain_successful": float(
                    pd.to_numeric(
                        successful["probability_gain"], errors="coerce"
                    ).mean()
                )
                if successes
                else math.nan,
                "mean_normalized_search_cost_successful": float(
                    pd.to_numeric(
                        successful["normalized_search_cost"], errors="coerce"
                    ).mean()
                )
                if successes
                else math.nan,
                "mean_changed_features_successful": float(
                    pd.to_numeric(
                        successful["n_changed_features"], errors="coerce"
                    ).mean()
                )
                if successes
                else math.nan,
            }
        )
        uncertainty_rows.append(
            {
                **identity,
                "candidate_feature_scope": scope,
                "budget_id": budget_id,
                "metric": "heuristic_search_success_rate",
                "n_observations": denominator,
                "estimate": successes / denominator if denominator else math.nan,
                "ci_low": low,
                "ci_high": high,
                "method": "wilson_95_ci",
                "n_bootstrap_draws": 0,
            }
        )
        for metric_index, metric in enumerate(
            (
                "probability_gain",
                "normalized_search_cost",
                "n_changed_features",
            )
        ):
            values = (
                pd.to_numeric(successful[metric], errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )
            boot_low, boot_high, completed_draws = _bootstrap_mean_interval(
                values,
                n_resamples=n_resamples,
                confidence=confidence,
                seed=seed + group_index * 10 + metric_index,
            )
            uncertainty_rows.append(
                {
                    **identity,
                    "candidate_feature_scope": scope,
                    "budget_id": budget_id,
                    "metric": f"mean_{metric}_successful",
                    "n_observations": len(values),
                    "estimate": float(values.mean()) if len(values) else math.nan,
                    "ci_low": boot_low,
                    "ci_high": boot_high,
                    "method": "percentile_bootstrap_95_ci",
                    "n_bootstrap_draws": completed_draws,
                }
            )
    failures = (
        cases[
            cases["eligible_for_upward_shift"].astype(bool)
            & ~cases["search_success"].astype(bool)
        ]
        .groupby(
            ["candidate_feature_scope", "budget_id", "search_failure_reason"],
            dropna=False,
        )
        .size()
        .reset_index(name="n_failures")
    )
    if not failures.empty:
        totals = failures.groupby(["candidate_feature_scope", "budget_id"])[
            "n_failures"
        ].transform("sum")
        failures["failure_share_within_scope_budget"] = (
            failures["n_failures"] / totals
        )
        for field in reversed(SUMMARY_IDENTITY_FIELDS):
            failures.insert(0, field, cases[field].iloc[0])
    return pd.DataFrame(summary_rows), pd.DataFrame(uncertainty_rows), failures


def representative_examples(
    cases: pd.DataFrame,
    *,
    primary_budget_id: str,
    limit_per_scope: int = 3,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    primary = cases[cases["budget_id"] == primary_budget_id]
    for _, group in primary.groupby("candidate_feature_scope", sort=False):
        successful = (
            group[group["search_success"].astype(bool)]
            .sort_values(["normalized_search_cost", "sample_index"])
            .head(limit_per_scope)
        )
        failed = (
            group[
                group["eligible_for_upward_shift"].astype(bool)
                & ~group["search_success"].astype(bool)
            ]
            .sort_values("sample_index")
            .head(1)
        )
        selected = pd.concat([successful, failed])
        if selected.empty:
            selected = group.sort_values("sample_index").head(1)
        for row in selected.to_dict(orient="records"):
            row["qualitative_example_only"] = True
            row["warning"] = (
                "Heuristic model-input scenario only; not causal recourse or employee advice."
            )
            rows.append(row)
    return rows


def write_interpretation(
    path: Path,
    summary: pd.DataFrame,
    *,
    run_id: str,
    config_hash: str,
) -> None:
    lines = [
        "# Supplementary OOF Heuristic Counterfactual-Search Interpretation",
        "",
        f"Run ID: `{run_id}`  ",
        f"Config hash: `{config_hash}`",
        "",
        "All search-success estimates use fold-specific models. Each evaluated case is excluded from model fitting, prototype selection, scale estimation, and domain construction. Desired-class prototypes come only from that case's outer training fold.",
        "",
        "## Primary-Budget Results by Candidate Feature Scope",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- `{row.candidate_feature_scope}`: {row.n_search_successes}/{row.n_eligible_oof_cases} search successes "
            f"({row.heuristic_search_success_rate:.4f}; Wilson 95% CI {row.search_success_ci_low:.4f}-{row.search_success_ci_high:.4f}); "
            f"total OOF cases {row.n_total_oof_cases}."
        )
    lines.extend(
        [
            "",
            "## Claim Boundaries",
            "",
            "Search success means only that the bounded heuristic found a model-input scenario whose fold-specific prediction reached the desired or a higher class.",
            "These scenarios are not causal recourse, employee advice, intervention evidence, or autonomous HR recommendations.",
            "Candidate feature scopes are reported independently because prototype ordering and candidate inclusion are not guaranteed across scopes.",
            "Restricted, primary and expanded budgets are nested only within each feature scope through one shared maximum candidate pool.",
            "Observed training-fold values and tenure constraints limit the search domain but do not establish feasibility outside the evaluated model input space.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    run_id: str,
    config_hash: str | None = None,
    scientific_input_hash: str,
    source_tree_hash: str,
    max_cases: int | None = None,
) -> Dict[str, Path]:
    raw_config = load_config(config_path)
    settings = raw_config.get("manuscript_final", raw_config)
    validate_counterfactual_search_contract(settings)
    observed_config_hash = canonical_config_hash(raw_config)
    config_hash = config_hash or observed_config_hash
    if config_hash != observed_config_hash:
        raise CounterfactualProtocolError(
            "Supplied config_hash does not match the canonical configuration."
        )
    for field, value in (
        ("scientific_input_hash", scientific_input_hash),
        ("source_tree_hash", source_tree_hash),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise CounterfactualProtocolError(
                f"{field} must be a lowercase SHA-256 digest."
            )
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise CounterfactualProtocolError(
            "Heuristic-search output directory must be absent or empty."
        )
    output = ensure_dir(output)
    primary_policy = str(settings["feature_policies"]["primary_policy"])
    definition = settings["feature_policies"]["definitions"][primary_policy]
    target = settings.get("target", {})
    target_column = str(target.get("column", "PerformanceRating"))
    labels = [int(value) for value in target.get("labels", [2, 3, 4])]
    identifier_fields = settings.get("governance_fields", {}).get("identifier_fields", ["EmpNumber"])
    id_column = str(identifier_fields[0] if identifier_fields else "EmpNumber")
    canonical = load_canonical_dataset(config_path, "inx_primary")
    data = canonical.frame
    dataset_sha256 = str(canonical.receipt.get("actual_sha256", ""))
    if len(dataset_sha256) != 64:
        raise CounterfactualProtocolError(
            "Canonical dataset receipt has no byte SHA-256 identity."
        )
    X, excluded = exact_policy_frame(data, primary_policy, definition, target_column=target_column, id_column=id_column)
    y = data[target_column].astype(int)

    protocol = settings.get("counterfactuals", {})
    if protocol.get("model_evaluation_scope") != "out_of_fold_only":
        raise CounterfactualProtocolError("Canonical counterfactual evaluation must be out-of-fold only.")
    if protocol.get("prototype_scope") != "outer_training_partition_only":
        raise CounterfactualProtocolError("Counterfactual prototypes must be restricted to each outer training fold.")
    configured_population = str(protocol["evaluation_population"])
    if max_cases is None and configured_population != "all_eligible_oof_cases":
        raise CounterfactualProtocolError("A sampled run requires an explicit predeclared sampling contract.")
    feature_scopes = protocol["candidate_feature_scopes"]
    search_budgets = protocol["search_budgets"]
    primary_budget_id = str(protocol["primary_budget_id"])
    cv = settings.get("evaluation", {}).get("cv", {})
    n_splits = int(cv.get("n_splits", 10))
    if n_splits != 10:
        raise CounterfactualProtocolError(
            "Supplementary heuristic search requires exactly ten outer folds."
        )
    seed = resolve_seed(settings, "counterfactual")
    parameters = _model_parameters(settings)
    taxonomy = taxonomy_by_feature()
    resolved_scope_features = {
        scope: candidate_scope_features(scope, X.columns, feature_scopes, taxonomy)
        for scope in feature_scopes
    }
    if any(not values for values in resolved_scope_features.values()):
        raise CounterfactualProtocolError(
            "Every candidate feature scope must resolve to at least one primary feature."
        )
    resolved_sets = [tuple(values) for values in resolved_scope_features.values()]
    if len(resolved_sets) != len(set(resolved_sets)):
        raise CounterfactualProtocolError(
            "Redundant candidate feature scopes are prohibited."
        )
    feature_policy_sha256 = _canonical_sha256(
        {
            "policy": primary_policy,
            "excluded_features": list(excluded),
            "feature_order": list(X.columns),
        }
    )
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=bool(cv.get("shuffle", True)),
        random_state=resolve_seed(settings, cv.get("seed", "cv")),
    )
    splits = list(splitter.split(X, y))
    fold_assignment = [0] * len(X)
    for fold, (_, test_positions) in enumerate(splits, start=1):
        for position in test_positions:
            if fold_assignment[int(position)] != 0:
                raise CounterfactualProtocolError("Outer-fold assignment is duplicated.")
            fold_assignment[int(position)] = fold
    if set(fold_assignment) != set(range(1, n_splits + 1)):
        raise CounterfactualProtocolError("Outer-fold assignment is incomplete.")
    fold_contract_sha256 = _canonical_sha256(
        {
            "dataset_sha256": dataset_sha256,
            "n_splits": n_splits,
            "shuffle": bool(cv.get("shuffle", True)),
            "random_state": resolve_seed(settings, cv.get("seed", "cv")),
            "assignments": [
                {"sample_index": int(index), "outer_fold": fold_assignment[position]}
                for position, index in enumerate(X.index)
            ],
        }
    )

    identity = {
        "run_id": run_id,
        "config_hash": config_hash,
        "scientific_input_hash": scientific_input_hash,
        "source_tree_hash": source_tree_hash,
        "dataset_sha256": dataset_sha256,
        "fold_contract_sha256": fold_contract_sha256,
        "feature_policy_sha256": feature_policy_sha256,
        "policy": primary_policy,
        "dataset_key": "inx_primary",
        "task_type": "ordinal_multiclass_performance",
        "evidence_role": "supplementary_heuristic_search_only",
    }
    budget_contract_sha256 = _canonical_sha256(search_budgets)
    scope_contract_sha256 = _canonical_sha256(
        {
            "declared": feature_scopes,
            "resolved_features": resolved_scope_features,
        }
    )
    rows: list[Dict[str, Any]] = []
    oof_rows: list[Dict[str, Any]] = []
    fold_receipts: list[Dict[str, Any]] = []
    eligible_processed = 0
    candidate_pool_evaluations = 0
    started = time.perf_counter()
    for fold, (train_positions, test_positions) in enumerate(splits, start=1):
        X_train = X.iloc[train_positions]
        y_train = y.iloc[train_positions]
        X_test = X.iloc[test_positions]
        y_test = y.iloc[test_positions]
        model_seed = int(resolve_seed(settings, "model") + fold)
        fit_receipt = {
            **identity,
            "outer_fold": fold,
            "model_name": "xgboost",
            "model_seed": model_seed,
            "n_train": int(len(train_positions)),
            "n_test": int(len(test_positions)),
            "train_sample_index_sha256": _index_sha256(X.index[train_positions]),
            "test_sample_index_sha256": _index_sha256(X.index[test_positions]),
            "feature_order_sha256": _canonical_sha256(list(X.columns)),
            "parameters_sha256": _canonical_sha256(parameters),
            "outer_test_used_for_fit_or_selection": False,
        }
        model_fit_contract_sha256 = _canonical_sha256(fit_receipt)
        pipeline = _fit_supplementary_pipeline(
            X_train,
            y_train,
            parameters,
            model_seed,
            forbidden_features=excluded,
        )
        model_sha256 = _serialized_model_sha256(pipeline)
        model_fit_receipt_sha256 = _canonical_sha256(
            {
                **fit_receipt,
                "model_fit_contract_sha256": model_fit_contract_sha256,
                "model_sha256": model_sha256,
            }
        )
        probability = align_proba(pipeline.predict_proba(X_test), pipeline.named_steps["model"].classes_, labels)
        predicted = predict_labels_from_proba(probability, labels)
        fold_probability_sha256 = _canonical_sha256(
            [
                {
                    "sample_index": int(sample_index),
                    "predicted_class": int(predicted[position]),
                    "probabilities": [float(value) for value in probability[position]],
                }
                for position, sample_index in enumerate(X_test.index)
            ]
        )
        fold_receipts.append(
            {
                **fit_receipt,
                "model_fit_contract_sha256": model_fit_contract_sha256,
                "model_fit_receipt_sha256": model_fit_receipt_sha256,
                "model_sha256": model_sha256,
                "outer_test_probability_sha256": fold_probability_sha256,
            }
        )
        scales = training_scales(X_train)
        for position, sample_index in enumerate(X_test.index):
            predicted_class = int(predicted[position])
            eligible = predicted_class < max(labels)
            if eligible and max_cases is not None and eligible_processed >= max_cases:
                continue
            if eligible:
                eligible_processed += 1
            desired_class = min(predicted_class + 1, max(labels))
            prototypes = X_train.loc[y_train[y_train == desired_class].index]
            oof_rows.append(
                {
                    **identity,
                    "sample_index": int(sample_index),
                    "outer_fold": fold,
                    "source_model_fit_receipt_sha256": model_fit_receipt_sha256,
                    "source_outer_model_sha256": model_sha256,
                    "true_class": int(y_test.loc[sample_index]),
                    "predicted_class": predicted_class,
                    **{
                        f"probability_{label}": float(probability[position, label_index])
                        for label_index, label in enumerate(labels)
                    },
                }
            )
            for scope, allowed in resolved_scope_features.items():
                if not eligible:
                    results = {
                        str(budget["budget_id"]): {
                            "search_success": False,
                            "search_failure_reason": (
                                "ineligible_already_predicted_highest_class"
                            ),
                            "candidate_pool_size": 0,
                            "candidates_within_budget": 0,
                            "actual_probability_evaluations": 0,
                            "prototypes_considered": 0,
                            "domain_rejections": 0,
                        }
                        for budget in search_budgets
                    }
                elif not allowed:
                    results = {
                        str(budget["budget_id"]): {
                            "search_success": False,
                            "search_failure_reason": "no_allowed_features",
                            "candidate_pool_size": 0,
                            "candidates_within_budget": 0,
                            "actual_probability_evaluations": 0,
                            "prototypes_considered": 0,
                            "domain_rejections": 0,
                        }
                        for budget in search_budgets
                    }
                else:
                    results = find_search_scenarios(
                        pipeline,
                        X.loc[sample_index],
                        prototypes,
                        allowed,
                        scales,
                        labels,
                        desired_class,
                        probability[position],
                        taxonomy,
                        search_budgets=search_budgets,
                    )
                candidate_pool_evaluations += int(
                    next(iter(results.values()))["actual_probability_evaluations"]
                )
                for budget in search_budgets:
                    budget_id = str(budget["budget_id"])
                    result = results[budget_id]
                    rows.append(
                        {
                            **identity,
                            "search_budget_contract_sha256": budget_contract_sha256,
                            "candidate_scope_contract_sha256": scope_contract_sha256,
                            "source_model_fit_receipt_sha256": model_fit_receipt_sha256,
                            "source_outer_model_sha256": model_sha256,
                            "sample_index": int(sample_index),
                            "outer_fold": fold,
                            "prototype_scope": "outer_training_partition_only",
                            "domain_scope": "outer_training_partition_only",
                            "scaling_scope": "outer_training_partition_only",
                            "case_excluded_from_training": True,
                            "true_class": int(y_test.loc[sample_index]),
                            "predicted_class": predicted_class,
                            "desired_class": desired_class,
                            "correct": bool(y_test.loc[sample_index] == predicted_class),
                            "confidence": float(np.max(probability[position])),
                            "eligible_for_upward_shift": eligible,
                            "candidate_feature_scope": scope,
                            "allowed_feature_sha256": _canonical_sha256(allowed),
                            "n_allowed_features": len(allowed),
                            "budget_id": budget_id,
                            "budget_role": str(budget["role"]),
                            "max_prototypes": int(budget["max_prototypes"]),
                            "max_features_changed": int(
                                budget["max_features_changed"]
                            ),
                            "search_success": bool(
                                result.get("search_success", False)
                            ),
                            "scenario_class": result.get("scenario_class", math.nan),
                            "probability_gain": result.get(
                                "probability_gain", math.nan
                            ),
                            "desired_class_probability": result.get(
                                "desired_class_probability", math.nan
                            ),
                            "normalized_search_cost": result.get(
                                "normalized_search_cost", math.nan
                            ),
                            "n_changed_features": result.get(
                                "n_changed_features", 0
                            ),
                            "changed_features": ";".join(
                                result.get("changed_features", [])
                            ),
                            "changed_control_types": ";".join(
                                result.get("changed_control_types", [])
                            ),
                            "changes_json": json.dumps(
                                result.get("changes", []),
                                sort_keys=True,
                                default=str,
                            ),
                            "search_failure_reason": result.get(
                                "search_failure_reason", ""
                            ),
                            "prototypes_considered": result.get(
                                "prototypes_considered", 0
                            ),
                            "candidate_pool_size": result.get(
                                "candidate_pool_size", 0
                            ),
                            "candidates_within_budget": result.get(
                                "candidates_within_budget", 0
                            ),
                            "actual_probability_evaluations": result.get(
                                "actual_probability_evaluations", 0
                            ),
                            "domain_rejections": result.get(
                                "domain_rejections", 0
                            ),
                        }
                    )
        if max_cases is not None and eligible_processed >= max_cases:
            break

    cases = pd.DataFrame(rows)
    if cases.empty:
        raise CounterfactualProtocolError("Heuristic-search protocol produced no case rows.")
    oof_predictions = pd.DataFrame(oof_rows)
    fold_model_receipts = pd.DataFrame(fold_receipts)
    model_set_sha256 = _canonical_sha256(
        [
            {
                "outer_fold": int(row["outer_fold"]),
                "model_sha256": str(row["model_sha256"]),
            }
            for row in fold_receipts
        ]
    )
    cases["model_set_sha256"] = model_set_sha256
    oof_predictions["model_set_sha256"] = model_set_sha256
    fold_model_receipts["model_set_sha256"] = model_set_sha256
    if max_cases is None:
        expected_rows = len(X) * len(feature_scopes) * len(search_budgets)
        if len(cases) != expected_rows:
            raise CounterfactualProtocolError(
                "Complete eligible-case coverage produced the wrong case/scope/budget count."
            )
        if len(oof_predictions) != len(X) or oof_predictions["sample_index"].duplicated().any():
            raise CounterfactualProtocolError(
                "OOF prediction coverage must contain each sample exactly once."
            )
        if cases.duplicated(
            ["sample_index", "candidate_feature_scope", "budget_id"]
        ).any():
            raise CounterfactualProtocolError(
                "A case/scope/budget combination is duplicated."
            )
        counts = cases.groupby(["candidate_feature_scope", "budget_id"]).size()
        if set(counts.astype(int)) != {len(X)}:
            raise CounterfactualProtocolError(
                "Every scope and budget must cover all OOF cases."
            )
    if max_cases is None and len(fold_model_receipts) != n_splits:
        raise CounterfactualProtocolError("Exactly ten fold-model receipts are required.")
    if max_cases is not None and not 1 <= len(fold_model_receipts) <= n_splits:
        raise CounterfactualProtocolError(
            "A bounded diagnostic must retain every fold receipt it actually evaluated."
        )
    if not fold_model_receipts["model_sha256"].astype(str).str.fullmatch(
        r"[0-9a-f]{64}"
    ).all():
        raise CounterfactualProtocolError(
            "Every evaluated fold must expose its actual serialized model SHA-256."
        )
    model_hash_by_fold = fold_model_receipts.set_index("outer_fold")[
        "model_sha256"
    ].astype(str)
    for table_name, table in (("OOF", oof_predictions), ("case", cases)):
        expected_model_hash = table["outer_fold"].map(model_hash_by_fold)
        if expected_model_hash.isna().any() or not expected_model_hash.equals(
            table["source_outer_model_sha256"].astype(str)
        ):
            raise CounterfactualProtocolError(
                f"{table_name} rows are not bound to their exact evaluated fold model."
            )
    budget_limits = {
        str(row["budget_id"]): int(row["max_prototypes"])
        * int(row["max_features_changed"])
        for row in search_budgets
    }
    for budget_id, group in cases.groupby("budget_id"):
        if (group["candidates_within_budget"] > budget_limits[str(budget_id)]).any():
            raise CounterfactualProtocolError(
                f"Candidate count exceeds declared budget {budget_id!r}."
            )
        if (group["n_changed_features"] > group["max_features_changed"]).any():
            raise CounterfactualProtocolError(
                f"Changed-feature count exceeds declared budget {budget_id!r}."
            )
    budget_order = [str(row["budget_id"]) for row in search_budgets]
    candidate_pivot = cases.pivot(
        index=["sample_index", "candidate_feature_scope"],
        columns="budget_id",
        values="candidates_within_budget",
    )[budget_order]
    if (candidate_pivot.diff(axis=1).iloc[:, 1:] < 0).any().any():
        raise CounterfactualProtocolError(
            "Within-scope candidate counts are not nested across search budgets."
        )
    success_pivot = cases.pivot(
        index=["sample_index", "candidate_feature_scope"],
        columns="budget_id",
        values="search_success",
    )[budget_order].astype(int)
    if (success_pivot.diff(axis=1).iloc[:, 1:] < 0).any().any():
        raise CounterfactualProtocolError(
            "Within-scope search success is inconsistent with nested budgets."
        )
    source_oof_probability_sha256 = _canonical_sha256(
        [
            {
                "sample_index": int(row["sample_index"]),
                "outer_fold": int(row["outer_fold"]),
                "source_model_fit_receipt_sha256": row[
                    "source_model_fit_receipt_sha256"
                ],
                "source_outer_model_sha256": row["source_outer_model_sha256"],
                "true_class": int(row["true_class"]),
                "predicted_class": int(row["predicted_class"]),
                **{
                    f"probability_{label}": float(row[f"probability_{label}"])
                    for label in labels
                },
            }
            for row in oof_rows
        ]
    )
    cases["source_oof_probability_sha256"] = source_oof_probability_sha256
    oof_predictions["source_oof_probability_sha256"] = (
        source_oof_probability_sha256
    )
    fold_model_receipts["source_oof_probability_sha256"] = (
        source_oof_probability_sha256
    )
    uncertainty = protocol["uncertainty"]
    confidence = float(uncertainty["confidence_level"])
    n_resamples = int(
        settings.get("evaluation", {}).get("bootstrap", {}).get("n_resamples", 5000)
    )
    if n_resamples != 5000:
        raise CounterfactualProtocolError(
            "Supplementary heuristic search requires exactly 5,000 bootstrap draws."
        )
    all_summary, uncertainty_table, failures = summarize_search_success(
        cases,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    primary_summary = all_summary[
        all_summary["budget_id"] == primary_budget_id
    ].reset_index(drop=True)
    if len(primary_summary) != len(feature_scopes):
        raise CounterfactualProtocolError(
            "Primary-budget summary must contain exactly one row per feature scope."
        )
    examples = representative_examples(
        cases,
        primary_budget_id=primary_budget_id,
    )
    feature_scope_rows = [
        {
            **identity,
            "source_oof_probability_sha256": source_oof_probability_sha256,
            "model_set_sha256": model_set_sha256,
            "candidate_scope_contract_sha256": scope_contract_sha256,
            "candidate_feature_scope": scope,
            "feature_order": order,
            "feature": feature,
            "control_type": taxonomy.get(feature, {}).get("control_type", "unknown"),
            "diagnostic_includes_immutable_history": (
                scope == "diagnostic_including_immutable_history"
            ),
        }
        for scope, features in resolved_scope_features.items()
        for order, feature in enumerate(features, start=1)
    ]
    feature_scope_table = pd.DataFrame(feature_scope_rows)
    if failures.empty:
        failures = pd.DataFrame(
            columns=[
                *SUMMARY_IDENTITY_FIELDS,
                "candidate_feature_scope",
                "budget_id",
                "search_failure_reason",
                "n_failures",
                "failure_share_within_scope_budget",
            ]
        )
    elapsed = time.perf_counter() - started
    population_status = "diagnostic_cost_benchmark" if max_cases is not None else "all_eligible_oof_cases"
    protocol_payload = {
        **identity,
        "source_oof_probability_sha256": source_oof_probability_sha256,
        "model_set_sha256": model_set_sha256,
        "search_budget_contract_sha256": budget_contract_sha256,
        "candidate_scope_contract_sha256": scope_contract_sha256,
        "terminology": protocol["terminology"],
        "excluded_features": list(excluded),
        "model_evaluation_scope": "out_of_fold_only",
        "prototype_scope": "outer_training_partition_only",
        "domain_scope": "outer_training_partition_only",
        "scaling_scope": "outer_training_partition_only",
        "case_excluded_from_training": True,
        "evaluation_population": population_status,
        "max_cases": max_cases,
        "n_splits": n_splits,
        "candidate_feature_scopes": feature_scopes,
        "resolved_scope_features": resolved_scope_features,
        "cross_scope_comparison": protocol["cross_scope_comparison"],
        "search_budgets": search_budgets,
        "primary_budget_id": primary_budget_id,
        "budget_candidate_inclusion": protocol["budget_candidate_inclusion"],
        "maximum_candidate_pool_per_scope_case": protocol[
            "maximum_candidate_pool_per_scope_case"
        ],
        "cost_scaling": "training-fold IQR, range fallback; categorical unit cost; sparsity penalty 0.15",
        "domain_constraints": "observed training-fold prototype values plus relational tenure constraints",
        "search_success_rule": "fold-specific predicted class reaches desired or higher class",
        "n_total_case_scope_budget_rows": len(cases),
        "n_unique_cases": int(cases["sample_index"].nunique()),
        "n_unique_eligible_cases": int(cases[cases["eligible_for_upward_shift"]]["sample_index"].nunique()),
        "candidate_pool_probability_evaluations": candidate_pool_evaluations,
        "n_fold_model_receipts": int(len(fold_model_receipts)),
        "n_bootstrap_draws": n_resamples,
        "elapsed_seconds": elapsed,
        "seed": seed,
        "warning": protocol["warning"],
    }
    paths = {
        "protocol": output / "heuristic_search_protocol.json",
        "oof_predictions": output / "heuristic_search_oof_predictions.csv",
        "fold_model_receipts": output / "heuristic_search_fold_model_receipts.csv",
        "feature_scopes": output / "heuristic_search_feature_scopes.csv",
        "by_case": output / "heuristic_search_by_case.csv",
        "summary": output / "heuristic_search_summary.csv",
        "budget_sensitivity": output / "heuristic_search_budget_sensitivity.csv",
        "failures": output / "heuristic_search_failure_reasons.csv",
        "uncertainty": output / "heuristic_search_uncertainty.csv",
        "examples": output / "representative_heuristic_scenarios.jsonl",
        "interpretation": output / "heuristic_search_interpretation.md",
        "inventory": output / "heuristic_search_artifact_inventory.json",
    }
    write_json(paths["protocol"], protocol_payload)
    oof_predictions.to_csv(paths["oof_predictions"], index=False)
    fold_model_receipts.to_csv(paths["fold_model_receipts"], index=False)
    feature_scope_table.to_csv(paths["feature_scopes"], index=False)
    cases.to_csv(paths["by_case"], index=False)
    primary_summary.to_csv(paths["summary"], index=False)
    all_summary.to_csv(paths["budget_sensitivity"], index=False)
    failures.to_csv(paths["failures"], index=False)
    uncertainty_table.to_csv(paths["uncertainty"], index=False)
    write_jsonl(paths["examples"], examples)
    write_interpretation(
        paths["interpretation"],
        primary_summary,
        run_id=run_id,
        config_hash=config_hash,
    )
    inventory_rows = [
        {
            "path": path.relative_to(output).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for key, path in sorted(paths.items())
        if key != "inventory"
    ]
    if any(row["size_bytes"] <= 0 for row in inventory_rows):
        raise CounterfactualProtocolError(
            "Heuristic-search publication artifact is empty."
        )
    write_json(
        paths["inventory"],
        {
            **identity,
            "source_oof_probability_sha256": source_oof_probability_sha256,
            "model_set_sha256": model_set_sha256,
            "status": "complete",
            "inventory_mode": "closed_world",
            "path_basis": "heuristic_counterfactual_relative",
            "artifact_count": len(inventory_rows),
            "artifacts": inventory_rows,
        },
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supplementary OOF heuristic counterfactual-search evaluation."
    )
    parser.add_argument("--config", default="configs/manuscript_final.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-hash", default=None)
    parser.add_argument("--scientific-input-hash", required=True)
    parser.add_argument("--source-tree-hash", required=True)
    parser.add_argument("--max-cases", type=int, default=None, help="Diagnostic cost benchmark only; final runs use all eligible cases.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            {
                key: str(value)
                for key, value in run(
                    arguments.config,
                    output_dir=arguments.output_dir,
                    run_id=arguments.run_id,
                    config_hash=arguments.config_hash,
                    scientific_input_hash=arguments.scientific_input_hash,
                    source_tree_hash=arguments.source_tree_hash,
                    max_cases=arguments.max_cases,
                ).items()
            },
            indent=2,
            sort_keys=True,
        )
    )
