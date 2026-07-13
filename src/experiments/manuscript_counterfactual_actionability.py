from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from src.core.io_utils import ensure_dir, write_json, write_jsonl
from src.data.canonical_loader import load_canonical_dataset
from src.experiments.final_evidence_common import align_proba, predict_labels_from_proba
from src.experiments.manuscript_policy_ablation import _model_parameters, exact_policy_frame, resolve_seed
from src.features.feature_sets import taxonomy_by_feature
from src.governance.manuscript_contract import canonical_config_hash
from src.models.canonical_models import CanonicalModelError, build_model_pipeline
from src.utils.config_loader import load_config


RELATIONAL_CONSTRAINTS = (
    ("ExperienceYearsInCurrentRole", "ExperienceYearsAtThisCompany"),
    ("YearsWithCurrManager", "ExperienceYearsAtThisCompany"),
    ("YearsSinceLastPromotion", "ExperienceYearsAtThisCompany"),
    ("ExperienceYearsAtThisCompany", "TotalWorkExperienceInYears"),
)


class CounterfactualProtocolError(RuntimeError):
    """Raised when actionability evidence would violate the OOF protocol."""


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


def intervention_features(
    mode: str,
    available_features: Iterable[str],
    intervention_modes: Mapping[str, Sequence[str]],
    taxonomy: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    if mode not in intervention_modes:
        raise CounterfactualProtocolError(f"Unknown intervention mode: {mode}")
    allowed_control_types = set(intervention_modes[mode])
    if mode == "diagnostic_full_default":
        allowed_control_types.add("immutable")
    features = [
        feature
        for feature in available_features
        if taxonomy.get(feature, {}).get("control_type", "unknown") in allowed_control_types
        and taxonomy.get(feature, {}).get("control_type") != "forbidden"
    ]
    if mode == "no_salary":
        features = [feature for feature in features if feature != "EmpLastSalaryHikePercent"]
    return features


def _prototype_distance(
    sample: pd.Series,
    prototype: pd.Series,
    features: Sequence[str],
    scales: Mapping[str, float],
) -> float:
    costs = [change_cost(feature, sample[feature], prototype[feature], scales) for feature in features]
    return float(np.mean(costs)) if costs else math.inf


def build_candidates(
    sample: pd.Series,
    prototypes: pd.DataFrame,
    allowed_features: Sequence[str],
    scales: Mapping[str, float],
    *,
    max_features_changed: int,
    max_prototypes: int,
) -> tuple[pd.DataFrame, list[list[Dict[str, Any]]], Dict[str, int]]:
    if prototypes.empty or not allowed_features:
        return pd.DataFrame(columns=sample.index), [], {"prototypes_considered": 0, "domain_rejections": 0}
    ordered = prototypes.copy()
    ordered["__distance"] = [
        _prototype_distance(sample, row, allowed_features, scales)
        for _, row in ordered.iterrows()
    ]
    ordered = ordered.sort_values(["__distance"], kind="mergesort").head(max_prototypes).drop(columns="__distance")
    candidate_rows: list[pd.Series] = []
    change_sets: list[list[Dict[str, Any]]] = []
    domain_rejections = 0
    seen: set[tuple[str, ...]] = set()
    for _, prototype in ordered.iterrows():
        differences = []
        for feature in allowed_features:
            cost = change_cost(feature, sample[feature], prototype[feature], scales)
            if cost > 0:
                differences.append((feature, cost, sample[feature], prototype[feature]))
        differences.sort(key=lambda item: (item[1], item[0]))
        for count in range(1, min(max_features_changed, len(differences)) + 1):
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
    frame = pd.DataFrame(candidate_rows, columns=sample.index) if candidate_rows else pd.DataFrame(columns=sample.index)
    return frame, change_sets, {
        "prototypes_considered": int(len(ordered)),
        "domain_rejections": domain_rejections,
    }


def find_counterfactual(
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
    max_features_changed: int,
    max_prototypes: int,
) -> Dict[str, Any]:
    candidates, change_sets, diagnostics = build_candidates(
        sample,
        prototypes,
        allowed_features,
        scales,
        max_features_changed=max_features_changed,
        max_prototypes=max_prototypes,
    )
    if candidates.empty:
        reason = "no_training_fold_desired_class_prototypes" if prototypes.empty else "no_domain_valid_candidate_changes"
        return {"valid": False, "failure_reason": reason, **diagnostics}
    probabilities = align_proba(
        pipeline.predict_proba(candidates),
        pipeline.named_steps["model"].classes_,
        list(labels),
    )
    predicted = predict_labels_from_proba(probabilities, list(labels))
    valid_positions = np.where(predicted >= desired_class)[0]
    if len(valid_positions) == 0:
        return {
            "valid": False,
            "failure_reason": "no_candidate_reached_desired_or_higher_class",
            "candidates_evaluated": int(len(candidates)),
            **diagnostics,
        }
    desired_index = list(labels).index(desired_class)
    ranked: list[Dict[str, Any]] = []
    for position in valid_positions:
        changes = change_sets[int(position)]
        total_cost = float(sum(float(change["normalized_cost"]) for change in changes) + 0.15 * len(changes))
        desired_probability = float(probabilities[int(position), desired_index])
        ranked.append(
            {
                "valid": True,
                "achieved_class": int(predicted[int(position)]),
                "desired_probability": desired_probability,
                "probability_gain": desired_probability - float(original_probability[desired_index]),
                "cost": total_cost,
                "num_changed_features": len(changes),
                "changes": changes,
                "changed_features": [change["feature"] for change in changes],
                "changed_control_types": [
                    taxonomy.get(change["feature"], {}).get("control_type", "unknown")
                    for change in changes
                ],
                "candidates_evaluated": int(len(candidates)),
                **diagnostics,
            }
        )
    return min(ranked, key=lambda item: (item["cost"], -item["desired_probability"], item["num_changed_features"]))


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


def summarize_actionability(
    cases: pd.DataFrame,
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[Dict[str, Any]] = []
    uncertainty_rows: list[Dict[str, Any]] = []
    for mode, group in cases.groupby("intervention_mode", sort=False):
        eligible = group[group["eligible_for_upward_shift"].astype(bool)]
        valid = eligible[eligible["valid"].astype(bool)]
        denominator = int(len(eligible))
        successes = int(len(valid))
        low, high = wilson_interval(successes, denominator, confidence)
        summary_rows.append(
            {
                "run_id": group["run_id"].iloc[0],
                "config_hash": group["config_hash"].iloc[0],
                "policy": group["policy"].iloc[0],
                "intervention_mode": mode,
                "n_total_oof_cases": int(len(group)),
                "n_eligible_oof_cases": denominator,
                "n_valid_counterfactuals": successes,
                "validity_rate": successes / denominator if denominator else math.nan,
                "validity_ci_low": low,
                "validity_ci_high": high,
                "validity_interval_method": "wilson_95_ci",
                "mean_probability_gain_valid": float(pd.to_numeric(valid["probability_gain"], errors="coerce").mean()) if successes else math.nan,
                "mean_cost_valid": float(pd.to_numeric(valid["cost"], errors="coerce").mean()) if successes else math.nan,
                "mean_sparsity_valid": float(pd.to_numeric(valid["num_changed_features"], errors="coerce").mean()) if successes else math.nan,
            }
        )
        uncertainty_rows.append(
            {
                "run_id": group["run_id"].iloc[0],
                "config_hash": group["config_hash"].iloc[0],
                "intervention_mode": mode,
                "metric": "validity_rate",
                "n": denominator,
                "estimate": successes / denominator if denominator else math.nan,
                "ci_low": low,
                "ci_high": high,
                "method": "wilson_95_ci",
                "valid_resamples": denominator,
            }
        )
        for metric in ("probability_gain", "cost", "num_changed_features"):
            values = pd.to_numeric(valid[metric], errors="coerce").dropna().to_numpy(dtype=float)
            boot_low, boot_high, valid_resamples = _bootstrap_mean_interval(
                values,
                n_resamples=n_resamples,
                confidence=confidence,
                seed=seed,
            )
            uncertainty_rows.append(
                {
                    "run_id": group["run_id"].iloc[0],
                    "config_hash": group["config_hash"].iloc[0],
                    "intervention_mode": mode,
                    "metric": f"mean_{metric}_valid",
                    "n": len(values),
                    "estimate": float(values.mean()) if len(values) else math.nan,
                    "ci_low": boot_low,
                    "ci_high": boot_high,
                    "method": "percentile_bootstrap_95_ci",
                    "valid_resamples": valid_resamples,
                }
            )
    failures = (
        cases[cases["eligible_for_upward_shift"].astype(bool) & ~cases["valid"].astype(bool)]
        .groupby(["intervention_mode", "failure_reason"], dropna=False)
        .size()
        .reset_index(name="n_failures")
    )
    if not failures.empty:
        totals = failures.groupby("intervention_mode")["n_failures"].transform("sum")
        failures["failure_share_within_mode"] = failures["n_failures"] / totals
        failures.insert(0, "config_hash", cases["config_hash"].iloc[0])
        failures.insert(0, "run_id", cases["run_id"].iloc[0])
    return pd.DataFrame(summary_rows), pd.DataFrame(uncertainty_rows), failures


def representative_examples(cases: pd.DataFrame, limit_per_mode: int = 3) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for mode, group in cases.groupby("intervention_mode", sort=False):
        valid = group[group["valid"].astype(bool)].sort_values(["cost", "sample_index"]).head(limit_per_mode)
        failed = group[group["eligible_for_upward_shift"].astype(bool) & ~group["valid"].astype(bool)].sort_values("sample_index").head(1)
        for row in pd.concat([valid, failed]).to_dict(orient="records"):
            row["qualitative_example_only"] = True
            row["warning"] = "Model scenario only; not a causal finding or employee prescription."
            rows.append(row)
    return rows


def write_interpretation(path: Path, summary: pd.DataFrame, *, run_id: str, config_hash: str) -> None:
    lines = [
        "# OOF Counterfactual Actionability Interpretation",
        "",
        f"Run ID: `{run_id}`  ",
        f"Config hash: `{config_hash}`",
        "",
        "All validity estimates use fold-specific models. Each evaluated case is excluded from model fitting, prototype selection, scale estimation, and domain construction. Desired-class prototypes come only from that case's outer training fold.",
        "",
        "## Results by Intervention Mode",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- `{row.intervention_mode}`: {row.n_valid_counterfactuals}/{row.n_eligible_oof_cases} valid "
            f"({row.validity_rate:.4f}; Wilson 95% CI {row.validity_ci_low:.4f}–{row.validity_ci_high:.4f}); "
            f"total OOF cases {row.n_total_oof_cases}."
        )
    lines.extend(
        [
            "",
            "## Claim Boundaries",
            "",
            "Validity means only that a constrained model input scenario changed the fold-specific model prediction to the desired or a higher class.",
            "Counterfactuals are not causal findings, guaranteed feasible interventions, employee prescriptions, or autonomous HR recommendations.",
            "Employee, manager, and organisation modes must be interpreted separately. Diagnostic full-default results are an upper-bound diagnostic and may include immutable/history features; they are never actionable evidence.",
            "Prototype values remain within observed training-fold domains and relational tenure constraints, but observational plausibility does not establish real-world feasibility.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    run_id: str,
    config_hash: str | None = None,
    max_cases: int | None = None,
) -> Dict[str, Path]:
    raw_config = load_config(config_path)
    settings = raw_config.get("manuscript_final", raw_config)
    config_hash = config_hash or canonical_config_hash(raw_config)
    output = ensure_dir(Path(output_dir))
    primary_policy = str(settings["feature_policies"]["primary_policy"])
    definition = settings["feature_policies"]["definitions"][primary_policy]
    target = settings.get("target", {})
    target_column = str(target.get("column", "PerformanceRating"))
    labels = [int(value) for value in target.get("labels", [2, 3, 4])]
    identifier_fields = settings.get("governance_fields", {}).get("identifier_fields", ["EmpNumber"])
    id_column = str(identifier_fields[0] if identifier_fields else "EmpNumber")
    data = load_canonical_dataset(config_path, "inx_primary").frame
    X, excluded = exact_policy_frame(data, primary_policy, definition, target_column=target_column, id_column=id_column)
    y = data[target_column].astype(int)

    protocol = settings.get("counterfactuals", {})
    if protocol.get("model_evaluation_scope") != "out_of_fold_only":
        raise CounterfactualProtocolError("Canonical counterfactual evaluation must be out-of-fold only.")
    if protocol.get("prototype_scope") != "outer_training_partition_only":
        raise CounterfactualProtocolError("Counterfactual prototypes must be restricted to each outer training fold.")
    configured_population = str(protocol.get("evaluation_population", "all_eligible_oof_cases"))
    if max_cases is None and configured_population != "all_eligible_oof_cases":
        raise CounterfactualProtocolError("A sampled run requires an explicit predeclared sampling contract.")
    intervention_modes = protocol.get("intervention_modes", {})
    if not isinstance(intervention_modes, Mapping) or not intervention_modes:
        raise CounterfactualProtocolError("Counterfactual intervention modes are missing.")
    max_features_changed = int(protocol.get("max_features_changed", 3))
    max_prototypes = int(protocol.get("max_prototypes", 250))
    cv = settings.get("evaluation", {}).get("cv", {})
    n_splits = int(cv.get("n_splits", 10))
    seed = resolve_seed(settings, "counterfactual")
    parameters = _model_parameters(settings)
    taxonomy = taxonomy_by_feature()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=bool(cv.get("shuffle", True)), random_state=resolve_seed(settings, cv.get("seed", "cv")))

    rows: list[Dict[str, Any]] = []
    eligible_processed = 0
    candidate_evaluations = 0
    started = time.perf_counter()
    for fold, (train_positions, test_positions) in enumerate(splitter.split(X, y), start=1):
        X_train = X.iloc[train_positions]
        y_train = y.iloc[train_positions]
        X_test = X.iloc[test_positions]
        y_test = y.iloc[test_positions]
        pipeline = _fit_supplementary_pipeline(
            X_train,
            y_train,
            parameters,
            seed,
            forbidden_features=excluded,
        )
        probability = align_proba(pipeline.predict_proba(X_test), pipeline.named_steps["model"].classes_, labels)
        predicted = predict_labels_from_proba(probability, labels)
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
            for mode in intervention_modes:
                allowed = intervention_features(mode, X.columns, intervention_modes, taxonomy)
                if not eligible:
                    result: Dict[str, Any] = {"valid": False, "failure_reason": "already_predicted_highest_class"}
                elif not allowed:
                    result = {"valid": False, "failure_reason": "no_allowed_features"}
                else:
                    result = find_counterfactual(
                        pipeline,
                        X.loc[sample_index],
                        prototypes,
                        allowed,
                        scales,
                        labels,
                        desired_class,
                        probability[position],
                        taxonomy,
                        max_features_changed=max_features_changed,
                        max_prototypes=max_prototypes,
                    )
                candidate_evaluations += int(result.get("candidates_evaluated", 0))
                rows.append(
                    {
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "policy": primary_policy,
                        "sample_index": int(sample_index),
                        "fold": fold,
                        "prototype_scope": "outer_training_partition_only",
                        "case_excluded_from_training": True,
                        "true_class": int(y_test.loc[sample_index]),
                        "predicted_class": predicted_class,
                        "desired_class": desired_class,
                        "correct": bool(y_test.loc[sample_index] == predicted_class),
                        "confidence": float(np.max(probability[position])),
                        "eligible_for_upward_shift": eligible,
                        "intervention_mode": mode,
                        "n_allowed_features": len(allowed),
                        "valid": bool(result.get("valid", False)),
                        "achieved_class": result.get("achieved_class", math.nan),
                        "probability_gain": result.get("probability_gain", math.nan),
                        "desired_probability": result.get("desired_probability", math.nan),
                        "cost": result.get("cost", math.nan),
                        "num_changed_features": result.get("num_changed_features", 0),
                        "changed_features": ";".join(result.get("changed_features", [])),
                        "changed_control_types": ";".join(result.get("changed_control_types", [])),
                        "changes_json": json.dumps(result.get("changes", []), sort_keys=True, default=str),
                        "failure_reason": result.get("failure_reason", ""),
                        "prototypes_considered": result.get("prototypes_considered", 0),
                        "candidates_evaluated": result.get("candidates_evaluated", 0),
                        "domain_rejections": result.get("domain_rejections", 0),
                    }
                )
        if max_cases is not None and eligible_processed >= max_cases:
            break

    cases = pd.DataFrame(rows)
    if cases.empty:
        raise CounterfactualProtocolError("Counterfactual protocol produced no case rows.")
    uncertainty = protocol.get("uncertainty", {})
    confidence = float(uncertainty.get("confidence_level", 0.95))
    n_resamples = int(settings.get("evaluation", {}).get("bootstrap", {}).get("n_resamples", 5000))
    summary, uncertainty_table, failures = summarize_actionability(
        cases,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    examples = representative_examples(cases)
    elapsed = time.perf_counter() - started
    population_status = "diagnostic_cost_benchmark" if max_cases is not None else "all_eligible_oof_cases"
    protocol_payload = {
        "run_id": run_id,
        "config_hash": config_hash,
        "policy": primary_policy,
        "excluded_features": excluded,
        "model_evaluation_scope": "out_of_fold_only",
        "prototype_scope": "outer_training_partition_only",
        "scale_and_domain_scope": "outer_training_partition_only",
        "case_excluded_from_training": True,
        "evaluation_population": population_status,
        "max_cases": max_cases,
        "n_splits": n_splits,
        "intervention_modes": intervention_modes,
        "diagnostic_full_default_includes_immutable": True,
        "max_features_changed": max_features_changed,
        "max_prototypes": max_prototypes,
        "cost_scaling": "training-fold IQR, range fallback; categorical unit cost; sparsity penalty 0.15",
        "domain_constraints": "observed training-fold prototype values plus relational tenure constraints",
        "validity_rule": "fold-specific predicted class reaches desired or higher class",
        "n_total_case_mode_rows": len(cases),
        "n_unique_cases": int(cases["sample_index"].nunique()),
        "n_unique_eligible_cases": int(cases[cases["eligible_for_upward_shift"]]["sample_index"].nunique()),
        "candidate_evaluations": candidate_evaluations,
        "elapsed_seconds": elapsed,
        "seed": seed,
        "warning": "Model scenarios only; not causal findings or employee prescriptions.",
    }
    paths = {
        "protocol": output / "actionability_protocol.json",
        "by_case": output / "actionability_by_case.csv",
        "summary": output / "actionability_summary.csv",
        "failures": output / "actionability_failure_reasons.csv",
        "uncertainty": output / "actionability_uncertainty.csv",
        "examples": output / "representative_counterfactual_examples.jsonl",
        "interpretation": output / "actionability_interpretation.md",
    }
    write_json(paths["protocol"], protocol_payload)
    cases.to_csv(paths["by_case"], index=False)
    summary.to_csv(paths["summary"], index=False)
    failures.to_csv(paths["failures"], index=False)
    uncertainty_table.to_csv(paths["uncertainty"], index=False)
    write_jsonl(paths["examples"], examples)
    write_interpretation(paths["interpretation"], summary, run_id=run_id, config_hash=config_hash)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical OOF counterfactual actionability evaluation.")
    parser.add_argument("--config", default="configs/manuscript_final.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-hash", default=None)
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
                    max_cases=arguments.max_cases,
                ).items()
            },
            indent=2,
            sort_keys=True,
        )
    )
