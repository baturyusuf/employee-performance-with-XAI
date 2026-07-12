from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.core.io_utils import ensure_dir, write_json
from src.data.preprocess import load_validated_or_raw_data
from src.experiments.leakage_safe_cv import LabelEncodedXGBClassifier, make_preprocessor
from src.models.evaluate import classification_metrics
from src.governance.manuscript_contract import canonical_config_hash
from src.utils.config_loader import load_config


DEFAULT_CONFIG = Path("configs/manuscript_final.yaml")
REQUIRED_POLICIES = (
    "full_feature_upper_bound",
    "no_salary_hike",
    "no_salary_hike_no_attrition",
    "no_salary_hike_no_attrition_no_department",
    "no_salary_hike_no_attrition_no_department_no_job_role",
)
SUMMARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "quadratic_weighted_kappa",
    "ordinal_mae",
    "severe_error_rate",
    "nll_log_loss",
    "multiclass_brier",
    "ece_confidence",
)
PAIRWISE_METRICS = (
    "macro_f1",
    "quadratic_weighted_kappa",
    "ordinal_mae",
    "nll_log_loss",
)
HIGHER_IS_BETTER = {
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "quadratic_weighted_kappa",
}


class PolicyAblationError(RuntimeError):
    """Raised when the canonical policy ablation contract is invalid."""


def _settings(config_path: str | Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    raw = load_config(config_path)
    settings = raw.get("manuscript_final", raw)
    if not isinstance(settings, dict):
        raise PolicyAblationError("Canonical config must contain a manuscript_final mapping.")
    return raw, settings


def _policy_definitions(settings: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    feature_policies = settings.get("feature_policies", {})
    definitions = feature_policies.get("definitions", {}) if isinstance(feature_policies, Mapping) else {}
    if not isinstance(definitions, dict):
        raise PolicyAblationError("feature_policies.definitions must be a mapping.")
    missing = [name for name in REQUIRED_POLICIES if name not in definitions]
    if missing:
        raise PolicyAblationError(f"Canonical config is missing required policies: {missing}")
    return definitions


def _selected_policies(definitions: Mapping[str, Mapping[str, Any]]) -> list[str]:
    selected = list(REQUIRED_POLICIES)
    selected.extend(
        name
        for name, definition in definitions.items()
        if name not in selected and bool(definition.get("audit_only", False))
    )
    return selected


def exact_policy_frame(
    frame: pd.DataFrame,
    policy_name: str,
    definition: Mapping[str, Any],
    *,
    target_column: str,
    id_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    excluded = [str(value) for value in definition.get("excluded_features", [])]
    required_exclusions = {target_column, id_column}
    if not required_exclusions.issubset(excluded):
        raise PolicyAblationError(
            f"Policy {policy_name!r} must explicitly exclude target and identifier: "
            f"{sorted(required_exclusions)}"
        )
    unknown = sorted(set(excluded).difference(frame.columns))
    if unknown:
        raise PolicyAblationError(f"Policy {policy_name!r} excludes unknown columns: {unknown}")
    features = [column for column in frame.columns if column not in set(excluded)]
    if not features:
        raise PolicyAblationError(f"Policy {policy_name!r} leaves no model features.")
    return frame.loc[:, features].copy(), excluded


def resolve_seed(settings: Mapping[str, Any], value_or_name: Any, *, default: int = 42) -> int:
    if isinstance(value_or_name, (int, np.integer)):
        return int(value_or_name)
    if isinstance(value_or_name, str):
        seeds = settings.get("seeds", {})
        if isinstance(seeds, Mapping) and value_or_name in seeds:
            return int(seeds[value_or_name])
        try:
            return int(value_or_name)
        except ValueError as exc:
            raise PolicyAblationError(f"Unknown seed reference: {value_or_name!r}") from exc
    return default


def _model_parameters(settings: Mapping[str, Any]) -> Dict[str, Any]:
    model = settings.get("model", {})
    params: Dict[str, Any] = {}
    if isinstance(model, Mapping):
        source = model.get("hyperparameters", model.get("xgboost", {}))
        if isinstance(source, Mapping):
            params = dict(source)
    params.pop("random_state_seed", None)
    allowed = {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "objective",
        "eval_metric",
        "random_state",
        "n_jobs",
    }
    unexpected = sorted(set(params).difference(allowed))
    if unexpected:
        raise PolicyAblationError(f"Unsupported canonical XGBoost parameters: {unexpected}")
    return params


def _fit_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    parameters: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    params = dict(parameters)
    params["random_state"] = seed
    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(X_train)),
            ("model", LabelEncodedXGBClassifier(**params)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline.predict(X_test), pipeline.predict_proba(X_test)


def _mean_ci(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan, math.nan, math.nan
    mean = float(array.mean())
    if len(array) == 1:
        return mean, 0.0, mean, mean
    std = float(array.std(ddof=1))
    alpha = 1.0 - confidence
    critical = float(student_t.ppf(1.0 - alpha / 2.0, df=len(array) - 1))
    half_width = critical * std / math.sqrt(len(array))
    return mean, std, mean - half_width, mean + half_width


def summarize_policies(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for policy, group in fold_metrics.groupby("policy", sort=False):
        row: Dict[str, Any] = {
            "run_id": group["run_id"].iloc[0],
            "config_hash": group["config_hash"].iloc[0],
            "policy": policy,
            "role": group["role"].iloc[0],
            "audit_only": bool(group["audit_only"].iloc[0]),
            "n_folds": int(group["fold"].nunique()),
            "n_features": int(group["n_features"].iloc[0]),
            "excluded_features": group["excluded_features"].iloc[0],
        }
        for metric in SUMMARY_METRICS:
            if metric not in group:
                continue
            mean, std, low, high = _mean_ci(pd.to_numeric(group[metric], errors="coerce"))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    if count == 0:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(p_values[int(index)]))
        running = max(running, value)
        adjusted[int(index)] = running
    return adjusted.tolist()


def policy_pairwise_tests(fold_metrics: pd.DataFrame, alpha: float) -> pd.DataFrame:
    records: list[Dict[str, Any]] = []
    policies = fold_metrics["policy"].drop_duplicates().tolist()
    for metric in PAIRWISE_METRICS:
        metric_records: list[Dict[str, Any]] = []
        for policy_a, policy_b in itertools.combinations(policies, 2):
            a = fold_metrics[fold_metrics["policy"] == policy_a].set_index("fold")
            b = fold_metrics[fold_metrics["policy"] == policy_b].set_index("fold")
            common = sorted(set(a.index).intersection(b.index))
            values_a = pd.to_numeric(a.loc[common, metric], errors="coerce").to_numpy(dtype=float)
            values_b = pd.to_numeric(b.loc[common, metric], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values_a) & np.isfinite(values_b)
            values_a, values_b = values_a[finite], values_b[finite]
            differences = values_a - values_b
            if len(differences) == 0:
                statistic, p_value, test_status = math.nan, math.nan, "no_valid_pairs"
            elif np.allclose(differences, 0.0):
                statistic, p_value, test_status = 0.0, 1.0, "all_differences_zero"
            else:
                result = wilcoxon(values_a, values_b, alternative="two-sided", zero_method="wilcox")
                statistic, p_value, test_status = float(result.statistic), float(result.pvalue), "ok"
            metric_records.append(
                {
                    "run_id": fold_metrics["run_id"].iloc[0],
                    "config_hash": fold_metrics["config_hash"].iloc[0],
                    "metric": metric,
                    "policy_a": policy_a,
                    "policy_b": policy_b,
                    "n_paired_folds": int(len(differences)),
                    "mean_a": float(values_a.mean()) if len(values_a) else math.nan,
                    "mean_b": float(values_b.mean()) if len(values_b) else math.nan,
                    "mean_paired_difference_a_minus_b": float(differences.mean()) if len(differences) else math.nan,
                    "wilcoxon_statistic": statistic,
                    "p_value_raw": p_value,
                    "test_status": test_status,
                }
            )
        finite_positions = [i for i, row in enumerate(metric_records) if np.isfinite(row["p_value_raw"])]
        adjusted = _holm_adjust([metric_records[i]["p_value_raw"] for i in finite_positions])
        for position, value in zip(finite_positions, adjusted):
            metric_records[position]["p_value_holm"] = value
            metric_records[position]["reject_holm"] = bool(value < alpha)
        for row in metric_records:
            row.setdefault("p_value_holm", math.nan)
            row.setdefault("reject_holm", False)
        records.extend(metric_records)
    return pd.DataFrame(records)


def leakage_sensitivity_indices(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    reference = fold_metrics[fold_metrics["policy"] == "full_feature_upper_bound"].set_index("fold")
    if reference.empty:
        raise PolicyAblationError("full_feature_upper_bound is required for leakage sensitivity indices.")
    rows: list[Dict[str, Any]] = []
    for policy, group in fold_metrics.groupby("policy", sort=False):
        current = group.set_index("fold")
        common = sorted(set(reference.index).intersection(current.index))
        for metric in PAIRWISE_METRICS:
            ref_values = pd.to_numeric(reference.loc[common, metric], errors="coerce").to_numpy(dtype=float)
            cur_values = pd.to_numeric(current.loc[common, metric], errors="coerce").to_numpy(dtype=float)
            denominator = np.maximum(np.abs(ref_values), 1e-12)
            if metric in HIGHER_IS_BETTER:
                fold_index = (ref_values - cur_values) / denominator
            else:
                fold_index = (cur_values - ref_values) / denominator
            mean, std, low, high = _mean_ci(fold_index)
            rows.append(
                {
                    "run_id": fold_metrics["run_id"].iloc[0],
                    "config_hash": fold_metrics["config_hash"].iloc[0],
                    "reference_policy": "full_feature_upper_bound",
                    "policy": policy,
                    "metric": metric,
                    "definition": "positive values indicate degradation relative to the diagnostic full-feature upper bound",
                    "n_folds": len(common),
                    "index_mean": mean,
                    "index_std": std,
                    "index_ci_low": low,
                    "index_ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def manuscript_policy_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "run_id",
        "config_hash",
        "policy",
        "role",
        "audit_only",
        "n_folds",
        "n_features",
        "macro_f1_mean",
        "macro_f1_ci_low",
        "macro_f1_ci_high",
        "quadratic_weighted_kappa_mean",
        "quadratic_weighted_kappa_ci_low",
        "quadratic_weighted_kappa_ci_high",
        "ordinal_mae_mean",
        "ordinal_mae_ci_low",
        "ordinal_mae_ci_high",
        "severe_error_rate_mean",
        "excluded_features",
    ]
    return summary.loc[:, [column for column in columns if column in summary.columns]].copy()


def write_interpretation(summary: pd.DataFrame, path: Path) -> None:
    full = summary[summary["policy"] == "full_feature_upper_bound"].iloc[0]
    lines = [
        "# Canonical Feature-Policy Interpretation",
        "",
        f"Run ID: `{full['run_id']}`  ",
        f"Config hash: `{full['config_hash']}`",
        "",
        "All policy estimates use the same predeclared stratified folds, preprocessing implementation, XGBoost settings, and target labels. Full-feature results are diagnostic leakage-warning upper bounds only and are not deployable evidence.",
        "",
        "## Policy Results",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- `{row.policy}` ({row.role}; audit_only={bool(row.audit_only)}): "
            f"macro-F1 {row.macro_f1_mean:.4f} (95% CI {row.macro_f1_ci_low:.4f}–{row.macro_f1_ci_high:.4f}); "
            f"QWK {row.quadratic_weighted_kappa_mean:.4f}; ordinal MAE {row.ordinal_mae_mean:.4f}; "
            f"features {row.n_features}."
        )
    lines.extend(
        [
            "",
            "## Claim Boundaries",
            "",
            "- Performance changes across compound policies must not be attributed to one removed field unless a dedicated audit-only contrast isolates that field.",
            "- The sensitive-retaining audit policy is diagnostic only; it is included to separate leakage-variable effects from demographic-governance exclusions.",
            "- Excluding sensitive or organisational fields does not prove fairness, eliminate proxy risk, or identify causal effects.",
            "- These models are research-grade decision support and must not be used for autonomous HR decisions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tradeoff_figure(summary: pd.DataFrame, output_dir: Path, metadata: Mapping[str, str]) -> Dict[str, Path]:
    source = summary[
        [
            "run_id",
            "config_hash",
            "policy",
            "role",
            "audit_only",
            "macro_f1_mean",
            "macro_f1_ci_low",
            "macro_f1_ci_high",
            "quadratic_weighted_kappa_mean",
            "ordinal_mae_mean",
        ]
    ].copy()
    source_path = output_dir / "figure_leakage_policy_tradeoff_source.csv"
    source.to_csv(source_path, index=False)

    display = source[~source["audit_only"].astype(bool)].reset_index(drop=True)
    labels = [name.replace("no_salary_hike_no_attrition", "no salary/attrition").replace("_", " ") for name in display["policy"]]
    positions = np.arange(len(display))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    macro_error = np.vstack(
        [
            display["macro_f1_mean"] - display["macro_f1_ci_low"],
            display["macro_f1_ci_high"] - display["macro_f1_mean"],
        ]
    )
    axes[0].errorbar(display["macro_f1_mean"], positions, xerr=macro_error, fmt="o", capsize=4, color="#176B87")
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Macro-F1 (fold mean and 95% t interval)")
    axes[0].set_title("Predictive utility under feature restrictions")
    axes[0].grid(axis="x", alpha=0.25)

    scatter = axes[1].scatter(
        display["ordinal_mae_mean"],
        display["quadratic_weighted_kappa_mean"],
        c=np.arange(len(display)),
        cmap="viridis",
        s=75,
    )
    del scatter
    for i, label in enumerate(labels):
        axes[1].annotate(label, (display.loc[i, "ordinal_mae_mean"], display.loc[i, "quadratic_weighted_kappa_mean"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("Ordinal MAE (lower is better)")
    axes[1].set_ylabel("Quadratic weighted kappa (higher is better)")
    axes[1].set_title("Ordinal error–agreement trade-off")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Canonical feature-policy trade-off (full-feature model is diagnostic only)", fontsize=13)

    png = output_dir / "figure_leakage_policy_tradeoff.png"
    svg = output_dir / "figure_leakage_policy_tradeoff.svg"
    description = f"run_id={metadata['run_id']}; config_hash={metadata['config_hash']}"
    fig.savefig(png, dpi=300, metadata={"Description": description})
    fig.savefig(svg, format="svg", metadata={"Title": "Canonical feature-policy trade-off", "Description": description})
    plt.close(fig)
    return {"png": png, "svg": svg, "source": source_path}


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    output_dir: str | Path,
    run_id: str,
    config_hash: str | None = None,
) -> Dict[str, Path]:
    raw_config, settings = _settings(config_path)
    config_hash = config_hash or canonical_config_hash(raw_config)
    output = ensure_dir(Path(output_dir))
    definitions = _policy_definitions(settings)
    policies = _selected_policies(definitions)

    target = settings.get("target", {})
    target_column = str(target.get("column", "PerformanceRating"))
    labels = [int(value) for value in target.get("labels", [2, 3, 4])]
    identifier_fields = settings.get("governance_fields", {}).get("identifier_fields", ["EmpNumber"])
    id_column = str(identifier_fields[0] if identifier_fields else "EmpNumber")
    data = load_validated_or_raw_data()
    y = data[target_column].astype(int)
    if sorted(y.unique().tolist()) != sorted(labels):
        raise PolicyAblationError(
            f"Observed labels {sorted(y.unique().tolist())} do not match canonical labels {sorted(labels)}."
        )

    cv = settings.get("evaluation", {}).get("cv", {})
    n_splits = int(cv.get("n_splits", 10))
    seed = resolve_seed(settings, cv.get("seed", "cv"))
    alpha = float(settings.get("evaluation", {}).get("statistical_tests", {}).get("alpha", 0.05))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=bool(cv.get("shuffle", True)), random_state=seed)
    folds = list(splitter.split(data, y))
    fold_assignment = np.empty(len(data), dtype=int)
    for fold, (_, test_positions) in enumerate(folds, start=1):
        fold_assignment[test_positions] = fold
    assignment_df = pd.DataFrame(
        {
            "run_id": run_id,
            "config_hash": config_hash,
            "sample_index": data.index,
            "fold": fold_assignment,
            "target": y.to_numpy(),
        }
    )
    assignment_path = output / "fold_assignments.csv"
    assignment_df.to_csv(assignment_path, index=False)

    parameters = _model_parameters(settings)
    rows: list[Dict[str, Any]] = []
    for policy in policies:
        definition = definitions[policy]
        X, excluded = exact_policy_frame(
            data,
            policy,
            definition,
            target_column=target_column,
            id_column=id_column,
        )
        for fold, (train_positions, test_positions) in enumerate(folds, start=1):
            prediction, probability = _fit_predict(
                X.iloc[train_positions],
                y.iloc[train_positions],
                X.iloc[test_positions],
                parameters=parameters,
                seed=seed,
            )
            metrics = classification_metrics(y.iloc[test_positions], prediction, probability, labels=labels)
            rows.append(
                {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "policy": policy,
                    "role": str(definition.get("role", "unspecified")),
                    "audit_only": bool(definition.get("audit_only", False)),
                    "model": "xgboost",
                    "fold": fold,
                    "n_train": len(train_positions),
                    "n_test": len(test_positions),
                    "n_features": X.shape[1],
                    "excluded_features": ";".join(excluded),
                    **metrics,
                }
            )

    fold_df = pd.DataFrame(rows)
    summary_df = summarize_policies(fold_df)
    pairwise_df = policy_pairwise_tests(fold_df, alpha=alpha)
    sensitivity_df = leakage_sensitivity_indices(fold_df)
    manuscript_df = manuscript_policy_table(summary_df)

    paths = {
        "fold_metrics": output / "fold_metrics.csv",
        "policy_summary": output / "policy_summary.csv",
        "policy_pairwise_tests": output / "policy_pairwise_tests.csv",
        "leakage_sensitivity_index": output / "leakage_sensitivity_index.csv",
        "policy_interpretation": output / "policy_interpretation.md",
        "manuscript_policy_table": output / "manuscript_policy_table.csv",
        "fold_assignments": assignment_path,
        "metadata": output / "policy_ablation_metadata.json",
    }
    fold_df.to_csv(paths["fold_metrics"], index=False)
    summary_df.to_csv(paths["policy_summary"], index=False)
    pairwise_df.to_csv(paths["policy_pairwise_tests"], index=False)
    sensitivity_df.to_csv(paths["leakage_sensitivity_index"], index=False)
    manuscript_df.to_csv(paths["manuscript_policy_table"], index=False)
    write_interpretation(summary_df, paths["policy_interpretation"])
    figure_paths = write_tradeoff_figure(summary_df, output, {"run_id": run_id, "config_hash": config_hash})
    paths.update({f"figure_{key}": value for key, value in figure_paths.items()})
    write_json(
        paths["metadata"],
        {
            "stage": "policy_ablation",
            "run_id": run_id,
            "config_hash": config_hash,
            "config_path": str(config_path),
            "policies": policies,
            "primary_policy": settings.get("feature_policies", {}).get("primary_policy"),
            "n_splits": n_splits,
            "same_fold_assignments_for_all_policies": True,
            "model": "xgboost",
            "model_parameters": parameters,
            "seed": seed,
            "labels": labels,
            "outputs": {key: str(value) for key, value in paths.items() if key != "metadata"},
            "claim_boundary": "full_feature_upper_bound is diagnostic leakage-warning evidence only",
        },
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical common-fold XGBoost feature-policy ablation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-hash", default=None)
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
                ).items()
            },
            indent=2,
            sort_keys=True,
        )
    )
