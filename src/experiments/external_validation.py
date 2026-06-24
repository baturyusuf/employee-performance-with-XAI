from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.external_adapters import (
    ExternalDataset,
    audit_attribute_columns,
    build_feature_columns,
    load_external_dataset,
    role_columns,
)
from src.data.external_audit import dataset_report_dir, run_dataset_audit
from src.data.preprocess import load_validated_or_raw_data, split_features_and_target
from src.experiments.fairness_sensitivity import (
    compute_disparity_summary,
    compute_group_metrics,
    compute_small_group_warnings,
)
from src.experiments.final_evidence_common import align_proba, predict_labels_from_proba
from src.experiments.final_shap_stability import (
    get_group_mapping,
    group_shap_values,
    normalize_shap_values,
)
from src.experiments.leakage_safe_cv import LabelEncodedXGBClassifier, infer_columns, make_preprocessor
from src.experiments.proxy_analysis import run_proxy_classifier_cv, summarize_cv
from src.features.feature_sets import apply_feature_set
from src.llm.evidence_schema import (
    CalibrationEvidence,
    CompleteCaseEvidence,
    CounterfactualEvidence,
    FairnessEvidence,
    GovernanceEvidence,
    LeakageEvidence,
    PredictionEvidence,
    ShapEvidence,
)
from src.llm.governed_explainer import GovernedExplainer
from src.llm.runtime_config import LLMRuntimeConfig
from src.agents.openai_agents_runtime import OpenAIAgentsSDKGovernanceRuntime
from src.models.evaluate import classification_metrics
from src.utils.config import SETTINGS
from src.utils.experiment_registry import (
    append_registry_row,
    collect_package_versions,
    get_git_commit,
    utc_now_iso,
)


MODEL_NAME = "xgboost"
DEFAULT_SEED = 42
DEFAULT_N_SPLITS = 5
PRIMARY_INX_FEATURE_SET = "no_salary_hike_no_attrition_no_department"
INX_BASELINE_FEATURE_SET = "no_salary_hike_no_attrition"
INX_STRICT_FEATURE_SET = "no_salary_hike_no_attrition_no_department_no_job_role"
EXTERNAL_REPORT_ROOT = SETTINGS.reports_dir / "external_validation"
GOVERNANCE_REPORT_DIR = SETTINGS.reports_dir / "governance_reports"
MANUSCRIPT_ASSET_DIR = SETTINGS.reports_dir / "manuscript_assets"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def fit_xgb_pipeline(X_train: pd.DataFrame, y_train: pd.Series, seed: int = DEFAULT_SEED) -> Any:
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(X_train)),
            ("model", LabelEncodedXGBClassifier(random_state=seed)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def run_external_dataset_experiment(
    dataset_name: str,
    *,
    target_kind: str = "primary",
    policies: Optional[List[str]] = None,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    min_support: Optional[int] = None,
    write_registry: bool = True,
) -> Dict[str, Path]:
    run_dataset_audit(dataset_name, target_kind=target_kind, write_registry=write_registry)
    dataset = load_external_dataset(dataset_name, target_kind=target_kind)
    output_dir = dataset_report_dir(dataset_name, target_kind)
    ensure_dir(output_dir)

    selected_policies = policies or list(dataset.config.feature_policy_variants.keys())
    labels = dataset.labels
    min_support = min_support or external_min_support(len(dataset.canonical))

    all_prediction_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    feature_policy_rows: List[Dict[str, Any]] = []

    for policy in selected_policies:
        feature_cols = build_feature_columns(dataset, policy)
        X = dataset.canonical.loc[:, feature_cols].copy()
        y = dataset.canonical[dataset.target_column].astype(int).copy()
        predictions = generate_oof_predictions(
            dataset=dataset,
            policy_name=policy,
            X=X,
            y=y,
            labels=labels,
            n_splits=n_splits,
            seed=seed,
        )
        all_prediction_rows.extend(predictions.to_dict(orient="records"))
        proba = predictions[[f"prob_class_{label}" for label in labels]].to_numpy(dtype=float)
        metrics = classification_metrics(predictions["y_true"], predictions["y_pred"], proba, labels=labels)
        metric_rows.append(
            {
                "dataset": dataset_name,
                "target_kind": target_kind,
                "task_type": dataset.task_type,
                "policy": policy,
                "model": MODEL_NAME,
                "n_rows": int(len(dataset.canonical)),
                "n_features": int(len(feature_cols)),
                "labels": ";".join(str(label) for label in labels),
                **metrics,
            }
        )
        feature_policy_rows.append(
            {
                "dataset": dataset_name,
                "target_kind": target_kind,
                "policy": policy,
                "n_features": len(feature_cols),
                "feature_columns": ";".join(feature_cols),
                "excluded_leakage_columns": ";".join(role_columns(dataset, "leakage")),
                "excluded_sensitive_columns": ";".join(role_columns(dataset, "sensitive")),
                "proxy_risk_columns_available": ";".join(role_columns(dataset, "proxy")),
            }
        )
        print(
            f"[external] dataset={dataset_name} target={target_kind} policy={policy} "
            f"macro_f1={metrics.get('macro_f1'):.4f} qwk={metrics.get('quadratic_weighted_kappa')}"
        )

    prediction_df = pd.DataFrame(all_prediction_rows)
    metrics_df = pd.DataFrame(metric_rows)
    feature_policy_df = pd.DataFrame(feature_policy_rows)

    prediction_path = output_dir / "model_predictions.csv"
    metrics_path = output_dir / "performance_metrics.csv"
    policy_path = output_dir / "feature_policy_audit.csv"
    prediction_df.to_csv(prediction_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    feature_policy_df.to_csv(policy_path, index=False)

    fairness_outputs = write_fairness_outputs(dataset, prediction_df, labels, output_dir, min_support)
    calibration_path = write_calibration_bins(prediction_df, labels, output_dir)
    proxy_path = write_proxy_outputs(dataset, selected_policies, output_dir, seed=seed)
    shap_outputs = write_shap_outputs(dataset, selected_policies, output_dir, seed=seed)
    actionability_path = write_actionability_outputs(dataset, shap_outputs, output_dir)
    representative_path = write_representative_cases(dataset, prediction_df, shap_outputs, output_dir, limit=8)
    interpretation_path = write_experiment_interpretation(
        dataset=dataset,
        target_kind=target_kind,
        metrics_df=metrics_df,
        output_dir=output_dir,
        min_support=min_support,
    )
    metadata_path = output_dir / "experiment_metadata.json"
    save_json(
        {
            "dataset": dataset_name,
            "target_kind": target_kind,
            "display_name": dataset.config.display_name,
            "recommended_role": dataset.config.recommended_role,
            "task_type": dataset.task_type,
            "labels": labels,
            "policies": selected_policies,
            "n_splits_requested": n_splits,
            "min_support": min_support,
            "seed": seed,
            "source_url": dataset.config.source_url,
            "package_versions": collect_package_versions(["numpy", "pandas", "scikit-learn", "xgboost", "shap"]),
            "outputs": {
                "predictions": prediction_path,
                "performance_metrics": metrics_path,
                "feature_policy_audit": policy_path,
                "fairness": fairness_outputs,
                "calibration_bins": calibration_path,
                "proxy_summary": proxy_path,
                "shap": shap_outputs,
                "actionability": actionability_path,
                "representative_cases": representative_path,
                "interpretation": interpretation_path,
            },
        },
        metadata_path,
    )
    if write_registry:
        append_registry_row(
            {
                "run_id": f"external_experiment_{dataset_name}_{target_kind}_{utc_now_iso()}",
                "date_time": utc_now_iso(),
                "git_commit_if_available": get_git_commit(),
                "script": "python -m src.experiments.external_validation",
                "config": f"data/external/{dataset_name}/schema_mapping.json",
                "feature_set": "; ".join(selected_policies),
                "model": MODEL_NAME,
                "seed": seed,
                "cv_strategy": "StratifiedKFold with split count capped by smallest class",
                "primary_metrics": metrics_df.to_dict(orient="records"),
                "output_dir": _rel(output_dir),
                "notes": f"External validation/robustness run for {dataset_name} ({dataset.task_type}).",
                "decision_status": "candidate",
            }
        )
    return {
        "predictions": prediction_path,
        "metrics": metrics_path,
        "feature_policy": policy_path,
        "calibration_bins": calibration_path,
        "proxy_summary": proxy_path,
        "actionability": actionability_path,
        "representative_cases": representative_path,
        "interpretation": interpretation_path,
        "metadata": metadata_path,
    }


def generate_oof_predictions(
    *,
    dataset: ExternalDataset,
    policy_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    labels: List[int],
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    min_class = int(y.value_counts().min())
    splits = min(n_splits, min_class)
    if splits < 2:
        raise ValueError(
            f"Not enough class support for stratified CV in {dataset.config.dataset_name}/{policy_name}: "
            f"minimum class count={min_class}"
        )

    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    rows: List[Dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()
        pipeline = fit_xgb_pipeline(X_train, y_train, seed=seed)
        classes = [int(c) for c in pipeline.named_steps["model"].classes_]
        proba = align_proba(pipeline.predict_proba(X_test), classes, labels)
        pred = predict_labels_from_proba(proba, labels)
        for row_pos, sample_index in enumerate(X_test.index):
            row = {
                "dataset": dataset.config.dataset_name,
                "target_kind": dataset.task_type,
                "feature_set": policy_name,
                "policy": policy_name,
                "model": MODEL_NAME,
                "fold": fold,
                "sample_index": int(sample_index),
                "external_sample_id": int(dataset.canonical.loc[sample_index, "ExternalSampleId"]),
                "y_true": int(y_test.loc[sample_index]),
                "y_pred": int(pred[row_pos]),
                "correct": bool(int(y_test.loc[sample_index]) == int(pred[row_pos])),
                "confidence": float(np.max(proba[row_pos])),
            }
            for label_idx, label in enumerate(labels):
                row[f"prob_class_{label}"] = float(proba[row_pos, label_idx])
            rows.append(row)
    return pd.DataFrame(rows)


def write_fairness_outputs(
    dataset: ExternalDataset,
    predictions: pd.DataFrame,
    labels: List[int],
    output_dir: Path,
    min_support: int,
) -> Dict[str, Path]:
    attributes = audit_attribute_columns(dataset)
    fairness_dir = output_dir / "fairness_proxy"
    ensure_dir(fairness_dir)
    if attributes:
        group_df = compute_group_metrics(predictions, dataset.canonical, attributes, labels, min_support)
    else:
        group_df = pd.DataFrame()
    if group_df.empty:
        disparity_df = pd.DataFrame(
            columns=[
                "feature_set",
                "model",
                "attribute",
                "metric",
                "class_label",
                "max_gap",
                "min_group_support_threshold",
            ]
        )
        warnings_df = pd.DataFrame(columns=["attribute", "group_value", "n_samples", "warning"])
    else:
        disparity_df = compute_disparity_summary(group_df, min_support)
        warnings_df = compute_small_group_warnings(dataset.canonical, attributes, min_support)

    paths = {
        "group_metrics": fairness_dir / "fairness_group_metrics.csv",
        "disparity_summary": fairness_dir / "fairness_disparity_summary.csv",
        "small_group_warnings": fairness_dir / "small_group_warnings.csv",
        "interpretation": fairness_dir / "fairness_proxy_interpretation.md",
    }
    group_df.to_csv(paths["group_metrics"], index=False)
    disparity_df.to_csv(paths["disparity_summary"], index=False)
    warnings_df.to_csv(paths["small_group_warnings"], index=False)
    write_fairness_interpretation(dataset, attributes, min_support, disparity_df, warnings_df, paths["interpretation"])
    return paths


def write_fairness_interpretation(
    dataset: ExternalDataset,
    attributes: List[str],
    min_support: int,
    disparity_df: pd.DataFrame,
    warnings_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        f"# External Fairness and Proxy Audit: {dataset.config.display_name}",
        "",
        f"Minimum support threshold: {min_support}",
        f"Audit attributes: {', '.join(attributes) if attributes else 'none available'}",
        "",
        "These subgroup metrics are diagnostic model-governance evidence. They do not prove fairness, discrimination, or absence of discrimination.",
        "",
        "## Largest Support-Filtered Gaps",
        "",
    ]
    if disparity_df.empty or "max_gap" not in disparity_df.columns:
        lines.append("No support-filtered disparity rows were available.")
    else:
        for row in disparity_df.sort_values("max_gap", ascending=False).head(12).itertuples(index=False):
            class_part = "" if not hasattr(row, "class_label") or pd.isna(row.class_label) else f", class={row.class_label}"
            lines.append(
                f"- policy={getattr(row, 'feature_set')}, attribute={row.attribute}, metric={row.metric}{class_part}: "
                f"gap={row.max_gap:.4f}"
            )
    lines.extend(["", "## Small-Group Warnings", ""])
    if warnings_df.empty:
        lines.append("No small support warnings were detected under this threshold.")
    else:
        for row in warnings_df.itertuples(index=False):
            lines.append(f"- {row.attribute}={row.group_value}: n={row.n_samples}, {row.warning}")
    lines.extend(
        [
            "",
            "## Required Claim Limits",
            "",
            "- Removing direct group variables is not evidence that the model is fair.",
            "- Department, role, salary, tenure, and evaluation-history features can act as organisational proxies.",
            "- Any subgroup result with low support must be treated as unstable.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_calibration_bins(predictions: pd.DataFrame, labels: List[int], output_dir: Path, n_bins: int = 10) -> Path:
    rows: List[Dict[str, Any]] = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for policy, group in predictions.groupby("policy"):
        confidence = group["confidence"].to_numpy(dtype=float)
        correct = group["correct"].astype(bool).to_numpy()
        for bin_idx, (low, high) in enumerate(zip(bins[:-1], bins[1:]), start=1):
            mask = ((confidence >= low) & (confidence <= high)) if bin_idx == 1 else ((confidence > low) & (confidence <= high))
            rows.append(
                {
                    "policy": policy,
                    "bin": bin_idx,
                    "bin_low": low,
                    "bin_high": high,
                    "n_samples": int(mask.sum()),
                    "mean_confidence": float(confidence[mask].mean()) if np.any(mask) else np.nan,
                    "accuracy": float(correct[mask].mean()) if np.any(mask) else np.nan,
                }
            )
    out = output_dir / "calibration_bins.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def write_proxy_outputs(dataset: ExternalDataset, policies: List[str], output_dir: Path, seed: int) -> Path:
    rows: List[Dict[str, Any]] = []
    if "EmpDepartment" not in dataset.canonical.columns or dataset.canonical["EmpDepartment"].nunique(dropna=True) < 2:
        rows.append({"policy": "", "proxy_target": "EmpDepartment", "status": "not_available"})
    else:
        y_department = dataset.canonical["EmpDepartment"].astype(str)
        for policy in policies:
            feature_cols = build_feature_columns(dataset, policy)
            feature_cols = [col for col in feature_cols if col != "EmpDepartment"]
            if not feature_cols:
                rows.append({"policy": policy, "proxy_target": "EmpDepartment", "status": "no_features"})
                continue
            try:
                fold_df = run_proxy_classifier_cv(
                    dataset.canonical.loc[:, feature_cols].copy(),
                    y_department,
                    n_splits=5,
                    random_state=seed,
                )
                summary = summarize_cv(fold_df).set_index("metric")["mean"].to_dict()
                rows.append(
                    {
                        "policy": policy,
                        "proxy_target": "EmpDepartment",
                        "status": "completed",
                        "n_features": len(feature_cols),
                        "accuracy": summary.get("accuracy"),
                        "balanced_accuracy": summary.get("balanced_accuracy"),
                        "macro_f1": summary.get("macro_f1"),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "policy": policy,
                        "proxy_target": "EmpDepartment",
                        "status": "failed",
                        "error": str(exc),
                    }
                )
    out = output_dir / "department_proxy_reconstruction.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def write_shap_outputs(
    dataset: ExternalDataset,
    policies: List[str],
    output_dir: Path,
    seed: int,
    top_k: int = 20,
) -> Dict[str, Dict[str, Path]]:
    import shap

    labels = dataset.labels
    outputs: Dict[str, Dict[str, Path]] = {}
    shap_root = output_dir / "shap"
    ensure_dir(shap_root)
    for policy in policies:
        policy_dir = shap_root / policy
        ensure_dir(policy_dir)
        feature_cols = build_feature_columns(dataset, policy)
        X = dataset.canonical.loc[:, feature_cols].copy()
        y = dataset.canonical[dataset.target_column].astype(int).copy()
        pipeline = fit_xgb_pipeline(X, y, seed=seed)
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["model"]
        numeric_cols, categorical_cols = infer_columns(X)
        X_t = preprocessor.transform(X)
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()
        group_names, mapping = get_group_mapping(preprocessor, numeric_cols, categorical_cols)
        explainer = shap.TreeExplainer(classifier.model_)
        raw_shap = explainer.shap_values(X_t)
        shap_arr = normalize_shap_values(raw_shap, X_t.shape[0], X_t.shape[1], len(classifier.classes_))
        grouped = group_shap_values(shap_arr, group_names, mapping)
        importance = np.mean(np.abs(grouped), axis=(0, 1))
        global_df = pd.DataFrame({"feature": group_names, "mean_abs_grouped_shap": importance})
        global_df = global_df.sort_values("mean_abs_grouped_shap", ascending=False).reset_index(drop=True)
        global_df["rank"] = np.arange(1, len(global_df) + 1)
        global_path = policy_dir / "global_grouped_shap_importance.csv"
        global_df.to_csv(global_path, index=False)

        proba = align_proba(pipeline.predict_proba(X), classifier.classes_, labels)
        pred = predict_labels_from_proba(proba, labels)
        local_rows: List[Dict[str, Any]] = []
        for row_pos, sample_index in enumerate(X.index):
            pred_class = int(pred[row_pos])
            class_idx = labels.index(pred_class)
            for feature_idx, feature in enumerate(group_names):
                value = grouped[row_pos, class_idx, feature_idx]
                local_rows.append(
                    {
                        "policy": policy,
                        "sample_index": int(sample_index),
                        "external_sample_id": int(dataset.canonical.loc[sample_index, "ExternalSampleId"]),
                        "predicted_class": pred_class,
                        "true_class": int(y.loc[sample_index]),
                        "confidence": float(np.max(proba[row_pos])),
                        "feature": feature,
                        "grouped_shap_value": float(value),
                        "abs_grouped_shap_value": float(abs(value)),
                    }
                )
        local_df = pd.DataFrame(local_rows)
        local_path = policy_dir / "local_grouped_shap_values.csv"
        local_df.to_csv(local_path, index=False)
        top_path = policy_dir / "top_grouped_shap_features.csv"
        global_df.head(top_k).to_csv(top_path, index=False)
        outputs[policy] = {"global": global_path, "local": local_path, "top": top_path}
    return outputs


def write_actionability_outputs(
    dataset: ExternalDataset,
    shap_outputs: Dict[str, Dict[str, Path]],
    output_dir: Path,
) -> Path:
    rows: List[Dict[str, Any]] = []
    proxy_cols = set(role_columns(dataset, "proxy"))
    for policy, paths in shap_outputs.items():
        top_df = pd.read_csv(paths["top"])
        top_df = top_df.head(15).copy()
        top_df["control_type"] = top_df["feature"].map(lambda feature: feature_control_type(str(feature), proxy_cols))
        counts = top_df["control_type"].value_counts().to_dict()
        top_n = int(len(top_df))
        employee_count = int(counts.get("employee_controllable", 0))
        manager_org_count = int(sum(counts.get(key, 0) for key in ["manager_controllable", "organisation_controllable"]))
        proxy_count = int(sum(counts.get(key, 0) for key in ["proxy_or_historical", "evaluation_proxy"]))
        rows.append(
            {
                "policy": policy,
                "top_feature_count": top_n,
                "employee_controllable_count": employee_count,
                "employee_controllable_share": float(employee_count / top_n) if top_n else np.nan,
                "manager_or_organisation_count": manager_org_count,
                "manager_or_organisation_share": float(manager_org_count / top_n) if top_n else np.nan,
                "high_caution_proxy_count": proxy_count,
                "high_caution_proxy_share": float(proxy_count / top_n) if top_n else np.nan,
                "actionability_status": actionability_status(employee_count, manager_org_count, proxy_count, top_n),
                "warning": "Counterfactuals are model scenarios, not employee prescriptions; many high-attribution features require manager or organisation action.",
            }
        )
    summary = pd.DataFrame(rows)
    out = output_dir / "actionability_summary.csv"
    summary.to_csv(out, index=False)
    detail_rows = []
    for policy, paths in shap_outputs.items():
        top_df = pd.read_csv(paths["top"]).head(15).copy()
        proxy_cols = set(role_columns(dataset, "proxy"))
        top_df["policy"] = policy
        top_df["control_type"] = top_df["feature"].map(lambda feature: feature_control_type(str(feature), proxy_cols))
        detail_rows.extend(top_df.to_dict(orient="records"))
    pd.DataFrame(detail_rows).to_csv(output_dir / "actionability_feature_audit.csv", index=False)
    return out


def feature_control_type(feature: str, proxy_cols: set[str]) -> str:
    employee = {
        "EmpJobInvolvement",
        "TrainingTimesLastYear",
        "ProjectCount",
        "Absences",
        "DaysLateLast30",
        "WorkAccident",
    }
    manager = {
        "EmpJobSatisfaction",
        "EmpEnvironmentSatisfaction",
        "EmpRelationshipSatisfaction",
        "EmpWorkLifeBalance",
        "EngagementSurvey",
        "AverageMonthlyHours",
    }
    organisation = {
        "EmpDepartment",
        "EmpJobRole",
        "EmpJobLevel",
        "Salary",
        "MonthlyIncome",
        "MonthlyRate",
        "DailyRate",
        "EmpHourlyRate",
        "SalaryBand",
        "SpecialProjectsCount",
        "PromotionLast5Years",
    }
    historical = {
        "ExperienceYearsAtThisCompany",
        "ExperienceYearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
        "TotalWorkExperienceInYears",
        "NumCompaniesWorked",
    }
    if feature == "LastEvaluation":
        return "evaluation_proxy"
    if feature in employee:
        return "employee_controllable"
    if feature in manager:
        return "manager_controllable"
    if feature in organisation:
        return "organisation_controllable"
    if feature in historical or feature in proxy_cols:
        return "proxy_or_historical"
    return "unknown_or_context_dependent"


def actionability_status(employee_count: int, manager_org_count: int, proxy_count: int, top_n: int) -> str:
    if not top_n:
        return "not_available"
    if employee_count / top_n >= 0.50:
        return "partly_employee_actionable_with_warnings"
    if (manager_org_count + proxy_count) / top_n >= 0.50:
        return "mostly_manager_organisation_or_proxy_constrained"
    return "mixed_context_dependent"


def write_representative_cases(
    dataset: ExternalDataset,
    predictions: pd.DataFrame,
    shap_outputs: Dict[str, Dict[str, Path]],
    output_dir: Path,
    limit: int = 8,
) -> Path:
    rows: List[Dict[str, Any]] = []
    proxy_cols = set(role_columns(dataset, "proxy"))
    for policy, group in predictions.groupby("policy"):
        selected: List[Tuple[str, int]] = []
        correct = group[group["correct"]]
        wrong = group[~group["correct"]]
        if not correct.empty:
            selected.append(("correct_high_confidence", int(correct.sort_values("confidence", ascending=False).iloc[0]["sample_index"])))
            selected.append(("correct_low_confidence", int(correct.sort_values("confidence", ascending=True).iloc[0]["sample_index"])))
        if not wrong.empty:
            selected.append(("misclassification", int(wrong.sort_values("confidence", ascending=False).iloc[0]["sample_index"])))
        selected.extend(minority_group_cases(dataset, group))
        selected.extend(proxy_warning_cases(policy, shap_outputs.get(policy, {}), proxy_cols))

        seen = set()
        for case_type, sample_index in selected:
            if sample_index in seen:
                continue
            seen.add(sample_index)
            pred_row = group[group["sample_index"] == sample_index].iloc[0]
            rows.append(
                {
                    "dataset": dataset.config.dataset_name,
                    "policy": policy,
                    "case_type": case_type,
                    "sample_index": sample_index,
                    "external_sample_id": int(pred_row["external_sample_id"]),
                    "true_class": int(pred_row["y_true"]),
                    "predicted_class": int(pred_row["y_pred"]),
                    "confidence": float(pred_row["confidence"]),
                    "correct": bool(pred_row["correct"]),
                }
            )
            if len([r for r in rows if r["policy"] == policy]) >= limit:
                break
    out = output_dir / "representative_cases.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def minority_group_cases(dataset: ExternalDataset, prediction_group: pd.DataFrame) -> List[Tuple[str, int]]:
    cases: List[Tuple[str, int]] = []
    for attr in audit_attribute_columns(dataset):
        if attr not in dataset.canonical.columns:
            continue
        values = dataset.canonical.loc[prediction_group["sample_index"].astype(int), attr].astype("string").fillna("__MISSING__")
        counts = values.value_counts()
        if counts.empty or len(counts) < 2:
            continue
        group_value = counts.sort_values().index[0]
        idx = values[values == group_value].index
        if len(idx):
            cases.append((f"small_support_group_{attr}", int(idx[0])))
            break
    return cases


def proxy_warning_cases(policy: str, paths: Dict[str, Path], proxy_cols: set[str]) -> List[Tuple[str, int]]:
    local_path = paths.get("local")
    if local_path is None or not Path(local_path).exists() or not proxy_cols:
        return []
    local_df = pd.read_csv(local_path)
    proxy_df = local_df[local_df["feature"].isin(proxy_cols)].copy()
    if proxy_df.empty:
        return []
    row = proxy_df.sort_values("abs_grouped_shap_value", ascending=False).iloc[0]
    return [("proxy_warning_high_shap", int(row["sample_index"]))]


def write_experiment_interpretation(
    dataset: ExternalDataset,
    target_kind: str,
    metrics_df: pd.DataFrame,
    output_dir: Path,
    min_support: int,
) -> Path:
    out = output_dir / "external_experiment_interpretation.md"
    lines = [
        f"# External Experiment Interpretation: {dataset.config.display_name}",
        "",
        f"Dataset role: {dataset.config.recommended_role}",
        f"Task type: `{dataset.task_type}`",
        f"Target kind: `{target_kind}`",
        f"Minimum subgroup support threshold: {min_support}",
        "",
        "## Performance Summary",
        "",
    ]
    for row in metrics_df.itertuples(index=False):
        qwk = getattr(row, "quadratic_weighted_kappa", np.nan)
        lines.append(
            f"- `{row.policy}`: macro-F1={row.macro_f1:.4f}, balanced accuracy={row.balanced_accuracy:.4f}, "
            f"QWK={qwk:.4f}, log loss={row.nll_log_loss:.4f}, ECE={row.ece_confidence:.4f}."
        )
    lines.extend(["", "## Interpretation", ""])
    if dataset.config.dataset_name == "ibm_hr_analytics" and target_kind == "primary":
        lines.append(
            "IBM `PerformanceRating` contains a restricted target space in this run. Treat results as schema-compatible performance robustness, not direct 2/3/4 external validation."
        )
    elif dataset.config.dataset_name == "employee_turnover":
        lines.append(
            "This is HR task-transfer robustness for turnover prediction. It must not be described as performance external validation."
        )
    elif dataset.config.dataset_name == "hrdataset_v14":
        lines.append(
            "This supports independent replication on a mappable external performance target, subject to dataset provenance and sample-size limitations."
        )
    lines.extend(
        [
            "",
            "## Required Claim Limits",
            "",
            "- The model is research-grade decision support only.",
            "- Full-feature or leakage-risk variables are not deployable final-model evidence.",
            "- SHAP is attribution, not causality.",
            "- Counterfactual/actionability outputs are not employee prescriptions.",
            "- Removing sensitive or group variables does not prove fairness.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def run_hrdataset_cross_dataset_feasibility(seed: int = DEFAULT_SEED) -> Dict[str, Path]:
    output_dir = EXTERNAL_REPORT_ROOT / "hrdataset_v14" / "cross_dataset_inx_to_hrdataset"
    ensure_dir(output_dir)
    hr = load_external_dataset("hrdataset_v14")
    hr_features = set(build_feature_columns(hr, "department_free"))

    inx_df = load_validated_or_raw_data()
    inx_X_raw, inx_y = split_features_and_target(inx_df, drop_sensitive=True)
    inx_X = apply_feature_set(inx_X_raw.copy(), PRIMARY_INX_FEATURE_SET)
    common = sorted(set(inx_X.columns).intersection(hr_features))
    all_overlap_rows = []
    for feature in sorted(set(inx_X.columns).union(hr_features)):
        all_overlap_rows.append(
            {
                "feature": feature,
                "in_inx_primary": feature in inx_X.columns,
                "in_hrdataset_department_free": feature in hr_features,
                "common": feature in common,
            }
        )
    overlap_path = output_dir / "feature_overlap.csv"
    pd.DataFrame(all_overlap_rows).to_csv(overlap_path, index=False)

    feasible = len(common) >= 5 and set(hr.labels) == {2, 3, 4}
    result_path = output_dir / "cross_dataset_validation.md"
    metrics_path = output_dir / "cross_dataset_metrics.csv"
    if feasible:
        X_train = inx_X.loc[:, common].copy()
        y_train = inx_y.astype(int)
        X_test = hr.canonical.loc[:, common].copy()
        y_test = hr.canonical[hr.target_column].astype(int)
        pipeline = fit_xgb_pipeline(X_train, y_train, seed=seed)
        labels = [2, 3, 4]
        proba = align_proba(pipeline.predict_proba(X_test), pipeline.named_steps["model"].classes_, labels)
        pred = predict_labels_from_proba(proba, labels)
        metrics = classification_metrics(y_test, pred, proba, labels)
        pd.DataFrame([{**metrics, "n_common_features": len(common), "common_features": ";".join(common)}]).to_csv(metrics_path, index=False)
        lines = [
            "# INX-to-HRDataset Cross-Dataset Validation",
            "",
            "Status: completed with a limited common feature set.",
            f"Common features: {', '.join(common)}",
            "",
            "This result should be interpreted as a domain-shift stress test, not a definitive transportability estimate.",
        ]
    else:
        pd.DataFrame([{"status": "infeasible_or_too_limited", "n_common_features": len(common), "common_features": ";".join(common)}]).to_csv(metrics_path, index=False)
        lines = [
            "# INX-to-HRDataset Cross-Dataset Validation",
            "",
            "Status: reported as infeasible/too limited.",
            "",
            f"Common department-free safe features found: {len(common)}",
            f"Common features: {', '.join(common) if common else 'none'}",
            "",
            "The overlap is too weak for a scientifically defensible train-on-INX/test-on-HRDataset performance claim. Forcing this experiment would primarily measure schema mismatch rather than model transportability.",
            "",
            "Decision: do not claim cross-dataset external validation from this feature overlap. Use HRDataset_v14 as independent external replication instead.",
        ]
    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_registry_row(
        {
            "run_id": f"external_cross_dataset_feasibility_{utc_now_iso()}",
            "date_time": utc_now_iso(),
            "git_commit_if_available": get_git_commit(),
            "script": "python -m src.experiments.external_validation --cross-dataset",
            "config": "configs/feature_sets.yaml; data/external/hrdataset_v14/schema_mapping.json",
            "feature_set": PRIMARY_INX_FEATURE_SET,
            "model": MODEL_NAME if feasible else "not_trained",
            "seed": seed,
            "cv_strategy": "train_on_inx_test_on_hrdataset if feasible",
            "primary_metrics": {"feasible": feasible, "n_common_features": len(common)},
            "output_dir": _rel(output_dir),
            "notes": "Cross-dataset feasibility gate based on common safe canonical features.",
            "decision_status": "candidate",
        }
    )
    return {"feature_overlap": overlap_path, "metrics": metrics_path, "interpretation": result_path}


def run_external_llm_agent_evaluation(
    dataset_name: str,
    *,
    target_kind: str = "primary",
    policy: Optional[str] = None,
    limit: int = 5,
    provider: str = "openai",
    model: Optional[str] = None,
    require_real_llm: bool = True,
    agent_runtime: str = "openai-agents",
) -> Dict[str, Path]:
    dataset = load_external_dataset(dataset_name, target_kind=target_kind)
    output_dir = dataset_report_dir(dataset_name, target_kind)
    cases_path = output_dir / "representative_cases.csv"
    if not cases_path.exists():
        raise FileNotFoundError(f"Representative cases not found. Run experiment first: {cases_path}")
    cases = pd.read_csv(cases_path)
    selected_policy = policy or default_policy_for(dataset)
    cases = cases[cases["policy"] == selected_policy].head(limit)
    if len(cases) < limit:
        raise RuntimeError(f"Only {len(cases)} representative cases available for {dataset_name}/{selected_policy}; required {limit}.")

    runtime_config = LLMRuntimeConfig(
        provider=provider,  # type: ignore[arg-type]
        model=model or LLMRuntimeConfig.from_env().model,
        temperature=0.0,
        max_tokens=1200,
        require_real_llm=require_real_llm,
    )
    explainer = GovernedExplainer(runtime_config=runtime_config)
    agent_runner = OpenAIAgentsSDKGovernanceRuntime(runtime_config=runtime_config)
    llm_dir = output_dir / "llm_agent_governance"
    ensure_dir(llm_dir)

    eval_rows: List[Dict[str, Any]] = []
    for case in cases.itertuples(index=False):
        evidence = build_external_case_evidence(
            dataset=dataset,
            output_dir=output_dir,
            policy=selected_policy,
            sample_index=int(case.sample_index),
            case_type=str(case.case_type),
        )
        case_dir = llm_dir / f"case_{int(case.sample_index)}"
        ensure_dir(case_dir)
        evidence_path = case_dir / "structured_evidence.json"
        evidence_path.write_text(evidence.to_json(), encoding="utf-8")
        explanation = explainer.generate(evidence)
        explanation_path = case_dir / "governed_explanation.json"
        save_json(explanation, explanation_path)
        agent_payload = agent_runner.run(evidence)
        agent_path = case_dir / "openai_agents_governance_audit.json"
        save_json(agent_payload, agent_path)
        write_case_markdown(evidence, explanation, agent_payload, case_dir / "governed_explanation_and_agent_audit.md")
        faithfulness = explanation.get("faithfulness_check", {})
        agent_statuses = [
            str(item.get("status", "unknown"))
            for item in agent_payload.get("agent_syntheses", [])
            if isinstance(item, dict)
        ]
        accepted_agent_statuses = {"pass", "pass_with_warnings"}
        agent_governance_pass = bool(agent_statuses) and all(status in accepted_agent_statuses for status in agent_statuses)
        eval_rows.append(
            {
                "dataset": dataset_name,
                "target_kind": target_kind,
                "policy": selected_policy,
                "case_id": evidence.prediction.case_id,
                "case_type": str(case.case_type),
                "model_provider": provider,
                "model": runtime_config.model,
                "faithfulness_pass": bool(faithfulness.get("faithfulness_pass", False)),
                "unsupported_claim_count": len(faithfulness.get("unsupported_claims", [])),
                "forbidden_claim_count": len(faithfulness.get("forbidden_claims", [])),
                "missing_warning_count": len(faithfulness.get("missing_warnings", [])),
                "agent_runtime_success": bool(agent_payload.get("supervisor_synthesis")),
                "agent_governance_pass": agent_governance_pass,
                "agent_failure_count": sum(1 for status in agent_statuses if status not in accepted_agent_statuses),
                "warning_count": len(explanation.get("warnings", [])),
            }
        )
        print(f"[external-llm] dataset={dataset_name} policy={selected_policy} case={case.sample_index}")

    eval_df = pd.DataFrame(eval_rows)
    eval_path = llm_dir / "external_llm_agent_eval.csv"
    eval_df.to_csv(eval_path, index=False)
    summary = {
        "dataset": dataset_name,
        "target_kind": target_kind,
        "policy": selected_policy,
        "n_cases": int(len(eval_df)),
        "case_ids": ";".join(eval_df["case_id"].astype(str).tolist()),
        "model_provider": provider,
        "model": runtime_config.model,
        "faithfulness_pass_rate": float(eval_df["faithfulness_pass"].mean()),
        "unsupported_claim_rate": float((eval_df["unsupported_claim_count"] > 0).mean()),
        "forbidden_claim_rate": float((eval_df["forbidden_claim_count"] > 0).mean()),
        "missing_warning_rate": float((eval_df["missing_warning_count"] > 0).mean()),
        "agent_runtime_success_rate": float(eval_df["agent_runtime_success"].mean()),
        "agent_success_rate": float(eval_df["agent_governance_pass"].mean()),
        "agent_failure_rate": float((eval_df["agent_failure_count"] > 0).mean()),
    }
    summary_path = llm_dir / "external_llm_agent_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    summary_md = llm_dir / "external_llm_agent_summary.md"
    write_llm_summary_markdown(summary, summary_md)
    append_registry_row(
        {
            "run_id": f"external_llm_agent_{dataset_name}_{target_kind}_{utc_now_iso()}",
            "date_time": utc_now_iso(),
            "git_commit_if_available": get_git_commit(),
            "script": "python -m src.experiments.external_validation --run-llm",
            "config": f"data/external/{dataset_name}/schema_mapping.json",
            "feature_set": selected_policy,
            "model": runtime_config.model,
            "seed": "temperature=0.0",
            "cv_strategy": "not_applicable",
            "primary_metrics": summary,
            "output_dir": _rel(llm_dir),
            "notes": "Real OpenAI governed explanations and OpenAI Agents SDK audits for external representative cases.",
            "decision_status": "candidate",
        }
    )
    return {"eval": eval_path, "summary": summary_path, "summary_md": summary_md}


def build_external_case_evidence(
    *,
    dataset: ExternalDataset,
    output_dir: Path,
    policy: str,
    sample_index: int,
    case_type: str,
) -> CompleteCaseEvidence:
    predictions = pd.read_csv(output_dir / "model_predictions.csv")
    pred_row = predictions[(predictions["policy"] == policy) & (predictions["sample_index"] == sample_index)].iloc[0]
    labels = dataset.labels
    probabilities = {
        str(label): float(pred_row[f"prob_class_{label}"])
        for label in labels
        if f"prob_class_{label}" in pred_row.index
    }
    prediction = PredictionEvidence(
        case_id=f"{dataset.config.dataset_name}_{policy}_{sample_index}",
        predicted_class=int(pred_row["y_pred"]),
        true_class=int(pred_row["y_true"]),
        class_probabilities=probabilities,
        confidence=float(pred_row["confidence"]),
        uncertainty_flag=bool(float(pred_row["confidence"]) < 0.60),
        model_name=MODEL_NAME,
        feature_policy=policy,
        leakage_safe_status="external_research_candidate",
    )

    local_path = output_dir / "shap" / policy / "local_grouped_shap_values.csv"
    local_df = pd.read_csv(local_path)
    local_case = local_df[local_df["sample_index"] == sample_index].copy()
    local_case = local_case.sort_values("abs_grouped_shap_value", ascending=False)
    positive = local_case[local_case["grouped_shap_value"] >= 0].head(5)
    negative = local_case[local_case["grouped_shap_value"] < 0].head(5)
    grouped = {
        str(row.feature): float(row.grouped_shap_value)
        for row in local_case.head(12).itertuples(index=False)
    }
    shap = ShapEvidence(
        top_positive_features=[
            {"feature": str(row.feature), "grouped_shap_value": float(row.grouped_shap_value)}
            for row in positive.itertuples(index=False)
        ],
        top_negative_features=[
            {"feature": str(row.feature), "grouped_shap_value": float(row.grouped_shap_value)}
            for row in negative.itertuples(index=False)
        ],
        grouped_shap_values=grouped,
        class_specific_shap_values=grouped,
        shap_stability_summary={"external_case_type": case_type},
        explanation_stability_warning="External SHAP values are model attributions, not causal effects.",
    )

    fairness_df = _read_optional_csv(output_dir / "fairness_proxy" / "fairness_disparity_summary.csv")
    small_df = _read_optional_csv(output_dir / "fairness_proxy" / "small_group_warnings.csv")
    fairness = FairnessEvidence(
        audited_groups=audit_attribute_columns(dataset),
        subgroup_metrics={},
        disparity_gaps=_top_disparity_payload(fairness_df, policy),
        bootstrap_ci={},
        low_support_warnings=small_df.head(10).to_dict(orient="records") if not small_df.empty else [],
        proxy_risk_warnings=[
            "Removing direct sensitive or group variables does not prove fairness.",
            "Department, job role, salary, tenure, and evaluation-history fields may act as proxies depending on dataset context.",
        ],
    )

    metrics_df = pd.read_csv(output_dir / "performance_metrics.csv")
    metric_row = metrics_df[metrics_df["policy"] == policy].iloc[0]
    calibration = CalibrationEvidence(
        log_loss=safe_float(metric_row.get("nll_log_loss")),
        brier_score=safe_float(metric_row.get("multiclass_brier")),
        expected_calibration_error=safe_float(metric_row.get("ece_confidence")),
        calibration_warning="External probabilities are diagnostic confidence estimates and require calibration caution.",
    )

    action_df = _read_optional_csv(output_dir / "actionability_summary.csv")
    action_row = action_df[action_df["policy"] == policy].iloc[0] if not action_df.empty and len(action_df[action_df["policy"] == policy]) else pd.Series(dtype=object)
    counterfactual = CounterfactualEvidence(
        counterfactual_mode="external_actionability_classification",
        validity=safe_float(action_row.get("employee_controllable_share")),
        changed_features=[],
        probability_gain=None,
        proximity_cost=None,
        actionability_label=str(action_row.get("actionability_status", "not_available")),
        failed_reason="Prototype counterfactual recourse was not claimed for this external evidence package.",
        warning="Counterfactuals and actionability summaries are model scenarios, not employee prescriptions.",
    )

    leakage = LeakageEvidence(
        feature_policy=policy,
        excluded_leakage_features=role_columns(dataset, "leakage"),
        full_feature_score=None,
        leakage_safe_score=safe_float(metric_row.get("macro_f1")),
        leakage_sensitivity_index=None,
        leakage_warning="External models exclude target, identifiers, direct sensitive/audit-only fields, and mapped leakage-risk columns for the selected policy.",
    )
    governance = GovernanceEvidence(
        intended_use="Research-grade external validation and robustness evidence for HR XAI governance.",
        prohibited_use="No autonomous hiring, firing, promotion, compensation, discipline, or individual employment decisions.",
        model_card_summary=external_model_card_summary(dataset, policy),
        deployment_status="research_only_external_validation_or_robustness",
        required_warnings=[
            "SHAP is attribution, not causality.",
            "Prediction requires human review.",
            "Not for autonomous HR decisions.",
            "Removing group variables does not prove fairness.",
            "Counterfactuals may not be employee-actionable.",
            "External dataset role limits must be respected.",
        ],
    )
    return CompleteCaseEvidence(
        prediction=prediction,
        shap=shap,
        fairness=fairness,
        calibration=calibration,
        counterfactual=counterfactual,
        leakage=leakage,
        governance=governance,
        evidence_sources=[
            str(output_dir / "performance_metrics.csv"),
            str(local_path),
            str(output_dir / "fairness_proxy" / "fairness_disparity_summary.csv"),
            str(output_dir / "actionability_summary.csv"),
        ],
    )


def write_case_markdown(
    evidence: CompleteCaseEvidence,
    explanation: Dict[str, Any],
    agent_payload: Dict[str, Any],
    output_path: Path,
) -> None:
    lines = [
        f"# External Governed Explanation and Agent Audit: {evidence.prediction.case_id}",
        "",
        "## Prediction Evidence",
        "",
        f"- Predicted class: {evidence.prediction.predicted_class}",
        f"- True class: {evidence.prediction.true_class}",
        f"- Confidence: {evidence.prediction.confidence}",
        f"- Feature policy: `{evidence.prediction.feature_policy}`",
        "",
        "## Governed Explanation",
        "",
        str(explanation.get("short_explanation", "")),
        "",
        "## Agent Supervisor Status",
        "",
        str(agent_payload.get("supervisor_synthesis", {}).get("overall_status", "unknown")),
        "",
        "## Warnings",
        "",
    ]
    for warning in explanation.get("warnings", []):
        lines.append(f"- {warning.get('message')}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_llm_summary_markdown(summary: Dict[str, Any], output_path: Path) -> None:
    lines = [
        "# External LLM-Agent Governance Summary",
        "",
        f"Dataset: `{summary['dataset']}`",
        f"Policy: `{summary['policy']}`",
        f"Cases: {summary['n_cases']} ({summary['case_ids']})",
        f"Model/provider: {summary['model_provider']} / {summary['model']}",
        "",
        "## Results",
        "",
        f"- Faithfulness pass rate: {summary['faithfulness_pass_rate']:.6f}",
        f"- Unsupported claim rate: {summary['unsupported_claim_rate']:.6f}",
        f"- Forbidden claim rate: {summary['forbidden_claim_rate']:.6f}",
        f"- Missing warning rate: {summary['missing_warning_rate']:.6f}",
        f"- Agent success rate: {summary['agent_success_rate']:.6f}",
        "",
        "## Claim Limits",
        "",
        "- Small-batch OpenAI evaluation is technical governance evidence, not human-subject validation.",
        "- The LLM interprets structured evidence and is not the performance predictor.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_policy_for(dataset: ExternalDataset) -> str:
    if "department_free" in dataset.config.feature_policy_variants:
        return "department_free"
    if "without_last_evaluation" in dataset.config.feature_policy_variants:
        return "without_last_evaluation"
    return next(iter(dataset.config.feature_policy_variants))


def _top_disparity_payload(disparity_df: pd.DataFrame, policy: str) -> Dict[str, Any]:
    if disparity_df.empty:
        return {}
    subset = disparity_df[disparity_df["feature_set"] == policy] if "feature_set" in disparity_df.columns else disparity_df
    if subset.empty or "max_gap" not in subset.columns:
        return {}
    top = subset.sort_values("max_gap", ascending=False).head(5)
    return {
        f"{row.attribute}_{row.metric}_{getattr(row, 'class_label', '')}": safe_float(row.max_gap)
        for row in top.itertuples(index=False)
    }


def external_model_card_summary(dataset: ExternalDataset, policy: str) -> str:
    if dataset.config.dataset_name == "hrdataset_v14":
        role = "independent external performance replication"
    elif dataset.config.dataset_name == "ibm_hr_analytics":
        role = "schema-compatible robustness with restricted performance target space"
    else:
        role = "related HR task-transfer robustness, not performance validation"
    return (
        f"External dataset {dataset.config.display_name} is used for {role}. "
        f"Policy {policy} excludes mapped leakage-risk and sensitive/audit-only fields. "
        "The system is research-only decision support and cannot make autonomous HR decisions."
    )


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def external_min_support(n_rows: int) -> int:
    return max(10, min(30, int(round(n_rows * 0.05))))


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(SETTINGS.project_root))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external validation and robustness experiments.")
    parser.add_argument("--dataset", default="hrdataset_v14")
    parser.add_argument("--target-kind", default="primary", choices=["primary", "attrition"])
    parser.add_argument("--policies", default="all")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cross-dataset", action="store_true")
    parser.add_argument("--run-llm", action="store_true")
    parser.add_argument("--llm-limit", type=int, default=5)
    parser.add_argument("--llm-policy", default=None)
    parser.add_argument("--provider", default="openai", choices=["openai"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--require-real-llm", action="store_true", default=True)
    parser.add_argument("--no-registry", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cross_dataset:
        print(run_hrdataset_cross_dataset_feasibility(seed=args.seed))
    elif args.run_llm:
        print(
            run_external_llm_agent_evaluation(
                args.dataset,
                target_kind=args.target_kind,
                policy=args.llm_policy,
                limit=args.llm_limit,
                provider=args.provider,
                model=args.model,
                require_real_llm=args.require_real_llm,
            )
        )
    else:
        dataset = load_external_dataset(args.dataset, target_kind=args.target_kind)
        policies = None if args.policies == "all" else [part.strip() for part in args.policies.split(",") if part.strip()]
        print(
            run_external_dataset_experiment(
                args.dataset,
                target_kind=args.target_kind,
                policies=policies,
                n_splits=args.n_splits,
                seed=args.seed,
                write_registry=not args.no_registry,
            )
        )
