from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.experiments.final_evidence_common import (
    CALIBRATION_DIR,
    COUNTERFACTUAL_DIR,
    FAIRNESS_DIR,
    FINAL_FEATURE_SETS,
    LABELS,
    MODEL_CARD_DIR,
    MODEL_NAME,
    MODEL_SELECTION_DIR,
    PRIMARY_FEATURE_SET,
    XAI_DIR,
    append_task_registry,
    dashboard_required_columns,
    ensure_dir,
    save_json,
)
from src.experiments.final_fairness_bootstrap import get_ci_row, validate_or_create_oof_predictions
from src.models.evaluate import classification_metrics
from src.utils.experiment_registry import utc_now_iso


def oof_performance() -> pd.DataFrame:
    predictions = pd.read_csv(validate_or_create_oof_predictions(FAIRNESS_DIR))
    rows = []
    for feature_set, group in predictions.groupby("feature_set"):
        proba = group[[f"prob_class_{label}" for label in LABELS]].to_numpy(dtype=float)
        rows.append({"feature_set": feature_set, **classification_metrics(group["y_true"], group["y_pred"], proba, LABELS)})
    return pd.DataFrame(rows)


def best_calibration_rows() -> pd.DataFrame:
    path = CALIBRATION_DIR / "calibration_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for feature_set, group in df.groupby("feature_set"):
        ranked = group.assign(
            rank_sum=group["nll_log_loss_mean"].rank()
            + group["multiclass_brier_mean"].rank()
            + group["ece_confidence_mean"].rank()
        ).sort_values("rank_sum")
        rows.append(ranked.iloc[0])
    return pd.DataFrame(rows)


def proxy_values(feature_set: str) -> Dict[str, Any]:
    if feature_set == "no_salary_hike_no_attrition":
        return {
            "department_proxy_macro_f1": 1.0,
            "department_proxy_balanced_accuracy": 1.0,
            "proxy_status": "EmpDepartment present directly",
            "emp_job_role_present": True,
        }
    if feature_set == "no_salary_hike_no_attrition_no_department":
        path = Path("reports/fairness/proxy_with_job_role/proxy_analysis_department_cv_summary.csv")
        present = True
    else:
        path = Path("reports/fairness/proxy_no_job_role/proxy_analysis_department_cv_summary.csv")
        present = False
    if not path.exists():
        return {
            "department_proxy_macro_f1": np.nan,
            "department_proxy_balanced_accuracy": np.nan,
            "proxy_status": "proxy output missing",
            "emp_job_role_present": present,
        }
    df = pd.read_csv(path).set_index("metric")
    return {
        "department_proxy_macro_f1": float(df.loc["macro_f1", "mean"]),
        "department_proxy_balanced_accuracy": float(df.loc["balanced_accuracy", "mean"]),
        "proxy_status": "department reconstructability measured",
        "emp_job_role_present": present,
    }


def build_dashboard() -> pd.DataFrame:
    perf = oof_performance().set_index("feature_set")
    cal = best_calibration_rows().set_index("feature_set") if (CALIBRATION_DIR / "calibration_summary.csv").exists() else pd.DataFrame()
    ci = pd.read_csv(FAIRNESS_DIR / "bootstrap_disparity_ci.csv") if (FAIRNESS_DIR / "bootstrap_disparity_ci.csv").exists() else pd.DataFrame()
    shap_summary = pd.read_csv(XAI_DIR / "shap_stability_summary.csv") if (XAI_DIR / "shap_stability_summary.csv").exists() else pd.DataFrame()
    cf = pd.read_csv(COUNTERFACTUAL_DIR / "actionability_summary.csv") if (COUNTERFACTUAL_DIR / "actionability_summary.csv").exists() else pd.DataFrame()
    rows = []
    for feature_set in FINAL_FEATURE_SETS:
        p = perf.loc[feature_set]
        c: Optional[pd.Series] = cal.loc[feature_set] if not cal.empty and feature_set in cal.index else None
        dept = get_ci_row(ci, feature_set, "EmpDepartment", "macro_f1") if not ci.empty else None
        shap_top10 = None
        if not shap_summary.empty:
            subset = shap_summary[(shap_summary["feature_set"] == feature_set) & (shap_summary["top_k"] == 10)]
            if len(subset):
                shap_top10 = subset.iloc[0]
        cf_map = {}
        if not cf.empty:
            cf_map = cf[cf["feature_set"] == feature_set].set_index("intervention_mode")["validity_rate"].to_dict()
        proxy = proxy_values(feature_set)
        if feature_set == PRIMARY_FEATURE_SET:
            role = "recommended primary research model candidate"
            category = "recommended primary research model"
        elif feature_set == "no_salary_hike_no_attrition":
            role = "main leakage-safe comparison baseline"
            category = "diagnostic comparison baseline"
        else:
            role = "strict fairness/proxy sensitivity baseline"
            category = "strict fairness/proxy sensitivity model"
        warnings = [
            "decision support only",
            "no causal SHAP claims",
            "external validation required",
            "probability calibration warning",
        ]
        if feature_set != "no_salary_hike_no_attrition_no_department_no_job_role":
            warnings.append("department or job-role proxy risk")
        rows.append(
            {
                "feature_set": feature_set,
                "model": MODEL_NAME,
                "role": role,
                "macro_f1": p.get("macro_f1"),
                "balanced_accuracy": p.get("balanced_accuracy"),
                "qwk": p.get("quadratic_weighted_kappa"),
                "ordinal_mae": p.get("ordinal_mae"),
                "severe_error_rate": p.get("severe_error_rate"),
                "adjacent_accuracy": p.get("adjacent_accuracy"),
                "calibration_method": c["method"] if c is not None else "missing",
                "log_loss": c["nll_log_loss_mean"] if c is not None else np.nan,
                "multiclass_brier": c["multiclass_brier_mean"] if c is not None else np.nan,
                "ece": c["ece_confidence_mean"] if c is not None else np.nan,
                "empdepartment_macro_f1_gap": dept["point_estimate"] if dept is not None else np.nan,
                "empdepartment_macro_f1_gap_ci_low": dept["ci_low"] if dept is not None else np.nan,
                "empdepartment_macro_f1_gap_ci_high": dept["ci_high"] if dept is not None else np.nan,
                "department_proxy_macro_f1": proxy["department_proxy_macro_f1"],
                "department_proxy_balanced_accuracy": proxy["department_proxy_balanced_accuracy"],
                "proxy_status": proxy["proxy_status"],
                "emp_job_role_present": proxy["emp_job_role_present"],
                "top10_shap_jaccard": shap_top10["mean_jaccard"] if shap_top10 is not None else np.nan,
                "shap_spearman": shap_top10["mean_spearman"] if shap_top10 is not None else np.nan,
                "employee_only_validity": cf_map.get("employee_only", np.nan),
                "employee_manager_validity": cf_map.get("employee_manager", np.nan),
                "organization_allowed_validity": cf_map.get("organization_allowed", np.nan),
                "full_default_validity": cf_map.get("full_default", np.nan),
                "no_salary_validity": cf_map.get("no_salary", np.nan),
                "salary_hike_excluded": True,
                "attrition_excluded": True,
                "age_excluded": True,
                "leakage_safe": True,
                "required_warnings": "; ".join(warnings),
                "recommendation_category": category,
            }
        )
    dashboard = pd.DataFrame(rows)
    missing = [col for col in dashboard_required_columns() if col not in dashboard.columns]
    if missing:
        raise ValueError(f"Dashboard missing required columns: {missing}")
    return dashboard


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def write_recommendation(dashboard: pd.DataFrame, output_path: Path) -> None:
    primary = dashboard[dashboard["feature_set"] == PRIMARY_FEATURE_SET].iloc[0]
    baseline = dashboard[dashboard["feature_set"] == "no_salary_hike_no_attrition"].iloc[0]
    strict = dashboard[dashboard["feature_set"] == "no_salary_hike_no_attrition_no_department_no_job_role"].iloc[0]
    lines = [
        "# Final Model Recommendation",
        "",
        "## Recommendation Categories",
        "",
        f"- Recommended primary research model: `{PRIMARY_FEATURE_SET}` + XGBoost.",
        "- Main leakage-safe comparison baseline: `no_salary_hike_no_attrition` + XGBoost.",
        "- Strict fairness/proxy sensitivity model: `no_salary_hike_no_attrition_no_department_no_job_role` + XGBoost.",
        "- Full-feature models: historical leakage-warning upper-bound only, not deployable final models.",
        "",
        "## Scientific Rationale",
        "",
        f"The primary candidate preserves most leakage-safe utility (macro-F1 {primary['macro_f1']:.4f}, QWK {primary['qwk']:.4f}) while excluding direct department membership. It does not eliminate proxy risk because EmpJobRole remains present.",
        "",
        f"The department-including baseline has slightly higher macro-F1 ({baseline['macro_f1']:.4f}) and QWK ({baseline['qwk']:.4f}) but directly uses EmpDepartment, so it is a diagnostic comparison baseline.",
        "",
        f"The strict job-role-free model lowers department proxy reconstructability but reduces macro-F1 to {strict['macro_f1']:.4f} and QWK to {strict['qwk']:.4f}. It is a strict sensitivity model unless the researcher chooses proxy minimization over utility.",
        "",
        "## Required Warnings",
        "",
        "- The model is decision support only, not an autonomous employee evaluator.",
        "- Removing EmpDepartment does not prove fairness.",
        "- EmpJobRole may proxy EmpDepartment and requires reason-code warnings.",
        "- SHAP is attribution, not causality.",
        "- Counterfactuals may require manager or organisation intervention.",
        "- Probability estimates require calibration warnings or probability bands.",
        "- External validation is required before operational use.",
        "",
        "## Composite Score Policy",
        "",
        "No single composite G-XAIR score is used as the sole decision rule. Components remain visible because performance, proxy risk, calibration, actionability, and subgroup gaps trade off against one another.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dashboard() -> Dict[str, Path]:
    ensure_dir(MODEL_SELECTION_DIR)
    dashboard = build_dashboard()
    csv_path = MODEL_SELECTION_DIR / "final_candidate_dashboard.csv"
    md_path = MODEL_SELECTION_DIR / "final_candidate_dashboard.md"
    recommendation_path = MODEL_SELECTION_DIR / "final_recommendation.md"
    rationale_path = MODEL_SELECTION_DIR / "model_selection_rationale.json"
    dashboard.to_csv(csv_path, index=False)
    md_path.write_text("# Final Candidate Dashboard\n\n" + dataframe_to_markdown(dashboard) + "\n", encoding="utf-8")
    write_recommendation(dashboard, recommendation_path)
    save_json(
        {
            "recommended_primary_research_model": PRIMARY_FEATURE_SET,
            "model_family": MODEL_NAME,
            "decision_rule": "multi-criteria governance review; no single composite score used as sole selection criterion",
            "candidate_roles": dashboard[["feature_set", "role", "recommendation_category", "required_warnings"]].to_dict(orient="records"),
        },
        rationale_path,
    )
    append_task_registry(
        run_id=f"final_model_selection_dashboard_{utc_now_iso()}",
        script="python -m src.experiments.final_model_selection dashboard",
        feature_set="; ".join(FINAL_FEATURE_SETS),
        primary_metrics="performance; calibration; fairness CI; proxy risk; SHAP stability; actionability; leakage safety",
        output_dir=MODEL_SELECTION_DIR,
        notes="Final multi-criteria dashboard and nuanced model recommendation for XGBoost final candidates.",
    )
    return {"dashboard_csv": csv_path, "dashboard_md": md_path, "recommendation": recommendation_path, "rationale": rationale_path}


def run_model_card() -> Dict[str, Path]:
    ensure_dir(MODEL_CARD_DIR)
    dashboard_path = MODEL_SELECTION_DIR / "final_candidate_dashboard.csv"
    dashboard = pd.read_csv(dashboard_path) if dashboard_path.exists() else build_dashboard()
    primary = dashboard[dashboard["feature_set"] == PRIMARY_FEATURE_SET].iloc[0]
    path = MODEL_CARD_DIR / "hr_xai_model_card.md"
    lines = [
        "# HR XAI Model Card",
        "",
        "## Model Name",
        f"XGBoost HR performance decision-support model using `{PRIMARY_FEATURE_SET}`.",
        "",
        "## Intended Use",
        "Research-grade decision support for auditing employee-performance prediction under leakage, fairness, explanation, calibration, and actionability constraints. Human review is required.",
        "",
        "## Prohibited Use",
        "This is not an autonomous employee evaluator. It should not be used for hiring, firing, compensation, promotion, disciplinary action, or individual employment decisions without independent validation, legal review, and governance approval.",
        "",
        "## Dataset and Target",
        "Public cross-sectional INX employee performance dataset. Target is ordinal `PerformanceRating` with classes 2, 3, and 4. Causal claims are not supported.",
        "",
        "## Feature Exclusions and Leakage Policy",
        "Age, Gender, MaritalStatus, EmpLastSalaryHikePercent, Attrition, EmpDepartment, EmpNumber, and PerformanceRating are excluded from the primary candidate input. Full-feature models are leakage-warning upper-bound baselines only.",
        "",
        "## Model Family and Evaluation Protocol",
        "XGBoost multiclass classifier with fold-safe one-hot preprocessing. Evidence uses config-backed CV/OOF predictions and final-candidate scripts under `src/experiments/`.",
        "",
        "## Performance Summary",
        f"Macro-F1: {primary['macro_f1']:.4f}; balanced accuracy: {primary['balanced_accuracy']:.4f}; QWK: {primary['qwk']:.4f}; ordinal MAE: {primary['ordinal_mae']:.4f}; severe error rate: {primary['severe_error_rate']:.4f}.",
        "",
        "## Calibration Summary",
        f"Dashboard calibration method: `{primary['calibration_method']}`. Log loss: {primary['log_loss']:.4f}; Brier: {primary['multiclass_brier']:.4f}; ECE: {primary['ece']:.4f}. Probability bands and warnings are recommended.",
        "",
        "## Fairness and Proxy-Risk Summary",
        f"EmpDepartment macro-F1 gap: {primary['empdepartment_macro_f1_gap']:.4f}. Department proxy macro-F1: {primary['department_proxy_macro_f1']:.4f}. EmpJobRole remains present and may proxy department. Removing EmpDepartment does not prove fairness.",
        "",
        "## SHAP and Explanation Summary",
        f"Top-10 grouped SHAP Jaccard: {primary['top10_shap_jaccard']:.4f}; Spearman rank stability: {primary['shap_spearman']:.4f}. SHAP is attribution, not causality.",
        "",
        "## Counterfactual and Actionability Summary",
        f"Employee-only validity: {primary['employee_only_validity']:.4f}; employee+manager validity: {primary['employee_manager_validity']:.4f}; organization-allowed validity: {primary['organization_allowed_validity']:.4f}. Counterfactuals are intervention hypotheses and may require manager or organisation action.",
        "",
        "## Known Limitations",
        "Public cross-sectional data, no external validation, possible organisational proxy effects, imperfect probability calibration, sparse class-4 support, and no causal identification.",
        "",
        "## Ethical and Governance Warnings",
        "Decision support only; no autonomous evaluation; no causal SHAP claims; no proof of fairness; proxy risk remains; external validation required before deployment.",
        "",
        "## Artifact Paths",
        "- `reports/model_selection/final_candidate_dashboard.csv`",
        "- `reports/model_selection/final_recommendation.md`",
        "- `reports/fairness/feature_set_sensitivity/bootstrap_disparity_ci.csv`",
        "- `reports/calibration/final_candidates/calibration_summary.csv`",
        "- `reports/xai/final_candidates/shap_stability_summary.csv`",
        "- `reports/counterfactuals/final_candidates/actionability_summary.csv`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_task_registry(
        run_id=f"final_model_card_{utc_now_iso()}",
        script="python -m src.experiments.final_model_selection model-card",
        feature_set=PRIMARY_FEATURE_SET,
        primary_metrics="model governance card",
        output_dir=MODEL_CARD_DIR,
        notes="Generated manuscript-ready governance model card for the recommended primary candidate.",
    )
    return {"model_card": path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final dashboard, recommendation, and model card.")
    parser.add_argument("command", choices=["dashboard", "model-card", "all"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "dashboard":
        print(run_dashboard())
    elif args.command == "model-card":
        print(run_model_card())
    else:
        print(run_dashboard())
        print(run_model_card())

