from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


OUTPUT_DIR = SETTINGS.reports_dir / "governance_reports"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def load_primary_dashboard() -> pd.Series:
    path = SETTINGS.reports_dir / "model_selection" / "final_candidate_dashboard.csv"
    df = pd.read_csv(path)
    return df[df["feature_set"] == "no_salary_hike_no_attrition_no_department"].iloc[0]


def load_llm_eval() -> Dict[str, float]:
    eval_path = SETTINGS.reports_dir / "llm_explanations" / "llm_governance_eval_summary.csv"
    if eval_path.exists():
        row = pd.read_csv(eval_path).iloc[0].to_dict()
        return {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}
    detail_path = SETTINGS.reports_dir / "llm_explanations" / "governed_explanation_eval.csv"
    if detail_path.exists():
        df = pd.read_csv(detail_path)
        return {
            "faithfulness_pass_rate": float(df["faithfulness_pass"].mean()),
            "unsupported_claim_rate": float((df["unsupported_claim_count"] > 0).mean()),
            "forbidden_claim_rate": float((df["forbidden_claim_count"] > 0).mean()),
            "missing_warning_rate": float((df["missing_warning_count"] > 0).mean()),
        }
    return {}


def compute_component_scores() -> pd.DataFrame:
    row = load_primary_dashboard()
    llm_eval = load_llm_eval()
    performance = clamp01((row["macro_f1"] + row["balanced_accuracy"] + row["qwk"] + (1 - row["ordinal_mae"])) / 4)
    leakage = 1.0 if row["salary_hike_excluded"] and row["attrition_excluded"] else 0.0
    explanation = clamp01((row["top10_shap_jaccard"] + row["shap_spearman"]) / 2)
    calibration = clamp01(1 - ((row["ece"] + row["multiclass_brier"]) / 2))
    fairness = clamp01(1 - row["empdepartment_macro_f1_gap"])
    actionability = clamp01((row["employee_only_validity"] + row["employee_manager_validity"] + row["organization_allowed_validity"]) / 3)
    proxy_penalty = clamp01(1 - row["department_proxy_macro_f1"])
    llm_score = clamp01(
        llm_eval.get("faithfulness_pass_rate", 0.0)
        * (1 - llm_eval.get("unsupported_claim_rate", 1.0))
        * (1 - llm_eval.get("forbidden_claim_rate", 1.0))
        * (1 - llm_eval.get("missing_warning_rate", 1.0))
    )
    rows = [
        ("Performance Adequacy Score", performance, "Macro-F1, balanced accuracy, QWK, ordinal MAE."),
        ("Leakage Robustness Score", leakage, "Salary hike and attrition exclusion status."),
        ("Explanation Stability Score", explanation, "Top-k Jaccard and Spearman SHAP stability."),
        ("Calibration Reliability Score", calibration, "ECE and Brier transformed so higher is better."),
        ("Fairness Robustness Score", fairness, "Support-filtered EmpDepartment macro-F1 gap transformed so higher is better."),
        ("Actionability Realism Score", actionability, "Employee, manager, and organisation-validity rates."),
        ("Proxy Risk Penalty Component", proxy_penalty, "Inverse of department reconstruction macro-F1; low is risky."),
        ("LLM Governance Compliance Score", llm_score, "Faithfulness pass, unsupported claim, forbidden claim, and missing warning rates."),
    ]
    return pd.DataFrame(rows, columns=["component", "score", "explanation"])


def readiness_label(scores: pd.DataFrame) -> str:
    score_map = scores.set_index("component")["score"].to_dict()
    if score_map.get("LLM Governance Compliance Score", 0) < 0.80:
        return "research_only_llm_layer_needs_review"
    if score_map.get("Proxy Risk Penalty Component", 0) < 0.30:
        return "research_only_proxy_risk_high"
    return "decision_support_with_strong_warnings"


def run() -> Dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = compute_component_scores()
    label = readiness_label(scores)
    csv_path = OUTPUT_DIR / "gxair_llm_agent_dashboard.csv"
    scores["overall_readiness_label"] = label
    scores.to_csv(csv_path, index=False)
    json_path = OUTPUT_DIR / "gxair_llm_agent_dashboard.json"
    json_path.write_text(json.dumps({"overall_readiness_label": label, "components": scores.to_dict(orient="records")}, indent=2), encoding="utf-8")
    append_registry_row(
        {
            "run_id": f"gxair_llm_agent_dashboard_{utc_now_iso()}",
            "date_time": utc_now_iso(),
            "git_commit_if_available": get_git_commit(),
            "script": "python -m src.governance.gxair_score",
            "config": "final dashboard + LLM governance eval",
            "feature_set": "no_salary_hike_no_attrition_no_department",
            "model": "gxair_dashboard",
            "seed": "deterministic",
            "cv_strategy": "not_applicable",
            "primary_metrics": "transparent G-XAIR component scores with LLM compliance",
            "output_dir": "reports/governance_reports",
            "notes": "Generated transparent G-XAIR dashboard extension with LLM governance component.",
            "decision_status": "candidate",
        }
    )
    return {"csv": str(csv_path), "json": str(json_path)}


if __name__ == "__main__":
    print(run())

