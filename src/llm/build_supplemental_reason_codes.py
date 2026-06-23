from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.utils.config import SETTINGS
from src.utils.config_loader import CONFIG_DIR, load_config
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


SOURCE_DIR = SETTINGS.reports_dir / "leakage_safe" / "xgboost_no_salary_hike_no_attrition_no_department" / "shap"
OUTPUT_DIR = SETTINGS.reports_dir / "xai" / "final_candidates" / "reason_code_examples"
FEATURE_SET = "no_salary_hike_no_attrition_no_department"
MODEL = "xgboost"

MANDATORY_WARNINGS = [
    "SHAP is attribution, not causality.",
    "This model is decision support, not autonomous employee evaluation.",
    "Job role may proxy department.",
    "Department removal does not prove fairness.",
    "Counterfactuals may require manager or organisation intervention.",
    "Probability estimates may be imperfectly calibrated.",
    "Small subgroup results may be unstable.",
    "External validation is required before deployment.",
]

CASE_TABLE_MAP = {
    "high_conf_class_2": "high_conf_class_2_local_grouped_shap_table.csv",
    "high_conf_class_3": "high_conf_class_3_local_grouped_shap_table.csv",
    "high_conf_class_4": "high_conf_class_4_local_grouped_shap_table.csv",
    "most_uncertain": "most_uncertain_local_grouped_shap_table.csv",
    "misclassified_example": "misclassified_example_local_grouped_shap_table.csv",
}


def run(
    source_dir: Path = SOURCE_DIR,
    output_dir: Path = OUTPUT_DIR,
    append_registry: bool = True,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    representative_path = source_dir / "representative_cases.csv"
    if not representative_path.exists():
        raise FileNotFoundError(f"Representative cases not found: {representative_path}")
    representative = pd.read_csv(representative_path)
    taxonomy = load_feature_taxonomy()
    written: List[str] = []
    skipped: List[str] = []
    for row in representative.to_dict(orient="records"):
        case_name = str(row["case"])
        table_name = CASE_TABLE_MAP.get(case_name)
        if not table_name:
            skipped.append(f"{case_name}: no local SHAP table mapping")
            continue
        table_path = source_dir / table_name
        if not table_path.exists():
            skipped.append(f"{case_name}: missing {table_name}")
            continue
        shap_df = pd.read_csv(table_path)
        payload = build_reason_code_payload(row, shap_df, taxonomy)
        output_path = output_dir / f"supplemental_{case_name}_{payload['sample_index']}.json"
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        md_path = output_dir / f"supplemental_{case_name}_{payload['sample_index']}.md"
        md_path.write_text(render_markdown(payload), encoding="utf-8")
        written.append(str(output_path))
    if append_registry:
        append_registry_row(
            {
                "run_id": f"supplemental_reason_codes_{utc_now_iso()}",
                "date_time": utc_now_iso(),
                "git_commit_if_available": get_git_commit(),
                "script": "python -m src.llm.build_supplemental_reason_codes",
                "config": f"source_dir={source_dir}; final-policy forbidden features filtered",
                "feature_set": FEATURE_SET,
                "model": MODEL,
                "seed": "deterministic_existing_shap_outputs",
                "cv_strategy": "not_applicable",
                "primary_metrics": f"written={len(written)}; skipped={len(skipped)}",
                "output_dir": "reports/xai/final_candidates/reason_code_examples",
                "notes": "Built supplemental reason-code evidence from existing XGBoost local grouped SHAP tables for larger LLM evaluation.",
                "decision_status": "accepted",
            }
        )
    return {"written": written, "skipped": skipped}


def load_feature_taxonomy() -> Dict[str, Dict[str, Any]]:
    payload = load_config(CONFIG_DIR / "feature_taxonomy.yaml")
    rows = payload.get("feature_taxonomy", [])
    return {str(row["feature_name"]): row for row in rows}


def build_reason_code_payload(
    case_row: Dict[str, Any],
    shap_df: pd.DataFrame,
    taxonomy: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    allowed = shap_df[
        shap_df["feature"].map(lambda feature: bool(taxonomy.get(str(feature), {}).get("allowed_for_final_model", False)))
    ].copy()
    allowed["abs_grouped_shap_value"] = allowed["grouped_shap_value"].abs()
    positive = allowed[allowed["grouped_shap_value"] >= 0].sort_values("abs_grouped_shap_value", ascending=False).head(5)
    negative = allowed[allowed["grouped_shap_value"] < 0].sort_values("abs_grouped_shap_value", ascending=False).head(5)
    case_name = str(case_row["case"])
    confidence = float(case_row["confidence"])
    return {
        "case_type": f"supplemental_{case_name}",
        "confidence_band": confidence_band(confidence),
        "feature_set": FEATURE_SET,
        "mandatory_warnings": MANDATORY_WARNINGS,
        "model": MODEL,
        "predicted_class": int(case_row["predicted_class"]),
        "predicted_probability": confidence,
        "sample_index": int(case_row["sample_index"]),
        "top_supporting_features": feature_rows(positive, taxonomy),
        "top_opposing_features": feature_rows(negative, taxonomy),
        "true_class": int(case_row["true_class"]),
        "evidence_note": "Supplemental case built from existing XGBoost local grouped SHAP outputs; final-policy forbidden features were filtered before LLM exposure.",
    }


def feature_rows(df: pd.DataFrame, taxonomy: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in df.to_dict(orient="records"):
        feature = str(item["feature"])
        meta = taxonomy.get(feature, {})
        rows.append(
            {
                "feature": feature,
                "feature_value": item.get("feature_value"),
                "grouped_shap_value": float(item["grouped_shap_value"]),
                "control_type": str(meta.get("control_type", "unknown")),
                "sensitive_or_proxy": str(meta.get("sensitive_or_proxy", "unknown")),
                "leakage_risk": str(meta.get("leakage_risk", "unknown")),
                "notes": str(meta.get("notes", "")),
            }
        )
    return rows


def confidence_band(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low_or_uncertain"


def render_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        f"# Supplemental Reason Code Example: {payload['sample_index']}",
        "",
        f"- Case type: {payload['case_type']}",
        f"- Predicted class: {payload['predicted_class']}",
        f"- True class: {payload['true_class']}",
        f"- Predicted probability: {payload['predicted_probability']}",
        f"- Evidence note: {payload['evidence_note']}",
        "",
        "## Supporting Features",
    ]
    for item in payload["top_supporting_features"]:
        lines.append(f"- {item['feature']}: {item['grouped_shap_value']:.4f}")
    lines.append("")
    lines.append("## Opposing Features")
    for item in payload["top_opposing_features"]:
        lines.append(f"- {item['feature']}: {item['grouped_shap_value']:.4f}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build supplemental LLM reason-code examples.")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-registry", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run(source_dir=args.source_dir, output_dir=args.output_dir, append_registry=not args.no_registry))
