from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.experiments.final_counterfactual_actionability import select_representative_cases
from src.experiments.final_evidence_common import (
    LABELS,
    MODEL_NAME,
    PRIMARY_FEATURE_SET,
    XAI_DIR,
    align_proba,
    append_task_registry,
    ensure_dir,
    fit_xgb_pipeline,
    get_current_xy,
    predict_labels_from_proba,
    save_json,
)
from src.experiments.final_shap_stability import get_group_mapping, group_shap_values, normalize_shap_values
from src.experiments.leakage_safe_cv import infer_columns
from src.features.feature_sets import taxonomy_by_feature
from src.utils.experiment_registry import utc_now_iso


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


def confidence_band(probability: float) -> str:
    if probability >= 0.80:
        return "high"
    if probability >= 0.60:
        return "medium"
    return "low"


def feature_governance(feature: str) -> Dict[str, Any]:
    row = taxonomy_by_feature().get(feature, {})
    return {
        "control_type": row.get("control_type", "unknown"),
        "sensitive_or_proxy": row.get("sensitive_or_proxy", "unknown"),
        "leakage_risk": row.get("leakage_risk", "unknown"),
        "notes": row.get("notes", ""),
    }


def encode_feature_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "feature": row.feature,
                "feature_value": row.feature_value,
                "grouped_shap_value": float(row.grouped_shap_value),
                **feature_governance(row.feature),
            }
        )
    return rows


def write_case_markdown(payload: Dict[str, Any], path: Path) -> None:
    lines = [
        f"# Reason Code Example: {payload['case_type']}",
        "",
        f"Sample index: {payload['sample_index']}",
        f"True class: {payload['true_class']}",
        f"Predicted class: {payload['predicted_class']}",
        f"Predicted probability: {payload['predicted_probability']:.4f} ({payload['confidence_band']} confidence band)",
        "",
        "## Top Supporting Features",
    ]
    for item in payload["top_supporting_features"]:
        lines.append(f"- {item['feature']} = {item['feature_value']} | SHAP {item['grouped_shap_value']:.4f} | {item['control_type']} | {item['sensitive_or_proxy']}")
    lines.extend(["", "## Top Opposing Features"])
    for item in payload["top_opposing_features"]:
        lines.append(f"- {item['feature']} = {item['feature_value']} | SHAP {item['grouped_shap_value']:.4f} | {item['control_type']} | {item['sensitive_or_proxy']}")
    lines.extend(["", "## Governance Warnings"])
    for warning in payload["mandatory_warnings"]:
        lines.append(f"- {warning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(seed: int = 42, top_k: int = 5) -> Dict[str, Path]:
    import shap

    examples_dir = XAI_DIR / "reason_code_examples"
    ensure_dir(examples_dir)
    notes_path = XAI_DIR / "reason_code_governance_notes.md"
    X, y, _ = get_current_xy(PRIMARY_FEATURE_SET, drop_sensitive=True)
    pipeline = fit_xgb_pipeline(X, y, random_state=seed)
    proba = align_proba(pipeline.predict_proba(X), pipeline.named_steps["model"].classes_, LABELS)
    pred = predict_labels_from_proba(proba, LABELS)
    cases = select_representative_cases(y, pred, proba)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["model"]
    numeric_cols, categorical_cols = infer_columns(X)
    group_names, mapping = get_group_mapping(preprocessor, numeric_cols, categorical_cols)
    X_t = preprocessor.transform(X)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    explainer = shap.TreeExplainer(classifier.model_)
    raw_shap = explainer.shap_values(X_t)
    shap_arr = normalize_shap_values(raw_shap, X_t.shape[0], X_t.shape[1], len(classifier.classes_))
    grouped = group_shap_values(shap_arr, group_names, mapping)

    for case in cases.itertuples(index=False):
        idx = int(case.sample_index)
        pos = X.index.get_loc(idx)
        predicted_class = int(pred[pos])
        class_idx = LABELS.index(predicted_class)
        local = pd.DataFrame(
            {
                "feature": group_names,
                "feature_value": [X.loc[idx, f] for f in group_names],
                "grouped_shap_value": grouped[pos, class_idx, :],
            }
        )
        local["abs_grouped_shap_value"] = local["grouped_shap_value"].abs()
        local = local.sort_values("abs_grouped_shap_value", ascending=False)
        payload = {
            "feature_set": PRIMARY_FEATURE_SET,
            "model": MODEL_NAME,
            "case_type": case.case_type,
            "sample_index": idx,
            "true_class": int(y.loc[idx]),
            "predicted_class": predicted_class,
            "predicted_probability": float(proba[pos, class_idx]),
            "confidence_band": confidence_band(float(proba[pos, class_idx])),
            "top_supporting_features": encode_feature_rows(local[local["grouped_shap_value"] > 0].head(top_k)),
            "top_opposing_features": encode_feature_rows(local[local["grouped_shap_value"] < 0].head(top_k)),
            "mandatory_warnings": MANDATORY_WARNINGS,
        }
        json_path = examples_dir / f"{case.case_type}_{idx}.json"
        md_path = examples_dir / f"{case.case_type}_{idx}.md"
        save_json(payload, json_path)
        write_case_markdown(payload, md_path)
    notes_path.write_text(
        "# Reason-Code Governance Notes\n\n"
        "Reason codes are model attribution summaries, not causal explanations or employee prescriptions. "
        "They must be shown with proxy, actionability, calibration, subgroup-support, and external-validation warnings. "
        "The primary candidate excludes EmpDepartment but retains EmpJobRole, so every reason-code surface must disclose department proxy risk.\n",
        encoding="utf-8",
    )
    append_task_registry(
        run_id=f"final_reason_codes_{utc_now_iso()}",
        script="python -m src.experiments.final_reason_codes",
        feature_set=PRIMARY_FEATURE_SET,
        primary_metrics="representative local grouped SHAP reason codes with governance warnings",
        output_dir=examples_dir,
        notes="Generated governance-safe reason-code examples for the recommended primary candidate.",
    )
    return {"examples_dir": examples_dir, "notes": notes_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Governance-safe reason-code examples for the primary candidate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run(seed=args.seed, top_k=args.top_k))

