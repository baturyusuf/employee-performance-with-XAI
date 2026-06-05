from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.experiments.final_evidence_common import (
    COUNTERFACTUAL_DIR,
    FINAL_FEATURE_SETS,
    LABELS,
    MODEL_NAME,
    align_proba,
    append_task_registry,
    ensure_dir,
    fit_xgb_pipeline,
    get_current_xy,
    predict_labels_from_proba,
    save_json,
)
from src.features.feature_sets import taxonomy_by_feature
from src.utils.experiment_registry import utc_now_iso


MODES = ["employee_only", "employee_manager", "organization_allowed", "full_default", "no_salary"]


def intervention_features(mode: str, available_columns: Iterable[str]) -> List[str]:
    taxonomy = taxonomy_by_feature()
    available = list(available_columns)

    def control(feature: str) -> str:
        return taxonomy.get(feature, {}).get("control_type", "unknown")

    if mode == "employee_only":
        return [f for f in available if control(f) == "employee_controllable"]
    if mode == "employee_manager":
        return [f for f in available if control(f) in {"employee_controllable", "manager_controllable"}]
    if mode == "organization_allowed":
        return [f for f in available if control(f) in {"employee_controllable", "manager_controllable", "organisation_controllable"}]
    if mode == "full_default":
        return [f for f in available if control(f) != "forbidden"]
    if mode == "no_salary":
        return [f for f in available if control(f) != "forbidden" and f != "EmpLastSalaryHikePercent"]
    raise ValueError(f"Unknown mode: {mode}")


def feature_scale_stats(X: pd.DataFrame) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            series = pd.to_numeric(X[col], errors="coerce").dropna()
            scale = float(series.std()) if len(series) > 1 else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = float(series.max() - series.min()) if len(series) else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            stats[col] = scale
        else:
            stats[col] = 1.0
    return stats


def change_cost(feature: str, old_value: Any, new_value: Any, scales: Dict[str, float]) -> float:
    if str(old_value) == str(new_value):
        return 0.0
    try:
        return float(abs(float(new_value) - float(old_value)) / max(scales.get(feature, 1.0), 1e-8))
    except Exception:
        return 1.0


def select_representative_cases(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    pred_s = pd.Series(pred, index=y_true.index)
    confidence = pd.Series(np.max(proba, axis=1), index=y_true.index)
    rows: List[Dict[str, Any]] = []
    for label in LABELS:
        mask = (y_true == label) & (pred_s == label)
        if mask.any():
            rows.append({"case_type": f"correct_class_{label}", "sample_index": int(confidence[mask].idxmax())})
    wrong = pred_s[pred_s != y_true]
    if len(wrong):
        rows.append({"case_type": "misclassified_high_confidence", "sample_index": int(confidence.loc[wrong.index].idxmax())})
    rows.append({"case_type": "most_uncertain", "sample_index": int(confidence.idxmin())})
    return pd.DataFrame(rows).drop_duplicates("sample_index")


def build_candidate_rows(
    sample: pd.Series,
    prototypes: pd.DataFrame,
    allowed_features: List[str],
    scales: Dict[str, float],
    max_features_changed: int,
    max_prototypes: int,
) -> Tuple[pd.DataFrame, List[List[Dict[str, Any]]]]:
    candidate_rows = []
    candidate_changes: List[List[Dict[str, Any]]] = []
    for _, proto in prototypes.head(max_prototypes).iterrows():
        diffs = []
        for feature in allowed_features:
            if feature not in sample.index or feature not in proto.index:
                continue
            if str(sample[feature]) == str(proto[feature]):
                continue
            cost = change_cost(feature, sample[feature], proto[feature], scales)
            diffs.append((feature, cost, sample[feature], proto[feature]))
        diffs = sorted(diffs, key=lambda item: item[1])
        for k in range(1, min(max_features_changed, len(diffs)) + 1):
            modified = sample.copy()
            changes = []
            for feature, cost, old_value, new_value in diffs[:k]:
                modified[feature] = new_value
                changes.append({"feature": feature, "old_value": old_value, "new_value": new_value, "cost": float(cost)})
            candidate_rows.append(modified)
            candidate_changes.append(changes)
    if not candidate_rows:
        return pd.DataFrame(columns=sample.index), []
    return pd.DataFrame(candidate_rows, columns=sample.index), candidate_changes


def find_best_counterfactual(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    sample_index: int,
    desired_class: int,
    allowed_features: List[str],
    original_proba: np.ndarray,
    max_features_changed: int,
    max_prototypes: int,
) -> Dict[str, Any]:
    sample = X.loc[sample_index].copy()
    prototypes = X.loc[y[y == desired_class].index].drop(index=sample_index, errors="ignore").copy()
    if prototypes.empty:
        return {"valid": False, "failed_reason": "no_desired_class_prototypes"}
    rows, changes_list = build_candidate_rows(sample, prototypes, allowed_features, feature_scale_stats(X), max_features_changed, max_prototypes)
    if rows.empty:
        return {"valid": False, "failed_reason": "no_candidate_changes_under_constraints"}
    proba = align_proba(pipeline.predict_proba(rows), pipeline.named_steps["model"].classes_, LABELS)
    pred = predict_labels_from_proba(proba, LABELS)
    valid_positions = np.where(pred == desired_class)[0]
    if len(valid_positions) == 0:
        return {"valid": False, "failed_reason": "no_candidate_reached_desired_class"}

    taxonomy = taxonomy_by_feature()
    original_desired = float(original_proba[LABELS.index(desired_class)])
    candidates = []
    for pos in valid_positions:
        changes = changes_list[int(pos)]
        total_cost = float(sum(c["cost"] for c in changes) + 0.15 * len(changes))
        desired_prob = float(proba[pos, LABELS.index(desired_class)])
        candidates.append(
            {
                "valid": True,
                "desired_probability": desired_prob,
                "probability_gain": desired_prob - original_desired,
                "cost": total_cost,
                "num_changed_features": len(changes),
                "changes": changes,
                "changed_features": [c["feature"] for c in changes],
                "changed_control_types": [taxonomy.get(c["feature"], {}).get("control_type", "unknown") for c in changes],
            }
        )
    return sorted(candidates, key=lambda c: (c["cost"], -c["desired_probability"], c["num_changed_features"]))[0]


def write_interpretation(summary_df: pd.DataFrame, output_path: Path) -> None:
    primary = summary_df[summary_df["feature_set"] == "no_salary_hike_no_attrition_no_department"].set_index("intervention_mode")

    def rate(mode: str) -> float:
        return float(primary.loc[mode, "validity_rate"]) if mode in primary.index else float("nan")

    lines = [
        "# Counterfactual Actionability Interpretation",
        "",
        "Counterfactuals are intervention hypotheses, not employee prescriptions. Validity means the fitted model prediction changes under constrained feature modifications; it does not imply feasibility, causality, fairness, or recommended employee behavior.",
        "",
        "## Required Answers",
        "",
        "### Can employees realistically change the model prediction through employee-only features?",
        f"For the primary candidate, employee-only validity is {rate('employee_only'):.4f}. Low validity should be interpreted as limited employee-side recourse, not as an implementation failure.",
        "",
        "### Are performance-class shifts mostly dependent on managerial or organisational variables?",
        f"For the primary candidate, employee+manager validity is {rate('employee_manager'):.4f}, while organization-allowed validity is {rate('organization_allowed'):.4f}. A gain after adding manager/organisation controls means recourse depends on workplace context.",
        "",
        "### Which counterfactual modes are valid but not practically actionable?",
        "`full_default` may change immutable or historical variables and is a diagnostic upper bound. `organization_allowed` may be valid for workforce planning but is not an employee prescription.",
        "",
        "### How should the paper phrase counterfactual explanations responsibly?",
        "Use: 'Under constrained feature changes, the model prediction would change if these workplace/context variables were different.' Do not write 'the employee should change X' unless employee-only validity is strong and the changed feature is genuinely employee-controllable.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(seed: int = 42, max_features_changed: int = 3, max_prototypes: int = 250) -> Dict[str, Path]:
    ensure_dir(COUNTERFACTUAL_DIR)
    by_sample_rows: List[Dict[str, Any]] = []
    for feature_set in FINAL_FEATURE_SETS:
        X, y, _ = get_current_xy(feature_set, drop_sensitive=True)
        pipeline = fit_xgb_pipeline(X, y, random_state=seed)
        proba = align_proba(pipeline.predict_proba(X), pipeline.named_steps["model"].classes_, LABELS)
        pred = predict_labels_from_proba(proba, LABELS)
        cases = select_representative_cases(y, pred, proba)
        for case in cases.itertuples(index=False):
            sample_index = int(case.sample_index)
            pos = X.index.get_loc(sample_index)
            predicted_class = int(pred[pos])
            true_class = int(y.loc[sample_index])
            desired_class = min(predicted_class + 1, max(LABELS))
            eligible = predicted_class < max(LABELS)
            for mode in MODES:
                allowed = intervention_features(mode, X.columns)
                if not eligible:
                    result = {"valid": False, "failed_reason": "already_predicted_highest_class"}
                elif not allowed:
                    result = {"valid": False, "failed_reason": "no_allowed_features"}
                else:
                    result = find_best_counterfactual(
                        pipeline,
                        X,
                        y,
                        sample_index,
                        desired_class,
                        allowed,
                        proba[pos],
                        max_features_changed,
                        max_prototypes,
                    )
                by_sample_rows.append(
                    {
                        "feature_set": feature_set,
                        "case_type": case.case_type,
                        "sample_index": sample_index,
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "desired_class": desired_class,
                        "eligible_for_upward_shift": bool(eligible),
                        "confidence": float(np.max(proba[pos])),
                        "intervention_mode": mode,
                        "n_allowed_features": len(allowed),
                        "valid": bool(result.get("valid", False)),
                        "probability_gain": result.get("probability_gain", np.nan),
                        "desired_probability": result.get("desired_probability", np.nan),
                        "cost": result.get("cost", np.nan),
                        "num_changed_features": result.get("num_changed_features", 0),
                        "changed_features": "; ".join(result.get("changed_features", [])),
                        "changed_control_types": "; ".join(result.get("changed_control_types", [])),
                        "failed_reason": result.get("failed_reason", ""),
                    }
                )
            print(f"[counterfactual] feature_set={feature_set} sample={sample_index}")

    by_sample = pd.DataFrame(by_sample_rows)
    summary_rows = []
    for (feature_set, mode), group in by_sample.groupby(["feature_set", "intervention_mode"]):
        eligible = group[group["eligible_for_upward_shift"]]
        valid = eligible[eligible["valid"]]
        summary_rows.append(
            {
                "feature_set": feature_set,
                "intervention_mode": mode,
                "n_cases": int(len(group)),
                "eligible_cases": int(len(eligible)),
                "valid_cases": int(len(valid)),
                "validity_rate": float(len(valid) / len(eligible)) if len(eligible) else np.nan,
                "mean_probability_gain": float(pd.to_numeric(valid["probability_gain"], errors="coerce").mean()) if len(valid) else np.nan,
                "mean_best_cost": float(pd.to_numeric(valid["cost"], errors="coerce").mean()) if len(valid) else np.nan,
                "mean_best_num_changes": float(pd.to_numeric(valid["num_changed_features"], errors="coerce").mean()) if len(valid) else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = COUNTERFACTUAL_DIR / "actionability_summary.csv"
    sample_path = COUNTERFACTUAL_DIR / "actionability_by_sample.csv"
    interp_path = COUNTERFACTUAL_DIR / "actionability_interpretation.md"
    meta_path = COUNTERFACTUAL_DIR / "metadata.json"
    summary.to_csv(summary_path, index=False)
    by_sample.to_csv(sample_path, index=False)
    write_interpretation(summary, interp_path)
    save_json(
        {
            "task": "final_candidate_counterfactual_actionability",
            "feature_sets": FINAL_FEATURE_SETS,
            "model": MODEL_NAME,
            "seed": seed,
            "max_features_changed": max_features_changed,
            "max_prototypes": max_prototypes,
            "modes": MODES,
        },
        meta_path,
    )
    append_task_registry(
        run_id=f"final_counterfactual_actionability_{utc_now_iso()}",
        script="python -m src.experiments.final_counterfactual_actionability",
        feature_set="; ".join(FINAL_FEATURE_SETS),
        primary_metrics="counterfactual validity by intervention mode; probability gain; cost; changed control types",
        output_dir=COUNTERFACTUAL_DIR,
        notes="Prototype-based constrained counterfactual actionability diagnostics for final XGBoost candidates.",
    )
    return {"summary": summary_path, "by_sample": sample_path, "interpretation": interp_path, "metadata": meta_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterfactual actionability for final XGBoost candidates.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features-changed", type=int, default=3)
    parser.add_argument("--max-prototypes", type=int, default=250)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run(seed=args.seed, max_features_changed=args.max_features_changed, max_prototypes=args.max_prototypes))

