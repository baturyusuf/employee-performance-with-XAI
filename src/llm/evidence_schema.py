from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.config import SETTINGS


FINAL_FEATURE_SET = "no_salary_hike_no_attrition_no_department"
MODEL_NAME = "xgboost"
REPORTS = SETTINGS.reports_dir


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
        if pd.isna(out):
            return None
        return out
    except Exception:
        return None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _first_existing_reason_code(case_id: Optional[str] = None) -> Optional[Path]:
    root = REPORTS / "xai" / "final_candidates" / "reason_code_examples"
    if not root.exists():
        return None
    if case_id is not None:
        direct = sorted(root.glob(f"*_{case_id}.json"))
        if direct:
            return direct[0]
    files = sorted(root.glob("*.json"))
    return files[0] if files else None


@dataclass
class WarningItem:
    type: str
    severity: str
    message: str

    def __post_init__(self) -> None:
        if self.severity not in {"low", "medium", "high"}:
            raise ValueError(f"Invalid warning severity: {self.severity}")


@dataclass
class PredictionEvidence:
    case_id: str
    predicted_class: Optional[int]
    true_class: Optional[int]
    class_probabilities: Dict[str, float]
    confidence: Optional[float]
    uncertainty_flag: bool
    model_name: str
    feature_policy: str
    leakage_safe_status: str

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("PredictionEvidence.case_id is required")
        if self.model_name.lower() == "llm":
            raise ValueError("LLM must not be represented as the performance predictor")


@dataclass
class ShapEvidence:
    top_positive_features: List[Dict[str, Any]]
    top_negative_features: List[Dict[str, Any]]
    grouped_shap_values: Dict[str, float]
    class_specific_shap_values: Dict[str, float]
    shap_stability_summary: Dict[str, Optional[float]]
    explanation_stability_warning: str


@dataclass
class FairnessEvidence:
    audited_groups: List[str]
    subgroup_metrics: Dict[str, Any]
    disparity_gaps: Dict[str, Any]
    bootstrap_ci: Dict[str, Any]
    low_support_warnings: List[Dict[str, Any]]
    proxy_risk_warnings: List[str]


@dataclass
class CalibrationEvidence:
    log_loss: Optional[float]
    brier_score: Optional[float]
    expected_calibration_error: Optional[float]
    calibration_warning: str


@dataclass
class CounterfactualEvidence:
    counterfactual_mode: str
    validity: Optional[float]
    changed_features: List[str]
    probability_gain: Optional[float]
    proximity_cost: Optional[float]
    actionability_label: str
    failed_reason: str
    warning: str


@dataclass
class LeakageEvidence:
    feature_policy: str
    excluded_leakage_features: List[str]
    full_feature_score: Optional[float]
    leakage_safe_score: Optional[float]
    leakage_sensitivity_index: Optional[float]
    leakage_warning: str


@dataclass
class GovernanceEvidence:
    intended_use: str
    prohibited_use: str
    model_card_summary: str
    deployment_status: str
    required_warnings: List[str]


@dataclass
class CompleteCaseEvidence:
    prediction: PredictionEvidence
    shap: Optional[ShapEvidence]
    fairness: Optional[FairnessEvidence]
    calibration: Optional[CalibrationEvidence]
    counterfactual: Optional[CounterfactualEvidence]
    leakage: Optional[LeakageEvidence]
    governance: GovernanceEvidence
    evidence_sources: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        forbidden_predictor_names = {"EmpLastSalaryHikePercent", "Attrition", "PerformanceRating", "EmpNumber", "Age"}
        if self.shap is not None:
            features = {row.get("feature") for row in self.shap.top_positive_features + self.shap.top_negative_features}
            leaked = sorted(forbidden_predictor_names.intersection(features))
            if leaked:
                raise ValueError(f"Forbidden final-model feature exposed to LLM evidence: {leaked}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_reports(cls, case_id: Optional[str] = None) -> "CompleteCaseEvidence":
        sources: List[str] = []
        dashboard_path = REPORTS / "model_selection" / "final_candidate_dashboard.csv"
        dashboard = _read_csv(dashboard_path)
        row = dashboard[dashboard["feature_set"] == FINAL_FEATURE_SET].iloc[0] if not dashboard.empty else pd.Series(dtype=object)
        if dashboard_path.exists():
            sources.append(str(dashboard_path))

        reason_path = _first_existing_reason_code(case_id)
        reason_payload: Dict[str, Any] = {}
        if reason_path is not None:
            reason_payload = json.loads(reason_path.read_text(encoding="utf-8"))
            sources.append(str(reason_path))

        sample_id = str(reason_payload.get("sample_index", case_id or "model_level"))
        predicted_probability = _safe_float(reason_payload.get("predicted_probability"))
        probabilities = {}
        if predicted_probability is not None and reason_payload.get("predicted_class") is not None:
            probabilities[str(reason_payload["predicted_class"])] = predicted_probability
        prediction = PredictionEvidence(
            case_id=sample_id,
            predicted_class=reason_payload.get("predicted_class"),
            true_class=reason_payload.get("true_class"),
            class_probabilities=probabilities,
            confidence=predicted_probability,
            uncertainty_flag=bool(predicted_probability is not None and predicted_probability < 0.60),
            model_name=MODEL_NAME,
            feature_policy=FINAL_FEATURE_SET,
            leakage_safe_status="final_candidate_leakage_safe",
        )

        shap_summary_path = REPORTS / "xai" / "final_candidates" / "shap_stability_summary.csv"
        shap_summary = _read_csv(shap_summary_path)
        if shap_summary_path.exists():
            sources.append(str(shap_summary_path))
        shap_row = pd.Series(dtype=object)
        if not shap_summary.empty:
            subset = shap_summary[(shap_summary["feature_set"] == FINAL_FEATURE_SET) & (shap_summary["top_k"] == 10)]
            if len(subset):
                shap_row = subset.iloc[0]
        pos = reason_payload.get("top_supporting_features", [])
        neg = reason_payload.get("top_opposing_features", [])
        grouped = {item["feature"]: _safe_float(item.get("grouped_shap_value")) or 0.0 for item in pos + neg if "feature" in item}
        shap = ShapEvidence(
            top_positive_features=pos,
            top_negative_features=neg,
            grouped_shap_values=grouped,
            class_specific_shap_values=grouped,
            shap_stability_summary={
                "top10_jaccard": _safe_float(shap_row.get("mean_jaccard")),
                "spearman": _safe_float(shap_row.get("mean_spearman")),
            },
            explanation_stability_warning="SHAP is attribution, not causality; discuss only stable grouped features.",
        )

        fairness_path = REPORTS / "fairness" / "feature_set_sensitivity" / "bootstrap_disparity_ci.csv"
        fairness_df = _read_csv(fairness_path)
        if fairness_path.exists():
            sources.append(str(fairness_path))
        dept_macro = pd.Series(dtype=object)
        if not fairness_df.empty:
            subset = fairness_df[
                (fairness_df["feature_set"] == FINAL_FEATURE_SET)
                & (fairness_df["attribute"] == "EmpDepartment")
                & (fairness_df["metric"] == "macro_f1")
                & (fairness_df["class_label"].isna())
            ]
            if len(subset):
                dept_macro = subset.iloc[0]
        small_path = REPORTS / "fairness" / "feature_set_sensitivity" / "small_group_warnings.csv"
        small_df = _read_csv(small_path)
        if small_path.exists():
            sources.append(str(small_path))
        fairness = FairnessEvidence(
            audited_groups=["Gender", "MaritalStatus", "EmpDepartment", "EducationBackground", "BusinessTravelFrequency"],
            subgroup_metrics={},
            disparity_gaps={"EmpDepartment_macro_f1_gap": _safe_float(dept_macro.get("point_estimate"))},
            bootstrap_ci={
                "EmpDepartment_macro_f1_gap_ci_low": _safe_float(dept_macro.get("ci_low")),
                "EmpDepartment_macro_f1_gap_ci_high": _safe_float(dept_macro.get("ci_high")),
            },
            low_support_warnings=small_df.to_dict(orient="records") if not small_df.empty else [],
            proxy_risk_warnings=[
                "EmpJobRole may proxy EmpDepartment.",
                "Removing EmpDepartment does not prove fairness.",
            ],
        )

        calibration = CalibrationEvidence(
            log_loss=_safe_float(row.get("log_loss")),
            brier_score=_safe_float(row.get("multiclass_brier")),
            expected_calibration_error=_safe_float(row.get("ece")),
            calibration_warning="Probabilities should be interpreted as approximate confidence bands, not calibrated truth.",
        )

        cf_path = REPORTS / "counterfactuals" / "final_candidates" / "actionability_summary.csv"
        cf = _read_csv(cf_path)
        if cf_path.exists():
            sources.append(str(cf_path))
        employee_only = pd.Series(dtype=object)
        if not cf.empty:
            subset = cf[(cf["feature_set"] == FINAL_FEATURE_SET) & (cf["intervention_mode"] == "employee_only")]
            if len(subset):
                employee_only = subset.iloc[0]
        counterfactual = CounterfactualEvidence(
            counterfactual_mode="employee_only",
            validity=_safe_float(employee_only.get("validity_rate")),
            changed_features=[],
            probability_gain=None,
            proximity_cost=None,
            actionability_label="not_employee_actionable",
            failed_reason="employee_only validity is zero or unavailable",
            warning="Counterfactuals are model scenarios, not employee prescriptions.",
        )

        leakage_path = REPORTS / "leakage" / "leakage_sensitivity_index.csv"
        leakage_df = _read_csv(leakage_path)
        if leakage_path.exists():
            sources.append(str(leakage_path))
        lsi_row = pd.Series(dtype=object)
        if not leakage_df.empty:
            subset = leakage_df[
                (leakage_df["model"].astype(str).str.lower() == "xgboost")
                & (leakage_df["metric"] == "macro_f1")
                & (leakage_df["comparison_feature_set"] == FINAL_FEATURE_SET)
            ]
            if len(subset):
                lsi_row = subset.iloc[0]
        leakage = LeakageEvidence(
            feature_policy=FINAL_FEATURE_SET,
            excluded_leakage_features=["EmpLastSalaryHikePercent", "Attrition", "Age", "EmpDepartment"],
            full_feature_score=_safe_float(lsi_row.get("full_feature_score")),
            leakage_safe_score=_safe_float(lsi_row.get("comparison_score")),
            leakage_sensitivity_index=_safe_float(lsi_row.get("lsi")),
            leakage_warning="Full-feature models are leakage-warning upper-bound baselines only, not deployable models.",
        )

        model_card_path = REPORTS / "model_card" / "hr_xai_model_card.md"
        model_card = model_card_path.read_text(encoding="utf-8") if model_card_path.exists() else "Model card unavailable."
        if model_card_path.exists():
            sources.append(str(model_card_path))
        governance = GovernanceEvidence(
            intended_use="Research-grade HR decision support and model governance review.",
            prohibited_use="No autonomous hiring, firing, promotion, compensation, disciplinary, or employee evaluation decisions.",
            model_card_summary=model_card[:1200],
            deployment_status="research_only_decision_support_with_strong_warnings",
            required_warnings=[
                "SHAP is attribution, not causality.",
                "Prediction requires human review.",
                "Not for autonomous HR decisions.",
                "Full-feature models are leakage-warning only.",
                "Department removal does not prove fairness.",
                "Counterfactuals may not be employee-actionable.",
                "External validation is required.",
            ],
        )
        return cls(
            prediction=prediction,
            shap=shap,
            fairness=fairness,
            calibration=calibration,
            counterfactual=counterfactual,
            leakage=leakage,
            governance=governance,
            evidence_sources=sources,
        )


def load_complete_case_evidence(case_id: Optional[str] = None) -> CompleteCaseEvidence:
    return CompleteCaseEvidence.from_reports(case_id=case_id)


def write_evidence_json(output_path: Path, case_id: Optional[str] = None) -> Path:
    evidence = load_complete_case_evidence(case_id=case_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(evidence.to_json(), encoding="utf-8")
    return output_path

