from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class WarningDefinition:
    warning_id: str
    category: str
    severity: str
    canonical_message: str
    match_terms: tuple[str, ...]


WARNING_TAXONOMY: tuple[WarningDefinition, ...] = (
    WarningDefinition(
        warning_id="deployment.not_autonomous",
        category="deployment",
        severity="high",
        canonical_message="This model is decision support only and is not for autonomous HR decisions.",
        match_terms=("autonomous", "decision support", "hr decisions"),
    ),
    WarningDefinition(
        warning_id="deployment.human_review_required",
        category="deployment",
        severity="high",
        canonical_message="Prediction requires human review.",
        match_terms=("human review", "review required"),
    ),
    WarningDefinition(
        warning_id="causality.shap_not_causal",
        category="causality",
        severity="high",
        canonical_message="SHAP is attribution, not causality.",
        match_terms=("shap", "causality", "causal"),
    ),
    WarningDefinition(
        warning_id="leakage.full_feature_upper_bound_only",
        category="leakage",
        severity="high",
        canonical_message="Full-feature models are leakage-warning upper-bound baselines only, not deployable models.",
        match_terms=("full-feature", "full feature", "upper-bound", "leakage-warning", "not deployable"),
    ),
    WarningDefinition(
        warning_id="leakage.salary_attrition_excluded",
        category="leakage",
        severity="high",
        canonical_message="EmpLastSalaryHikePercent and Attrition are excluded from final candidates because they are leakage-risk or outcome-proximal variables.",
        match_terms=("emplastsalaryhikepercent", "salary hike", "attrition", "outcome-proximal"),
    ),
    WarningDefinition(
        warning_id="fairness.department_removal_not_fairness_proof",
        category="fairness",
        severity="high",
        canonical_message="Removing EmpDepartment does not prove fairness.",
        match_terms=("department removal", "empdepartment", "does not prove fairness", "fairness"),
    ),
    WarningDefinition(
        warning_id="proxy.jobrole_department_proxy",
        category="proxy",
        severity="high",
        canonical_message="EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.",
        match_terms=("empjobrole", "jobrole", "job role", "proxy", "department"),
    ),
    WarningDefinition(
        warning_id="calibration.probabilities_approximate",
        category="calibration",
        severity="medium",
        canonical_message="Probability estimates should be interpreted as approximate confidence, not objective certainty.",
        match_terms=("calibration", "probability", "confidence", "ece", "brier", "log loss"),
    ),
    WarningDefinition(
        warning_id="actionability.counterfactual_not_prescription",
        category="actionability",
        severity="high",
        canonical_message="Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.",
        match_terms=("counterfactual", "employee-actionable", "employee actionable", "prescription"),
    ),
    WarningDefinition(
        warning_id="fairness.small_subgroup_instability",
        category="fairness",
        severity="medium",
        canonical_message="Small subgroup findings may be unstable and should be treated as diagnostic audit evidence.",
        match_terms=("small subgroup", "low support", "support", "unstable"),
    ),
    WarningDefinition(
        warning_id="validation.external_validation_required",
        category="validation",
        severity="high",
        canonical_message="External validation is required before deployment.",
        match_terms=("external validation", "deployment", "validation required"),
    ),
)

MANDATORY_WARNING_IDS: tuple[str, ...] = (
    "deployment.not_autonomous",
    "deployment.human_review_required",
    "causality.shap_not_causal",
    "leakage.full_feature_upper_bound_only",
    "fairness.department_removal_not_fairness_proof",
    "proxy.jobrole_department_proxy",
    "actionability.counterfactual_not_prescription",
    "validation.external_validation_required",
)


def warning_definition(warning_id: str) -> WarningDefinition:
    for definition in WARNING_TAXONOMY:
        if definition.warning_id == warning_id:
            return definition
    raise KeyError(f"Unknown warning_id: {warning_id}")


def normalize_warning_text(text: str) -> Dict[str, str]:
    lowered = text.lower()
    best_match: WarningDefinition | None = None
    best_score = 0
    for definition in WARNING_TAXONOMY:
        score = sum(1 for term in definition.match_terms if term in lowered)
        if score > best_score:
            best_match = definition
            best_score = score
    if best_match is None:
        return {
            "warning_id": "other.unmapped",
            "category": "other",
            "severity": "medium",
            "message": text,
            "source_text": text,
        }
    return {
        "warning_id": best_match.warning_id,
        "category": best_match.category,
        "severity": best_match.severity,
        "message": best_match.canonical_message,
        "source_text": text,
    }


def normalize_warning_records(warnings: Iterable[Any]) -> List[Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    for warning in warnings:
        if isinstance(warning, dict):
            raw_text = str(warning.get("message") or warning.get("warning") or warning.get("text") or "")
            if warning.get("warning_id"):
                warning_id = str(warning["warning_id"])
                try:
                    definition = warning_definition(warning_id)
                    record = {
                        "warning_id": definition.warning_id,
                        "category": definition.category,
                        "severity": definition.severity,
                        "message": definition.canonical_message,
                        "source_text": raw_text or definition.canonical_message,
                    }
                except KeyError:
                    record = normalize_warning_text(raw_text or warning_id)
            else:
                record = normalize_warning_text(raw_text)
        else:
            record = normalize_warning_text(str(warning))
        records.setdefault(record["warning_id"], record)
    return list(records.values())


def add_mandatory_warning_records(
    warnings: Iterable[Any],
    mandatory_ids: Iterable[str] = MANDATORY_WARNING_IDS,
) -> List[Dict[str, str]]:
    records = {item["warning_id"]: item for item in normalize_warning_records(warnings)}
    for warning_id in mandatory_ids:
        definition = warning_definition(warning_id)
        records.setdefault(
            warning_id,
            {
                "warning_id": definition.warning_id,
                "category": definition.category,
                "severity": definition.severity,
                "message": definition.canonical_message,
                "source_text": "mandatory_taxonomy",
            },
        )
    return list(records.values())


def warning_messages(records: Iterable[Dict[str, str]]) -> List[str]:
    return [record["message"] for record in records]


def warning_taxonomy_rows() -> List[Dict[str, Any]]:
    rows = []
    for definition in WARNING_TAXONOMY:
        row = asdict(definition)
        row["match_terms"] = "; ".join(definition.match_terms)
        row["mandatory"] = definition.warning_id in MANDATORY_WARNING_IDS
        rows.append(row)
    return rows
