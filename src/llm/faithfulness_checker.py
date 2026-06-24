from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set

from src.features.feature_sets import taxonomy_by_feature


CAUSAL_PATTERNS = [
    r"\bcaused\b",
    r"\bleads to\b",
    r"\bimproves performance\b",
    r"\bbecause the employee\b",
    r"\bdue to the employee being\b",
    r"\bwill increase performance\b",
]
AUTONOMOUS_PATTERNS = [
    r"should be promoted",
    r"should be fired",
    r"should receive salary",
    r"should be disciplined",
    r"automatically decide",
]
FAIRNESS_OVERCLAIMS = [
    r"fair model",
    r"unbiased model",
    r"no discrimination",
    r"fairness solved",
]
EMPLOYEE_PRESCRIPTIONS = [
    r"the employee should",
    r"the employee must",
    r"change your",
    r"do this to improve",
]


@dataclass
class FaithfulnessResult:
    faithfulness_pass: bool
    score: int
    unsupported_claims: List[str] = field(default_factory=list)
    forbidden_claims: List[str] = field(default_factory=list)
    missing_warnings: List[str] = field(default_factory=list)
    suggested_revision: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness_pass": self.faithfulness_pass,
            "score": self.score,
            "unsupported_claims": self.unsupported_claims,
            "forbidden_claims": self.forbidden_claims,
            "missing_warnings": self.missing_warnings,
            "suggested_revision": self.suggested_revision,
        }


@dataclass
class FaithfulnessCategoryResult:
    unsupported_metric_claims: List[str] = field(default_factory=list)
    unsupported_feature_claims: List[str] = field(default_factory=list)
    causal_claims: List[str] = field(default_factory=list)
    autonomous_decision_claims: List[str] = field(default_factory=list)
    fairness_overclaims: List[str] = field(default_factory=list)
    employee_prescriptions: List[str] = field(default_factory=list)
    missing_warnings: List[str] = field(default_factory=list)
    parsing_error: str = ""

    @property
    def forbidden_claims(self) -> List[str]:
        return sorted(
            set(
                self.causal_claims
                + self.autonomous_decision_claims
                + self.fairness_overclaims
                + self.employee_prescriptions
            )
        )

    @property
    def unsupported_claims(self) -> List[str]:
        return sorted(set(self.unsupported_metric_claims + self.unsupported_feature_claims))

    @property
    def score(self) -> int:
        penalty = (
            15 * len(self.forbidden_claims)
            + 10 * len(self.unsupported_claims)
            + 8 * len(self.missing_warnings)
            + (25 if self.parsing_error else 0)
        )
        return max(0, 100 - penalty)

    @property
    def passed(self) -> bool:
        return (
            not self.forbidden_claims
            and not self.unsupported_claims
            and not self.missing_warnings
            and not self.parsing_error
            and self.score >= 80
        )

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness_pass": self.passed,
            "score": self.score,
            "unsupported_claims": self.unsupported_claims,
            "forbidden_claims": self.forbidden_claims,
            "missing_warnings": self.missing_warnings,
            "parsing_error": self.parsing_error,
        }

    def to_eval_row(
        self,
        *,
        run_id: str,
        dataset_name: str,
        case_id: str,
        evidence_hash: str,
        response_hash: str,
        notes: str = "",
    ) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "dataset_name": dataset_name,
            "case_id": case_id,
            "evidence_hash": evidence_hash,
            "response_hash": response_hash,
            "faithfulness_pass": self.passed,
            "faithfulness_score": self.score,
            "unsupported_metric_count": len(self.unsupported_metric_claims),
            "unsupported_feature_count": len(self.unsupported_feature_claims),
            "causal_claim_count": len(self.causal_claims),
            "autonomous_decision_claim_count": len(self.autonomous_decision_claims),
            "fairness_overclaim_count": len(self.fairness_overclaims),
            "employee_prescription_count": len(self.employee_prescriptions),
            "missing_warning_count": len(self.missing_warnings),
            "parsing_error": self.parsing_error,
            "notes": notes or "; ".join(self.unsupported_claims + self.forbidden_claims + self.missing_warnings),
        }


def flatten_text(output: Dict[str, Any] | str) -> str:
    if isinstance(output, str):
        return output
    parts: List[str] = []
    for key in ["short_explanation", "detailed_explanation"]:
        parts.append(str(output.get(key, "")))
    for warning in output.get("warnings", []):
        if isinstance(warning, dict):
            parts.append(str(warning.get("message", "")))
        else:
            parts.append(str(warning))
    return " ".join(parts)


def evidence_feature_names(evidence: Dict[str, Any]) -> Set[str]:
    names = set(taxonomy_by_feature().keys())

    def walk_keys(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                _add_name_variant(names, str(key))
                walk_keys(child)
        elif isinstance(value, list):
            for child in value:
                walk_keys(child)
        elif isinstance(value, str):
            stripped = value.strip()
            if stripped:
                _add_name_variant(names, stripped)

    walk_keys(evidence)
    shap = evidence.get("shap") or {}
    for key in ["top_positive_features", "top_negative_features"]:
        for row in shap.get(key, []) or []:
            if isinstance(row, dict) and row.get("feature"):
                names.add(str(row["feature"]))
    return names


def _add_name_variant(names: Set[str], value: str) -> None:
    names.add(value)
    names.add(re.sub(r"\.0\b", "", value))


def evidence_numbers(evidence: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                num = float(value)
                for candidate in {num, abs(num)}:
                    out.add(f"{candidate:.0f}")
                    out.add(f"{candidate:.1f}")
                    out.add(f"{candidate:.2f}")
                    out.add(f"{candidate:.3f}")
                    out.add(f"{candidate:.4f}")
            except Exception:
                pass
        elif isinstance(value, str):
            stripped = value.strip()
            if re.fullmatch(r"\d+(?:\.\d+)?", stripped):
                try:
                    num = float(stripped)
                    out.add(stripped)
                    out.add(f"{num:.0f}")
                    out.add(f"{num:.1f}")
                    out.add(f"{num:.2f}")
                    out.add(f"{num:.3f}")
                    out.add(f"{num:.4f}")
                except Exception:
                    pass
            for match in re.findall(r"(?<![A-Za-z0-9_\-\.])\d+(?:\.\d+)?(?![A-Za-z0-9_])", stripped):
                try:
                    num = float(match)
                    out.add(match)
                    out.add(f"{num:.0f}")
                    out.add(f"{num:.1f}")
                    out.add(f"{num:.2f}")
                    out.add(f"{num:.3f}")
                    out.add(f"{num:.4f}")
                except Exception:
                    pass

    walk(evidence)
    return out


def detect_patterns(text: str, patterns: Iterable[str]) -> List[str]:
    found = []
    lowered = text.lower()
    for pattern in patterns:
        if re.search(pattern, lowered):
            found.append(pattern)
    return found


def unsupported_metric_check(text: str, evidence: Dict[str, Any]) -> List[str]:
    allowed = evidence_numbers(evidence)
    claims = []
    for match in re.findall(r"(?<![A-Za-z0-9_\-\.])\d+(?:\.\d+)?(?![A-Za-z0-9_])", text):
        if match not in allowed:
            try:
                rounded = f"{float(match):.3f}"
            except Exception:
                rounded = match
            if rounded not in allowed and match not in {"2", "3", "4", "5", "10", "30", "100"}:
                claims.append(f"Unsupported numeric claim: {match}")
    return sorted(set(claims))


def unsupported_feature_check(text: str, evidence: Dict[str, Any]) -> List[str]:
    known = evidence_feature_names(evidence)
    known_lower = {name.lower() for name in known}
    candidates = set(re.findall(r"\b[A-Z][A-Za-z0-9_]{3,}\b", text))
    feature_like_prefixes = (
        "Emp",
        "Experience",
        "Years",
        "Total",
        "Training",
        "Business",
        "Distance",
        "Num",
        "OverTime",
        "Attrition",
        "Age",
        "Gender",
        "MaritalStatus",
        "Department",
        "JobRole",
    )
    feature_like = [
        name
        for name in candidates
        if name.startswith(feature_like_prefixes) or "_" in name
    ]
    unsupported = [
        name
        for name in feature_like
        if name not in known and name.lower() not in known_lower and name not in {"Department", "JobRole"}
    ]
    return sorted(set(unsupported))


def missing_warning_check(text: str, evidence: Dict[str, Any]) -> List[str]:
    lowered = text.lower()
    required = {
        "SHAP is not causal.": ["shap is attribution", "not causality", "not causal"],
        "Prediction requires human review.": ["human review"],
        "Not for autonomous HR decisions.": ["not for autonomous", "decision support only", "not autonomous"],
    }
    if evidence.get("leakage"):
        required["Full-feature models are leakage-warning only."] = ["leakage-warning", "upper-bound", "upper bound"]
    if evidence.get("fairness"):
        required["Department removal does not prove fairness."] = ["department removal does not prove fairness", "does not prove fairness"]
    if evidence.get("counterfactual"):
        required["Counterfactuals may not be employee-actionable."] = [
            "not employee",
            "not employee-actionable",
            "not be employee-actionable",
            "not employee instructions",
            "not employee prescriptions",
        ]
    missing = []
    for label, variants in required.items():
        if not any(v in lowered for v in variants):
            missing.append(label)
    return missing


def check_faithfulness(output: Dict[str, Any] | str, evidence: Dict[str, Any]) -> FaithfulnessResult:
    categorized = check_faithfulness_categories(output, evidence)
    revision = ""
    if not categorized.passed:
        revision = (
            "Revise explanation to use only structured evidence, remove causal/HR decision language, "
            "and include all required governance warnings."
        )
    return FaithfulnessResult(
        faithfulness_pass=categorized.passed,
        score=categorized.score,
        unsupported_claims=categorized.unsupported_claims,
        forbidden_claims=categorized.forbidden_claims,
        missing_warnings=categorized.missing_warnings,
        suggested_revision=revision,
    )


def check_faithfulness_categories(
    output: Dict[str, Any] | str,
    evidence: Dict[str, Any],
    *,
    parsing_error: str = "",
) -> FaithfulnessCategoryResult:
    text = flatten_text(output)
    return FaithfulnessCategoryResult(
        unsupported_metric_claims=unsupported_metric_check(text, evidence),
        unsupported_feature_claims=unsupported_feature_check(text, evidence),
        causal_claims=detect_patterns(text, CAUSAL_PATTERNS),
        autonomous_decision_claims=detect_patterns(text, AUTONOMOUS_PATTERNS),
        fairness_overclaims=detect_patterns(text, FAIRNESS_OVERCLAIMS),
        employee_prescriptions=detect_patterns(text, EMPLOYEE_PRESCRIPTIONS),
        missing_warnings=missing_warning_check(text, evidence),
        parsing_error=parsing_error,
    )
