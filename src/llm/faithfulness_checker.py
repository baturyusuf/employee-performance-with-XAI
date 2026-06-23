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
    shap = evidence.get("shap") or {}
    for key in ["top_positive_features", "top_negative_features"]:
        for row in shap.get(key, []) or []:
            if isinstance(row, dict) and row.get("feature"):
                names.add(str(row["feature"]))
    return names


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
    for match in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", text):
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
    unsupported = [name for name in feature_like if name not in known and name not in {"Department", "JobRole"}]
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
    text = flatten_text(output)
    forbidden = []
    forbidden.extend(detect_patterns(text, CAUSAL_PATTERNS))
    forbidden.extend(detect_patterns(text, AUTONOMOUS_PATTERNS))
    forbidden.extend(detect_patterns(text, FAIRNESS_OVERCLAIMS))
    forbidden.extend(detect_patterns(text, EMPLOYEE_PRESCRIPTIONS))
    unsupported = unsupported_metric_check(text, evidence) + unsupported_feature_check(text, evidence)
    missing = missing_warning_check(text, evidence)
    penalty = 15 * len(forbidden) + 10 * len(unsupported) + 8 * len(missing)
    score = max(0, 100 - penalty)
    passed = not forbidden and not unsupported and not missing and score >= 80
    revision = ""
    if not passed:
        revision = (
            "Revise explanation to use only structured evidence, remove causal/HR decision language, "
            "and include all required governance warnings."
        )
    return FaithfulnessResult(
        faithfulness_pass=passed,
        score=score,
        unsupported_claims=unsupported,
        forbidden_claims=sorted(set(forbidden)),
        missing_warnings=missing,
        suggested_revision=revision,
    )
