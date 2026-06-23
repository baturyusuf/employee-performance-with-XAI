from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.llm.evidence_schema import CompleteCaseEvidence
from src.utils.config import SETTINGS


REPORT_SNIPPETS = {
    "model_card": SETTINGS.reports_dir / "model_card" / "hr_xai_model_card.md",
    "recommendation": SETTINGS.reports_dir / "model_selection" / "final_recommendation.md",
    "fairness": SETTINGS.reports_dir / "fairness" / "feature_set_sensitivity" / "bootstrap_disparity_interpretation.md",
    "calibration": SETTINGS.reports_dir / "calibration" / "final_candidates" / "calibration_interpretation.md",
    "counterfactual": SETTINGS.reports_dir / "counterfactuals" / "final_candidates" / "actionability_interpretation.md",
    "shap": SETTINGS.reports_dir / "xai" / "final_candidates" / "shap_stability_interpretation.md",
    "agent_audit": SETTINGS.reports_dir / "agent_audits" / "multi_agent_governance_audit.md",
}


def load_report_text(key: str, max_chars: int = 2000) -> str:
    path = REPORT_SNIPPETS.get(key)
    if path is None or not path.exists():
        return f"{key} evidence is unavailable."
    return path.read_text(encoding="utf-8")[:max_chars]


def retrieve_context(question: str, case_id: str | None = None) -> Dict[str, str]:
    lowered = question.lower()
    context: Dict[str, str] = {}
    if any(word in lowered for word in ["prediction", "why", "class", "case"]):
        context["structured_evidence"] = CompleteCaseEvidence.from_reports(case_id=case_id).to_json()
    if "leak" in lowered or "full-feature" in lowered or "full feature" in lowered:
        context["recommendation"] = load_report_text("recommendation")
    if "fair" in lowered or "proxy" in lowered or "department" in lowered or "jobrole" in lowered or "job role" in lowered:
        context["fairness"] = load_report_text("fairness")
    if "calibration" in lowered or "probability" in lowered or "confidence" in lowered:
        context["calibration"] = load_report_text("calibration")
    if "counterfactual" in lowered or "action" in lowered or "employee act" in lowered:
        context["counterfactual"] = load_report_text("counterfactual")
    if "shap" in lowered or "feature" in lowered or "attribution" in lowered:
        context["shap"] = load_report_text("shap")
    if "safe" in lowered or "decision support" in lowered or "autonomous" in lowered:
        context["model_card"] = load_report_text("model_card")
    if not context:
        context["model_card"] = load_report_text("model_card")
        context["recommendation"] = load_report_text("recommendation")
    return context

