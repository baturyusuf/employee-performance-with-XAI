from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.chatbot.guardrails import check_question
from src.chatbot.retrieval import retrieve_context
from src.llm.evidence_schema import CompleteCaseEvidence
from src.llm.governed_explainer import GovernedExplainer


@dataclass
class ChatResponse:
    answer: str
    allowed: bool
    guardrail_reasons: list[str]
    context_keys: list[str]


class GuardrailedChatEngine:
    def __init__(self):
        self.explainer = GovernedExplainer()

    def answer(self, question: str, case_id: Optional[str] = None) -> ChatResponse:
        guard = check_question(question)
        if not guard.allowed:
            return ChatResponse(guard.response, False, guard.reasons, [])

        lowered = question.lower()
        context = retrieve_context(question, case_id=case_id)
        if "leak" in lowered or "full-feature" in lowered or "full feature" in lowered:
            answer = (
                "Full-feature models are leakage-warning upper-bound baselines only. "
                "EmpLastSalaryHikePercent and Attrition are outcome-proximal/leakage-risk variables and are excluded from final candidates."
            )
        elif "fair" in lowered or "proxy" in lowered or "department" in lowered:
            answer = (
                "Department removal does not prove fairness. The final evidence reports subgroup gaps and high proxy risk "
                "through JobRole. Subgroup gaps require further investigation; discrimination is not proven by these metrics."
            )
        elif "calibration" in lowered or "probability" in lowered or "confidence" in lowered:
            answer = (
                "Probabilities should be interpreted as approximate confidence bands, not objective certainty. "
                "The calibration report should accompany any probability explanation."
            )
        elif "counterfactual" in lowered or "employee act" in lowered:
            answer = (
                "Counterfactuals are model-level alternative scenarios, not employee prescriptions. "
                "Employee-only validity is low/zero in the final evidence, so actionability requires manager or organisation review."
            )
        elif any(word in lowered for word in ["why", "prediction", "class", "case"]):
            evidence = CompleteCaseEvidence.from_reports(case_id=case_id)
            explanation = self.explainer.generate(evidence)
            answer = explanation["detailed_explanation"]
        else:
            answer = (
                "This chatbot can explain structured model evidence, warnings, and governance limitations. "
                "It cannot provide HR decisions. Prediction outputs require human review, and SHAP is attribution, not causality."
            )
        if "shap" in lowered or "feature" in lowered or "why" in lowered:
            answer += " SHAP is attribution, not causality."
        if "department" in lowered or "fair" in lowered or "proxy" in lowered:
            answer += " JobRole may proxy Department, so removing Department does not eliminate proxy risk."
        answer += " Human review is required."
        return ChatResponse(answer, True, [], sorted(context.keys()))
