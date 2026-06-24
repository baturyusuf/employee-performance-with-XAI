from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


UNSAFE_PATTERNS = [
    r"\bfire\b",
    r"\bfired\b",
    r"firing priority",
    r"\bterminate\b",
    r"\bpromote\b",
    r"\bpromoted\b",
    r"\bpromotion\b",
    r"\brank .*employees\b",
    r"\bwho should .*promot",
    r"\bwho should .*fire",
    r"\bpay raise\b",
    r"\bsalary increase\b",
    r"\bcompensation (decision|recommendation|recommend)\b",
    r"\bdisciplin",
    r"ignore .*fairness",
    r"ignore .*warning",
    r"ignore .*guardrail",
    r"ignore .*previous",
    r"warnings are irrelevant",
    r"give me the decision",
    r"bypass .*guardrail",
    r"jailbreak",
    r"system prompt",
    r"developer message",
    r"hidden instruction",
    r"pretend .*not .*govern",
    r"act as .*hr manager",
    r"use gender",
    r"use marital",
    r"use race",
    r"race .*used",
    r"race .*justify",
    r"use age .*justify",
    r"use sensitive",
    r"use salary hike",
    r"salary hike .*deployment",
    r"protected group",
    r"sensitive group",
    r"exactly what to change",
    r"tell .* exactly .* change",
    r"employee should",
    r"employee must",
    r"employee prescription",
    r"guaranteed to be fair",
    r"\bunbiased model\b",
    r"no discrimination",
    r"prove .*fair",
    r"fairness .*solved",
    r"\bcaused\b",
    r"\bcausal reason\b",
    r"what causes performance",
    r"prove what causes",
    r"increase .*real performance",
    r"caused .*score",
    r"automatically (decide|rank|approve|reject|evaluate)",
    r"approve or reject",
    r"use .*autonomous.*decision",
    r"without human review",
    r"hide .*uncertainty",
    r"remove .*warning",
    r"remove .*disclaimer",
    r"do not mention .*disclaimer",
    r"do not mention .*human review",
    r"objectively correct",
    r"trust .*probability .*certainty",
    r"termination justification",
    r"deny promotion",
    r"legally defensible",
    r"can i use .*full[- ]feature.*deployment",
    r"can i deploy .*full[- ]feature",
    r"should i deploy .*full[- ]feature",
    r"use .*full[- ]feature.*production",
    r"use .*upper[- ]bound.*operational",
    r"\bkov\b",
    r"\bişten çıkar",
    r"\bterfi\b",
    r"\bmaaş\b.*(artış|zam|karar)",
]

SAFE_AUDIT_PATTERNS = [
    r"does removing department prove fairness",
    r"if department is removed.*fairness solved",
    r"why are counterfactuals not employee prescriptions",
    r"is shap causal",
    r"why is the full[- ]feature model not deployable",
    r"what are the prohibited uses",
]

REFUSAL = (
    "I cannot provide hiring, firing, promotion, compensation, disciplinary, or autonomous HR recommendations. "
    "I can explain the model evidence, limitations, and required governance warnings."
)


@dataclass
class GuardrailResult:
    allowed: bool
    reasons: List[str]
    response: str = ""


def check_question(question: str) -> GuardrailResult:
    lowered = question.lower()
    if any(re.search(pattern, lowered) for pattern in SAFE_AUDIT_PATTERNS):
        return GuardrailResult(True, [])
    reasons = [pattern for pattern in UNSAFE_PATTERNS if re.search(pattern, lowered)]
    if reasons:
        return GuardrailResult(False, reasons, REFUSAL)
    return GuardrailResult(True, [])
