from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


UNSAFE_PATTERNS = [
    r"\bfire\b",
    r"\bfired\b",
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
    r"use age .*justify",
    r"use sensitive",
    r"exactly what to change",
    r"tell .* exactly .* change",
    r"employee should",
    r"employee must",
    r"guaranteed to be fair",
    r"\bunbiased model\b",
    r"no discrimination",
    r"automatically (decide|rank|approve|reject|evaluate)",
    r"use .*autonomous.*decision",
    r"without human review",
    r"can i use .*full[- ]feature.*deployment",
    r"should i deploy .*full[- ]feature",
    r"\bkov\b",
    r"\bişten çıkar",
    r"\bterfi\b",
    r"\bmaaş\b.*(artış|zam|karar)",
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
    reasons = [pattern for pattern in UNSAFE_PATTERNS if re.search(pattern, lowered)]
    if reasons:
        return GuardrailResult(False, reasons, REFUSAL)
    return GuardrailResult(True, [])
