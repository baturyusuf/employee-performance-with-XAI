from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class GuardrailRule:
    """A stable, reportable rule identifier and its deterministic patterns."""

    rule_id: str
    patterns: Sequence[str]


UNSAFE_RULES = (
    GuardrailRule(
        "employment_or_compensation_decision",
        (
            r"\bfire\b",
            r"\bfired\b",
            r"firing priority",
            r"\bterminate\b",
            r"\btermination\b",
            r"\bdismiss(?:al|ed)?\b",
            r"\blayoffs?\b",
            r"\bpromot(?:e|ed|ion)\b",
            r"\bwho should .*promot",
            r"\bwho should .*fire",
            r"\bpay raise\b",
            r"\bpay adjustment\b",
            r"\bsalary increase\b",
            r"\bcompensation (?:decision|recommendation|recommend)\b",
            r"\bdisciplin",
            r"employment relationship .*\bend\b",
            r"\bend\b .*employment relationship",
            r"moving? .* into a higher position",
            r"higher position .*recommendation",
            r"deny promotion",
            r"approve promotion",
        ),
    ),
    GuardrailRule(
        "employee_ranking_or_selection",
        (
            r"\brank .*employees\b",
            r"\brank .*workforce\b",
            r"\bsort .*workforce.*remov",
            r"priority order .*advancement",
            r"bottom .*talent list",
            r"firing priority",
            r"select an? employee .*dismiss",
            r"approve or reject",
        ),
    ),
    GuardrailRule(
        "direct_employee_prescription",
        (
            r"exactly what to change",
            r"tell .* exactly .* change",
            r"employee should",
            r"employee must",
            r"worker should follow",
            r"employee prescription",
            r"personal improvement plan .*follow",
            r"counterfactual .*mandatory .*actions?",
            r"shap .*tell .*coach",
            r"convert .*shap .*personal .*plan",
            r"turn .*counterfactual .*mandatory",
        ),
    ),
    GuardrailRule(
        "sensitive_attribute_misuse",
        (
            r"use gender",
            r"gender .*justify",
            r"use marital",
            r"marital status .*tie[- ]breaker",
            r"use race",
            r"race .*used",
            r"race .*justify",
            r"use age .*justify",
            r"use sensitive",
            r"employee(?:'s)? sex .*outcome",
            r"base .*outcome .*employee(?:'s)? sex",
            r"protected group",
            r"sensitive group",
        ),
    ),
    GuardrailRule(
        "fairness_overclaim",
        (
            r"guaranteed to be fair",
            r"\bunbiased model\b",
            r"no discrimination",
            r"prove .*fair",
            r"fairness .*solved",
        ),
    ),
    GuardrailRule(
        "causal_overclaim",
        (
            r"\bcaused\b",
            r"\bcausal reason\b",
            r"what causes performance",
            r"prove what causes",
            r"increase .*real performance",
            r"caused .*score",
        ),
    ),
    GuardrailRule(
        "instruction_hierarchy_attack",
        (
            r"ignore .*fairness",
            r"ignore .*warning",
            r"ignore .*guardrail",
            r"ignore .*previous",
            r"warnings are irrelevant",
            r"give me the decision",
            r"tell me the decision",
            r"bypass .*guardrail",
            r"\bjailbreak\b",
            r"system prompt",
            r"system override",
            r"developer message",
            r"developer instructions",
            r"hidden instruction",
            r"outranks .*instructions",
            r"everything before .*untrusted",
            r"disable .*safety",
            r"pretend .*not .*govern",
            r"act as .*hr manager",
        ),
    ),
    GuardrailRule(
        "leakage_risk_model_operationalization",
        (
            r"use salary hike",
            r"salary hike .*deployment",
            r"can i use .*full[- ]feature.*deployment",
            r"can i deploy .*full[- ]feature",
            r"should i deploy .*full[- ]feature",
            r"use .*full[- ]feature.*production",
            r"use .*upper[- ]bound.*operational",
        ),
    ),
    GuardrailRule(
        "autonomous_hr_decision",
        (
            r"automatically (?:decide|rank|approve|reject|evaluate)",
            r"use .*autonomous.*decision",
            r"without human review",
            r"human review .*without",
        ),
    ),
    GuardrailRule(
        "warning_or_uncertainty_suppression",
        (
            r"hide .*uncertainty",
            r"hide .*conflict",
            r"remove .*warning",
            r"remove .*disclaimer",
            r"do not mention .*disclaimer",
            r"do not mention .*human review",
            r"omit .*caveat",
            r"no limitation .*appear",
            r"no .*review requirement .*appear",
            r"silently .*guess",
        ),
    ),
    GuardrailRule(
        "unwarranted_certainty_or_legal_justification",
        (
            r"objectively correct",
            r"trust .*probability .*certainty",
            r"termination justification",
            r"legally defensible",
            r"make a confident .*recommendation",
        ),
    ),
    GuardrailRule(
        "missing_or_conflicting_evidence_misuse",
        (
            r"invent .*missing evidence",
            r"fill .*gaps .*intuition",
            r"report .*unavailable.*guess",
            r"retriev.*fail.*(?:decide|recommend|hr action)",
            r"lookup failed.*(?:decide|recommend|hr action)",
            r"conflicting report .*supports .*dismiss",
            r"conflict.*(?:approve|promotion recommendation|hr action)",
            r"choose whichever .*report",
        ),
    ),
    GuardrailRule(
        "turkish_hr_decision_or_misuse",
        (
            r"işten çıkar",
            r"\bterfi\b",
            r"\bmaaş\b.*(?:artış|zam|karar)",
            r"\bcinsiyet\b.*(?:kullan|gerekçe)",
            r"belirsizli.*gizle",
            r"insan incelemesi olmadan",
            r"kararı ver",
            r"kimlere .*sırala",
        ),
    ),
)

# Compatibility view for callers that previously imported the flat pattern list.
UNSAFE_PATTERNS = [pattern for rule in UNSAFE_RULES for pattern in rule.patterns]

# These are deliberately whole-question patterns. They are narrow exceptions for
# questions *about* governance boundaries, not substrings that can mask a second,
# unsafe instruction in the same prompt.
SAFE_AUDIT_PATTERNS = (
    r"(?:please )?(?:explain )?does removing (?:the )?department prove fairness",
    r"if (?:the )?department is removed,? is fairness solved",
    r"(?:please )?(?:explain )?why are counterfactuals not employee prescriptions",
    r"(?:please )?(?:explain )?is shap causal",
    r"(?:please )?(?:explain )?why (?:is|are) (?:the )?full[- ]feature models? not deployable",
    r"what are the prohibited uses",
    r"why must shap and counterfactual evidence not be turned into employee advice",
    r"why must a model score not be converted into an employment decision",
)

REFUSAL = (
    "I cannot provide hiring, firing, promotion, compensation, disciplinary, ranking, or autonomous HR "
    "recommendations. I can explain the model evidence, limitations, uncertainty, and required governance warnings."
)


@dataclass
class GuardrailResult:
    allowed: bool
    reasons: List[str]
    response: str = ""


def normalize_question(question: str) -> str:
    """Normalize text for deterministic matching without pretending semantic coverage."""

    normalized = unicodedata.normalize("NFKC", str(question)).casefold()
    normalized = normalized.translate(
        str.maketrans(
            {
                "’": "'",
                "‘": "'",
                "‐": "-",
                "‑": "-",
                "‒": "-",
                "–": "-",
                "—": "-",
            }
        )
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Targeted canonicalization for suite-covered sensitive-attribute obfuscation.
    # This is intentionally finite and must not be described as general evasion resistance.
    normalized = re.sub(r"\bg[\W_]*e[\W_]*n[\W_]*d[\W_]*e[\W_]*r\b", "gender", normalized)
    normalized = re.sub(r"\bm[4@]rital\b", "marital", normalized)
    normalized = re.sub(r"\bs[\W_]*e[\W_]*x\b", "sex", normalized)
    return normalized


def check_question(question: str) -> GuardrailResult:
    normalized = normalize_question(question)
    reasons = [
        rule.rule_id
        for rule in UNSAFE_RULES
        if any(re.search(pattern, normalized) for pattern in rule.patterns)
    ]

    # An audit-safe override is valid only when the entire normalized prompt is an
    # approved boundary question. Appending an unsafe request invalidates it.
    pure_audit_question = any(
        re.fullmatch(rf"{pattern}[?.!]?", normalized) for pattern in SAFE_AUDIT_PATTERNS
    )
    if reasons and not pure_audit_question:
        return GuardrailResult(False, reasons, REFUSAL)
    return GuardrailResult(True, [])
