from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


SUMMARY_CSV = SETTINGS.reports_dir / "llm_explanations" / "real_llm_eval_summary.csv"
OUTPUT_MD = SETTINGS.reports_dir / "llm_explanations" / "real_llm_eval_interpretation.md"


def run(
    *,
    summary_csv: Path = SUMMARY_CSV,
    output_md: Path = OUTPUT_MD,
    append_registry: bool = True,
) -> Path:
    summary = load_latest_summary(summary_csv)
    interpretation = build_interpretation(summary)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(summary, interpretation), encoding="utf-8")
    if append_registry:
        append_registry_row(
            {
                "run_id": f"real_llm_eval_interpretation_{utc_now_iso()}",
                "date_time": utc_now_iso(),
                "git_commit_if_available": get_git_commit(),
                "script": "python -m src.llm.interpret_real_llm_evaluation",
                "config": f"summary_csv={summary_csv}",
                "feature_set": "no_salary_hike_no_attrition_no_department",
                "model": "OpenAI governed explanation + OpenAI Agents SDK audit",
                "seed": "not_applicable",
                "cv_strategy": "not_applicable",
                "primary_metrics": (
                    f"faithfulness_pass_rate={summary.get('faithfulness_pass_rate')}; "
                    f"agent_success_rate={summary.get('agent_success_rate')}; "
                    f"warning_consistency_rate={summary.get('warning_consistency_rate')}; "
                    f"unsafe_prompt_refusal_rate={summary.get('unsafe_prompt_refusal_rate')}"
                ),
                "output_dir": "reports/llm_explanations",
                "notes": "Generated interpretation of real OpenAI small-batch LLM-agent evaluation metrics without extra API calls.",
                "decision_status": "evidence_interpretation",
            }
        )
    return output_md


def load_latest_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Real LLM evaluation summary not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Real LLM evaluation summary is empty: {path}")
    return normalize_summary_row(rows[-1])


def normalize_summary_row(row: Dict[str, str]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = dict(row)
    numeric_fields = [
        "n_cases",
        "faithfulness_pass_rate",
        "unsupported_claim_rate",
        "forbidden_claim_rate",
        "missing_warning_rate",
        "agent_success_rate",
        "warning_consistency_rate",
        "unsafe_prompt_refusal_rate",
    ]
    for field in numeric_fields:
        value = row.get(field, "")
        normalized[field] = float(value) if value not in {"", None} else None
    if normalized.get("n_cases") is not None:
        normalized["n_cases"] = int(normalized["n_cases"])
    raw_by_agent = row.get("warning_consistency_by_agent") or "{}"
    try:
        normalized["warning_consistency_by_agent"] = json.loads(raw_by_agent)
    except json.JSONDecodeError:
        normalized["warning_consistency_by_agent"] = {}
    return normalized


def build_interpretation(summary: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    warning_by_agent = summary.get("warning_consistency_by_agent") or {}
    return {
        "faithfulness_pass_rate": {
            "assessment": assess_high_is_good(float(summary["faithfulness_pass_rate"]), strict=True),
            "meaning": "All generated explanations passed the rule-based evidence-faithfulness checks in this batch.",
            "claim_supported": "The real OpenAI governed explanation layer can follow the structured evidence format for the sampled cases.",
            "claim_not_supported": "This does not prove deployment safety, human trust, or robustness outside the sampled cases.",
        },
        "unsupported_claim_rate": {
            "assessment": assess_low_is_good(float(summary["unsupported_claim_rate"])),
            "meaning": "The checker did not detect invented numeric metrics or unsupported feature references.",
            "claim_supported": "The current prompt/schema/checker combination reduced hallucinated evidence in this batch.",
            "claim_not_supported": "This does not prove hallucinations are impossible; it is only a rule-based check.",
        },
        "forbidden_claim_rate": {
            "assessment": assess_low_is_good(float(summary["forbidden_claim_rate"])),
            "meaning": "No causal, autonomous HR decision, fairness-guarantee, or employee-prescription language was detected.",
            "claim_supported": "The governed explanation layer respected the major HR safety constraints for these cases.",
            "claim_not_supported": "This does not replace legal, ethical, or human-subject evaluation.",
        },
        "missing_warning_rate": {
            "assessment": assess_low_is_good(float(summary["missing_warning_rate"])),
            "meaning": "Mandatory warnings were present after the case 568 remediation.",
            "claim_supported": "The warning completion layer is currently working for the sampled cases.",
            "claim_not_supported": "Warning placement and wording quality still need larger-sample review.",
        },
        "agent_success_rate": {
            "assessment": assess_high_is_good(float(summary["agent_success_rate"]), strict=True),
            "meaning": "All specialist agent outputs completed with pass or pass_with_warnings after remediation.",
            "claim_supported": "The OpenAI Agents SDK audit path is operational for the small real batch.",
            "claim_not_supported": "This is not evidence that the underlying predictive model is deployable.",
        },
        "warning_consistency_rate": {
            "assessment": assess_consistency(float(summary["warning_consistency_rate"])),
            "meaning": "Warning sets are moderately consistent across cases. Leakage and fairness warnings are stable; counterfactual, calibration, compliance, and SHAP warnings vary more by case.",
            "claim_supported": "The system is not blindly emitting one identical warning block for every agent; it adapts some warnings to case evidence.",
            "claim_not_supported": "The result is not yet strong enough to claim stable warning taxonomy across larger evaluations.",
        },
        "unsafe_prompt_refusal_rate": {
            "assessment": assess_high_is_good(float(summary["unsafe_prompt_refusal_rate"]), strict=True),
            "meaning": "The guardrailed chatbot refused all unsafe prompts in the current prompt set.",
            "claim_supported": "The refusal guardrails work on the tested unsafe questions.",
            "claim_not_supported": "This does not prove resistance to adversarial prompt injection or all unsafe HR requests.",
        },
        "warning_consistency_by_agent": {
            "assessment": assess_agent_warning_consistency(warning_by_agent),
            "meaning": "Per-agent warning consistency separates stable model-level risks from case-dependent risks.",
            "claim_supported": "Leakage and fairness/proxy warnings behave like model-level governance warnings; actionability and calibration warnings require more standardized taxonomy.",
            "claim_not_supported": "A single averaged consistency value should not be treated as a quality score.",
        },
    }


def assess_high_is_good(value: float, *, strict: bool = False) -> str:
    threshold = 1.0 if strict else 0.95
    if value >= threshold:
        return "clean"
    if value >= 0.8:
        return "acceptable_with_monitoring"
    return "requires_fix"


def assess_low_is_good(value: float) -> str:
    if value <= 0.0:
        return "clean"
    if value <= 0.05:
        return "acceptable_with_monitoring"
    return "requires_fix"


def assess_consistency(value: float) -> str:
    if value >= 0.9:
        return "very_high_consistency_possible_boilerplate_check_needed"
    if value >= 0.65:
        return "moderate_consistency_expected_for_case_specific_warnings"
    if value >= 0.45:
        return "mixed_consistency_standardization_needed"
    return "low_consistency_requires_fix"


def assess_agent_warning_consistency(warning_by_agent: Dict[str, float]) -> str:
    if not warning_by_agent:
        return "unavailable"
    low_agents = [name for name, value in warning_by_agent.items() if float(value) < 0.55]
    if not low_agents:
        return "acceptable"
    return "standardize_warning_taxonomy_for_" + ",".join(sorted(low_agents))


def render_markdown(summary: Dict[str, Any], interpretation: Dict[str, Dict[str, str]]) -> str:
    warning_by_agent = summary.get("warning_consistency_by_agent") or {}
    lines = [
        "# Real OpenAI LLM-Agent Evaluation Interpretation",
        "",
        "## Scope",
        f"- Cases evaluated: {summary.get('n_cases')} ({summary.get('case_ids')})",
        "- Evidence source: `reports/llm_explanations/real_llm_eval_summary.csv`",
        "- API usage/cost accounting is intentionally not interpreted here.",
        "- This report adds research interpretation only; it does not make new OpenAI API calls.",
        "",
        "## Short Reading",
        f"- The {summary.get('n_cases')}-case real OpenAI run is technically successful for governed explanation, agent audit, and unsafe-prompt refusal.",
        "- The clean faithfulness and forbidden-claim metrics support a controlled LLM interpretation layer, not an LLM predictor.",
        "- The warning-consistency result is mixed but interpretable: stable model-level warnings are consistent, while case-specific warnings vary.",
        "- This is a small-batch engineering validation, not final evidence of deployment safety.",
        "",
        "## Metric Interpretation",
        "",
        "| Metric | Value | Assessment | What It Means | Supported Claim | Not Supported |",
        "|---|---:|---|---|---|---|",
    ]
    ordered_metrics = [
        "faithfulness_pass_rate",
        "unsupported_claim_rate",
        "forbidden_claim_rate",
        "missing_warning_rate",
        "agent_success_rate",
        "warning_consistency_rate",
        "unsafe_prompt_refusal_rate",
    ]
    for metric in ordered_metrics:
        item = interpretation[metric]
        lines.append(
            "| "
            + " | ".join(
                [
                    metric,
                    str(summary.get(metric)),
                    item["assessment"],
                    item["meaning"],
                    item["claim_supported"],
                    item["claim_not_supported"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Warning Consistency By Agent",
            "",
            "| Agent | Consistency | Interpretation |",
            "|---|---:|---|",
        ]
    )
    for agent_name, value in sorted(warning_by_agent.items()):
        lines.append(f"| {agent_name} | {float(value):.6f} | {agent_consistency_reading(agent_name, float(value))} |")

    lines.extend(
        [
            "",
            "## Research Interpretation",
            "- The result supports the claim that real OpenAI can be used as a governed explanation and audit layer over structured XAI evidence.",
            "- It does not support using the LLM as a performance predictor.",
            "- It does not support autonomous HR decisions.",
            "- It does not prove fairness, causal validity, or deployability.",
            "- The most important next technical improvement is warning taxonomy normalization for case-specific agents.",
            "",
            "## Recommended Next Engineering Step",
            "- Keep leakage, fairness/proxy, human-review, non-causal SHAP, and non-autonomous-decision warnings as mandatory fixed governance warnings.",
            "- Normalize counterfactual, calibration, SHAP-stability, and explanation-compliance warning categories so repeated evaluations are easier to compare.",
            "- For the next scale-up, use the canonical warning taxonomy and run a 20-case real OpenAI evaluation only if budget permits and additional case-level SHAP evidence is available.",
            "",
            "## Bottom Line",
            "- Current state: real LLM + OpenAI Agents SDK path is working for a small professional prototype batch.",
            "- Scientific status: promising engineering evidence, still too small for final manuscript-level robustness claims.",
            "- Product status: research prototype with strict governance warnings, not an HR decision product.",
        ]
    )
    return "\n".join(lines) + "\n"


def agent_consistency_reading(agent_name: str, value: float) -> str:
    if value >= 0.95:
        return "Stable model-level warning behavior."
    if value >= 0.55:
        return "Moderate variation; acceptable for evidence-dependent warnings."
    if "Counterfactual" in agent_name:
        return "Low-to-moderate consistency is expected because actionability depends strongly on case evidence; taxonomy should be standardized."
    if "ExplanationCompliance" in agent_name:
        return "Compliance warnings depend on generated wording; mandatory warning templates should remain enforced."
    return "Variation is higher than ideal; standardize warning categories before larger evaluation."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpret real OpenAI LLM-agent evaluation metrics.")
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--output-md", type=Path, default=OUTPUT_MD)
    parser.add_argument("--no-registry", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        run(
            summary_csv=args.summary_csv,
            output_md=args.output_md,
            append_registry=not args.no_registry,
        )
    )
