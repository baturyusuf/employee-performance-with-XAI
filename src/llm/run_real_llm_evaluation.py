from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.agents.run_llm_governance_audit import run as run_agent_audit
from src.chatbot.evaluate_chatbot import run as run_chatbot_guardrails
from src.llm.generate_governed_explanations import reason_case_ids, run as run_governed
from src.llm.usage_logger import USAGE_LOG_PATH
from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


OUTPUT_DIR = SETTINGS.reports_dir / "llm_explanations"


def run(
    *,
    limit: int = 5,
    provider: str = "openai",
    model: str | None = None,
    agent_runtime: str = "openai-agents",
    require_real_llm: bool = True,
) -> Dict[str, Path]:
    before_rows = _read_usage_rows()
    case_ids = reason_case_ids(limit)
    if not case_ids:
        raise RuntimeError("No reason-code case IDs found for real LLM evaluation.")

    run_governed(
        limit=limit,
        provider=provider,
        model=model,
        require_real_llm=require_real_llm,
    )

    agent_paths = []
    for case_id in case_ids:
        agent_paths.append(
            run_agent_audit(
                case_id=case_id,
                provider=provider,
                model=model,
                require_real_llm=require_real_llm,
                agent_runtime=agent_runtime,
            )
        )

    guardrail_paths = run_chatbot_guardrails()
    after_rows = _read_usage_rows()
    new_usage_rows = after_rows[len(before_rows):]

    summary = _build_summary(case_ids, new_usage_rows, agent_paths)
    summary_path = OUTPUT_DIR / "real_llm_eval_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    md_path = OUTPUT_DIR / "real_llm_eval_summary.md"
    _write_summary_markdown(summary, md_path, agent_paths, guardrail_paths)

    append_registry_row(
        {
            "run_id": f"real_llm_eval_{utc_now_iso()}",
            "date_time": utc_now_iso(),
            "git_commit_if_available": get_git_commit(),
            "script": "python -m src.llm.run_real_llm_evaluation",
            "config": f"limit={limit}; provider={provider}; agent_runtime={agent_runtime}; require_real_llm={require_real_llm}",
            "feature_set": "no_salary_hike_no_attrition_no_department",
            "model": model or "env_default",
            "seed": "temperature=0.0",
            "cv_strategy": "not_applicable",
            "primary_metrics": (
                f"faithfulness_pass_rate={summary['faithfulness_pass_rate']}; "
                f"agent_success_rate={summary['agent_success_rate']}; "
                f"estimated_cost_usd={summary['estimated_cost_usd']}"
            ),
            "output_dir": "reports/llm_explanations; reports/agent_audits; reports/chatbot_eval",
            "notes": "Ran real OpenAI governed explanations and OpenAI Agents SDK audits for a small case batch.",
            "decision_status": "candidate",
        }
    )
    return {"summary_csv": summary_path, "summary_md": md_path}


def _build_summary(
    case_ids: List[str],
    usage_rows: List[Dict[str, str]],
    agent_paths: List[Dict[str, Path]],
) -> Dict[str, Any]:
    eval_path = OUTPUT_DIR / "governed_explanation_eval.csv"
    eval_df = pd.read_csv(eval_path)
    faithfulness_pass_rate = float(eval_df["faithfulness_pass"].mean()) if len(eval_df) else 0.0
    unsupported_claim_rate = float((eval_df["unsupported_claim_count"] > 0).mean()) if len(eval_df) else 0.0
    forbidden_claim_rate = float((eval_df["forbidden_claim_count"] > 0).mean()) if len(eval_df) else 0.0
    missing_warning_rate = float((eval_df["missing_warning_count"] > 0).mean()) if len(eval_df) else 0.0

    agent_statuses = []
    warning_sets_by_agent: Dict[str, List[tuple[str, ...]]] = {}
    for paths in agent_paths:
        payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
        for item in payload.get("agent_syntheses", []):
            agent_name = str(item.get("agent_name", "unknown"))
            status = str(item.get("status", "unknown"))
            agent_statuses.append(status)
            warning_ids = item.get("normalized_warning_ids")
            warning_values = warning_ids if isinstance(warning_ids, list) else item.get("required_warnings", [])
            warning_sets_by_agent.setdefault(agent_name, []).append(
                tuple(sorted(str(value) for value in warning_values))
            )

    accepted_statuses = {"pass", "pass_with_warnings"}
    agent_success_rate = (
        sum(1 for status in agent_statuses if status in accepted_statuses) / len(agent_statuses)
        if agent_statuses
        else 0.0
    )
    per_agent_warning_consistency = {
        agent_name: _warning_consistency_rate(warning_sets)
        for agent_name, warning_sets in sorted(warning_sets_by_agent.items())
    }
    warning_consistency_rate = (
        sum(per_agent_warning_consistency.values()) / len(per_agent_warning_consistency)
        if per_agent_warning_consistency
        else 1.0
    )

    guardrail_path = SETTINGS.reports_dir / "chatbot_eval" / "guardrail_evaluation.csv"
    guardrail_df = pd.read_csv(guardrail_path)
    if "prompt_type" in guardrail_df.columns:
        unsafe_guardrail_df = guardrail_df[guardrail_df["prompt_type"].isin(["unsafe", "adversarial"])]
    else:
        unsafe_guardrail_df = guardrail_df
    unsafe_prompt_refusal_rate = (
        float(unsafe_guardrail_df["pass"].mean()) if len(unsafe_guardrail_df) else 0.0
    )

    total_input = sum(int(float(row.get("input_tokens") or 0)) for row in usage_rows)
    total_output = sum(int(float(row.get("output_tokens") or 0)) for row in usage_rows)
    total_tokens = sum(int(float(row.get("total_tokens") or 0)) for row in usage_rows)
    estimated_cost = round(sum(float(row.get("estimated_cost_usd") or 0.0) for row in usage_rows), 6)

    return {
        "n_cases": len(case_ids),
        "case_ids": ";".join(case_ids),
        "faithfulness_pass_rate": round(faithfulness_pass_rate, 6),
        "unsupported_claim_rate": round(unsupported_claim_rate, 6),
        "forbidden_claim_rate": round(forbidden_claim_rate, 6),
        "missing_warning_rate": round(missing_warning_rate, 6),
        "agent_success_rate": round(agent_success_rate, 6),
        "warning_consistency_rate": round(warning_consistency_rate, 6),
        "warning_consistency_by_agent": json.dumps(per_agent_warning_consistency, sort_keys=True),
        "unsafe_prompt_refusal_rate": round(unsafe_prompt_refusal_rate, 6),
        "usage_log_rows": len(usage_rows),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost,
        "notes": "Real OpenAI small-batch evaluation; warning consistency uses same-batch warning-set overlap, not repeated stochastic reruns.",
    }


def _warning_consistency_rate(warning_sets: List[tuple[str, ...]]) -> float:
    if len(warning_sets) <= 1:
        return 1.0
    reference = set(warning_sets[0])
    scores = []
    for item in warning_sets[1:]:
        current = set(item)
        union = reference | current
        if not union:
            scores.append(1.0)
        else:
            scores.append(len(reference & current) / len(union))
    return sum(scores) / len(scores) if scores else 1.0


def _write_summary_markdown(
    summary: Dict[str, Any],
    output_path: Path,
    agent_paths: List[Dict[str, Path]],
    guardrail_paths: Dict[str, str],
) -> None:
    lines = [
        "# Real OpenAI LLM-Agent Evaluation Summary",
        "",
        "## Scope",
        f"- Cases: {summary['n_cases']} ({summary['case_ids']})",
        "- Model/provider: OpenAI API with OpenAI Agents SDK for governance audit.",
        "- Cost policy: small-batch evaluation with gpt-5.4-mini and temperature 0.",
        "",
        "## Results",
        f"- Faithfulness pass rate: {summary['faithfulness_pass_rate']}",
        f"- Unsupported claim rate: {summary['unsupported_claim_rate']}",
        f"- Forbidden claim rate: {summary['forbidden_claim_rate']}",
        f"- Missing warning rate: {summary['missing_warning_rate']}",
        f"- Agent success rate: {summary['agent_success_rate']}",
        f"- Warning consistency rate: {summary['warning_consistency_rate']}",
        f"- Warning consistency by agent: `{summary['warning_consistency_by_agent']}`",
        f"- Unsafe prompt refusal rate: {summary['unsafe_prompt_refusal_rate']}",
        "",
        "## Usage",
        f"- Usage log rows: {summary['usage_log_rows']}",
        f"- Input tokens: {summary['input_tokens']}",
        f"- Output tokens: {summary['output_tokens']}",
        f"- Total tokens: {summary['total_tokens']}",
        f"- Estimated cost USD: {summary['estimated_cost_usd']}",
        "",
        "## Artifacts",
        "- Governed explanations: `reports/llm_explanations/governed_explanation_examples.md`",
        "- Governed explanation eval: `reports/llm_explanations/governed_explanation_eval.csv`",
        "- Usage log: `reports/llm_explanations/llm_usage_log.csv`",
        f"- Guardrail eval: `{guardrail_paths['markdown']}`",
    ]
    for paths in agent_paths:
        lines.append(f"- Agent audit: `{paths['markdown']}`")
    lines.extend(
        [
            "",
            "## Limitations",
            "- This is a small-batch technical validation, not a human evaluation.",
            "- Warning consistency is computed across same-batch warning sets, not repeated LLM reruns.",
            "- Cost is estimated from logged token usage and configured model prices; billing dashboard remains the source of truth.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_usage_rows() -> List[Dict[str, str]]:
    if not USAGE_LOG_PATH.exists():
        return []
    with USAGE_LOG_PATH.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small real OpenAI LLM-agent evaluation.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--provider", choices=["openai"], default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--agent-runtime", choices=["openai-agents"], default="openai-agents")
    parser.add_argument("--require-real-llm", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        run(
            limit=args.limit,
            provider=args.provider,
            model=args.model,
            agent_runtime=args.agent_runtime,
            require_real_llm=args.require_real_llm,
        )
    )
