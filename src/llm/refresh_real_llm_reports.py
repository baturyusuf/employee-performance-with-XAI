from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.llm.evidence_schema import CompleteCaseEvidence
from src.llm.faithfulness_checker import check_faithfulness
from src.llm.generate_governed_explanations import OUTPUT_DIR, reason_case_ids, write_examples_markdown
from src.llm.run_real_llm_evaluation import _warning_consistency_rate, _write_summary_markdown
from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


SUMMARY_CSV = OUTPUT_DIR / "real_llm_eval_summary.csv"


def run(limit: int = 10, append_registry: bool = True) -> Dict[str, str]:
    case_ids = [case_id for case_id in reason_case_ids(limit) if (OUTPUT_DIR / f"case_{case_id}" / "governed_explanation.json").exists()]
    if not case_ids:
        raise RuntimeError("No saved governed explanations found to refresh.")

    rows = []
    eval_rows = []
    for case_id in case_ids:
        evidence = CompleteCaseEvidence.from_reports(case_id=case_id)
        path = OUTPUT_DIR / f"case_{case_id}" / "governed_explanation.json"
        output = json.loads(path.read_text(encoding="utf-8"))
        compliance = check_faithfulness(output, evidence.to_dict()).to_dict()
        output["faithfulness_check"] = compliance
        output["unsupported_claims_detected"] = compliance["unsupported_claims"]
        path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
        rows.append({"case_id": case_id, "evidence": evidence.to_dict(), "output": output})
        eval_rows.append(
            {
                "case_id": case_id,
                "faithfulness_pass": compliance["faithfulness_pass"],
                "score": compliance["score"],
                "unsupported_claim_count": len(compliance["unsupported_claims"]),
                "forbidden_claim_count": len(compliance["forbidden_claims"]),
                "missing_warning_count": len(compliance["missing_warnings"]),
            }
        )

    write_examples_markdown(rows, OUTPUT_DIR / "governed_explanation_examples.md")
    pd.DataFrame(eval_rows).to_csv(OUTPUT_DIR / "governed_explanation_eval.csv", index=False)
    summary = build_refreshed_summary(case_ids, eval_rows)
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)
    agent_paths = [
        {
            "json": SETTINGS.reports_dir / "agent_audits" / f"openai_agents_sdk_case_{case_id}_governance_audit.json",
            "markdown": SETTINGS.reports_dir / "agent_audits" / f"openai_agents_sdk_case_{case_id}_governance_audit.md",
        }
        for case_id in case_ids
    ]
    guardrail_paths = {
        "markdown": str(SETTINGS.reports_dir / "chatbot_eval" / "guardrail_evaluation.md"),
        "csv": str(SETTINGS.reports_dir / "chatbot_eval" / "guardrail_evaluation.csv"),
    }
    _write_summary_markdown(summary, OUTPUT_DIR / "real_llm_eval_summary.md", agent_paths, guardrail_paths)
    if append_registry:
        append_registry_row(
            {
                "run_id": f"real_llm_report_refresh_{utc_now_iso()}",
                "date_time": utc_now_iso(),
                "git_commit_if_available": get_git_commit(),
                "script": "python -m src.llm.refresh_real_llm_reports",
                "config": f"limit={limit}; no API calls; recompute faithfulness and summary from saved outputs",
                "feature_set": "no_salary_hike_no_attrition_no_department",
                "model": "saved_openai_outputs",
                "seed": "not_applicable",
                "cv_strategy": "not_applicable",
                "primary_metrics": (
                    f"faithfulness_pass_rate={summary['faithfulness_pass_rate']}; "
                    f"agent_success_rate={summary['agent_success_rate']}; "
                    f"warning_consistency_rate={summary['warning_consistency_rate']}"
                ),
                "output_dir": "reports/llm_explanations",
                "notes": "Refreshed real LLM reports after checker warning-variant remediation without new API calls.",
                "decision_status": "accepted",
            }
        )
    return {
        "summary_csv": str(SUMMARY_CSV),
        "summary_md": str(OUTPUT_DIR / "real_llm_eval_summary.md"),
        "eval_csv": str(OUTPUT_DIR / "governed_explanation_eval.csv"),
    }


def build_refreshed_summary(case_ids: List[str], eval_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    eval_df = pd.DataFrame(eval_rows)
    agent_statuses = []
    warning_sets_by_agent: Dict[str, List[tuple[str, ...]]] = {}
    for case_id in case_ids:
        path = SETTINGS.reports_dir / "agent_audits" / f"openai_agents_sdk_case_{case_id}_governance_audit.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("agent_syntheses", []):
            agent_name = str(item.get("agent_name", "unknown"))
            agent_statuses.append(str(item.get("status", "unknown")))
            values = item.get("normalized_warning_ids") or item.get("required_warnings") or []
            warning_sets_by_agent.setdefault(agent_name, []).append(tuple(sorted(str(value) for value in values)))
    per_agent_warning_consistency = {
        agent_name: _warning_consistency_rate(warning_sets)
        for agent_name, warning_sets in sorted(warning_sets_by_agent.items())
    }
    accepted_statuses = {"pass", "pass_with_warnings"}
    guardrail_path = SETTINGS.reports_dir / "chatbot_eval" / "guardrail_evaluation.csv"
    guardrail_df = pd.read_csv(guardrail_path) if guardrail_path.exists() else pd.DataFrame()
    if "prompt_type" in guardrail_df.columns:
        unsafe_guardrail_df = guardrail_df[guardrail_df["prompt_type"].isin(["unsafe", "adversarial"])]
    else:
        unsafe_guardrail_df = guardrail_df
    old_summary = read_existing_summary()
    return {
        "n_cases": len(case_ids),
        "case_ids": ";".join(case_ids),
        "faithfulness_pass_rate": round(float(eval_df["faithfulness_pass"].mean()), 6),
        "unsupported_claim_rate": round(float((eval_df["unsupported_claim_count"] > 0).mean()), 6),
        "forbidden_claim_rate": round(float((eval_df["forbidden_claim_count"] > 0).mean()), 6),
        "missing_warning_rate": round(float((eval_df["missing_warning_count"] > 0).mean()), 6),
        "agent_success_rate": round(
            sum(1 for status in agent_statuses if status in accepted_statuses) / len(agent_statuses),
            6,
        )
        if agent_statuses
        else 0.0,
        "warning_consistency_rate": round(
            sum(per_agent_warning_consistency.values()) / len(per_agent_warning_consistency),
            6,
        )
        if per_agent_warning_consistency
        else 1.0,
        "warning_consistency_by_agent": json.dumps(per_agent_warning_consistency, sort_keys=True),
        "unsafe_prompt_refusal_rate": round(float(unsafe_guardrail_df["pass"].mean()), 6) if len(unsafe_guardrail_df) else 0.0,
        "usage_log_rows": old_summary.get("usage_log_rows", ""),
        "input_tokens": old_summary.get("input_tokens", ""),
        "output_tokens": old_summary.get("output_tokens", ""),
        "total_tokens": old_summary.get("total_tokens", ""),
        "estimated_cost_usd": old_summary.get("estimated_cost_usd", ""),
        "notes": "Refreshed from saved real OpenAI outputs after checker warning-variant remediation; no new API calls.",
    }


def read_existing_summary() -> Dict[str, str]:
    if not SUMMARY_CSV.exists():
        return {}
    with SUMMARY_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh real LLM reports from saved outputs without API calls.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--no-registry", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run(limit=args.limit, append_registry=not args.no_registry))
