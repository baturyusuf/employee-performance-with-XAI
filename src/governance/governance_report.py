from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


def run() -> dict[str, str]:
    output_dir = SETTINGS.reports_dir / "governance_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    gxair_path = output_dir / "gxair_llm_agent_dashboard.csv"
    eval_path = SETTINGS.reports_dir / "llm_explanations" / "llm_governance_eval_summary.csv"
    gxair = pd.read_csv(gxair_path) if gxair_path.exists() else pd.DataFrame()
    eval_df = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
    path = output_dir / "final_llm_agent_research_summary.md"
    readiness = gxair["overall_readiness_label"].iloc[0] if not gxair.empty else "unavailable"
    lines = [
        "# Final LLM-Agent Research Summary",
        "",
        "## Positioning",
        "The LLM is not the predictive model. XGBoost remains the performance predictor, SHAP and audit modules produce structured evidence, and the LLM/agent layer interprets that evidence under governance constraints.",
        "",
        "## Evidence Flow",
        "XGBoost prediction -> grouped SHAP and reliability evidence -> structured evidence schema -> governed explanation -> multi-agent audit -> guardrailed chatbot.",
        "",
        "## Agent Audit Functions",
        "- Leakage: checks leakage-risk exclusions and labels full-feature baselines as diagnostic only.",
        "- Fairness/proxy: checks subgroup gaps, low support, and JobRole/Department proxy risk.",
        "- Calibration: warns that probabilities are approximate.",
        "- SHAP stability: checks attribution stability and non-causal wording.",
        "- Counterfactual actionability: separates technical validity from employee actionability.",
        "- Explanation compliance: detects unsupported or forbidden explanation claims.",
        "",
        "## Chatbot Guardrails",
        "The chatbot refuses hiring, firing, promotion, compensation, disciplinary, autonomous decision, fairness-guarantee, sensitive-attribute justification, and employee-prescription prompts.",
        "",
        "## Automatic Evaluation",
    ]
    if not eval_df.empty:
        row = eval_df.iloc[0]
        lines.extend(
            [
                f"- Faithfulness pass rate: {row.get('faithfulness_pass_rate', 'unavailable')}",
                f"- Unsupported claim rate: {row.get('unsupported_claim_rate', 'unavailable')}",
                f"- Forbidden claim rate: {row.get('forbidden_claim_rate', 'unavailable')}",
                f"- Missing warning rate: {row.get('missing_warning_rate', 'unavailable')}",
                f"- Unsafe prompt refusal rate: {row.get('unsafe_prompt_refusal_rate', 'unavailable')}",
            ]
        )
    else:
        lines.append("Automatic evaluation summary is unavailable.")
    lines.extend(
        [
            "",
            "## G-XAIR Readiness",
            f"Overall readiness label: `{readiness}`.",
            "",
            "## Limitations",
            "- No human-subject evaluation was performed.",
            "- The offline LLM stub is deterministic and conservative; external LLM behavior must be separately validated.",
            "- The evidence is based on a public cross-sectional dataset and does not support causal claims.",
            "- The system is a research prototype, not an HR decision system.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_registry_row(
        {
            "run_id": f"final_llm_agent_research_summary_{utc_now_iso()}",
            "date_time": utc_now_iso(),
            "git_commit_if_available": get_git_commit(),
            "script": "python -m src.governance.governance_report",
            "config": "final LLM-agent generated reports",
            "feature_set": "no_salary_hike_no_attrition_no_department",
            "model": "governance_report",
            "seed": "deterministic",
            "cv_strategy": "not_applicable",
            "primary_metrics": "readiness label; automatic evaluation summary",
            "output_dir": "reports/governance_reports",
            "notes": "Generated final LLM-agent research summary.",
            "decision_status": "candidate",
        }
    )
    return {"summary": str(path)}


if __name__ == "__main__":
    print(run())

