from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


OUTPUT_DIR = SETTINGS.reports_dir / "manuscript_assets"


AGENT_ROLE_ROWS: List[Dict[str, str]] = [
    {
        "agent": "LeakageAuditAgent",
        "input": "LeakageEvidence, feature policy, leakage-safe/full-feature metrics",
        "output": "Leakage risk level, leakage warnings, deployability status",
        "governance_function": "Prevents full-feature leakage baselines from being presented as deployable models.",
    },
    {
        "agent": "FairnessProxyAuditAgent",
        "input": "FairnessEvidence, subgroup gaps, proxy warnings, feature policy",
        "output": "Fairness/proxy risk level, subgroup warnings, proxy warnings",
        "governance_function": "Separates diagnostic subgroup gaps from unsupported discrimination/fairness claims.",
    },
    {
        "agent": "CalibrationAuditAgent",
        "input": "CalibrationEvidence and probability quality metrics",
        "output": "Calibration risk level and probability-use warnings",
        "governance_function": "Prevents raw probabilities from being treated as objective certainty.",
    },
    {
        "agent": "ShapStabilityAuditAgent",
        "input": "SHAP stability metrics and feature rankings",
        "output": "Explanation stability assessment and non-causality warnings",
        "governance_function": "Distinguishes model attribution from causal explanation.",
    },
    {
        "agent": "CounterfactualActionabilityAgent",
        "input": "Counterfactual validity, changed features, actionability labels",
        "output": "Actionability risk level and intervention-scope warnings",
        "governance_function": "Prevents counterfactuals from becoming employee prescriptions.",
    },
    {
        "agent": "ExplanationComplianceAgent",
        "input": "LLM explanation and structured evidence",
        "output": "Compliance score, unsupported claims, missing warnings",
        "governance_function": "Checks faithfulness and forbidden HR/causal/fairness claims.",
    },
    {
        "agent": "SupervisorGovernanceAgent",
        "input": "All specialist agent outputs",
        "output": "Overall readiness label and required human-review warnings",
        "governance_function": "Aggregates governance evidence without making HR decisions.",
    },
]


def run(output_dir: Path = OUTPUT_DIR, append_registry: bool = True) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    agent_roles = pd.DataFrame(AGENT_ROLE_ROWS)
    paths["agent_roles_csv"] = str(output_dir / "agent_roles_table.csv")
    agent_roles.to_csv(paths["agent_roles_csv"], index=False)

    warning_taxonomy_path = SETTINGS.reports_dir / "governance_reports" / "warning_taxonomy.csv"
    warning_taxonomy = pd.read_csv(warning_taxonomy_path) if warning_taxonomy_path.exists() else pd.DataFrame()
    paths["warning_taxonomy_csv"] = str(output_dir / "warning_taxonomy_table.csv")
    warning_taxonomy.to_csv(paths["warning_taxonomy_csv"], index=False)

    llm_metrics = _llm_metric_rows()
    paths["llm_eval_csv"] = str(output_dir / "llm_eval_metrics_table.csv")
    pd.DataFrame(llm_metrics).to_csv(paths["llm_eval_csv"], index=False)

    guardrail_table = _guardrail_summary_table()
    paths["guardrail_eval_csv"] = str(output_dir / "guardrail_eval_table.csv")
    guardrail_table.to_csv(paths["guardrail_eval_csv"], index=False)

    tables_md = output_dir / "llm_agent_extension_tables.md"
    tables_md.write_text(
        render_tables_markdown(agent_roles, warning_taxonomy, pd.DataFrame(llm_metrics), guardrail_table),
        encoding="utf-8",
    )
    paths["tables_md"] = str(tables_md)

    summary_md = output_dir / "llm_agent_extension_summary.md"
    summary_md.write_text(render_summary_markdown(llm_metrics, guardrail_table), encoding="utf-8")
    paths["summary_md"] = str(summary_md)

    if append_registry:
        append_registry_row(
            {
                "run_id": f"llm_agent_manuscript_assets_{utc_now_iso()}",
                "date_time": utc_now_iso(),
                "git_commit_if_available": get_git_commit(),
                "script": "python -m src.governance.manuscript_assets",
                "config": "agent roles + warning taxonomy + LLM eval + guardrail tables",
                "feature_set": "no_salary_hike_no_attrition_no_department",
                "model": "manuscript_assets",
                "seed": "deterministic",
                "cv_strategy": "not_applicable",
                "primary_metrics": "; ".join(f"{row['metric']}={row['value']}" for row in llm_metrics),
                "output_dir": "reports/manuscript_assets",
                "notes": "Generated manuscript-ready LLM-agent extension tables and summary.",
                "decision_status": "candidate",
            }
        )
    return paths


def _llm_metric_rows() -> List[Dict[str, Any]]:
    real_path = SETTINGS.reports_dir / "llm_explanations" / "real_llm_eval_summary.csv"
    if not real_path.exists():
        return [{"metric": "real_openai_eval", "value": "unavailable", "interpretation": "Real OpenAI evaluation not run."}]
    row = pd.read_csv(real_path).tail(1).iloc[0].to_dict()
    return [
        {
            "metric": "n_cases",
            "value": row.get("n_cases"),
            "interpretation": "Small real OpenAI batch size; not human-subject validation.",
        },
        {
            "metric": "faithfulness_pass_rate",
            "value": row.get("faithfulness_pass_rate"),
            "interpretation": "Generated explanations passed rule-based evidence-faithfulness checks.",
        },
        {
            "metric": "unsupported_claim_rate",
            "value": row.get("unsupported_claim_rate"),
            "interpretation": "No detected invented metrics or unsupported features.",
        },
        {
            "metric": "forbidden_claim_rate",
            "value": row.get("forbidden_claim_rate"),
            "interpretation": "No detected causal/autonomous/fairness-guarantee/employee-prescription language.",
        },
        {
            "metric": "agent_success_rate",
            "value": row.get("agent_success_rate"),
            "interpretation": "Specialist OpenAI Agents SDK audits completed with acceptable statuses.",
        },
        {
            "metric": "warning_consistency_rate",
            "value": row.get("warning_consistency_rate"),
            "interpretation": "Moderate case-level warning consistency; taxonomy normalization is required for larger claims.",
        },
        {
            "metric": "unsafe_prompt_refusal_rate",
            "value": row.get("unsafe_prompt_refusal_rate"),
            "interpretation": "Unsafe/adversarial prompt refusal rate from the latest guardrail report.",
        },
    ]


def _guardrail_summary_table() -> pd.DataFrame:
    path = SETTINGS.reports_dir / "chatbot_eval" / "guardrail_evaluation.csv"
    if not path.exists():
        return pd.DataFrame([{"prompt_type": "unavailable", "n": 0, "pass_rate": 0.0}])
    df = pd.read_csv(path)
    return (
        df.groupby("prompt_type")
        .agg(n=("pass", "size"), pass_rate=("pass", "mean"))
        .reset_index()
    )


def render_tables_markdown(
    agent_roles: pd.DataFrame,
    warning_taxonomy: pd.DataFrame,
    llm_metrics: pd.DataFrame,
    guardrail_table: pd.DataFrame,
) -> str:
    lines = [
        "# LLM-Agent Extension Manuscript Tables",
        "",
        "## Table A. Agent Roles, Inputs, Outputs, and Governance Functions",
        "",
        dataframe_to_markdown(agent_roles),
        "",
        "## Table B. Required Warnings and Canonical Warning Taxonomy",
        "",
        dataframe_to_markdown(warning_taxonomy[["warning_id", "category", "severity", "mandatory", "canonical_message"]])
        if not warning_taxonomy.empty
        else "Warning taxonomy unavailable.",
        "",
        "## Table C. Real OpenAI LLM-Agent Evaluation Metrics",
        "",
        dataframe_to_markdown(llm_metrics),
        "",
        "## Table D. Guardrailed Chatbot Evaluation",
        "",
        dataframe_to_markdown(guardrail_table),
        "",
        "## Figure Captions",
        "",
        "- Figure A. Evidence flow from leakage-safe XGBoost to structured XAI evidence, governed LLM explanation, multi-agent audit, and guardrailed chatbot.",
        "- Figure B. Multi-agent governance architecture showing specialist audit agents and supervisor aggregation.",
        "- Figure C. Guardrailed chatbot workflow separating safe audit questions from prohibited HR decision requests.",
        "- Figure D. G-XAIR dashboard with performance, leakage, explanation stability, calibration, fairness/proxy, actionability, and LLM governance compliance components.",
    ]
    return "\n".join(lines) + "\n"


def render_summary_markdown(llm_metrics: List[Dict[str, Any]], guardrail_table: pd.DataFrame) -> str:
    metric_map = {row["metric"]: row["value"] for row in llm_metrics}
    guardrail_text = dataframe_to_markdown(guardrail_table)
    lines = [
        "# LLM-Agent Extension Manuscript Support",
        "",
        "## Contribution Statement",
        "This extension proposes an LLM-assisted multi-agent XAI governance framework for employee performance prediction. The predictive model remains leakage-safe XGBoost. SHAP, calibration diagnostics, subgroup fairness/proxy analysis, leakage analysis, and counterfactual actionability modules generate structured evidence. OpenAI-backed LLM and agent components interpret this evidence under explicit governance constraints, while a guardrailed chatbot exposes the results without allowing autonomous HR decisions or unsupported claims.",
        "",
        "## Research Questions",
        "RQ1. How much does apparent employee performance prediction performance depend on leakage-risk variables?",
        "",
        "RQ2. Can leakage-safe XAI evidence support a governance-aware model selection process?",
        "",
        "RQ3. Can an LLM-assisted multi-agent governance layer generate faithful and compliant explanations from structured XAI evidence?",
        "",
        "RQ4. Can the agent system detect and communicate leakage, proxy fairness, calibration, explanation stability, and counterfactual actionability risks?",
        "",
        "RQ5. Can a guardrailed chatbot answer model-audit questions while refusing unsafe HR decision requests?",
        "",
        "## LLM/Agent Methodology",
        "The LLM receives constrained JSON evidence rather than raw uncontrolled data. Evidence includes prediction metadata, grouped SHAP attribution summaries, leakage status, fairness/proxy diagnostics, calibration metrics, counterfactual actionability summaries, and governance warnings. The real implementation uses OpenAI API structured outputs and OpenAI Agents SDK specialist agents. The deterministic offline stub is retained for tests and reproducibility.",
        "",
        "## Real OpenAI Small-Batch Evaluation",
        f"- Cases: {metric_map.get('n_cases', 'unavailable')}.",
        f"- Faithfulness pass rate: {metric_map.get('faithfulness_pass_rate', 'unavailable')}.",
        f"- Unsupported claim rate: {metric_map.get('unsupported_claim_rate', 'unavailable')}.",
        f"- Forbidden claim rate: {metric_map.get('forbidden_claim_rate', 'unavailable')}.",
        f"- Agent success rate: {metric_map.get('agent_success_rate', 'unavailable')}.",
        f"- Warning consistency rate: {metric_map.get('warning_consistency_rate', 'unavailable')}.",
        f"- Unsafe prompt refusal rate: {metric_map.get('unsafe_prompt_refusal_rate', 'unavailable')}.",
        "",
        "## Chatbot Guardrail Evaluation",
        "",
        guardrail_text,
        "",
        "## Limitations",
        "- The LLM layer does not establish causality, fairness, or deployment readiness.",
        "- The system is evaluated on a public cross-sectional dataset and requires external validation.",
        "- Automatic checks are not a substitute for human-subject evaluation.",
        "- Guardrails reduce unsafe behavior on tested prompts but do not prove robustness to all adversarial prompts.",
        "",
        "## Suggested Tables and Figures",
        "- Table A. Agent roles, inputs, outputs, and governance functions.",
        "- Table B. Required warnings and forbidden claims in governed explanations.",
        "- Table C. Real OpenAI faithfulness and guardrail compliance results.",
        "- Table D. Unsafe/adversarial chatbot prompt refusal evaluation.",
        "- Figure A. LLM-assisted multi-agent XAI governance architecture.",
        "- Figure B. Evidence flow from XGBoost and SHAP to agents and chatbot.",
        "- Figure C. Guardrailed chatbot workflow.",
        "- Figure D. G-XAIR component dashboard including LLM governance compliance.",
    ]
    return "\n".join(lines) + "\n"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.astype(object).where(pd.notnull(df), "").to_dict(orient="records"):
        values = [escape_markdown_cell(str(row[col])) for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def escape_markdown_cell(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript assets for LLM-agent extension.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-registry", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run(output_dir=args.output_dir, append_registry=not args.no_registry))
