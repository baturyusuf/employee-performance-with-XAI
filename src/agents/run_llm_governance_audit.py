from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.agents.llm_agent_orchestrator import LLMAgentGovernanceOrchestrator
from src.agents.openai_agents_runtime import OpenAIAgentsSDKGovernanceRuntime
from src.llm.evidence_schema import CompleteCaseEvidence
from src.llm.openai_client import OpenAIClientAPIError, OpenAIClientConfigurationError
from src.llm.runtime_config import LLMRuntimeConfig
from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


OUTPUT_DIR = SETTINGS.reports_dir / "agent_audits"


def run(
    case_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    require_real_llm: bool = False,
    agent_runtime: str = "custom",
) -> Dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_config = LLMRuntimeConfig.from_env()
    runtime_config = LLMRuntimeConfig(
        provider=(provider or base_config.provider),  # type: ignore[arg-type]
        model=model or base_config.model,
        temperature=base_config.temperature,
        max_tokens=base_config.max_tokens,
        require_real_llm=require_real_llm or base_config.require_real_llm,
    )
    evidence = CompleteCaseEvidence.from_reports(case_id=case_id)
    if agent_runtime == "openai-agents":
        audit = OpenAIAgentsSDKGovernanceRuntime(runtime_config=runtime_config).run(evidence)
    elif agent_runtime == "custom":
        audit = LLMAgentGovernanceOrchestrator(runtime_config=runtime_config).run(evidence)
    else:
        raise ValueError("agent_runtime must be one of: custom, openai-agents")

    slug = "openai_agents_sdk" if agent_runtime == "openai-agents" else "llm_multi_agent"
    audit_case_id = str(audit.get("case_id") or "unknown")
    json_path = OUTPUT_DIR / f"{slug}_case_{audit_case_id}_governance_audit.json"
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    md_path = OUTPUT_DIR / f"{slug}_case_{audit_case_id}_governance_audit.md"
    _write_markdown(audit, md_path)
    latest_json_path = OUTPUT_DIR / f"{slug}_governance_audit.json"
    latest_md_path = OUTPUT_DIR / f"{slug}_governance_audit.md"
    latest_json_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(audit, latest_md_path)

    append_registry_row(
        {
            "run_id": f"llm_multi_agent_governance_audit_{utc_now_iso()}",
            "date_time": utc_now_iso(),
            "git_commit_if_available": get_git_commit(),
            "script": "python -m src.agents.run_llm_governance_audit",
            "config": f"agent_runtime={agent_runtime}; structured evidence + deterministic tools + LLM agent synthesis",
            "feature_set": "no_salary_hike_no_attrition_no_department",
            "model": f"{runtime_config.provider}:{runtime_config.model}",
            "seed": "temperature configured by HR_XAI_LLM_TEMPERATURE",
            "cv_strategy": "not_applicable",
            "primary_metrics": "agent synthesis schema validation; supervisor readiness status",
            "output_dir": "reports/agent_audits",
            "notes": "Generated LLM-assisted multi-agent governance audit from structured evidence.",
            "decision_status": "candidate",
        }
    )
    return {"json": json_path, "markdown": md_path}


def _write_markdown(audit: Dict[str, object], output_path: Path) -> None:
    supervisor = audit.get("supervisor_audit") or audit.get("supervisor_synthesis")  # type: ignore[assignment]
    syntheses = audit.get("llm_agent_syntheses") or audit.get("agent_syntheses")  # type: ignore[assignment]
    lines: List[str] = [
        "# LLM-Assisted Multi-Agent Governance Audit",
        "",
        "## Architecture",
        "XGBoost predicts -> XAI/audit tools create evidence -> deterministic agents audit evidence -> LLM agents synthesize governed findings -> supervisor aggregates readiness.",
        "",
        "## Runtime",
        f"- Provider/runtime: `{audit['runtime_provider']}`",
        f"- Model: `{audit['runtime_model']}`",
        "",
        "## Supervisor Readiness",
        f"- Overall status: `{supervisor['overall_status']}`",  # type: ignore[index]
        f"- Human review required: `{supervisor.get('required_human_review') or supervisor.get('required_human_review', True)}`",  # type: ignore[union-attr]
        "",
        "## LLM Agent Syntheses",
    ]
    for item in syntheses:  # type: ignore[assignment]
        lines.append(f"### {item['agent_name']}")
        lines.append(f"- Status: {item['status']}")
        lines.append(f"- Risk level: {item['risk_level']}")
        lines.append(f"- Summary: {item['governed_summary']}")
        if item.get("required_warnings"):
            lines.append("- Required warnings:")
            for warning in item["required_warnings"]:
                lines.append(f"  - {warning}")
        lines.append("")
    lines.extend(
        [
            "## Governance Constraint",
            "The LLM agents may interpret evidence but must not invent metrics, causal claims, fairness guarantees, or HR decisions.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-assisted multi-agent governance audit.")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--provider", choices=["auto", "offline", "openai"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--agent-runtime", choices=["custom", "openai-agents"], default="custom")
    parser.add_argument(
        "--require-real-llm",
        action="store_true",
        help="Fail instead of falling back to offline synthesis when a real LLM is unavailable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        print(
            run(
                case_id=args.case_id,
                provider=args.provider,
                model=args.model,
                require_real_llm=args.require_real_llm,
                agent_runtime=args.agent_runtime,
            )
        )
    except (OpenAIClientConfigurationError, OpenAIClientAPIError) as exc:
        raise SystemExit(f"LLM configuration error: {exc}") from exc
