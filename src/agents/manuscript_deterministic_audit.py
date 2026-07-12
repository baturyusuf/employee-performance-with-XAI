"""Run the canonical deterministic governance-agent audit without network use.

The generated explanation is the repository's explicit offline renderer.  Its
purpose is to validate evidence flow, warnings, and agent composition; outputs
are labelled as deterministic pipeline evidence and cannot support claims about
a real LLM's quality.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd

from src.agents.run_governance_audit import deterministic_agent_findings
from src.core.io_utils import ensure_dir, write_json, write_jsonl
from src.core.reporting import markdown_table
from src.llm.evidence_schema import (
    CalibrationEvidence,
    CompleteCaseEvidence,
    CounterfactualEvidence,
    FairnessEvidence,
    GovernanceEvidence,
    LeakageEvidence,
    PredictionEvidence,
    ShapEvidence,
)
from src.llm.governed_explainer import GovernedExplainer
from src.llm.runtime_config import LLMRuntimeConfig


class DeterministicAuditError(RuntimeError):
    """Raised when canonical evidence cannot be audited safely."""


def _optional(section: Any, cls: Any) -> Any:
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise DeterministicAuditError(f"{cls.__name__} evidence must be an object or null.")
    return cls(**dict(section))


def evidence_from_dict(payload: Mapping[str, Any]) -> CompleteCaseEvidence:
    required = {"prediction", "governance"}
    missing = sorted(required.difference(payload))
    if missing:
        raise DeterministicAuditError(f"CompleteCaseEvidence lacks required sections: {missing}")
    return CompleteCaseEvidence(
        prediction=PredictionEvidence(**dict(payload["prediction"])),
        shap=_optional(payload.get("shap"), ShapEvidence),
        fairness=_optional(payload.get("fairness"), FairnessEvidence),
        calibration=_optional(payload.get("calibration"), CalibrationEvidence),
        counterfactual=_optional(payload.get("counterfactual"), CounterfactualEvidence),
        leakage=_optional(payload.get("leakage"), LeakageEvidence),
        governance=GovernanceEvidence(**dict(payload["governance"])),
        evidence_sources=[str(value) for value in payload.get("evidence_sources", [])],
    )


def _read_evidence(path: Path, *, run_id: str, config_hash: str) -> list[CompleteCaseEvidence]:
    records: list[CompleteCaseEvidence] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DeterministicAuditError(f"Invalid evidence JSONL at line {line_number}: {exc}") from exc
            if record.get("run_id") != run_id or record.get("config_hash") != config_hash:
                raise DeterministicAuditError(
                    f"Evidence identity mismatch at line {line_number}; expected {run_id}/{config_hash}."
                )
            evidence = record.get("evidence")
            if not isinstance(evidence, Mapping):
                raise DeterministicAuditError(f"Evidence line {line_number} has no evidence object.")
            records.append(evidence_from_dict(evidence))
    if not records:
        raise DeterministicAuditError(f"No evidence records found at {path}")
    return records


def _markdown(
    path: Path,
    *,
    run_id: str,
    config_hash: str,
    results: pd.DataFrame,
    n_cases: int,
) -> None:
    supervisor = results[results["agent_name"] == "SupervisorGovernanceAgent"]
    agent_summary = (
        results.groupby(["agent_name", "status", "risk_level"], dropna=False)
        .size()
        .reset_index(name="n_findings")
        .sort_values(["agent_name", "status", "risk_level"])
    )
    supervisor_summary = supervisor["status"].value_counts().rename_axis("status").reset_index(name="n_cases")
    lines = [
        "# Canonical Deterministic Multi-Agent Governance Audit",
        "",
        f"Run ID: `{run_id}`  ",
        f"Config hash: `{config_hash}`  ",
        f"Cases audited: `{n_cases}`",
        "",
        "This is an offline, deterministic evidence-flow and governance-composition audit. It made zero API calls and is not evidence of real-LLM faithfulness or safety.",
        "",
        "## Supervisor readiness distribution",
        "",
        *markdown_table(supervisor_summary),
        "",
        "## Agent finding distribution",
        "",
        *markdown_table(agent_summary),
        "",
        "## Claim boundaries",
        "",
        "- Agents issue warnings and research-readiness diagnostics only; they make no HR decision.",
        "- The offline renderer is deterministic pipeline-validation infrastructure, not a substitute for an approved real-LLM evaluation.",
        "- Audit findings do not establish legal compliance, fairness, causality, deployment readiness, or zero failure probability.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    evidence_path: str | Path,
    *,
    output_dir: str | Path,
    run_id: str,
    config_hash: str,
) -> Dict[str, Path]:
    source = Path(evidence_path)
    if not source.is_file():
        raise FileNotFoundError(f"Canonical case evidence is missing: {source}")
    output = ensure_dir(Path(output_dir))
    evidence_items = _read_evidence(source, run_id=run_id, config_hash=config_hash)
    runtime = LLMRuntimeConfig(
        provider="offline",
        model="offline-stub",
        temperature=0.0,
        max_tokens=1200,
        require_real_llm=False,
    )
    renderer = GovernedExplainer(runtime_config=runtime)
    explanation_rows: list[dict[str, Any]] = []
    finding_rows: list[dict[str, Any]] = []
    for evidence in evidence_items:
        explanation = renderer.generate(evidence)
        evidence_dict = evidence.to_dict()
        findings, supervisor = deterministic_agent_findings(evidence_dict, explanation)
        case_id = evidence.prediction.case_id
        dataset = evidence.prediction.dataset_name
        explanation_rows.append(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "dataset_name": dataset,
                "case_id": case_id,
                "execution_mode": "offline_deterministic_pipeline_validation",
                "real_llm_used": False,
                "api_call_attempted": False,
                "explanation": explanation,
            }
        )
        for finding in findings:
            finding_rows.append(
                {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "dataset_name": dataset,
                    "case_id": case_id,
                    **finding.to_dict(),
                    "supervisor_overall_status": supervisor["overall_status"],
                    "real_llm_used": False,
                    "audit_scope": "deterministic_pipeline_validation_only",
                }
            )
        finding_rows.append(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "dataset_name": dataset,
                "case_id": case_id,
                "agent_name": "SupervisorGovernanceAgent",
                "status": supervisor["overall_status"],
                "risk_level": (
                    "high" if supervisor["overall_status"] in {"not_ready", "evidence_missing", "research_only"} else "medium"
                ),
                "summary": supervisor["readiness_summary"],
                "required_warnings": supervisor["critical_warnings"],
                "details": supervisor,
                "supervisor_overall_status": supervisor["overall_status"],
                "real_llm_used": False,
                "audit_scope": "deterministic_pipeline_validation_only",
            }
        )

    explanation_path = output / "deterministic_governed_explanations.jsonl"
    audit_jsonl = output / "agent_audit_results.jsonl"
    audit_csv = output / "agent_audit_results.csv"
    audit_md = output / "multi_agent_governance_audit.md"
    metadata = output / "agent_audit_metadata.json"
    write_jsonl(explanation_path, explanation_rows)
    write_jsonl(audit_jsonl, finding_rows)
    frame = pd.DataFrame(
        [
            {
                **row,
                "required_warnings": "; ".join(row.get("required_warnings", [])),
                "details": json.dumps(row.get("details", {}), sort_keys=True, ensure_ascii=False),
            }
            for row in finding_rows
        ]
    )
    frame.to_csv(audit_csv, index=False)
    _markdown(
        audit_md,
        run_id=run_id,
        config_hash=config_hash,
        results=frame,
        n_cases=len(evidence_items),
    )
    write_json(
        metadata,
        {
            "run_id": run_id,
            "config_hash": config_hash,
            "execution_mode": "offline_deterministic_pipeline_validation",
            "real_llm_used": False,
            "api_call_attempted": False,
            "paid_api_calls": 0,
            "n_cases": len(evidence_items),
            "dataset_counts": dict(Counter(item.prediction.dataset_name for item in evidence_items)),
            "n_agent_rows": len(finding_rows),
            "claim_boundary": "Deterministic pipeline/governance validation only; no real-LLM performance claim.",
        },
    )
    return {
        "explanations": explanation_path,
        "audit_jsonl": audit_jsonl,
        "audit_csv": audit_csv,
        "audit_markdown": audit_md,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical no-API deterministic agent audit.")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-hash", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(
        arguments.evidence,
        output_dir=arguments.output_dir,
        run_id=arguments.run_id,
        config_hash=arguments.config_hash,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2, sort_keys=True))
