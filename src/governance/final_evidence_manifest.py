from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.core.reporting import markdown_table, write_markdown
from src.core.run_registry import RunRegistryEntry, append_run_entry
from src.utils.config import SETTINGS
from src.utils.experiment_registry import utc_now_iso


OUTPUT_DIR = SETTINGS.reports_dir / "manuscript_assets" / "final_evidence_manifest"
FINAL_SUMMARY = SETTINGS.reports_dir / "llm_explanations" / "llm_agent_eval_summary.csv"
READINESS_REPORT = SETTINGS.reports_dir / "governance_reports" / "final_governance_readiness_report.md"
GXAIR_DASHBOARD = SETTINGS.reports_dir / "governance_reports" / "gxair_component_dashboard.csv"

STUB_DRY_RUN_DISCLAIMER = (
    "Stub/dry-run outputs generated with offline_stub_llm or run_mode=dry_run are reproducibility and "
    "pipeline-validation artifacts only. They are excluded from the manuscript-grade evidence package and "
    "must not be cited as real LLM evidence."
)


@dataclass(frozen=True)
class EvidenceFile:
    evidence_id: str
    category: str
    path: str
    file_type: str
    manuscript_grade_status: str
    run_scope: str
    real_llm_used: str
    claim_role: str
    row_count: str
    sha256: str
    required_for_package: bool
    notes: str


def run(output_dir: Path = OUTPUT_DIR) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_summary = load_final_summary()
    dashboard = load_dashboard()

    manifest_csv = output_dir / "final_evidence_manifest.csv"
    manifest_json = output_dir / "final_evidence_manifest.json"
    manifest_md = output_dir / "final_evidence_manifest.md"
    readiness_md = output_dir / "readiness_not_ready_explanation.md"

    append_run_entry(
        RunRegistryEntry(
            run_id=f"final_evidence_manifest_{utc_now_iso()}",
            command="python -m src.governance.final_evidence_manifest",
            dataset="inx_primary;hrdataset_v14",
            model="xgboost; openai:gpt-5.4-mini",
            feature_policy="no_salary_hike_no_attrition_no_department; department_free",
            llm_provider=str(final_summary.get("llm_provider", "")),
            llm_model=str(final_summary.get("llm_model_used", "")),
            output_files=[str(manifest_csv), str(manifest_json), str(manifest_md), str(readiness_md)],
        )
    )
    rows = build_manifest_rows(final_summary)
    write_manifest_csv(rows, manifest_csv)
    write_manifest_json(rows, final_summary, dashboard, manifest_json)
    write_manifest_markdown(rows, final_summary, dashboard, manifest_md)
    write_readiness_explanation(dashboard, readiness_md)
    return {
        "manifest_csv": manifest_csv,
        "manifest_json": manifest_json,
        "manifest_markdown": manifest_md,
        "readiness_explanation": readiness_md,
    }


def load_final_summary() -> Dict[str, Any]:
    if not FINAL_SUMMARY.exists():
        raise FileNotFoundError(f"Final LLM-agent summary is missing: {FINAL_SUMMARY}")
    summary = pd.read_csv(FINAL_SUMMARY)
    if summary.empty:
        raise RuntimeError(f"Final LLM-agent summary is empty: {FINAL_SUMMARY}")
    row = summary.iloc[0].to_dict()
    if str(row.get("run_mode", "")).lower() != "real" or str(row.get("real_llm_used", "")).lower() not in {"true", "1"}:
        raise RuntimeError("Final evidence package requires the root 80-case summary to be a real OpenAI run.")
    if int(row.get("n_cases", 0)) != 80:
        raise RuntimeError("Final evidence package requires the root final LLM-agent summary to contain 80 cases.")
    return row


def load_dashboard() -> pd.DataFrame:
    if not GXAIR_DASHBOARD.exists():
        raise FileNotFoundError(f"G-XAIR dashboard is missing: {GXAIR_DASHBOARD}")
    dashboard = pd.read_csv(GXAIR_DASHBOARD)
    if dashboard.empty:
        raise RuntimeError(f"G-XAIR dashboard is empty: {GXAIR_DASHBOARD}")
    return dashboard


def build_manifest_rows(final_summary: Dict[str, Any]) -> List[EvidenceFile]:
    real_scope = "final_80_real_openai_inx_primary_hrdataset_v14"
    real_llm = str(final_summary.get("real_llm_used", ""))
    files = [
        ("final_case_manifest", "llm_agent_final_real", "reports/llm_explanations/eval_case_manifest.csv", "csv", "primary_manuscript_grade", "Case manifest for 40 INX and 40 HRDataset_v14 final real OpenAI cases."),
        ("final_governed_explanations", "llm_agent_final_real", "reports/llm_explanations/governed_explanations.jsonl", "jsonl", "primary_manuscript_grade", "Structured real OpenAI governed explanation outputs for final cases."),
        ("final_faithfulness_eval", "llm_agent_final_real", "reports/llm_explanations/faithfulness_eval.csv", "csv", "primary_manuscript_grade", "Deterministic faithfulness/compliance evaluation for final real OpenAI outputs."),
        ("final_faithfulness_summary", "llm_agent_final_real", "reports/llm_explanations/faithfulness_eval_summary.md", "markdown", "primary_manuscript_grade", "Markdown summary of final real OpenAI faithfulness metrics."),
        ("final_llm_agent_summary_csv", "llm_agent_final_real", "reports/llm_explanations/llm_agent_eval_summary.csv", "csv", "primary_manuscript_grade", "Integrated final 80-case real OpenAI LLM-agent metrics."),
        ("final_llm_agent_summary_md", "llm_agent_final_real", "reports/llm_explanations/llm_agent_eval_summary.md", "markdown", "primary_manuscript_grade", "Markdown integrated final 80-case real OpenAI LLM-agent summary."),
        ("final_agent_audit_csv", "agent_audit_final", "reports/agent_audits/agent_audit_results.csv", "csv", "primary_manuscript_grade", "Deterministic batch governance agent outputs for final explanations."),
        ("final_agent_audit_jsonl", "agent_audit_final", "reports/agent_audits/agent_audit_results.jsonl", "jsonl", "primary_manuscript_grade", "Machine-readable batch governance agent outputs."),
        ("final_agent_audit_md", "agent_audit_final", "reports/agent_audits/multi_agent_governance_audit.md", "markdown", "primary_manuscript_grade", "Markdown batch governance agent audit summary."),
        ("chatbot_guardrail_eval_csv", "chatbot_guardrails", "reports/chatbot_eval/guardrail_evaluation.csv", "csv", "supporting_manuscript_grade", "Deterministic safe/unsafe chatbot guardrail evaluation."),
        ("chatbot_guardrail_summary", "chatbot_guardrails", "reports/chatbot_eval/guardrail_evaluation_summary.md", "markdown", "supporting_manuscript_grade", "Markdown chatbot guardrail evaluation summary."),
        ("unsafe_prompt_suite", "chatbot_guardrails", "reports/chatbot_eval/unsafe_prompt_suite.csv", "csv", "supporting_manuscript_grade", "Unsafe/adversarial prompt suite."),
        ("safe_prompt_suite", "chatbot_guardrails", "reports/chatbot_eval/safe_prompt_suite.csv", "csv", "supporting_manuscript_grade", "Safe governance/audit prompt suite."),
        ("gxair_dashboard_csv", "readiness", "reports/governance_reports/gxair_component_dashboard.csv", "csv", "primary_manuscript_grade", "Component readiness dashboard."),
        ("gxair_dashboard_md", "readiness", "reports/governance_reports/gxair_component_dashboard.md", "markdown", "primary_manuscript_grade", "Markdown component readiness dashboard."),
        ("final_readiness_report", "readiness", "reports/governance_reports/final_governance_readiness_report.md", "markdown", "primary_manuscript_grade", "Final readiness report explaining not_ready status."),
        ("statistical_uncertainty_summary", "statistical_reliability", "reports/statistical_reliability/uncertainty_summary.md", "markdown", "supporting_manuscript_grade", "Statistical reliability and uncertainty summary."),
        ("llm_guardrail_ci", "statistical_reliability", "reports/statistical_reliability/llm_guardrail_ci.csv", "csv", "supporting_manuscript_grade", "Binomial CI estimates for final real LLM and guardrail checks."),
        ("external_validation_summary", "external_validation", "reports/external_validation/external_validation_summary.md", "markdown", "supporting_manuscript_grade", "External validation and related-task robustness summary with claim boundaries."),
        ("external_validation_tables", "external_validation", "reports/manuscript_assets/external_validation_tables.md", "markdown", "supporting_manuscript_grade", "Manuscript-support external validation tables."),
        ("external_governance_summary", "external_validation", "reports/governance_reports/external_validation_governance_summary.md", "markdown", "supporting_manuscript_grade", "Governance interpretation for external validation."),
        ("model_card", "governance_documentation", "reports/model_card/hr_xai_model_card.md", "markdown", "supporting_manuscript_grade", "Updated model card with final real LLM scope and limitations."),
        ("decision_log", "research_traceability", "RESEARCH_DECISION_LOG.md", "markdown", "supporting_traceability", "Research decision log documenting final real run and claim boundaries."),
        ("handoff", "research_traceability", "PROJECT_CONTINUATION_HANDOFF.md", "markdown", "supporting_traceability", "Continuation handoff documenting final evidence status."),
        ("run_registry", "research_traceability", "reports/research_log/run_registry.csv", "csv", "supporting_traceability", "Run registry for reproducibility."),
        ("llm_usage_log", "cost_accounting", "reports/llm_explanations/llm_usage_log.csv", "csv", "supporting_traceability", "OpenAI usage/cost log. Provider billing dashboard remains source of truth."),
        ("dry_run_config", "excluded_stub_dry_run", "configs/llm_agent_eval.yaml", "json", "excluded_not_manuscript_grade", "Default dry-run config retained for reproducibility; not real LLM evidence."),
        ("offline_stub_client", "excluded_stub_dry_run", "src/llm/offline_stub_llm.py", "python", "excluded_not_manuscript_grade", "Offline stub implementation for tests/dry-runs only."),
        ("pilot_40_summary", "pilot_quality_control", "reports/llm_explanations/openai_pilot_40/llm_agent_eval_summary.csv", "csv", "pilot_not_final_manuscript_evidence", "Real 40-case pilot used as staged quality control before final run."),
    ]
    rows = []
    for evidence_id, category, raw_path, file_type, status, notes in files:
        path = Path(raw_path)
        rows.append(
            EvidenceFile(
                evidence_id=evidence_id,
                category=category,
                path=raw_path,
                file_type=file_type,
                manuscript_grade_status=status,
                run_scope=real_scope if "excluded" not in category and "pilot" not in category else category,
                real_llm_used=real_llm if "llm" in category or "agent" in category or category == "readiness" else "",
                claim_role=claim_role_for(category, status),
                row_count=row_count(path, file_type),
                sha256=file_sha256(path),
                required_for_package=status.startswith("primary") or evidence_id in {"model_card", "decision_log"},
                notes=notes,
            )
        )
    missing_required = [row.path for row in rows if row.required_for_package and row.sha256 == "missing"]
    if missing_required:
        raise FileNotFoundError(f"Required final evidence files are missing: {missing_required}")
    return rows


def claim_role_for(category: str, status: str) -> str:
    if status == "excluded_not_manuscript_grade":
        return "excluded; do not cite as real LLM evidence"
    if status == "pilot_not_final_manuscript_evidence":
        return "pilot quality-control evidence; not final 80-case result"
    if category == "readiness":
        return "readiness interpretation and deployment-blocker evidence"
    if category == "external_validation":
        return "supports external validation and related-task claim boundaries"
    if category == "chatbot_guardrails":
        return "supports automated guardrail compliance evidence"
    return "supports final 80-case manuscript-grade technical evidence"


def row_count(path: Path, file_type: str) -> str:
    if not path.exists():
        return "missing"
    if file_type == "csv":
        with path.open("r", newline="", encoding="utf-8") as f:
            return str(max(0, sum(1 for _ in csv.reader(f)) - 1))
    if file_type == "jsonl":
        with path.open("r", encoding="utf-8") as f:
            return str(sum(1 for line in f if line.strip()))
    return ""


def file_sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest_csv(rows: Iterable[EvidenceFile], path: Path) -> None:
    frame = pd.DataFrame([asdict(row) for row in rows])
    frame.to_csv(path, index=False)


def write_manifest_json(rows: List[EvidenceFile], final_summary: Dict[str, Any], dashboard: pd.DataFrame, path: Path) -> None:
    payload = {
        "created_at": utc_now_iso(),
        "package_name": "final_evidence_manifest",
        "source_run": final_summary,
        "readiness_label": readiness_label(dashboard),
        "readiness_blockers": readiness_blockers(dashboard),
        "stub_dry_run_disclaimer": STUB_DRY_RUN_DISCLAIMER,
        "files": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_manifest_markdown(rows: List[EvidenceFile], final_summary: Dict[str, Any], dashboard: pd.DataFrame, path: Path) -> None:
    frame = pd.DataFrame([asdict(row) for row in rows])
    primary = frame[
        ~frame["manuscript_grade_status"].isin(
            ["excluded_not_manuscript_grade", "pilot_not_final_manuscript_evidence"]
        )
    ]
    excluded = frame[frame["manuscript_grade_status"].str.contains("excluded|pilot", regex=True)]
    lines = [
        "# Final Evidence Manifest",
        "",
        "This package binds the final 80-case real OpenAI LLM-agent outputs and supporting governance artifacts for manuscript-support use.",
        "",
        f"Run ID: `{final_summary.get('run_id')}`",
        f"Run mode: `{final_summary.get('run_mode')}`",
        f"Real LLM used: `{final_summary.get('real_llm_used')}`",
        f"LLM provider/model: `{final_summary.get('llm_provider')}` / `{final_summary.get('llm_model_used')}`",
        f"Cases/datasets: `{final_summary.get('n_cases')}` / `{final_summary.get('n_datasets')}`",
        "",
        "## Manuscript-Grade Scope",
        "",
        "- Included final scope: 40 INX primary cases and 40 HRDataset_v14 independent-replication cases.",
        "- The LLM interprets structured ML/XAI/governance evidence; it is not the predictive model.",
        "- IBM performance, IBM attrition, and Employee Turnover LLM regeneration are second-stage robustness work.",
        "- Related-task datasets must not be described as direct employee-performance validation.",
        "",
        "## Stub / Dry-Run Exclusion",
        "",
        STUB_DRY_RUN_DISCLAIMER,
        "",
        "## Readiness",
        "",
        f"Final readiness label: `{readiness_label(dashboard)}`",
        "",
        "The readiness label remains `not_ready` because two high-severity blockers remain:",
        "",
        *[f"- {item}" for item in readiness_blockers(dashboard)],
        "",
        "## Included Evidence and Traceability Files",
        "",
        *markdown_table(primary[["evidence_id", "category", "path", "file_type", "manuscript_grade_status", "row_count", "claim_role"]]),
        "",
        "## Excluded or Non-Final Files",
        "",
        *markdown_table(excluded[["evidence_id", "category", "path", "file_type", "manuscript_grade_status", "claim_role"]]),
        "",
        "## Integrity",
        "",
        "The CSV and JSON manifests include SHA-256 hashes and row counts for machine-readable traceability.",
    ]
    write_markdown(path, lines)


def write_readiness_explanation(dashboard: pd.DataFrame, path: Path) -> None:
    blockers = readiness_blockers(dashboard)
    lines = [
        "# Readiness Explanation: not_ready",
        "",
        "The final evidence package does not support a deployment-ready claim. The framework remains research-grade decision support only.",
        "",
        "## Primary Blockers",
        "",
        *[f"- {item}" for item in blockers],
        "",
        "## Interpretation",
        "",
        "The real 80-case OpenAI LLM-agent evaluation supports technical evidence-interpretation quality for the stated INX and HRDataset_v14 scope. It does not remove the underlying ML governance blockers. Proxy risk remains high because department can still be reconstructed from remaining variables, and counterfactual actionability remains weak because technically valid counterfactuals often depend on manager or organisation-controlled changes rather than employee-actionable changes.",
        "",
        "## Claim Boundary",
        "",
        "- No autonomous HR decision capability.",
        "- No fairness guarantee.",
        "- No causal SHAP or counterfactual claim.",
        "- No deployment readiness without independent data provenance, human validation, legal review, and organisation-specific governance.",
        "",
        "## Stub / Dry-Run Exclusion",
        "",
        STUB_DRY_RUN_DISCLAIMER,
    ]
    write_markdown(path, lines)


def readiness_label(dashboard: pd.DataFrame) -> str:
    if "final_readiness_label" in dashboard.columns and not dashboard["final_readiness_label"].dropna().empty:
        return str(dashboard["final_readiness_label"].dropna().iloc[0])
    return "unknown"


def readiness_blockers(dashboard: pd.DataFrame) -> List[str]:
    blockers = []
    for component in ["Proxy Risk Penalty", "Counterfactual Actionability"]:
        subset = dashboard[dashboard["component"] == component]
        if subset.empty:
            blockers.append(f"{component}: evidence missing.")
            continue
        row = subset.iloc[0]
        blockers.append(f"{component}: {row.get('status')} ({row.get('severity')}). {row.get('limitations')}")
    return blockers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final manuscript evidence manifest.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run(Path(args.output_dir)))
