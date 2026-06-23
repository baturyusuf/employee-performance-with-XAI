from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.agents.run_governance_audit import run as run_agent_audit
from src.chatbot.evaluate_chatbot import run as run_chatbot_eval
from src.llm.evidence_schema import CompleteCaseEvidence
from src.llm.governed_explainer import GovernedExplainer
from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


def run() -> dict[str, str]:
    output_dir = SETTINGS.reports_dir / "llm_explanations"
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "governed_explanation_eval.csv"
    detail = pd.read_csv(detail_path) if detail_path.exists() else pd.DataFrame()
    if detail.empty:
        from src.llm.generate_governed_explanations import run as generate

        generate(limit=5)
        detail = pd.read_csv(detail_path)

    chatbot_paths = run_chatbot_eval()
    chatbot_eval = pd.read_csv(chatbot_paths["csv"])
    audit_paths = run_agent_audit()
    audit_payload = json.loads(Path(audit_paths["json"]).read_text(encoding="utf-8"))

    explainer = GovernedExplainer()
    evidence = CompleteCaseEvidence.from_reports()
    first = explainer.generate(evidence)
    second = explainer.generate(evidence)
    consistency_rate = 1.0 if first["warnings"] == second["warnings"] and first["short_explanation"] == second["short_explanation"] else 0.0

    missing = evidence.to_dict()
    missing["calibration"] = None
    missing_output = explainer.llm_client.generate_json("", "", missing)
    missing_behavior_pass = "unavailable" in missing_output["detailed_explanation"].lower() or any(
        "calibration" in w.get("type", "") for w in missing_output.get("warnings", [])
    )

    explanation_agent = audit_payload["supervisor_audit"]["agent_findings"].get("ExplanationComplianceAgent", {})
    rule_based_agreement = 1.0 if explanation_agent.get("status") == "pass" else 0.0

    n_cases = int(len(detail))
    summary = {
        "n_cases": n_cases,
        "faithfulness_pass_rate": float(detail["faithfulness_pass"].mean()) if n_cases else 0.0,
        "unsupported_claim_rate": float((detail["unsupported_claim_count"] > 0).mean()) if n_cases else 0.0,
        "forbidden_claim_rate": float((detail["forbidden_claim_count"] > 0).mean()) if n_cases else 0.0,
        "missing_warning_rate": float((detail["missing_warning_count"] > 0).mean()) if n_cases else 0.0,
        "unsafe_prompt_refusal_rate": float(chatbot_eval["pass"].mean()) if len(chatbot_eval) else 0.0,
        "consistency_rate": consistency_rate,
        "rule_based_agent_agreement": rule_based_agreement,
        "missing_evidence_behavior_pass": float(missing_behavior_pass),
        "notes": "Deterministic offline stub evaluation; no human study.",
    }
    summary_path = output_dir / "llm_governance_eval_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    append_registry_row(
        {
            "run_id": f"llm_governance_eval_{utc_now_iso()}",
            "date_time": utc_now_iso(),
            "git_commit_if_available": get_git_commit(),
            "script": "python -m src.llm.evaluate_llm_governance",
            "config": "governed explanations + chatbot unsafe prompts + agent audit",
            "feature_set": "no_salary_hike_no_attrition_no_department",
            "model": "offline_stub_llm_interpreter",
            "seed": "deterministic",
            "cv_strategy": "not_applicable",
            "primary_metrics": "faithfulness pass rate; unsafe prompt refusal rate; consistency rate",
            "output_dir": "reports/llm_explanations",
            "notes": "Automatic evaluation of LLM governance layer without human study.",
            "decision_status": "candidate",
        }
    )
    return {"summary": str(summary_path)}


if __name__ == "__main__":
    print(run())

