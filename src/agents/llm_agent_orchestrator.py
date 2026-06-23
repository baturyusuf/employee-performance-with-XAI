from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.agents.agent_output_schema import (
    AGENT_AUDIT_RESPONSE_FORMAT,
    validate_agent_audit_synthesis,
)
from src.agents.base_agent import AgentFinding
from src.agents.calibration_agent import CalibrationAuditAgent
from src.agents.counterfactual_actionability_agent import CounterfactualActionabilityAgent
from src.agents.explanation_agent import ExplanationComplianceAgent
from src.agents.fairness_proxy_agent import FairnessProxyAuditAgent
from src.agents.leakage_agent import LeakageAuditAgent
from src.agents.shap_stability_agent import ShapStabilityAuditAgent
from src.agents.supervisor_agent import SupervisorGovernanceAgent
from src.llm.client_factory import build_llm_client
from src.llm.evidence_schema import CompleteCaseEvidence
from src.llm.governed_explainer import GovernedExplainer
from src.llm.llm_client import LLMClient
from src.llm.runtime_config import LLMRuntimeConfig
from src.governance.warning_taxonomy import normalize_warning_records, warning_messages


AGENT_SYSTEM_PROMPT = """You are an HR XAI governance audit agent.
You must only interpret the supplied deterministic audit finding and structured evidence.
Do not invent metrics, feature names, legal conclusions, causal claims, or HR decisions.
Use cautious research-governance language.
Return only JSON matching the requested schema."""


class LLMAgentGovernanceOrchestrator:
    """Runs deterministic audit tools and asks an LLM to synthesize each agent's governed finding."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        runtime_config: Optional[LLMRuntimeConfig] = None,
    ):
        self.runtime_config = runtime_config or LLMRuntimeConfig.from_env()
        self.llm_client = llm_client or build_llm_client(self.runtime_config)

    def run(self, evidence: CompleteCaseEvidence) -> Dict[str, Any]:
        evidence_dict = evidence.to_dict()
        explanation = GovernedExplainer(
            llm_client=self.llm_client,
            runtime_config=self.runtime_config,
        ).generate(evidence)
        findings = self._deterministic_findings(evidence_dict, explanation)
        supervisor = SupervisorGovernanceAgent().aggregate(findings)
        llm_syntheses = [
            self._synthesize_agent_finding(finding, evidence_dict, supervisor)
            for finding in findings
        ]
        llm_syntheses = [_normalize_synthesis_payload(item) for item in llm_syntheses]
        return {
            "case_id": evidence.prediction.case_id,
            "runtime_provider": self.runtime_config.provider,
            "runtime_model": self.runtime_config.model,
            "governed_explanation": explanation,
            "deterministic_agent_findings": [_normalize_finding_payload(finding) for finding in findings],
            "llm_agent_syntheses": llm_syntheses,
            "supervisor_audit": supervisor,
        }

    def _deterministic_findings(
        self,
        evidence_dict: Dict[str, Any],
        explanation: Dict[str, Any],
    ) -> List[AgentFinding]:
        agents = [
            LeakageAuditAgent(),
            FairnessProxyAuditAgent(),
            CalibrationAuditAgent(),
            ShapStabilityAuditAgent(),
            CounterfactualActionabilityAgent(),
        ]
        findings = [agent.audit(evidence_dict) for agent in agents]
        findings.append(ExplanationComplianceAgent().audit_explanation(explanation, evidence_dict))
        return findings

    def _synthesize_agent_finding(
        self,
        finding: AgentFinding,
        evidence_dict: Dict[str, Any],
        supervisor: Dict[str, Any],
    ) -> Dict[str, Any]:
        if hasattr(self.llm_client, "generate_structured_json"):
            payload = {
                "agent_name": finding.agent_name,
                "deterministic_finding": finding.to_dict(),
                "supervisor_status": supervisor.get("overall_status"),
                "evidence": evidence_dict,
            }
            user_prompt = (
                "Synthesize this deterministic audit finding as the named governance agent. "
                "Preserve the status, risk level, and required warnings unless evidence is missing. "
                f"AGENT_INPUT:\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
            )
            return self.llm_client.generate_structured_json(
                system_prompt=AGENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                evidence=payload,
                response_format=AGENT_AUDIT_RESPONSE_FORMAT,
                validator=validate_agent_audit_synthesis,
            )
        return self._offline_agent_synthesis(finding)

    @staticmethod
    def _offline_agent_synthesis(finding: AgentFinding) -> Dict[str, Any]:
        normalized_warnings = normalize_warning_records(finding.required_warnings)
        return {
            "agent_name": finding.agent_name,
            "status": finding.status,
            "risk_level": finding.risk_level,
            "governed_summary": finding.summary,
            "evidence_used": sorted(finding.details.keys()),
            "required_warnings": warning_messages(normalized_warnings),
            "normalized_warnings": normalized_warnings,
            "normalized_warning_ids": [item["warning_id"] for item in normalized_warnings],
            "unsupported_claims_detected": [],
            "requires_human_review": True,
            "recommended_research_actions": [
                "Use this audit finding as governance evidence, not as an HR decision."
            ],
            "_llm_backend": "offline_agent_synthesis",
        }


def _normalize_synthesis_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized_warnings = normalize_warning_records(payload.get("required_warnings", []))
    payload = dict(payload)
    payload["required_warnings"] = warning_messages(normalized_warnings)
    payload["normalized_warnings"] = normalized_warnings
    payload["normalized_warning_ids"] = [item["warning_id"] for item in normalized_warnings]
    return payload


def _normalize_finding_payload(finding: AgentFinding) -> Dict[str, Any]:
    payload = finding.to_dict()
    normalized_warnings = normalize_warning_records(payload.get("required_warnings", []))
    payload["required_warnings"] = warning_messages(normalized_warnings)
    payload["normalized_warnings"] = normalized_warnings
    payload["normalized_warning_ids"] = [item["warning_id"] for item in normalized_warnings]
    return payload
