from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List

from pydantic import BaseModel, Field

from src.agents.calibration_agent import CalibrationAuditAgent
from src.agents.counterfactual_actionability_agent import CounterfactualActionabilityAgent
from src.agents.explanation_agent import ExplanationComplianceAgent
from src.agents.fairness_proxy_agent import FairnessProxyAuditAgent
from src.agents.leakage_agent import LeakageAuditAgent
from src.agents.shap_stability_agent import ShapStabilityAuditAgent
from src.agents.supervisor_agent import SupervisorGovernanceAgent
from src.llm.evidence_schema import CompleteCaseEvidence
from src.llm.governed_explainer import GovernedExplainer
from src.llm.openai_client import OpenAIClientConfigurationError
from src.llm.runtime_config import LLMRuntimeConfig
from src.llm.usage_logger import append_llm_usage, normalize_usage_object
from src.governance.warning_taxonomy import normalize_warning_records, warning_messages
from src.utils.experiment_registry import utc_now_iso


class AgentAuditSynthesis(BaseModel):
    agent_name: str
    status: str
    risk_level: str
    governed_summary: str
    evidence_used: List[str] = Field(default_factory=list)
    required_warnings: List[str] = Field(default_factory=list)
    unsupported_claims_detected: List[str] = Field(default_factory=list)
    requires_human_review: bool = True
    recommended_research_actions: List[str] = Field(default_factory=list)


class SupervisorGovernanceSynthesis(BaseModel):
    overall_status: str
    readiness_summary: str
    critical_warnings: List[str] = Field(default_factory=list)
    prohibited_uses: List[str] = Field(default_factory=list)
    required_human_review: bool = True
    recommended_research_actions: List[str] = Field(default_factory=list)


def _load_json_arg(raw: str) -> Dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Tool input must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Tool input JSON must be an object")
    return payload


def _finding_json(finding: Any) -> str:
    return json.dumps(finding.to_dict(), ensure_ascii=False, sort_keys=True)


def _require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise OpenAIClientConfigurationError(
            "OPENAI_API_KEY is not set. Agents SDK runtime requires a real OpenAI API key."
        )


def _agents_imports() -> tuple[Any, Any, Any, Any]:
    try:
        from agents import Agent, ModelSettings, Runner, function_tool, trace
    except ImportError as exc:
        raise OpenAIClientConfigurationError(
            "openai-agents is not installed. Run: .\\myenv\\Scripts\\pip.exe install -r requirements.txt"
        ) from exc
    return Agent, ModelSettings, Runner, function_tool, trace


Agent, ModelSettings, Runner, function_tool, trace = _agents_imports()


SPECIALIST_AGENT_INSTRUCTIONS = """You are a specialist HR XAI governance audit agent.
You must call your deterministic audit tool before producing a final answer.
Use only the evidence JSON and tool output.
Do not invent metrics, feature names, legal claims, causal claims, or HR recommendations.
Preserve deterministic status, risk level, and required warnings unless the evidence is explicitly missing.
Use the exact specialist agent name supplied in the prompt as `agent_name`.
Return the requested structured output."""


SUPERVISOR_INSTRUCTIONS = """You are the supervisor governance agent for an HR XAI system.
Synthesize the specialist audit outputs into a cautious research-governance readiness assessment.
Do not make HR decisions. Do not claim fairness, causality, or deployment readiness.
The system is decision support only and requires human review."""


class OpenAIAgentsSDKGovernanceRuntime:
    """Production agent runtime using OpenAI Agents SDK tools, structured outputs, and tracing."""

    def __init__(self, runtime_config: LLMRuntimeConfig | None = None):
        self.runtime_config = runtime_config or LLMRuntimeConfig.from_env()

    def run(self, evidence: CompleteCaseEvidence) -> Dict[str, Any]:
        _require_openai_key()
        self._active_case_id = evidence.prediction.case_id
        self._usage_rows = []
        evidence_dict = evidence.to_dict()
        explanation = GovernedExplainer(runtime_config=self.runtime_config).generate(evidence)
        evidence_json = json.dumps(evidence_dict, ensure_ascii=False, sort_keys=True)
        explanation_json = json.dumps(explanation, ensure_ascii=False, sort_keys=True)
        tools = self._build_bound_tools(evidence_dict, explanation)

        with trace(
            "hr_xai_agents_governance_audit",
            metadata={
                "case_id": evidence.prediction.case_id,
                "model": self.runtime_config.model,
                "feature_policy": evidence.prediction.feature_policy,
            },
        ):
            specialist_outputs = [
                self._run_specialist(
                    name="LeakageAuditAgent",
                    tool=tools["run_leakage_audit_tool"],
                    prompt=self._specialist_prompt("LeakageAuditAgent", "leakage", evidence_json),
                ),
                self._run_specialist(
                    name="FairnessProxyAuditAgent",
                    tool=tools["run_fairness_proxy_audit_tool"],
                    prompt=self._specialist_prompt("FairnessProxyAuditAgent", "fairness/proxy", evidence_json),
                ),
                self._run_specialist(
                    name="CalibrationAuditAgent",
                    tool=tools["run_calibration_audit_tool"],
                    prompt=self._specialist_prompt("CalibrationAuditAgent", "calibration", evidence_json),
                ),
                self._run_specialist(
                    name="ShapStabilityAuditAgent",
                    tool=tools["run_shap_stability_audit_tool"],
                    prompt=self._specialist_prompt("ShapStabilityAuditAgent", "SHAP stability", evidence_json),
                ),
                self._run_specialist(
                    name="CounterfactualActionabilityAgent",
                    tool=tools["run_counterfactual_actionability_audit_tool"],
                    prompt=self._specialist_prompt(
                        "CounterfactualActionabilityAgent",
                        "counterfactual actionability",
                        evidence_json,
                    ),
                ),
                self._run_specialist(
                    name="ExplanationComplianceAgent",
                    tool=tools["run_explanation_compliance_audit_tool"],
                    prompt=(
                        "Audit explanation compliance as ExplanationComplianceAgent. "
                        "You must call the compliance audit tool with no arguments before final output.\n"
                        f"EXPLANATION_JSON:\n{explanation_json}\n\nEVIDENCE_JSON:\n{evidence_json}"
                    ),
                ),
            ]
            supervisor_output = self._run_supervisor(specialist_outputs, evidence_dict)
            usage_summary = getattr(self, "_usage_rows", [])

        return {
            "case_id": evidence.prediction.case_id,
            "runtime_provider": "openai_agents_sdk",
            "runtime_model": self.runtime_config.model,
            "governed_explanation": explanation,
            "agent_syntheses": [_agent_synthesis_payload(item) for item in specialist_outputs],
            "supervisor_synthesis": _supervisor_synthesis_payload(supervisor_output),
            "usage_rows": usage_summary,
        }

    def _run_specialist(self, name: str, tool: Any, prompt: str) -> AgentAuditSynthesis:
        agent = Agent(
            name=name,
            instructions=SPECIALIST_AGENT_INSTRUCTIONS,
            tools=[tool],
            model=self.runtime_config.model,
            model_settings=self._model_settings(),
            output_type=AgentAuditSynthesis,
        )
        result = Runner.run_sync(agent, prompt, max_turns=4)
        self._log_agents_usage(result, operation=f"agents_sdk_{name}")
        return result.final_output_as(AgentAuditSynthesis, raise_if_incorrect_type=True)

    def _run_supervisor(
        self,
        specialist_outputs: List[AgentAuditSynthesis],
        evidence_dict: Dict[str, Any],
    ) -> SupervisorGovernanceSynthesis:
        agent = Agent(
            name="SupervisorGovernanceAgent",
            instructions=SUPERVISOR_INSTRUCTIONS,
            model=self.runtime_config.model,
            model_settings=self._model_settings(),
            output_type=SupervisorGovernanceSynthesis,
        )
        deterministic_supervisor = SupervisorGovernanceAgent().aggregate(
            [
                _agent_synthesis_to_finding(output)
                for output in specialist_outputs
            ]
        )
        prompt = (
            "Synthesize these specialist agent outputs into final governance readiness. "
            "Preserve prohibited uses and human-review requirements.\n"
            f"SPECIALIST_OUTPUTS:\n{json.dumps([o.model_dump() for o in specialist_outputs], ensure_ascii=False, sort_keys=True)}\n\n"
            f"DETERMINISTIC_SUPERVISOR_BASELINE:\n{json.dumps(deterministic_supervisor, ensure_ascii=False, sort_keys=True)}\n\n"
            f"EVIDENCE_SUMMARY:\n{json.dumps(_evidence_summary(evidence_dict), ensure_ascii=False, sort_keys=True)}"
        )
        result = Runner.run_sync(agent, prompt, max_turns=2)
        self._log_agents_usage(result, operation="agents_sdk_SupervisorGovernanceAgent")
        return result.final_output_as(SupervisorGovernanceSynthesis, raise_if_incorrect_type=True)

    def _model_settings(self) -> Any:
        return ModelSettings(
            temperature=self.runtime_config.temperature,
            max_tokens=self.runtime_config.max_tokens,
            include_usage=True,
        )

    def _build_bound_tools(
        self,
        evidence_dict: Dict[str, Any],
        explanation: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "run_leakage_audit_tool": _make_no_arg_tool(
                "run_leakage_audit_tool",
                "Run deterministic leakage audit on the bound structured evidence.",
                lambda: _finding_json(LeakageAuditAgent().audit(evidence_dict)),
            ),
            "run_fairness_proxy_audit_tool": _make_no_arg_tool(
                "run_fairness_proxy_audit_tool",
                "Run deterministic fairness/proxy audit on the bound structured evidence.",
                lambda: _finding_json(FairnessProxyAuditAgent().audit(evidence_dict)),
            ),
            "run_calibration_audit_tool": _make_no_arg_tool(
                "run_calibration_audit_tool",
                "Run deterministic calibration audit on the bound structured evidence.",
                lambda: _finding_json(CalibrationAuditAgent().audit(evidence_dict)),
            ),
            "run_shap_stability_audit_tool": _make_no_arg_tool(
                "run_shap_stability_audit_tool",
                "Run deterministic SHAP stability audit on the bound structured evidence.",
                lambda: _finding_json(ShapStabilityAuditAgent().audit(evidence_dict)),
            ),
            "run_counterfactual_actionability_audit_tool": _make_no_arg_tool(
                "run_counterfactual_actionability_audit_tool",
                "Run deterministic counterfactual actionability audit on the bound structured evidence.",
                lambda: _finding_json(CounterfactualActionabilityAgent().audit(evidence_dict)),
            ),
            "run_explanation_compliance_audit_tool": _make_no_arg_tool(
                "run_explanation_compliance_audit_tool",
                "Run deterministic explanation compliance audit on the bound governed explanation and evidence.",
                lambda: _finding_json(
                    ExplanationComplianceAgent().audit_explanation(explanation, evidence_dict)
                ),
            ),
        }

    def _log_agents_usage(self, result: Any, operation: str) -> None:
        usage_obj = getattr(getattr(result, "context_wrapper", None), "usage", None)
        usage = normalize_usage_object(usage_obj)
        case_id = str(getattr(self, "_active_case_id", "unknown"))
        row = append_llm_usage(
            run_id=f"{operation}_{utc_now_iso()}",
            case_id=case_id,
            operation=operation,
            provider="openai_agents_sdk",
            model=self.runtime_config.model,
            usage=usage,
            notes="OpenAI Agents SDK run.",
        )
        if not hasattr(self, "_usage_rows"):
            self._usage_rows = []
        self._usage_rows.append(row)

    @staticmethod
    def _specialist_prompt(agent_name: str, topic: str, evidence_json: str) -> str:
        return (
            f"Perform the {topic} governance audit as {agent_name}. "
            "Call your deterministic audit tool with no arguments, then produce structured output. "
            f"The final `agent_name` must be exactly `{agent_name}`.\n"
            f"EVIDENCE_JSON:\n{evidence_json}"
        )


def _agent_synthesis_to_finding(output: AgentAuditSynthesis) -> Any:
    from src.agents.base_agent import AgentFinding

    normalized_warnings = normalize_warning_records(output.required_warnings)

    return AgentFinding(
        agent_name=output.agent_name,
        status=output.status,
        risk_level=output.risk_level,
        summary=output.governed_summary,
        required_warnings=warning_messages(normalized_warnings),
        details={
            "evidence_used": output.evidence_used,
            "recommended_research_actions": output.recommended_research_actions,
            "unsupported_claims_detected": output.unsupported_claims_detected,
            "normalized_warning_ids": [item["warning_id"] for item in normalized_warnings],
        },
    )


def _agent_synthesis_payload(output: AgentAuditSynthesis) -> Dict[str, Any]:
    payload = output.model_dump()
    normalized_warnings = normalize_warning_records(payload.get("required_warnings", []))
    payload["required_warnings"] = warning_messages(normalized_warnings)
    payload["normalized_warnings"] = normalized_warnings
    payload["normalized_warning_ids"] = [item["warning_id"] for item in normalized_warnings]
    return payload


def _supervisor_synthesis_payload(output: SupervisorGovernanceSynthesis) -> Dict[str, Any]:
    payload = output.model_dump()
    normalized_warnings = normalize_warning_records(payload.get("critical_warnings", []))
    payload["critical_warnings"] = warning_messages(normalized_warnings)
    payload["normalized_critical_warnings"] = normalized_warnings
    payload["normalized_critical_warning_ids"] = [item["warning_id"] for item in normalized_warnings]
    return payload


def _evidence_summary(evidence_dict: Dict[str, Any]) -> Dict[str, Any]:
    prediction = evidence_dict.get("prediction", {})
    fairness = evidence_dict.get("fairness", {})
    calibration = evidence_dict.get("calibration", {})
    counterfactual = evidence_dict.get("counterfactual", {})
    return {
        "case_id": prediction.get("case_id"),
        "feature_policy": prediction.get("feature_policy"),
        "predicted_class": prediction.get("predicted_class"),
        "confidence": prediction.get("confidence"),
        "fairness_proxy_warnings": fairness.get("proxy_risk_warnings"),
        "calibration_ece": calibration.get("expected_calibration_error"),
        "counterfactual_validity": counterfactual.get("validity"),
    }


def _make_no_arg_tool(name: str, description: str, callback: Callable[[], str]) -> Any:
    @function_tool(name_override=name, description_override=description)
    def bound_tool() -> str:
        """Run a deterministic bound governance audit tool."""
        return callback()

    return bound_tool


def assert_agents_sdk_runtime_available() -> None:
    _require_openai_key()
    _agents_imports()
