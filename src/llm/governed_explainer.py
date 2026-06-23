from __future__ import annotations

from typing import Any, Dict, Optional

from src.llm.evidence_schema import CompleteCaseEvidence, load_complete_case_evidence
from src.llm.faithfulness_checker import check_faithfulness
from src.llm.client_factory import build_llm_client
from src.llm.llm_client import LLMClient
from src.llm.prompt_templates import SYSTEM_PROMPT, build_case_prompt
from src.llm.runtime_config import LLMRuntimeConfig
from src.governance.warning_taxonomy import add_mandatory_warning_records


class GovernedExplainer:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        runtime_config: Optional[LLMRuntimeConfig] = None,
    ):
        self.runtime_config = runtime_config or LLMRuntimeConfig.from_env()
        self.llm_client = llm_client or build_llm_client(self.runtime_config)

    def generate(self, evidence: CompleteCaseEvidence) -> Dict[str, Any]:
        evidence_dict = evidence.to_dict()
        output = self.llm_client.generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_case_prompt(evidence.to_json()),
            evidence=evidence_dict,
        )
        self._ensure_mandatory_warnings(output, evidence_dict)
        compliance = check_faithfulness(output, evidence_dict).to_dict()
        output["faithfulness_check"] = compliance
        output["unsupported_claims_detected"] = compliance["unsupported_claims"]
        return output

    @staticmethod
    def _ensure_mandatory_warnings(output: Dict[str, Any], evidence: Dict[str, Any]) -> None:
        warnings = output.setdefault("warnings", [])
        if not isinstance(warnings, list):
            warnings = []
            output["warnings"] = warnings
        normalized = add_mandatory_warning_records(warnings)
        output["warnings"] = [
            {
                "warning_id": item["warning_id"],
                "type": item["category"],
                "category": item["category"],
                "severity": item["severity"],
                "message": item["message"],
            }
            for item in normalized
        ]


def generate_governed_explanation(case_id: Optional[str] = None) -> Dict[str, Any]:
    evidence = load_complete_case_evidence(case_id=case_id)
    return GovernedExplainer().generate(evidence)
