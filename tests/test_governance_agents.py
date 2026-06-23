from __future__ import annotations

import unittest

from src.agents.calibration_agent import CalibrationAuditAgent
from src.agents.fairness_proxy_agent import FairnessProxyAuditAgent
from src.agents.leakage_agent import LeakageAuditAgent
from src.agents.supervisor_agent import SupervisorGovernanceAgent
from src.llm.evidence_schema import CompleteCaseEvidence


class GovernanceAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = CompleteCaseEvidence.from_reports().to_dict()

    def test_leakage_agent_checks_required_exclusions(self) -> None:
        finding = LeakageAuditAgent().audit(self.evidence)
        self.assertIn(finding.status, {"pass", "pass_with_warnings"})
        self.assertIn("EmpLastSalaryHikePercent", self.evidence["leakage"]["excluded_leakage_features"])

    def test_fairness_agent_warns_proxy(self) -> None:
        finding = FairnessProxyAuditAgent().audit(self.evidence)
        self.assertIn("proxy", " ".join(finding.required_warnings).lower())

    def test_calibration_agent_outputs_risk_level(self) -> None:
        finding = CalibrationAuditAgent().audit(self.evidence)
        self.assertIn(finding.risk_level, {"low", "medium", "high"})

    def test_supervisor_aggregates_findings(self) -> None:
        findings = [
            LeakageAuditAgent().audit(self.evidence),
            FairnessProxyAuditAgent().audit(self.evidence),
            CalibrationAuditAgent().audit(self.evidence),
        ]
        result = SupervisorGovernanceAgent().aggregate(findings)
        self.assertTrue(result["required_human_review"])
        self.assertIn(result["overall_status"], {"research_only", "decision_support_with_strong_warnings", "not_ready", "evidence_missing"})


if __name__ == "__main__":
    unittest.main()

