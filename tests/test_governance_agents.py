from __future__ import annotations

import unittest

from src.agents.calibration_agent import CalibrationAuditAgent
from src.agents.counterfactual_actionability_agent import CounterfactualActionabilityAgent
from src.agents.explanation_agent import ExplanationComplianceAgent
from src.agents.fairness_proxy_agent import FairnessProxyAuditAgent
from src.agents.leakage_agent import LeakageAuditAgent
from src.agents.shap_stability_agent import ShapStabilityAuditAgent
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

    def test_missing_calibration_evidence_needs_evidence(self) -> None:
        evidence = dict(self.evidence)
        evidence["calibration"] = None
        finding = CalibrationAuditAgent().audit(evidence)
        self.assertEqual(finding.status, "needs_evidence")
        self.assertEqual(finding.risk_level, "high")

    def test_shap_missing_warning(self) -> None:
        evidence = dict(self.evidence)
        evidence["shap"] = None
        finding = ShapStabilityAuditAgent().audit(evidence)
        self.assertEqual(finding.status, "needs_evidence")

    def test_counterfactual_actionability_warns_invalid(self) -> None:
        evidence = dict(self.evidence)
        evidence["counterfactual"] = {"validity": 0.0, "actionability_label": "not_employee_actionable"}
        finding = CounterfactualActionabilityAgent().audit(evidence)
        self.assertEqual(finding.risk_level, "high")

    def test_explanation_agent_flags_unsafe_explanation(self) -> None:
        bad = {
            "short_explanation": "The employee should be promoted.",
            "detailed_explanation": "The model is a fair model and this caused performance.",
            "warnings": [],
        }
        finding = ExplanationComplianceAgent().audit_explanation(bad, self.evidence)
        self.assertEqual(finding.status, "fail")

    def test_supervisor_aggregates_findings(self) -> None:
        findings = [
            LeakageAuditAgent().audit(self.evidence),
            FairnessProxyAuditAgent().audit(self.evidence),
            CalibrationAuditAgent().audit(self.evidence),
        ]
        result = SupervisorGovernanceAgent().aggregate(findings)
        self.assertTrue(result["required_human_review"])
        self.assertIn(result["overall_status"], {"research_only", "decision_support_with_strong_warnings", "not_ready", "evidence_missing"})

    def test_supervisor_marks_missing_evidence(self) -> None:
        findings = [CalibrationAuditAgent().audit({"calibration": None})]
        result = SupervisorGovernanceAgent().aggregate(findings)
        self.assertEqual(result["overall_status"], "evidence_missing")


if __name__ == "__main__":
    unittest.main()
