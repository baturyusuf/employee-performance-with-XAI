from __future__ import annotations

import unittest

from src.llm.evidence_schema import CompleteCaseEvidence
from src.llm.faithfulness_checker import check_faithfulness
from src.llm.governed_explainer import GovernedExplainer


class FaithfulnessCheckerTests(unittest.TestCase):
    def test_offline_governed_explanation_passes(self) -> None:
        evidence = CompleteCaseEvidence.from_reports()
        output = GovernedExplainer().generate(evidence)
        self.assertTrue(output["faithfulness_check"]["faithfulness_pass"])

    def test_forbidden_claim_detection(self) -> None:
        evidence = CompleteCaseEvidence.from_reports().to_dict()
        bad = "The employee should be promoted because this feature caused performance. The model is a fair model."
        result = check_faithfulness(bad, evidence)
        self.assertFalse(result.faithfulness_pass)
        self.assertGreater(len(result.forbidden_claims), 0)

    def test_missing_warning_detection(self) -> None:
        evidence = CompleteCaseEvidence.from_reports().to_dict()
        weak = {"short_explanation": "The model predicted class 3.", "detailed_explanation": "The model predicted class 3.", "warnings": []}
        result = check_faithfulness(weak, evidence)
        self.assertFalse(result.faithfulness_pass)
        self.assertGreater(len(result.missing_warnings), 0)

    def test_negative_evidence_numbers_allow_positive_regex_match(self) -> None:
        evidence = CompleteCaseEvidence.from_reports().to_dict()
        evidence["shap"]["top_negative_features"] = [
            {"feature": "EmpWorkLifeBalance", "value": -0.10764}
        ]
        text = {
            "short_explanation": "SHAP is attribution, not causality. Human review is required. This is decision support only.",
            "detailed_explanation": (
                "EmpWorkLifeBalance had attribution -0.1076. "
                "Full-feature models are leakage-warning upper-bound baselines only. "
                "Removing EmpDepartment does not prove fairness. "
                "Counterfactuals are not employee-actionable."
            ),
            "warnings": [],
        }
        result = check_faithfulness(text, evidence)
        self.assertNotIn("Unsupported numeric claim: 0.1076", result.unsupported_claims)

    def test_counterfactual_not_be_employee_actionable_warning_is_accepted(self) -> None:
        evidence = CompleteCaseEvidence.from_reports().to_dict()
        text = {
            "short_explanation": "Prediction requires human review.",
            "detailed_explanation": (
                "SHAP is attribution, not causality. This is decision support only. "
                "Full-feature models are leakage-warning upper-bound baselines only. "
                "Removing EmpDepartment does not prove fairness. "
                "Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions."
            ),
            "warnings": [],
        }
        result = check_faithfulness(text, evidence)
        self.assertNotIn("Counterfactuals may not be employee-actionable.", result.missing_warnings)


if __name__ == "__main__":
    unittest.main()
