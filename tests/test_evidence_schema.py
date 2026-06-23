from __future__ import annotations

import unittest

from src.llm.evidence_schema import CompleteCaseEvidence, PredictionEvidence


class EvidenceSchemaTests(unittest.TestCase):
    def test_load_complete_case_evidence_from_reports(self) -> None:
        evidence = CompleteCaseEvidence.from_reports()
        self.assertEqual(evidence.prediction.model_name, "xgboost")
        self.assertEqual(evidence.prediction.feature_policy, "no_salary_hike_no_attrition_no_department")
        self.assertIn("Prediction requires human review.", evidence.governance.required_warnings)

    def test_llm_cannot_be_predictive_model(self) -> None:
        with self.assertRaises(ValueError):
            PredictionEvidence(
                case_id="x",
                predicted_class=3,
                true_class=None,
                class_probabilities={"3": 0.8},
                confidence=0.8,
                uncertainty_flag=False,
                model_name="llm",
                feature_policy="test",
                leakage_safe_status="unsafe",
            )


if __name__ == "__main__":
    unittest.main()

