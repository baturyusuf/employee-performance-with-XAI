from __future__ import annotations

import unittest

from src.llm.evidence_schema import CalibrationEvidence, CompleteCaseEvidence, CounterfactualEvidence, PredictionEvidence


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

    def test_prediction_requires_dataset_name(self) -> None:
        with self.assertRaises(ValueError):
            PredictionEvidence(
                dataset_name="",
                case_id="x",
                predicted_class=3,
                true_class=3,
                class_probabilities={"3": 0.9},
                confidence=0.9,
                uncertainty_flag=False,
                model_name="xgboost",
                feature_policy="test",
                leakage_safe_status="safe",
            )

    def test_alias_fields_are_serialized(self) -> None:
        calibration = CalibrationEvidence(
            log_loss=0.4,
            brier_score=0.2,
            expected_calibration_error=0.05,
            calibration_warning="Use caution.",
        )
        self.assertEqual(calibration.ece, 0.05)
        counterfactual = CounterfactualEvidence(
            counterfactual_mode="employee_only",
            validity=0.0,
            changed_features=[],
            probability_gain=None,
            proximity_cost=1.2,
            actionability_label="not_actionable",
            failed_reason="none",
            warning="Do not prescribe.",
        )
        self.assertEqual(counterfactual.mode, "employee_only")
        self.assertEqual(counterfactual.cost, 1.2)

    def test_missing_evidence_sections_are_explicit(self) -> None:
        evidence = CompleteCaseEvidence.from_reports()
        evidence.shap = None
        self.assertIn("shap", evidence.missing_evidence_sections())


if __name__ == "__main__":
    unittest.main()
