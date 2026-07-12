from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.agents.manuscript_deterministic_audit import run


class DeterministicAgentAuditOfflineTests(unittest.TestCase):
    def test_audit_is_explicitly_offline_and_run_bound(self) -> None:
        run_id = "canonical-test-run"
        config_hash = "c" * 64
        evidence = {
            "prediction": {
                "case_id": "inx_primary:1",
                "predicted_class": 3,
                "true_class": 3,
                "class_probabilities": {"2": 0.1, "3": 0.8, "4": 0.1},
                "confidence": 0.8,
                "uncertainty_flag": False,
                "model_name": "xgboost",
                "feature_policy": "no_salary_hike_no_attrition_no_department",
                "leakage_safe_status": "leakage_reduced_primary",
                "dataset_name": "inx_primary",
            },
            "shap": {
                "top_positive_features": [{"feature": "TrainingTimesLastYear", "grouped_shap_value": 0.2}],
                "top_negative_features": [{"feature": "DistanceFromHome", "grouped_shap_value": -0.1}],
                "grouped_shap_values": {"TrainingTimesLastYear": 0.2, "DistanceFromHome": -0.1},
                "class_specific_shap_values": {"TrainingTimesLastYear": 0.2, "DistanceFromHome": -0.1},
                "shap_stability_summary": {"top10_jaccard": 0.7, "spearman": 0.6},
                "explanation_stability_warning": "SHAP is attribution, not causality.",
            },
            "fairness": {
                "audited_groups": ["Gender"],
                "subgroup_metrics": {},
                "disparity_gaps": {},
                "bootstrap_ci": {},
                "low_support_warnings": [],
                "proxy_risk_warnings": ["Removing sensitive variables does not prove fairness."],
            },
            "calibration": {
                "log_loss": 0.5,
                "brier_score": 0.2,
                "expected_calibration_error": 0.05,
                "calibration_warning": "Approximate probability only.",
            },
            "counterfactual": {
                "counterfactual_mode": "employee_only",
                "validity": 0.4,
                "changed_features": ["TrainingTimesLastYear"],
                "probability_gain": 0.1,
                "proximity_cost": 0.4,
                "actionability_label": "model_scenario",
                "failed_reason": "",
                "warning": "Model scenario, not employee advice.",
            },
            "leakage": {
                "feature_policy": "no_salary_hike_no_attrition_no_department",
                "excluded_leakage_features": ["EmpLastSalaryHikePercent", "Attrition"],
                "full_feature_score": 0.9,
                "leakage_safe_score": 0.7,
                "leakage_sensitivity_index": 0.2,
                "leakage_warning": "Full-feature evidence is diagnostic only.",
            },
            "governance": {
                "intended_use": "Research-grade decision support only.",
                "prohibited_use": "No autonomous HR decisions.",
                "model_card_summary": "Research-only model.",
                "deployment_status": "research_only",
                "required_warnings": ["Human review required."],
            },
            "evidence_sources": ["canonical-test-fixture"],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "complete_case_evidence.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "evidence": evidence,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            outputs = run(
                source,
                output_dir=root / "audit",
                run_id=run_id,
                config_hash=config_hash,
            )
            metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
            self.assertFalse(metadata["real_llm_used"])
            self.assertFalse(metadata["api_call_attempted"])
            self.assertEqual(metadata["paid_api_calls"], 0)
            rows = pd.read_csv(outputs["audit_csv"])
            self.assertEqual(set(rows["run_id"]), {run_id})
            self.assertEqual(set(rows["config_hash"]), {config_hash})
            self.assertEqual(rows["agent_name"].nunique(), 7)


if __name__ == "__main__":
    unittest.main()
