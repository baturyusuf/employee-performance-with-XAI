from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

import pandas as pd

from src.data.external_adapters import build_feature_columns, load_external_dataset, role_columns
from src.llm.output_schema import validate_governed_explanation_payload
from src.models.evaluate import multiclass_brier
from src.utils.config import SETTINGS


class ExternalValidationTests(unittest.TestCase):
    def test_hrdataset_performance_score_mapping(self) -> None:
        dataset = load_external_dataset("hrdataset_v14")
        counts = dataset.canonical[dataset.target_column].value_counts().sort_index().to_dict()

        self.assertEqual(dataset.target_column, "PerformanceRating")
        self.assertEqual(counts, {2: 31, 3: 243, 4: 37})
        self.assertEqual(dataset.labels, [2, 3, 4])

    def test_ibm_performance_target_space_is_restricted(self) -> None:
        dataset = load_external_dataset("ibm_hr_analytics")

        self.assertEqual(dataset.labels, [3, 4])
        self.assertEqual(dataset.task_type, "restricted_ordinal_performance")

    def test_external_policy_excludes_leakage_and_sensitive_columns(self) -> None:
        dataset = load_external_dataset("hrdataset_v14")
        features = set(build_feature_columns(dataset, "department_free"))

        for col in role_columns(dataset, "leakage"):
            self.assertNotIn(col, features)
        for col in role_columns(dataset, "sensitive"):
            self.assertNotIn(col, features)
        self.assertNotIn("EmpDepartment", features)
        self.assertNotIn(dataset.target_column, features)
        self.assertNotIn("EmpNumber", features)

    def test_employee_turnover_last_evaluation_sensitivity(self) -> None:
        dataset = load_external_dataset("employee_turnover")
        with_eval = set(build_feature_columns(dataset, "with_last_evaluation"))
        without_eval = set(build_feature_columns(dataset, "without_last_evaluation"))

        self.assertIn("LastEvaluation", with_eval)
        self.assertNotIn("LastEvaluation", without_eval)

    def test_binary_brier_uses_two_column_target(self) -> None:
        value = multiclass_brier([0, 1], pd.DataFrame([[1.0, 0.0], [0.0, 1.0]]).to_numpy(), [0, 1])
        self.assertEqual(value, 0.0)

    def test_generated_external_reports_exist_with_required_columns(self) -> None:
        required = {
            "reports/external_validation/hrdataset_v14/performance_metrics.csv": {"policy", "macro_f1", "ece_confidence"},
            "reports/external_validation/ibm_hr_analytics/performance_metrics.csv": {"policy", "macro_f1", "multiclass_brier"},
            "reports/external_validation/employee_turnover/performance_metrics.csv": {"policy", "macro_f1", "nll_log_loss"},
            "reports/external_validation/external_llm_usage_summary.csv": {"dataset", "total_tokens", "estimated_cost_usd"},
            "reports/external_validation/external_validation_summary.md": None,
            "reports/manuscript_assets/external_validation_tables.md": None,
            "reports/governance_reports/external_validation_governance_summary.md": None,
        }
        for rel_path, columns in required.items():
            path = SETTINGS.project_root / rel_path
            self.assertTrue(path.exists(), rel_path)
            if columns is not None:
                observed = set(pd.read_csv(path).columns)
                self.assertTrue(columns.issubset(observed), rel_path)

    def test_external_llm_output_core_schema_validation(self) -> None:
        path = (
            SETTINGS.project_root
            / "reports"
            / "external_validation"
            / "hrdataset_v14"
            / "llm_agent_governance"
            / "case_49"
            / "governed_explanation.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        projected = {
            "case_id": payload["case_id"],
            "short_explanation": payload["short_explanation"],
            "detailed_explanation": payload["detailed_explanation"],
            "warnings": [
                {
                    "type": warning["type"],
                    "severity": warning["severity"],
                    "message": warning["message"],
                }
                for warning in payload["warnings"]
            ],
            "unsupported_claims_detected": payload["unsupported_claims_detected"],
            "requires_human_review": payload["requires_human_review"],
        }
        validate_governed_explanation_payload(projected)

    def test_no_forbidden_decision_language_in_external_governance_reports(self) -> None:
        report_paths = [
            SETTINGS.project_root / "reports" / "external_validation" / "external_validation_summary.md",
            SETTINGS.project_root / "reports" / "manuscript_assets" / "external_validation_tables.md",
            SETTINGS.project_root / "reports" / "governance_reports" / "external_validation_governance_summary.md",
        ]
        forbidden = [
            r"should be promoted",
            r"should be fired",
            r"should receive salary",
            r"should be disciplined",
            r"automatically decide",
            r"the employee should",
        ]
        for path in report_paths:
            text = path.read_text(encoding="utf-8").lower()
            for pattern in forbidden:
                self.assertIsNone(re.search(pattern, text), f"{pattern} in {path}")

    def test_no_api_key_pattern_in_project_text_artifacts(self) -> None:
        roots = ["src", "tests", "configs", "reports", "data/external"]
        pattern = re.compile(r"sk-[A-Za-z0-9_-]{20,}")
        for root in roots:
            for path in (SETTINGS.project_root / root).rglob("*"):
                if path.is_dir() or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".joblib", ".cbm", ".pyc"}:
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                self.assertIsNone(pattern.search(text), str(path))


if __name__ == "__main__":
    unittest.main()
