from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.governance.manuscript_contract import (
    ForbiddenFeatureError,
    load_manuscript_config,
    validate_artifact_forbidden_features,
    validate_primary_feature_names,
)


class ForbiddenFeaturesAbsentFromPrimaryArtifactsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_manuscript_config()

    def test_allowed_grouped_shap_features_pass(self) -> None:
        validate_primary_feature_names(
            ["EmpJobRole", "TrainingTimesLastYear", "cat__OverTime_Yes"],
            self.config,
            context="grouped SHAP fixture",
        )

    def test_raw_and_one_hot_forbidden_features_fail(self) -> None:
        for feature_name in ("Age", "cat__Gender_Female", "MaritalStatus_Single", "EmpDepartment_Sales"):
            with self.subTest(feature_name=feature_name):
                with self.assertRaises(ForbiddenFeatureError):
                    validate_primary_feature_names([feature_name], self.config)

    def test_structured_csv_artifact_is_scanned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            safe = root / "safe_shap.csv"
            unsafe = root / "unsafe_shap.csv"
            safe.write_text(
                "feature_name,mean_abs_shap\nEmpJobRole,0.2\nTrainingTimesLastYear,0.1\n",
                encoding="utf-8",
            )
            unsafe.write_text(
                "feature_name,mean_abs_shap\ncat__Gender_Female,0.4\nEmpJobRole,0.2\n",
                encoding="utf-8",
            )
            validate_artifact_forbidden_features(safe, self.config)
            with self.assertRaisesRegex(ForbiddenFeatureError, "Gender"):
                validate_artifact_forbidden_features(unsafe, self.config)

    def test_markdown_reason_code_can_be_scanned_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            reason_code = Path(temporary_directory) / "local_reason_code.md"
            reason_code.write_text("Top attribution: **Age**.\n", encoding="utf-8")
            with self.assertRaisesRegex(ForbiddenFeatureError, "Age"):
                validate_artifact_forbidden_features(reason_code, self.config, scan_text=True)


if __name__ == "__main__":
    unittest.main()
