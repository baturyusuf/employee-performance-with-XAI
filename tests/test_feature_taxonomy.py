from __future__ import annotations

import unittest

from src.features.feature_sets import taxonomy_by_feature


class FeatureTaxonomyTests(unittest.TestCase):
    def test_target_and_identifier_are_forbidden(self) -> None:
        taxonomy = taxonomy_by_feature()
        self.assertEqual(taxonomy["EmpNumber"]["control_type"], "forbidden")
        self.assertFalse(taxonomy["EmpNumber"]["allowed_for_final_model"])
        self.assertEqual(taxonomy["PerformanceRating"]["control_type"], "forbidden")
        self.assertFalse(taxonomy["PerformanceRating"]["allowed_for_final_model"])

    def test_primary_leakage_features_are_high_risk_and_not_allowed(self) -> None:
        taxonomy = taxonomy_by_feature()
        for feature in ["EmpLastSalaryHikePercent", "Attrition"]:
            self.assertEqual(taxonomy[feature]["leakage_risk"], "high")
            self.assertFalse(taxonomy[feature]["allowed_for_final_model"])

    def test_department_and_job_role_proxy_annotations_exist(self) -> None:
        taxonomy = taxonomy_by_feature()
        self.assertEqual(taxonomy["EmpDepartment"]["sensitive_or_proxy"], "organisational_group")
        self.assertEqual(taxonomy["EmpJobRole"]["sensitive_or_proxy"], "possible_proxy")


if __name__ == "__main__":
    unittest.main()
