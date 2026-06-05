from __future__ import annotations

import unittest

import pandas as pd

from src.features.feature_sets import apply_feature_set, build_feature_columns


ALL_COLUMNS = [
    "EmpNumber",
    "Age",
    "Gender",
    "EmpDepartment",
    "EmpJobRole",
    "EmpLastSalaryHikePercent",
    "Attrition",
    "TrainingTimesLastYear",
    "EmpJobInvolvement",
    "PerformanceRating",
]


class FeatureSetTests(unittest.TestCase):
    def test_leakage_safe_feature_set_excludes_target_id_and_leakage_features(self) -> None:
        columns = build_feature_columns(ALL_COLUMNS, "no_salary_hike_no_attrition")

        self.assertNotIn("EmpNumber", columns)
        self.assertNotIn("PerformanceRating", columns)
        self.assertNotIn("EmpLastSalaryHikePercent", columns)
        self.assertNotIn("Attrition", columns)
        self.assertNotIn("Age", columns)
        self.assertIn("EmpDepartment", columns)

    def test_department_free_feature_set_excludes_department(self) -> None:
        columns = build_feature_columns(ALL_COLUMNS, "no_salary_hike_no_attrition_no_department")
        self.assertNotIn("EmpDepartment", columns)
        self.assertNotIn("Age", columns)
        self.assertIn("EmpJobRole", columns)

    def test_age_included_audit_feature_set_retains_age(self) -> None:
        columns = build_feature_columns(ALL_COLUMNS, "no_salary_hike_no_attrition_with_age_audit")
        self.assertIn("Age", columns)
        self.assertNotIn("EmpLastSalaryHikePercent", columns)

    def test_job_role_free_sensitivity_feature_set_excludes_job_role(self) -> None:
        columns = build_feature_columns(
            ALL_COLUMNS,
            "no_salary_hike_no_attrition_no_department_no_job_role",
        )
        self.assertNotIn("Age", columns)
        self.assertNotIn("EmpDepartment", columns)
        self.assertNotIn("EmpJobRole", columns)
        self.assertNotIn("EmpLastSalaryHikePercent", columns)
        self.assertNotIn("Attrition", columns)

    def test_actionable_employee_only_uses_taxonomy_control_type(self) -> None:
        columns = build_feature_columns(ALL_COLUMNS, "actionable_employee_only")
        self.assertIn("TrainingTimesLastYear", columns)
        self.assertIn("EmpJobInvolvement", columns)
        self.assertNotIn("EmpDepartment", columns)

    def test_apply_feature_set_returns_dataframe_copy(self) -> None:
        df = pd.DataFrame([{col: 1 for col in ALL_COLUMNS}])
        out = apply_feature_set(df, "no_salary_hike_no_attrition")

        self.assertIsInstance(out, pd.DataFrame)
        self.assertNotIn("PerformanceRating", out.columns)
        self.assertNotIn("EmpLastSalaryHikePercent", out.columns)


if __name__ == "__main__":
    unittest.main()
