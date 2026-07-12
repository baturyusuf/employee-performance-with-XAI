from __future__ import annotations

import copy
import unittest

from src.governance.manuscript_contract import (
    FeaturePolicyConsistencyError,
    ManuscriptConfigError,
    canonical_policy_mapping,
    feature_policy_definitions,
    load_manuscript_config,
    primary_excluded_features,
    primary_policy_name,
    validate_manuscript_config,
    validate_policy_consistency,
)


EXPECTED_PRIMARY_EXCLUSIONS = {
    "Age",
    "Gender",
    "MaritalStatus",
    "EmpDepartment",
    "EmpLastSalaryHikePercent",
    "Attrition",
    "EmpNumber",
    "PerformanceRating",
}


class CanonicalFeaturePolicyConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_manuscript_config()

    def test_primary_policy_has_one_exact_governed_definition(self) -> None:
        self.assertEqual(
            primary_policy_name(self.config),
            "no_salary_hike_no_attrition_no_department",
        )
        self.assertEqual(set(primary_excluded_features(self.config)), EXPECTED_PRIMARY_EXCLUSIONS)
        definition = feature_policy_definitions(self.config)[primary_policy_name(self.config)]
        self.assertEqual(definition["role"], "canonical_primary")
        self.assertFalse(definition["audit_only"])

    def test_all_required_ablation_and_audit_policies_are_explicit(self) -> None:
        definitions = feature_policy_definitions(self.config)
        required = {
            "full_feature_upper_bound",
            "no_salary_hike",
            "no_salary_hike_no_attrition",
            "no_salary_hike_no_attrition_no_department",
            "no_salary_hike_no_attrition_no_department_no_job_role",
            "no_salary_hike_no_attrition_sensitive_retaining_audit",
        }
        self.assertTrue(required.issubset(definitions))
        for definition in definitions.values():
            self.assertIn("excluded_features", definition)
            self.assertIn("role", definition)
            self.assertIsInstance(definition["audit_only"], bool)

    def test_governed_no_attrition_family_excludes_sensitive_fields(self) -> None:
        definitions = feature_policy_definitions(self.config)
        sensitive = {"Age", "Gender", "MaritalStatus"}
        governed_family = (
            "no_salary_hike_no_attrition",
            "no_salary_hike_no_attrition_no_department",
            "no_salary_hike_no_attrition_no_department_no_job_role",
        )
        for name in governed_family:
            self.assertTrue(sensitive.issubset(definitions[name]["excluded_features"]), name)

    def test_sensitive_retaining_variant_is_non_primary_audit_only(self) -> None:
        definition = feature_policy_definitions(self.config)[
            "no_salary_hike_no_attrition_sensitive_retaining_audit"
        ]
        self.assertEqual(
            set(definition["excluded_features"]),
            {"EmpNumber", "PerformanceRating", "EmpLastSalaryHikePercent", "Attrition"},
        )
        self.assertTrue(definition["audit_only"])
        self.assertNotEqual(definition["role"], "canonical_primary")

    def test_matching_module_policy_mapping_is_accepted(self) -> None:
        mapping = canonical_policy_mapping(self.config)
        validate_policy_consistency(self.config, {"phase_2_policy_runner": mapping})

    def test_conflicting_module_policy_mapping_fails_fast(self) -> None:
        conflicting = {
            "no_salary_hike_no_attrition_no_department": [
                "EmpLastSalaryHikePercent",
                "Attrition",
                "EmpDepartment",
            ]
        }
        with self.assertRaisesRegex(FeaturePolicyConsistencyError, "missing=.*Age"):
            validate_policy_consistency(self.config, {"legacy_shap_module": conflicting})

    def test_internal_cross_section_mismatch_fails_validation(self) -> None:
        malformed = copy.deepcopy(self.config)
        malformed["manuscript_final"]["feature_policies"]["definitions"][
            "no_salary_hike_no_attrition_no_department"
        ]["excluded_features"].remove("Gender")
        with self.assertRaisesRegex(ManuscriptConfigError, "single exact union"):
            validate_manuscript_config(malformed)


if __name__ == "__main__":
    unittest.main()
