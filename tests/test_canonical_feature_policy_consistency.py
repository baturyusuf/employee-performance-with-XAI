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
    repository_feature_policy_projection,
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

    def test_repository_legacy_projection_matches_every_shared_policy_name(self) -> None:
        projection = repository_feature_policy_projection()
        validate_policy_consistency(
            self.config,
            {"configs/feature_sets.yaml legacy projection": projection},
        )
        canonical = canonical_policy_mapping(self.config)
        shared = set(projection).intersection(canonical)
        self.assertTrue(set(canonical).issubset(shared))

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

    def test_policy_comparison_protocol_cannot_enable_independent_tuning(self) -> None:
        malformed = copy.deepcopy(self.config)
        malformed["manuscript_final"]["feature_policies"]["comparison_protocol"][
            "independent_policy_tuning"
        ] = True
        with self.assertRaisesRegex(ManuscriptConfigError, "matched-OOF"):
            validate_manuscript_config(malformed)

    def test_calibration_contract_is_predeclared_cross_fitted_sigmoid(self) -> None:
        calibration = self.config["manuscript_final"]["calibration"]

        self.assertEqual(calibration["primary_method"], "sigmoid")
        self.assertEqual(calibration["comparison_systems"], ["raw", "sigmoid"])
        self.assertEqual(
            calibration["method_selection"],
            "predeclared_not_outer_test_selected",
        )
        self.assertFalse(calibration["selection_performed"])
        self.assertEqual(
            calibration["training_protocol"],
            "five_fold_cross_fitted_outer_training_only",
        )
        self.assertEqual(calibration["inner_splits"], 5)
        self.assertEqual(calibration["outer_test_usage"], "evaluation_only")
        self.assertFalse(calibration["outer_model_refit_in_calibration_stage"])
        self.assertFalse(
            calibration["outer_test_used_for_tuning_fitting_selection_or_thresholds"]
        )
        self.assertEqual(
            calibration["sigmoid"],
            {
                "algorithm": "one_vs_rest_platt_logit_then_row_renormalize",
                "implementation_dependency": "scikit-learn>=1.8,<1.9",
                "solver": "lbfgs",
                "regularization": "l2_via_l1_ratio_zero",
                "l1_ratio": 0.0,
                "C": 1.0,
                "fit_intercept": True,
                "max_iter": 1000,
                "tol": 1.0e-10,
                "probability_clip": 1.0e-6,
                "solver_threadpool_limit": 1,
            },
        )

    def test_calibration_contract_rejects_outer_test_use_or_protocol_drift(self) -> None:
        mutations = {
            "selection_performed": True,
            "outer_model_refit_in_calibration_stage": True,
            "outer_test_usage": "fit_and_evaluation",
            "outer_test_used_for_tuning_fitting_selection_or_thresholds": True,
            "inner_splits": 3,
            "primary_method": "isotonic",
            "comparison_systems": ["raw", "sigmoid", "isotonic"],
        }

        for field, value in mutations.items():
            with self.subTest(field=field):
                malformed = copy.deepcopy(self.config)
                malformed["manuscript_final"]["calibration"][field] = value
                with self.assertRaisesRegex(ManuscriptConfigError, "calibration differs"):
                    validate_manuscript_config(malformed)


if __name__ == "__main__":
    unittest.main()
