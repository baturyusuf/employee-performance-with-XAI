from __future__ import annotations

import inspect
import unittest

import pandas as pd

from src.experiments import manuscript_counterfactual_search as counterfactual_module
from src.experiments.manuscript_counterfactual_search import (
    build_candidates,
    candidate_scope_features,
    respects_relational_constraints,
    training_scales,
)


class CounterfactualProtocolIsOOFTests(unittest.TestCase):
    def test_supplementary_module_does_not_import_calibration_private_helpers(self) -> None:
        source = inspect.getsource(counterfactual_module)
        self.assertNotIn("manuscript_calibration", source)
        self.assertNotIn("_fit_pipeline", source)

    def test_candidates_use_supplied_training_prototypes_and_observed_values(self) -> None:
        training = pd.DataFrame(
            {
                "TrainingTimesLastYear": [1, 2, 3],
                "EmpJobInvolvement": [2, 3, 4],
                "ExperienceYearsAtThisCompany": [4, 5, 6],
                "ExperienceYearsInCurrentRole": [2, 3, 4],
            }
        )
        sample = training.iloc[0].copy()
        prototypes = training.iloc[1:].copy()
        candidates, changes, diagnostics = build_candidates(
            sample,
            prototypes,
            ["TrainingTimesLastYear", "EmpJobInvolvement"],
            training_scales(training),
            max_features_changed=2,
            max_prototypes=2,
        )
        self.assertFalse(candidates.empty)
        self.assertEqual(diagnostics["prototypes_considered"], 2)
        observed = set(prototypes["TrainingTimesLastYear"])
        changed_values = {
            change["new_value"]
            for change_set in changes
            for change in change_set
            if change["feature"] == "TrainingTimesLastYear"
        }
        self.assertTrue(changed_values.issubset(observed))

    def test_relational_tenure_constraints_are_enforced(self) -> None:
        valid = pd.Series(
            {
                "ExperienceYearsInCurrentRole": 3,
                "ExperienceYearsAtThisCompany": 5,
                "TotalWorkExperienceInYears": 8,
                "YearsWithCurrManager": 2,
                "YearsSinceLastPromotion": 1,
            }
        )
        invalid = valid.copy()
        invalid["ExperienceYearsInCurrentRole"] = 7
        self.assertTrue(respects_relational_constraints(valid))
        self.assertFalse(respects_relational_constraints(invalid))

    def test_employee_mode_excludes_immutable_and_organisation_fields(self) -> None:
        taxonomy = {
            "Employee": {"control_type": "employee_controllable"},
            "Manager": {"control_type": "manager_controllable"},
            "Organisation": {"control_type": "organisation_controllable"},
            "History": {"control_type": "immutable"},
        }
        scopes = {
            "employee_control_tagged": ["employee_controllable"],
            "diagnostic_including_immutable_history": [
                "employee_controllable",
                "manager_controllable",
                "organisation_controllable",
                "immutable",
            ],
        }
        employee = candidate_scope_features(
            "employee_control_tagged", taxonomy, scopes, taxonomy
        )
        diagnostic = candidate_scope_features(
            "diagnostic_including_immutable_history", taxonomy, scopes, taxonomy
        )
        self.assertEqual(employee, ["Employee"])
        self.assertIn("History", diagnostic)


if __name__ == "__main__":
    unittest.main()
