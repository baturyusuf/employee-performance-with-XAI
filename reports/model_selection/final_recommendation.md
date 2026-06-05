# Final Model Recommendation

## Recommendation Categories

- Recommended primary research model: `no_salary_hike_no_attrition_no_department` + XGBoost.
- Main leakage-safe comparison baseline: `no_salary_hike_no_attrition` + XGBoost.
- Strict fairness/proxy sensitivity model: `no_salary_hike_no_attrition_no_department_no_job_role` + XGBoost.
- Full-feature models: historical leakage-warning upper-bound only, not deployable final models.

## Scientific Rationale

The primary candidate preserves most leakage-safe utility (macro-F1 0.5988, QWK 0.6376) while excluding direct department membership. It does not eliminate proxy risk because EmpJobRole remains present.

The department-including baseline has slightly higher macro-F1 (0.6036) and QWK (0.6442) but directly uses EmpDepartment, so it is a diagnostic comparison baseline.

The strict job-role-free model lowers department proxy reconstructability but reduces macro-F1 to 0.5488 and QWK to 0.5351. It is a strict sensitivity model unless the researcher chooses proxy minimization over utility.

## Required Warnings

- The model is decision support only, not an autonomous employee evaluator.
- Removing EmpDepartment does not prove fairness.
- EmpJobRole may proxy EmpDepartment and requires reason-code warnings.
- SHAP is attribution, not causality.
- Counterfactuals may require manager or organisation intervention.
- Probability estimates require calibration warnings or probability bands.
- External validation is required before operational use.

## Composite Score Policy

No single composite G-XAIR score is used as the sole decision rule. Components remain visible because performance, proxy risk, calibration, actionability, and subgroup gaps trade off against one another.
