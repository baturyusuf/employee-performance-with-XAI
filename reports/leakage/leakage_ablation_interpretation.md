# Leakage Ablation Interpretation

This report derives Leakage Sensitivity Index (LSI) values from existing cross-validation ablation outputs.

Source summary: `reports\robustness\leakage_ablation\leakage_ablation_summary.csv`

Full-feature results are upper-bound/leakage-warning baselines, not deployable HR decision-support models.

LSI is computed as:

`LSI(metric) = (score_full_feature - score_leakage_safe) / abs(score_full_feature)`

A higher LSI indicates stronger model dependence on leakage-risk or outcome-proximal features. This is evidence of sensitivity to leakage-risk variables, not proof of causal leakage.

## Primary macro-F1 LSI vs no_salary_hike_no_attrition

- catboost: full=0.8578, safe=0.5567, absolute_drop=0.3011, LSI=0.3510
- xgboost: full=0.9038, safe=0.6020, absolute_drop=0.3018, LSI=0.3339
- lightgbm: full=0.9032, safe=0.6028, absolute_drop=0.3004, LSI=0.3326

## Scientific interpretation

`EmpLastSalaryHikePercent` and `Attrition` must remain excluded from primary decision-support feature sets. The full-feature benchmark should be presented as a cautionary upper bound showing how apparent performance can inflate when outcome-proximal variables are available.

## Required follow-up

- Regenerate this report from a config-backed leakage ablation script.
- Add synthetic leakage stress tests.
- Compare explanation shift between full-feature and leakage-safe models.
- Do not select a final model until calibration, fairness, SHAP stability, and actionability evidence are reviewed.
