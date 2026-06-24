# External Experiment Interpretation: IBM HR Analytics Employee Attrition and Performance

Dataset role: schema-compatible robustness
Task type: `restricted_ordinal_performance`
Target kind: `primary`
Minimum subgroup support threshold: 30

## Performance Summary

- `department_including`: macro-F1=0.4649, balanced accuracy=0.5000, QWK=0.0000, log loss=0.4955, ECE=0.0777.
- `department_free`: macro-F1=0.4603, balanced accuracy=0.4974, QWK=-0.0085, log loss=0.4978, ECE=0.0822.
- `department_job_role_free`: macro-F1=0.4692, balanced accuracy=0.5022, QWK=0.0072, log loss=0.4881, ECE=0.0795.

## Interpretation

IBM `PerformanceRating` contains a restricted target space in this run. Treat results as schema-compatible performance robustness, not direct 2/3/4 external validation.

## Required Claim Limits

- The model is research-grade decision support only.
- Full-feature or leakage-risk variables are not deployable final-model evidence.
- SHAP is attribution, not causality.
- Counterfactual/actionability outputs are not employee prescriptions.
- Removing sensitive or group variables does not prove fairness.
