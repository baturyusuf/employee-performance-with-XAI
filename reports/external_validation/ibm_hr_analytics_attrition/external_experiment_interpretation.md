# External Experiment Interpretation: IBM HR Analytics Employee Attrition and Performance

Dataset role: schema-compatible robustness
Task type: `binary_attrition`
Target kind: `attrition`
Minimum subgroup support threshold: 30

## Performance Summary

- `department_including`: macro-F1=0.6807, balanced accuracy=0.6444, QWK=0.3742, log loss=0.3547, ECE=0.0415.
- `department_free`: macro-F1=0.6749, balanced accuracy=0.6388, QWK=0.3639, log loss=0.3560, ECE=0.0388.
- `department_job_role_free`: macro-F1=0.6850, balanced accuracy=0.6530, QWK=0.3783, log loss=0.3688, ECE=0.0467.

## Interpretation


## Required Claim Limits

- The model is research-grade decision support only.
- Full-feature or leakage-risk variables are not deployable final-model evidence.
- SHAP is attribution, not causality.
- Counterfactual/actionability outputs are not employee prescriptions.
- Removing sensitive or group variables does not prove fairness.
