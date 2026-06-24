# External Experiment Interpretation: Human Resources Analytics / Employee Turnover

Dataset role: HR task-transfer robustness
Task type: `binary_turnover`
Target kind: `primary`
Minimum subgroup support threshold: 30

## Performance Summary

- `with_last_evaluation`: macro-F1=0.9703, balanced accuracy=0.9627, QWK=0.9406, log loss=0.0727, ECE=0.0059.
- `without_last_evaluation`: macro-F1=0.9666, balanced accuracy=0.9603, QWK=0.9331, log loss=0.0810, ECE=0.0077.

## Interpretation

This is HR task-transfer robustness for turnover prediction. It must not be described as performance external validation.

## Required Claim Limits

- The model is research-grade decision support only.
- Full-feature or leakage-risk variables are not deployable final-model evidence.
- SHAP is attribution, not causality.
- Counterfactual/actionability outputs are not employee prescriptions.
- Removing sensitive or group variables does not prove fairness.
