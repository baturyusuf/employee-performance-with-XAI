# External Experiment Interpretation: HRDataset_v14

Dataset role: independent replication and direct external performance validation
Task type: `ordinal_multiclass_performance`
Target kind: `primary`
Minimum subgroup support threshold: 16

## Performance Summary

- `department_including`: macro-F1=0.6389, balanced accuracy=0.6567, QWK=0.5945, log loss=0.5504, ECE=0.0920.
- `department_free`: macro-F1=0.6389, balanced accuracy=0.6567, QWK=0.5945, log loss=0.5508, ECE=0.0925.
- `department_job_role_free`: macro-F1=0.6368, balanced accuracy=0.6539, QWK=0.5832, log loss=0.5535, ECE=0.0934.

## Interpretation

This supports independent replication on a mappable external performance target, subject to dataset provenance and sample-size limitations.

## Required Claim Limits

- The model is research-grade decision support only.
- Full-feature or leakage-risk variables are not deployable final-model evidence.
- SHAP is attribution, not causality.
- Counterfactual/actionability outputs are not employee prescriptions.
- Removing sensitive or group variables does not prove fairness.
