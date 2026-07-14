# HRDataset_v14 independent mapped-target replication

Run ID: `stage_validation_hrdataset_20260713T175045Z_5af0262e83a3`  
Config hash: `5af0262e83a3720f8dca0b4d6795bdffc6bb2cefedc901ae0a47f9262d07f305`

This stage independently trains the leakage-aware XGBoost protocol on HRDataset_v14. It is not locked INX-model transport or universal external validation.

## Conservative primary result

- Raw macro-F1: 0.666355 (95% CI 0.628073–0.704690).
- Raw QWK: 0.541220 (95% CI 0.485143–0.598791).
- Predeclared sigmoid macro-F1: 0.625251 (95% CI 0.607003–0.643085).
- Fold summaries are descriptive variability only; sample-level intervals use 5,000 paired outer-fold/target-stratified bootstrap draws.

## Claim boundaries

- Engagement, satisfaction, project, lateness and attendance fields have unverified timing; the temporality-restricted audit is reported separately.
- SHAP values are exact-fold model attribution, not causal effects.
- Subgroup differences are support-aware descriptive diagnostics, not proof of fairness, discrimination, legal compliance or deployment readiness.
- Department reconstructability status is `not_estimated_insufficient_outer_training_class_support`; classes were not merged or dropped.
- No employee advice, prescription, autonomous HR decision, or causal intervention is supported.
- Dataset source authenticity and licence remain manual-review items.

## Locked-model transport feasibility

Common conservative features: 3 (EmpJobRole, EmpJobSatisfaction, ExperienceYearsAtThisCompany). Status: `infeasible_too_few_common_safe_features`. No locked model was transported.
