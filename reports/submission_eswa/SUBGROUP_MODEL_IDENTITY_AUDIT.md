# Subgroup model-identity audit

## Finding

The primary subgroup results are **P3 nominal-XGBoost subgroup diagnostics**.

P3 is the feature policy (`no_salary_hike_no_attrition_no_department`), not a model. The diagnostic package uses the persisted exactly-once out-of-fold predictions from the canonical nominal XGBoost system under P3. The evidence package binds these predictions to the canonical XGBoost model set through `xgboost_model_set_sha256=d483d8ddb93a47a99a2eab54fe3d138d8f614798dd418bca860ea3f20c813f51`.

## Evidence

- `reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use/subgroup_gap_sensitivity.csv` identifies the primary system as `system_id=no_salary_hike_no_attrition_no_department` and `v3_system_label=P3_PRIMARY_LEAKAGE_AWARE`.
- `reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use/provenance_receipt.json` binds the compact evidence to the Phase 2C run and the persisted XGBoost model set.
- The frozen diagnostic values match the nominal-XGBoost canonical metrics, including macro-F1 0.621021 and QWK 0.567602.

## Required wording

Methods §3.10, the subgroup Results subsection, Table 9, Discussion, and Supplementary Table S6 must use “P3 nominal-XGBoost subgroup diagnostics” at first mention. Later references may use “the P3 nominal-XGBoost diagnostics.”

The manuscript must keep these boundaries explicit:

- gaps are descriptive and support-dependent;
- intervals condition on fixed fitted models, folds, and eligibility;
- the displayed primary gaps do not describe all six trained systems;
- proxy reconstructability, performance-output dependence, and causal use are different questions;
- no result proves fairness, discrimination, causality, or legal compliance.

