# Canonical Feature-Policy Interpretation

Run ID: `manuscript_final_20260712T175251Z_c664ef152ff3`  
Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`

All policy estimates use the same predeclared stratified folds, preprocessing implementation, XGBoost settings, and target labels. Full-feature results are diagnostic leakage-warning upper bounds only and are not deployable evidence.

## Policy Results

- `full_feature_upper_bound` (diagnostic_upper_bound_never_deployable; audit_only=True): macro-F1 0.9051 (95% CI 0.8789–0.9312); QWK 0.8695; ordinal MAE 0.0658; features 26.
- `no_salary_hike` (leakage_ablation_non_primary; audit_only=True): macro-F1 0.6010 (95% CI 0.5834–0.6186); QWK 0.6389; ordinal MAE 0.1500; features 25.
- `no_salary_hike_no_attrition` (governed_leakage_ablation_non_primary; audit_only=False): macro-F1 0.6034 (95% CI 0.5829–0.6238); QWK 0.6442; ordinal MAE 0.1475; features 21.
- `no_salary_hike_no_attrition_no_department` (canonical_primary; audit_only=False): macro-F1 0.5987 (95% CI 0.5785–0.6190); QWK 0.6380; ordinal MAE 0.1525; features 20.
- `no_salary_hike_no_attrition_no_department_no_job_role` (strict_proxy_sensitivity_non_primary; audit_only=False): macro-F1 0.5476 (95% CI 0.5300–0.5652); QWK 0.5349; ordinal MAE 0.2050; features 19.
- `no_salary_hike_no_attrition_sensitive_retaining_audit` (sensitive_retaining_leakage_ablation_non_primary; audit_only=True): macro-F1 0.6018 (95% CI 0.5825–0.6210); QWK 0.6392; ordinal MAE 0.1500; features 24.

## Claim Boundaries

- Performance changes across compound policies must not be attributed to one removed field unless a dedicated audit-only contrast isolates that field.
- The sensitive-retaining audit policy is diagnostic only; it is included to separate leakage-variable effects from demographic-governance exclusions.
- Excluding sensitive or organisational fields does not prove fairness, eliminate proxy risk, or identify causal effects.
- These models are research-grade decision support and must not be used for autonomous HR decisions.
