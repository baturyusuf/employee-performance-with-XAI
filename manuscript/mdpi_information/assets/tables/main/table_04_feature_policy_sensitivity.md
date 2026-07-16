# Matched-fold feature-policy sensitivity

Same outer folds and selected candidate schedule. Audit-only policies are sensitivity models, not alternate primary systems.

| policy | role | status | feature_count | macro_f1_display | macro_f1_difference_vs_primary_display | qwk_display | qwk_difference_vs_primary_display | ordinal_mae_display | severe_error_rate_display | supported_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_feature_upper_bound | diagnostic_upper_bound_never_deployable | audit_only | 26 | 0.8914 | 0.2703 | 0.8496 | 0.2820 | 0.0775 | 0.0025 | matched feature-access sensitivity; not causal and not an alternative primary model |
| no_salary_hike | leakage_ablation_non_primary | audit_only | 25 | 0.6270 | 0.0060 | 0.5813 | 0.0137 | 0.2283 | 0.0017 | matched feature-access sensitivity; not causal and not an alternative primary model |
| no_salary_hike_no_attrition_sensitive_retaining_audit | sensitive_retaining_leakage_ablation_non_primary | audit_only | 24 | 0.6279 | 0.0069 | 0.5814 | 0.0138 | 0.2308 | 0.0017 | matched feature-access sensitivity; not causal and not an alternative primary model |
| no_salary_hike_no_attrition | governed_leakage_ablation_non_primary | matched_sensitivity | 21 | 0.6150 | -0.0060 | 0.5569 | -0.0107 | 0.2483 | 0.0025 | matched feature-access sensitivity; not causal and not an alternative primary model |
| no_salary_hike_no_attrition_no_department | canonical_primary | primary | 20 | 0.6210 | N/A | 0.5676 | N/A | 0.2433 | 0.0025 | prespecified primary |
| no_salary_hike_no_attrition_no_department_no_job_role | strict_proxy_sensitivity_non_primary | matched_sensitivity | 19 | 0.5880 | -0.0330 | 0.5338 | -0.0338 | 0.2800 | 0.0033 | matched feature-access sensitivity; not causal and not an alternative primary model |
