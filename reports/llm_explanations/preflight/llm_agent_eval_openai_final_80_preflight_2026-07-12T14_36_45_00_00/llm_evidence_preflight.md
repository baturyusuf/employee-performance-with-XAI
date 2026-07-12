# CompleteCaseEvidence Preflight

Run ID: `llm_agent_eval_openai_final_80_preflight_2026-07-12T14:36:45+00:00`
Run mode: `real`
Cases requested: 80
Cases selected: 80
Cases complete: 50
Cases incomplete: 30
Cases not selected: 0
Real API execution allowed: `False`

## Readiness Distribution

| readiness_status | count | denominator | rate | wilson_ci_low | wilson_ci_high | confidence_level |
| --- | --- | --- | --- | --- | --- | --- |
| complete_case_evidence | 50 | 80 | 0.625 | 0.5154872546314301 | 0.7230582129846882 | 0.95 |
| incomplete_evidence_blocking | 30 | 80 | 0.375 | 0.2769417870153118 | 0.48451274536857003 | 0.95 |

## Missing or Invalid Evidence

| field_or_error | case_count |
| --- | --- |
| shap | 30 |
| shap.class_specific_shap_values | 30 |
| shap.explanation_stability_warning | 30 |
| shap.grouped_shap_values | 30 |
| shap.shap_stability_summary | 30 |

## Diagnostic Classification

| diagnostic_category | case_count |
| --- | --- |
| complete_case_evidence | 50 |
| local_evidence_generation_coverage_gap | 30 |

## Blocking Reasons

- incomplete evidence is not assigned to an explicit separate stratum: ['inx_primary_no_salary_hike_no_attrition_no_department_1048', 'inx_primary_no_salary_hike_no_attrition_no_department_1072', 'inx_primary_no_salary_hike_no_attrition_no_department_1083', 'inx_primary_no_salary_hike_no_attrition_no_department_1126', 'inx_primary_no_salary_hike_no_attrition_no_department_1130', 'inx_primary_no_salary_hike_no_attrition_no_department_1174', 'inx_primary_no_salary_hike_no_attrition_no_department_189', 'inx_primary_no_salary_hike_no_attrition_no_department_216', 'inx_primary_no_salary_hike_no_attrition_no_department_234', 'inx_primary_no_salary_hike_no_attrition_no_department_29', 'inx_primary_no_salary_hike_no_attrition_no_department_314', 'inx_primary_no_salary_hike_no_attrition_no_department_32', 'inx_primary_no_salary_hike_no_attrition_no_department_327', 'inx_primary_no_salary_hike_no_attrition_no_department_445', 'inx_primary_no_salary_hike_no_attrition_no_department_457', 'inx_primary_no_salary_hike_no_attrition_no_department_546', 'inx_primary_no_salary_hike_no_attrition_no_department_609', 'inx_primary_no_salary_hike_no_attrition_no_department_610', 'inx_primary_no_salary_hike_no_attrition_no_department_69', 'inx_primary_no_salary_hike_no_attrition_no_department_699', 'inx_primary_no_salary_hike_no_attrition_no_department_702', 'inx_primary_no_salary_hike_no_attrition_no_department_74', 'inx_primary_no_salary_hike_no_attrition_no_department_77', 'inx_primary_no_salary_hike_no_attrition_no_department_796', 'inx_primary_no_salary_hike_no_attrition_no_department_826', 'inx_primary_no_salary_hike_no_attrition_no_department_849', 'inx_primary_no_salary_hike_no_attrition_no_department_889', 'inx_primary_no_salary_hike_no_attrition_no_department_933', 'inx_primary_no_salary_hike_no_attrition_no_department_960', 'inx_primary_no_salary_hike_no_attrition_no_department_962']

## Interpretation Boundary

Text faithfulness/compliance and evidence completeness are separate outcomes. A perfect compliance rate cannot repair missing case evidence, and a finite all-pass sample does not establish a zero population failure probability.
