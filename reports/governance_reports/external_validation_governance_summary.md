# External Validation Governance Summary

## What Can Be Claimed

- HRDataset_v14 provides independent replication on a directly mappable external performance target.
- IBM HR Analytics provides schema-compatible robustness, but its performance target is restricted to classes 3 and 4.
- IBM attrition and Employee Turnover provide related HR risk-prediction task-transfer evidence, not performance validation.
- Real OpenAI governed explanations and OpenAI Agents SDK audits ran on small representative external batches for HRDataset_v14, IBM performance robustness, and Employee Turnover.

## What Cannot Be Claimed

- No autonomous hiring, firing, promotion, compensation, discipline, or individual employment decision capability.
- No causal interpretation of SHAP or counterfactuals.
- No proof of fairness from removing sensitive or group variables.
- No direct performance external-validation claim for IBM or Employee Turnover.
- No deployment readiness without independent data provenance review, human validation, legal review, and organisation-specific governance.

## Remaining Deployment Blockers

- External data provenance and licensing should be independently verified before publication.
- Cross-dataset INX-to-HRDataset feature overlap is too weak for a defensible transportability result.
- Subgroup results remain sample-size and support-threshold sensitive.
- LLM-agent evaluation is small-batch technical evidence, not human-subject validation.

## Q3 Manuscript Positioning

The project can be positioned as a leakage-aware, fairness/proxy-audited, calibration-aware, actionability-constrained, SHAP/XAI, LLM-assisted governance framework with independent external replication and related-task robustness. It should not be positioned as a deployable HR decision system.

## Supporting Tables

| dataset | policy | macro_f1 | balanced_accuracy | qwk | ordinal_mae | severe_error_rate | log_loss | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INX primary model | no_salary_hike_no_attrition_no_department | 0.59881 | 0.62606 | 0.637613 | 0.1525 | 0.001667 | 0.455092 | 0.260802 | 0.063846 |
| HRDataset_v14 | department_free | 0.638943 | 0.656693 | 0.5945 | 0.138264 | 0 | 0.550791 | 0.256523 | 0.092454 |
| IBM HR Analytics performance | department_free | 0.460311 | 0.497389 | -0.008499 | 0.161224 | 0 | 0.497845 | 0.286919 | 0.082177 |
| IBM HR Analytics attrition | department_free | 0.674896 | 0.63884 | 0.363876 | 0.131293 | 0 | 0.356022 | 0.204166 | 0.03877 |
| Employee Turnover | without_last_evaluation | 0.966562 | 0.960323 | 0.933132 | 0.023935 | 0 | 0.081045 | 0.039546 | 0.007724 |

| dataset | case_ids | n_cases | faithfulness_pass_rate | unsupported_claim_rate | forbidden_claim_rate | missing_warning_rate | agent_success_rate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INX primary model | 528;376;568;18;392;405;125;176;662;906 | 10 | 1 | 0 | 0 | 0 | 1 | real OpenAI + OpenAI Agents SDK |
| HRDataset_v14 | hrdataset_v14_department_free_49;hrdataset_v14_department_free_209;hrdataset_v14_department_free_208;hrdataset_v14_department_free_1;hrdataset_v14_department_free_105 | 5 | 1 | 0 | 0 | 0 | 1 | real OpenAI + OpenAI Agents SDK |
| IBM HR Analytics performance | ibm_hr_analytics_department_free_45;ibm_hr_analytics_department_free_174;ibm_hr_analytics_department_free_918;ibm_hr_analytics_department_free_20;ibm_hr_analytics_department_free_272 | 5 | 1 | 0 | 0 | 0 | 1 | real OpenAI + OpenAI Agents SDK |
| IBM HR Analytics attrition |  | 0 |  |  |  |  |  | not run |
| Employee Turnover | employee_turnover_without_last_evaluation_2594;employee_turnover_without_last_evaluation_3989;employee_turnover_without_last_evaluation_1158;employee_turnover_without_last_evaluation_137;employee_turnover_without_last_evaluation_14587 | 5 | 1 | 0 | 0 | 0 | 1 | real OpenAI + OpenAI Agents SDK |

## External LLM Usage / Cost Summary

Usage is cumulative for the listed case IDs and includes remediation reruns; billing dashboards remain the source of truth.

| dataset | logged_usage_rows | input_tokens | output_tokens | total_tokens | estimated_cost_usd |
| --- | --- | --- | --- | --- | --- |
| INX primary model | 190 | 1122859 | 95024 | 1217883 | 0.951109 |
| HRDataset_v14 | 90 | 474550 | 47726 | 522276 | 0.451275 |
| IBM HR Analytics performance | 90 | 476167 | 49621 | 525788 | 0.460497 |
| IBM HR Analytics attrition | 0 | 0 | 0 | 0 | 0 |
| Employee Turnover | 135 | 653058 | 71461 | 724519 | 0.630792 |
