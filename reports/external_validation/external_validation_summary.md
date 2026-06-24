# External Validation Summary

This report extends the original INX HR XAI governance evidence with external validation and robustness datasets. Claims are intentionally conservative: research-grade decision support only, no autonomous HR decisions, no causal SHAP claims, no fairness guarantee, and no deployment readiness claim.

## Dataset Roles

| dataset | role | target_definition | default_policy | allowed_claim |
| --- | --- | --- | --- | --- |
| INX primary model | internal primary research benchmark | PerformanceRating 2/3/4 | no_salary_hike_no_attrition_no_department | internal benchmark only |
| HRDataset_v14 | independent replication / direct external performance validation | PerformanceScore mapped to 2/3/4 | department_free | independent external performance-target replication |
| IBM HR Analytics performance | schema-compatible restricted target-space robustness | PerformanceRating restricted to 3/4 | department_free | restricted 3/4 performance robustness |
| IBM HR Analytics attrition | optional related HR task-transfer robustness | Attrition mapped No/Yes to 0/1 | department_free | related HR attrition task transfer |
| Employee Turnover | related HR task-transfer robustness only | left 0/1 | without_last_evaluation | related HR turnover task transfer |

## Target Mapping

| dataset | raw_target | canonical_target | mapping | class_distribution |
| --- | --- | --- | --- | --- |
| INX primary model | PerformanceRating | PerformanceRating | 2/3/4 unchanged | see INX reports/model card |
| HRDataset_v14 | PerformanceScore | PerformanceRating | PerformanceScore mapped to 2/3/4 | 2:31; 3:243; 4:37 |
| IBM HR Analytics performance | PerformanceRating | PerformanceRating | PerformanceRating restricted to 3/4 | 3:1244; 4:226 |
| IBM HR Analytics attrition | Attrition | AttritionTarget | Attrition mapped No/Yes to 0/1 | 0:1233; 1:237 |
| Employee Turnover | left | AttritionTarget | left 0/1 | 0:11428; 1:3571 |

## Feature Overlap

| comparison | common_safe_feature_count | common_features | status |
| --- | --- | --- | --- |
| INX primary -> HRDataset_v14 department-free | 3 | EmpJobRole; EmpJobSatisfaction; ExperienceYearsAtThisCompany | infeasible/too limited |

## Performance Comparison

| dataset | policy | macro_f1 | balanced_accuracy | qwk | ordinal_mae | severe_error_rate | log_loss | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INX primary model | no_salary_hike_no_attrition_no_department | 0.59881 | 0.62606 | 0.637613 | 0.1525 | 0.001667 | 0.455092 | 0.260802 | 0.063846 |
| HRDataset_v14 | department_free | 0.638943 | 0.656693 | 0.5945 | 0.138264 | 0 | 0.550791 | 0.256523 | 0.092454 |
| IBM HR Analytics performance | department_free | 0.460311 | 0.497389 | -0.008499 | 0.161224 | 0 | 0.497845 | 0.286919 | 0.082177 |
| IBM HR Analytics attrition | department_free | 0.674896 | 0.63884 | 0.363876 | 0.131293 | 0 | 0.356022 | 0.204166 | 0.03877 |
| Employee Turnover | without_last_evaluation | 0.966562 | 0.960323 | 0.933132 | 0.023935 | 0 | 0.081045 | 0.039546 | 0.007724 |

## Fairness / Proxy Comparison

| dataset | policy | largest_disparity_gap | largest_disparity_attribute | department_proxy_macro_f1 | fairness_claim |
| --- | --- | --- | --- | --- | --- |
| INX primary model | no_salary_hike_no_attrition_no_department | 0.268859 | EmpDepartment macro-F1 | 0.972385 | audit evidence only; no fairness guarantee |
| HRDataset_v14 | department_free | 1 | Absences false_positive_rate |  | audit evidence only; no fairness guarantee |
| IBM HR Analytics performance | department_free | 1 | EmpJobRole precision | 0.929279 | audit evidence only; no fairness guarantee |
| IBM HR Analytics attrition | department_free | 1 | Age false_positive_rate | 0.929279 | audit evidence only; no fairness guarantee |
| Employee Turnover | without_last_evaluation | 1 | AverageMonthlyHours false_positive_rate | 0.0952739 | audit evidence only; no fairness guarantee |

## Calibration Comparison

| dataset | policy | log_loss | brier | ece | interpretation |
| --- | --- | --- | --- | --- | --- |
| INX primary model | no_salary_hike_no_attrition_no_department | 0.455092 | 0.260802 | 0.063846 | probability confidence requires calibration caution |
| HRDataset_v14 | department_free | 0.550791 | 0.256523 | 0.092454 | probability confidence requires calibration caution |
| IBM HR Analytics performance | department_free | 0.497845 | 0.286919 | 0.082177 | probability confidence requires calibration caution |
| IBM HR Analytics attrition | department_free | 0.356022 | 0.204166 | 0.03877 | probability confidence requires calibration caution |
| Employee Turnover | without_last_evaluation | 0.081045 | 0.039546 | 0.007724 | probability confidence requires calibration caution |

## Actionability Comparison

| dataset | policy | employee_controllable_share | manager_or_org_share | status |
| --- | --- | --- | --- | --- |
| INX primary model | no_salary_hike_no_attrition_no_department | 0 | 0.25 | counterfactual validity from INX final evidence |
| HRDataset_v14 | department_free | 0.133333 | 0.266667 | mixed_context_dependent |
| IBM HR Analytics performance | department_free | 0 | 0.466667 | mostly_manager_organisation_or_proxy_constrained |
| IBM HR Analytics attrition | department_free | 0.066667 | 0.4 | mostly_manager_organisation_or_proxy_constrained |
| Employee Turnover | without_last_evaluation | 0.25 | 0.625 | mostly_manager_organisation_or_proxy_constrained |

## LLM / Agent Governance Comparison

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

## Explicit Limitations

| area | limitation | allowed_claim |
| --- | --- | --- |
| Dataset provenance | External CSVs are retrieved from public mirrors and documented in dataset cards; source authenticity should be checked before publication submission. | Reproducible public-mirror robustness evidence, not audited data provenance. |
| HRDataset_v14 | Small external sample with mapped performance labels and public cross-sectional data. | Independent replication on a mappable external performance target. |
| Cross-dataset validation | Only three common department-free safe features were available. | Cross-dataset INX-to-HRDataset validation is infeasible/too limited. |
| IBM performance | PerformanceRating contains only classes 3 and 4. | Restricted target-space schema-compatible robustness. |
| Employee turnover | Target is attrition/turnover, not performance. | Related HR task-transfer robustness only. |
| LLM/agents | Small real OpenAI batches evaluate evidence interpretation, not human trust or deployment safety. | Technical LLM-agent governance evidence on structured ML/XAI artifacts. |
