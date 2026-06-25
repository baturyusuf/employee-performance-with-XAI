# Table 3. Internal and external performance metrics

Source data: reports/external_validation/external_validation_summary.md; reports/manuscript_assets/external_validation_tables.md

| Dataset / task               | Policy                                    | Macro-F1 | Balanced accuracy | QWK       | Log loss | Brier    | ECE      | Interpretation                                                              |
| ---------------------------- | ----------------------------------------- | -------- | ----------------- | --------- | -------- | -------- | -------- | --------------------------------------------------------------------------- |
| INX primary model | no_salary_hike_no_attrition_no_department | 0.59881 | 0.62606 | 0.637613 | 0.455092 | 0.260802 | 0.063846 | Internal benchmark for the selected primary policy. |
| HRDataset_v14 | department_free | 0.638943 | 0.656693 | 0.5945 | 0.550791 | 0.256523 | 0.092454 | Independent replication on a directly mappable external performance target. |
| IBM HR Analytics performance | department_free | 0.460311 | 0.497389 | -0.008499 | 0.497845 | 0.286919 | 0.082177 | Restricted 3/4 target-space robustness, not full 2/3/4 validation. |
| IBM HR Analytics attrition | department_free | 0.674896 | 0.63884 | 0.363876 | 0.356022 | 0.204166 | 0.03877 | Related HR attrition task transfer, not performance validation. |
| Employee Turnover | without_last_evaluation | 0.966562 | 0.960323 | 0.933132 | 0.081045 | 0.039546 | 0.007724 | Related HR turnover task transfer, not performance validation. |
