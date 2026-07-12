# Canonical Fairness and Proxy Audit

Run ID: `manuscript_final_20260712T175251Z_c664ef152ff3`  
Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`

Subgroup gaps use out-of-fold predictions, declared minimum support, metric-specific class denominators, and stratified bootstrap uncertainty. Sensitive audits and exploratory operational subgroup diagnostics are labelled separately.

## Largest support-qualified primary-policy gaps

- EmpDepartment, true_positive_rate, class 2: gap=0.7692, 95% CI [0.0462, 1.0000], minimum subgroup n=49, valid bootstrap replicates=5000/5000.
- EmpDepartment, precision, class 2: gap=0.7692, 95% CI [0.1048, 1.0000], minimum subgroup n=49, valid bootstrap replicates=5000/5000.
- EmpDepartment, false_positive_rate, class 3: gap=0.7368, 95% CI [0.6070, 0.9132], minimum subgroup n=49, valid bootstrap replicates=5000/5000.
- EducationBackground, false_positive_rate, class 3: gap=0.3977, 95% CI [0.1027, 0.6443], minimum subgroup n=66, valid bootstrap replicates=5000/5000.
- EmpDepartment, positive_prediction_rate, class 2: gap=0.2701, 95% CI [0.2040, 0.4037], minimum subgroup n=49, valid bootstrap replicates=5000/5000.
- EmpDepartment, macro_f1: gap=0.2689, 95% CI [0.1964, 0.3479], minimum subgroup n=49, valid bootstrap replicates=5000/5000.
- EmpDepartment, positive_prediction_rate, class 3: gap=0.2673, 95% CI [0.2045, 0.4020], minimum subgroup n=49, valid bootstrap replicates=5000/5000.
- EmpDepartment, mean_predicted_probability, class 2: gap=0.2211, 95% CI [0.1583, 0.3410], minimum subgroup n=49, valid bootstrap replicates=5000/5000.
- EmpDepartment, mean_predicted_probability, class 3: gap=0.2130, 95% CI [0.1473, 0.3205], minimum subgroup n=49, valid bootstrap replicates=5000/5000.
- EducationBackground, positive_prediction_rate, class 3: gap=0.2022, 95% CI [0.1176, 0.2907], minimum subgroup n=66, valid bootstrap replicates=5000/5000.

## Department reconstructability

- `no_salary_hike_no_attrition`: proxy macro-F1=0.9757 (95% CI 0.9645-0.9869); proxy target absent from predictors=True (explicit_proxy_safeguard).
- `no_salary_hike_no_attrition_no_department`: proxy macro-F1=0.9757 (95% CI 0.9645-0.9869); proxy target absent from predictors=True (already_excluded_by_performance_policy).
- `no_salary_hike_no_attrition_no_department_no_job_role`: proxy macro-F1=0.2540 (95% CI 0.2322-0.2759); proxy target absent from predictors=True (already_excluded_by_performance_policy).

## Claim boundaries

- Subgroup differences are descriptive audit evidence, not legal findings or proof of discrimination, fairness, or causality.
- Groups or class-specific denominators below threshold remain visible in the group-support file but are excluded from gap estimates.
- Department reconstructability is proxy-risk evidence; it is not proof that the performance model uses department causally or discriminatorily.
- Removing sensitive, department, or job-role fields does not establish fairness or eliminate indirect proxy information.
- The system is research-grade decision support only and must not make autonomous HR decisions.
