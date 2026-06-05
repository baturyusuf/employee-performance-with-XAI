# Bootstrap Fairness Disparity Interpretation

Bootstrap iterations: 500
Minimum group support: 30
Model: XGBoost with fold-safe preprocessing

These metrics are diagnostic audit evidence. They are not legal findings and do not prove discrimination or absence of discrimination.

## Department Disparity With Uncertainty

- `no_salary_hike_no_attrition`: EmpDepartment macro-F1 gap 0.2733 (95% CI 0.1944-0.3573); accuracy gap 0.0873 (95% CI 0.0392-0.1958).
- `no_salary_hike_no_attrition_no_department`: EmpDepartment macro-F1 gap 0.2689 (95% CI 0.1849-0.3492); accuracy gap 0.0984 (95% CI 0.0428-0.2172).
- `no_salary_hike_no_attrition_no_department_no_job_role`: EmpDepartment macro-F1 gap 0.2148 (95% CI 0.1744-0.3025); accuracy gap 0.1286 (95% CI 0.0755-0.2338).

## Required Answers

### Did removing EmpDepartment materially reduce department-related disparity?
No clear material reduction is supported. The department macro-F1 gap is nearly unchanged after direct department removal, and uncertainty should be treated as overlapping diagnostic evidence.

### Did removing EmpJobRole materially reduce proxy/fairness risk?
It materially reduced department reconstructability in the proxy audit, but subgroup disparity improvement is mixed. The strict set lowers the department macro-F1 gap point estimate while worsening utility and not uniformly improving every disparity metric.

### What was the utility cost?
Relative to the department-free candidate, strict job-role removal changes macro-F1 from 0.5988 to 0.5488, QWK from 0.6376 to 0.5351, and ordinal MAE from 0.1525 to 0.2050.

### Which disparities remain concerning?
Department-related macro-F1, class-2 TPR/precision, and class-3 FPR gaps remain prominent. Class-4 precision gaps are large but sparse-prediction-sensitive and should be interpreted cautiously.

### Which claims are not justified by the evidence?
The evidence does not justify claiming that removing EmpDepartment proves fairness, that removing EmpJobRole is automatically the best final policy, or that subgroup gaps prove causal discrimination.

## Small-Group Warnings

- EmpDepartment=Data Science: n=20, n_samples < 30
- EducationBackground=Human Resources: n=21, n_samples < 30
