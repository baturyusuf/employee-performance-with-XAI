# Open Research Questions

This file tracks decisions that need researcher review before they become final paper claims or final-model settings.

## 2026-06-05 - Feature taxonomy review

### Issue
The first taxonomy marks `Age` as `direct_sensitive` and not allowed for the final model by default. The original brief explicitly listed `Gender` and `MaritalStatus` in `no_sensitive_demographic`, but did not define whether `Age` should be excluded or only audited.

### Why it matters
Including age can improve predictive performance but creates legal, ethical, and interpretability risk in HR decision support. Excluding it may reduce performance but is more conservative.

### Options
1. Exclude `Age` from all final deployable models and keep it only for audit/sensitivity analysis.
2. Allow `Age` in leakage-safe candidates but require fairness and subgroup reporting.
3. Bin `Age` only for fairness audit and exclude raw `Age` from model features.

### Current recommendation
Use option 1 or 3 unless there is a domain-specific justification for including age.

### What I need from the researcher
Confirm whether `Age` should be excluded from final-model candidates or retained as a model feature with explicit audit warnings.

### Status
Resolved on 2026-06-05. Researcher approved the recommendation to exclude `Age` from final model candidates and keep it only for audit/sensitivity analysis.

## 2026-06-05 - Department and job-role proxy treatment

### Issue
The first taxonomy marks `EmpDepartment` as not allowed for final deployment by default and `EmpJobRole` as a possible proxy that remains allowed pending proxy audit.

### Why it matters
Removing department does not remove department-related information if job role reconstructs department. This affects fairness, proxy-risk, and final feature-set choice.

### Options
1. Treat `no_salary_hike_no_attrition` as primary and `no_salary_hike_no_attrition_no_department` as sensitivity only.
2. Treat department-free modeling as the primary final-model path.
3. Keep both as candidates until proxy analysis and fairness gaps are reviewed.

### Current recommendation
Use option 3 until the proxy analysis is implemented and reviewed.

### What I need from the researcher
Confirm whether department-free modeling should become the default final-model candidate if performance remains statistically similar.

### Status
Resolved on 2026-06-05. Researcher approved keeping both department-including and department-free models as candidates until proxy analysis, fairness gaps, and model-selection evidence are reviewed.

## 2026-06-05 - High department proxy reconstruction through job role

### Issue
The department proxy analysis shows that `EmpDepartment` can be reconstructed from the department-free final-candidate feature set with macro-F1 about 0.972. `EmpJobRole` is the dominant proxy feature.

### Why it matters
Removing `EmpDepartment` does not remove organisational group information if `EmpJobRole` almost perfectly recovers department. This affects fairness interpretation, reason-code warnings, and whether a stricter feature set should be evaluated.

### Options
1. Keep `EmpJobRole` in final candidates, but add explicit proxy warnings and fairness/proxy penalties.
2. Add a stricter `no_salary_hike_no_attrition_no_department_no_job_role` sensitivity feature set and compare performance/fairness.
3. Exclude `EmpJobRole` from final deployable candidates by policy.

### Current recommendation
Use option 2 next. Do not remove job role by policy until the performance, fairness, explanation, and actionability costs are measured.

### What I need from the researcher
Confirm whether to run a stricter job-role-free sensitivity experiment before any final model selection.

### Status
Resolved on 2026-06-05. Researcher approved option 2, and the job-role-free sensitivity experiment was run. Result: department proxy macro-F1 dropped from about 0.972 to about 0.241, while best employee-performance macro-F1 dropped from about 0.599 to about 0.561 and QWK dropped from about 0.638 to about 0.543.
