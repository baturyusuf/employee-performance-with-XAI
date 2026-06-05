# Fairness Feature-Set Sensitivity

Model used for controlled comparison: `xgboost`
Minimum group support threshold: 30

Feature sets audited:

- `no_salary_hike_no_attrition`
- `no_salary_hike_no_attrition_no_department`
- `no_salary_hike_no_attrition_no_department_no_job_role`

## Largest Support-Filtered Gaps

- feature_set=no_salary_hike_no_attrition, attribute=BusinessTravelFrequency, metric=precision, class=4.0: max_gap=1.0000, groups=3, min_group=Non-Travel, max_group=Travel_Rarely
- feature_set=no_salary_hike_no_attrition, attribute=EmpDepartment, metric=precision, class=4.0: max_gap=1.0000, groups=2, min_group=Development, max_group=Research & Development
- feature_set=no_salary_hike_no_attrition, attribute=MaritalStatus, metric=precision, class=4.0: max_gap=1.0000, groups=3, min_group=Divorced, max_group=Married
- feature_set=no_salary_hike_no_attrition_no_department, attribute=EducationBackground, metric=precision, class=4.0: max_gap=1.0000, groups=4, min_group=Life Sciences, max_group=Marketing
- feature_set=no_salary_hike_no_attrition_no_department, attribute=EmpDepartment, metric=precision, class=4.0: max_gap=1.0000, groups=4, min_group=Development, max_group=Sales
- feature_set=no_salary_hike_no_attrition, attribute=EmpDepartment, metric=true_positive_rate, class=2.0: max_gap=0.8462, groups=5, min_group=Development, max_group=Finance
- feature_set=no_salary_hike_no_attrition_no_department_no_job_role, attribute=EmpDepartment, metric=precision, class=2.0: max_gap=0.8167, groups=5, min_group=Development, max_group=Finance
- feature_set=no_salary_hike_no_attrition_no_department, attribute=EmpDepartment, metric=precision, class=2.0: max_gap=0.7692, groups=5, min_group=Development, max_group=Finance
- feature_set=no_salary_hike_no_attrition_no_department, attribute=EmpDepartment, metric=true_positive_rate, class=2.0: max_gap=0.7692, groups=5, min_group=Development, max_group=Finance
- feature_set=no_salary_hike_no_attrition, attribute=EmpDepartment, metric=false_positive_rate, class=3.0: max_gap=0.7544, groups=5, min_group=Finance, max_group=Development
- feature_set=no_salary_hike_no_attrition_no_department, attribute=EmpDepartment, metric=false_positive_rate, class=3.0: max_gap=0.7368, groups=5, min_group=Finance, max_group=Development
- feature_set=no_salary_hike_no_attrition, attribute=EmpDepartment, metric=precision, class=2.0: max_gap=0.6667, groups=5, min_group=Development, max_group=Finance

## Interpretation Rules

- These are audit metrics, not proof of discrimination or proof of fairness.
- Gaps below the support threshold are excluded from disparity summaries but retained in small-group warnings.
- Feature-set changes should be interpreted jointly with performance, calibration, proxy risk, SHAP stability, and actionability.
- Removing `EmpDepartment` and `EmpJobRole` can reduce direct/proxy organisational information but may also reduce model utility.
