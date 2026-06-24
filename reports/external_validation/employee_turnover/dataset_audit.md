# Dataset Audit: Human Resources Analytics / Employee Turnover

Dataset name: `employee_turnover`
Recommended role: HR task-transfer robustness
Task type: `binary_turnover`
Source URL: `https://raw.githubusercontent.com/ucg8j/kaggle_HR/master/HR_comma_sep.csv`
Source note: Public mirror of the Human Resources Analytics employee turnover dataset commonly circulated from Kaggle.

## Shape

- Rows: 14999
- Raw columns: 10
- Canonical columns: 12

## Target

- Raw target column: `left`
- Canonical target column: `AttritionTarget`
- Final target mapping: `{"identity_numeric_mapping": true}`

## Target Distribution

- 0: 11428 (0.7619)
- 1: 3571 (0.2381)

## Feature Mapping Highlights

- `average_montly_hours` -> `AverageMonthlyHours` (proxy_risk)
- `last_evaluation` -> `LastEvaluation` (proxy_risk)
- `left` -> `left` (target;leakage_risk)
- `promotion_last_5years` -> `PromotionLast5Years` (proxy_risk)
- `salary` -> `SalaryBand` (proxy_risk)
- `sales` -> `EmpDepartment` (proxy_risk)

## Leakage-Risk Columns

`AttritionTarget`, `left`

## Sensitive / Audit-Only Columns

None detected from mapping.

## Proxy-Risk Columns

`AverageMonthlyHours`, `EmpDepartment`, `LastEvaluation`, `PromotionLast5Years`, `SalaryBand`

## Available Audit Attributes

`EmpDepartment`, `SalaryBand`, `AverageMonthlyHours`, `LastEvaluation`, `PromotionLast5Years`

## Claim Boundaries

- This audit is schema and target evidence, not proof of fairness or causal validity.
- SHAP explanations in downstream reports must be attribution-only.
- Counterfactuals in downstream reports must be framed as model scenarios, not employee prescriptions.
- This project remains research-grade decision support only and must not make autonomous HR decisions.
