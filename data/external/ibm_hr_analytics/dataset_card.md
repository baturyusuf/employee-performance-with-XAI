# IBM HR Analytics Dataset Card

## Role

Schema-compatible robustness dataset. It should not be described as direct 3-class external performance validation unless the target audit supports that claim.

## Source

Raw CSV path: `data/external/ibm_hr_analytics/raw.csv`

Retrieval URL: `https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/master/WA_Fn-UseC_-HR-Employee-Attrition.csv`

This file is treated as a public mirror of the IBM HR Analytics Employee Attrition and Performance Kaggle dataset.

## Target

Primary robustness target: `PerformanceRating`.

Expected audit caveat: this dataset commonly contains only classes 3 and 4, so it supports restricted target-space robustness rather than direct 2/3/4 external validation.

Optional task-transfer target: `Attrition`, mapped `No` -> 0 and `Yes` -> 1.

## Claim Boundaries

Use this dataset for schema-compatible robustness and optional related-task transfer. Exclude `PercentSalaryHike` and `Attrition` from performance prediction. Treat income/salary variables, promotion history, job level, department, and job role as proxy-risk fields.
