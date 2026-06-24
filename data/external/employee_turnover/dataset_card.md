# Employee Turnover Dataset Card

## Role

HR task-transfer robustness only. This dataset does not support employee-performance external validation because the target is turnover/attrition (`left`), not performance rating.

## Source

Raw CSV path: `data/external/employee_turnover/raw.csv`

Retrieval URL: `https://raw.githubusercontent.com/ucg8j/kaggle_HR/master/HR_comma_sep.csv`

This file is treated as a public mirror of the Human Resources Analytics employee turnover dataset.

## Target

`left` is used as canonical `AttritionTarget`, with values 0 and 1 preserved.

## Claim Boundaries

Use only to test whether the governance pipeline transfers to a related HR risk-prediction task. `last_evaluation` is a high-caution evaluation/performance proxy and must be tested with and without that feature. Sensitive attributes are absent or limited, so fairness claims are correspondingly limited.
