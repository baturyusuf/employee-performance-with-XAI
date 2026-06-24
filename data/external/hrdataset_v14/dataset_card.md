# HRDataset_v14 Dataset Card

## Role

Primary independent replication and direct external performance-validation dataset because it includes the mappable `PerformanceScore` target.

## Source

Raw CSV path: `data/external/hrdataset_v14/raw.csv`

Retrieval URL: `https://raw.githubusercontent.com/pouyasattari/HR-Dataset-Analysis/main/HRDataset_v14.csv`

This file is treated as a public mirror of HRDataset_v14 / Human Resources Data Set. Reports must state the mirror provenance and avoid stronger source-authenticity claims than the repository evidence supports.

## Target

`PerformanceScore` is mapped to canonical `PerformanceRating`:

- `PIP` -> 2
- `Needs Improvement` -> 2
- `Fully Meets` -> 3
- `Exceeds` -> 4
- `Exceptional` -> 4 if present

## Modeling Role

Use for independent external replication of the leakage-safe HR XAI governance pipeline. Do not use termination status, employment status, termination dates/reasons, IDs, names, or target-adjacent review identifiers as predictors.

## Claim Boundaries

The dataset supports external performance-target replication, not causal validation, fairness proof, or deployment readiness. SHAP remains attribution only. Counterfactuals are model scenarios, not employee prescriptions.
