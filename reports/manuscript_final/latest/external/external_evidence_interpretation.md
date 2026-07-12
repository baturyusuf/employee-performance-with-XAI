# Canonical External Evidence Interpretation

Run ID: `manuscript_final_20260712T181754Z_c664ef152ff3`  
Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`

## Separated evidence roles

- HRDataset_v14 is independent external performance-target replication using independently trained, dataset-specific models.
- IBM PerformanceRating is restricted-target robustness because only classes 3 and 4 are observed.
- IBM attrition and Employee Turnover are related binary task-transfer evidence; they supply no validation evidence for the employee-performance model.
- These task families are reported in separate tables and must not be ranked as directly equivalent outcomes.

## Locked-model transport gate

Status: `infeasible_or_too_limited`. Common safe features: 3 (EmpJobRole, EmpJobSatisfaction, ExperienceYearsAtThisCompany).
No locked INX model was transported. The overlap result is a feasibility finding, not a transport performance estimate.

## Claim limits

- Research-grade decision support only; no autonomous hiring, firing, promotion, ranking, or discipline decisions.
- Target mappings and class support are recorded beside each task and constrain interpretation.
- Binary-task ordinal metrics are N/A, never zero.
- Restricted-target ordinal-distance metrics are N/A and non-comparable.
- Removing group variables does not establish fairness, and model attribution is not causality.
