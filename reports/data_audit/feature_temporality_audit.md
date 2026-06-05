# Feature Temporality Audit

Date: 2026-06-05

Rows audited: 1200

Target distribution: {2: 194, 3: 874, 4: 132}

## Summary

This audit classifies features by temporal and governance risk before model selection. It does not prove causal leakage. It identifies variables that are outcome-proximal, post-evaluation-risk, organisational grouping variables, or plausible proxies that must be tested through ablation, fairness analysis, proxy analysis, and explanation shift.

## High Leakage-Risk Features

### EmpLastSalaryHikePercent

`EmpLastSalaryHikePercent` is high leakage risk because salary increases are often determined after or during performance review cycles. If the model uses this field, performance can look artificially high because the feature may encode the outcome or a manager decision derived from the outcome. Full-feature models using this variable must be treated as upper-bound/leakage-warning baselines, not deployable HR decision-support models.

### Attrition

`Attrition` is high risk unless the prediction time is explicitly before attrition status is known. If attrition is observed after the performance period, it may encode downstream employee or organisational outcomes. The primary leakage-safe feature set excludes it.

## Organisational Group and Proxy Risk

### EmpDepartment

`EmpDepartment` is not necessarily a legally protected attribute by itself, but it encodes organisational group membership. Direct department dependence can create unequal error patterns across departments and can hide process differences between business units. The department-free feature set is therefore required as a fairness-aware sensitivity analysis.

### EmpJobRole

`EmpJobRole` may remain a proxy for department after `EmpDepartment` is removed. If job role predicts department well, a department-free model can still retain indirect department dependence. A proxy analysis should test whether remaining features, especially job role, reconstruct `EmpDepartment`.

## Concurrent Survey and Historical Decision Features

Satisfaction, involvement, overtime, work-life balance, years since promotion, and related features may be measured near the evaluation period or encode prior managerial decisions. They are not automatically removed, but reason codes and counterfactuals must avoid causal language and employee-blaming recommendations.

## Required Follow-Up

- Compare full-feature and leakage-safe feature sets.
- Compute Leakage Sensitivity Index for macro-F1 and QWK.
- Run proxy analysis for `EmpDepartment`.
- Review whether `Age` should be excluded from final model candidates.
- Use the taxonomy for counterfactual actionability constraints.
