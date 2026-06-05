# Job-Role Proxy Sensitivity Comparison

## Summary

Removing `EmpJobRole` sharply reduces department reconstructability, but it also reduces employee-performance predictive performance.

## Performance comparison

- With `EmpJobRole`: best model `xgboost`, macro-F1=0.5987, QWK=0.6380.
- Without `EmpJobRole`: best model `lightgbm`, macro-F1=0.5612, QWK=0.5430.
- Macro-F1 drop from removing job role: 0.0375.
- QWK drop from removing job role: 0.0950.

## Proxy-risk comparison

- Department proxy macro-F1 with `EmpJobRole`: 0.9724.
- Department proxy macro-F1 without `EmpJobRole`: 0.2412.
- Absolute proxy macro-F1 reduction: 0.7312.
- Relative proxy macro-F1 reduction: 75.2%.

## Interpretation

`EmpJobRole` is the dominant department proxy. Removing it makes `EmpDepartment` much harder to reconstruct, but the performance cost is non-trivial. This supports keeping the job-role-free model as a strict fairness/proxy sensitivity baseline, not automatically as the final model.

## Decision implication

Do not claim that removing `EmpDepartment` alone removes organisational proxy risk. If `EmpJobRole` remains in a final model, reason codes, model cards, and G-XAIR scoring should include explicit organisational proxy warnings.
