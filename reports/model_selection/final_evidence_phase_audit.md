# Final Model Selection Evidence Phase Audit

Generated: 2026-06-05

## Scope
This phase starts from the existing research decision log and config-backed feature policy. It does not restart the repository and does not overwrite earlier evidence. The goal is to build a final evidence package for three XGBoost feature-set candidates:

1. `no_salary_hike_no_attrition`
2. `no_salary_hike_no_attrition_no_department`
3. `no_salary_hike_no_attrition_no_department_no_job_role`

Full-feature models remain leakage-warning upper bounds only and are not final deployable candidates.

## Current Feature Policy
Source: `configs/feature_sets.yaml`

- `Age` is excluded from final model candidates and retained only for audit/sensitivity.
- `EmpLastSalaryHikePercent` and `Attrition` are excluded from final model candidates as leakage-risk/outcome-proximal variables.
- `no_salary_hike_no_attrition` is the main leakage-safe comparison baseline and retains `EmpDepartment`.
- `no_salary_hike_no_attrition_no_department` is the current primary governance-aware candidate and retains `EmpJobRole` with proxy warnings.
- `no_salary_hike_no_attrition_no_department_no_job_role` is the strict fairness/proxy sensitivity baseline.

## Reusable Current Code
- `src/features/feature_sets.py`: config-backed feature-set construction and exclusion checks. Reuse.
- `src/models/evaluate.py`: central performance, ordinal, and calibration metrics. Reuse and extend only if needed.
- `src/experiments/leakage_safe_cv.py`: fold-safe model fitting, XGBoost pipeline, CV metadata, registry writing. Reuse helpers.
- `src/experiments/fairness_sensitivity.py`: valid 10-fold OOF XGBoost predictions and support-filtered group metrics. Reuse for fairness bootstrap and calibration raw-probability evaluation where appropriate.
- `src/experiments/proxy_analysis.py`: current department proxy evidence. Reuse outputs in dashboard.
- `src/utils/experiment_registry.py`: registry and metadata helpers. Reuse.

## Historical or Non-Final Evidence
These outputs are valuable background, but they predate the current final-candidate framing or use older scripts/models. They should be labelled historical unless regenerated:

- `reports/robustness/calibration/*`: historical calibration outputs, not final-candidate XGBoost feature-set evidence.
- `reports/robustness/shap_stability/*`: historical SHAP stability outputs, not final-candidate grouped XGBoost evidence.
- `reports/robustness/counterfactual_actionability/*`: historical actionability outputs, not final-candidate XGBoost feature-set evidence.
- Most pre-existing `reports/xai/*`: older SHAP/local/counterfactual examples; useful as implementation references but not current final claims.
- `reports/leakage/*`: current LSI report is derived from older ablation outputs and should be used as leakage-risk context, not as a fresh rerun.

## Current Config-Backed Evidence
- `reports/leakage_safe_cv_config/`: current config-backed 10-fold CV for the two candidate sets excluding age and sensitive demographics.
- `reports/leakage_safe_cv_job_role_sensitivity/`: current config-backed 10-fold CV for strict job-role-free sensitivity.
- `reports/fairness/proxy_with_job_role/`: current EmpDepartment proxy analysis for department-free candidate retaining `EmpJobRole`.
- `reports/fairness/proxy_no_job_role/`: current EmpDepartment proxy analysis for strict job-role-free sensitivity.
- `reports/fairness/job_role_sensitivity_comparison.csv`: current joined utility/proxy comparison.
- `reports/fairness/feature_set_sensitivity/`: current 10-fold OOF XGBoost subgroup metrics for the three final evidence feature sets.
- `reports/research_log/experiment_registry.csv`: current run registry; duplicate early proxy/fairness entries exist and should be interpreted by timestamp/output path.

## Gaps To Fill In This Phase
- Bootstrap uncertainty for subgroup disparity metrics.
- Final-candidate calibration diagnostics and reliability curves.
- Config-backed grouped SHAP stability for the three XGBoost candidates.
- Final-candidate counterfactual actionability under taxonomy-constrained intervention modes.
- Combined G-XAIR/model-selection dashboard and nuanced recommendation.
- Governance-safe reason-code examples.
- Manuscript-ready model card for the recommended candidate.
- Tests for new utilities and output schema.

## Engineering Decision
Preserve the current repository structure. Add narrow, scriptable final-evidence modules under `src/experiments/` and final artifacts under `reports/`. Do not modify older historical outputs except by adding new clearly named final-candidate outputs.
