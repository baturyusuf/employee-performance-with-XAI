# HR XAI Model Card

## Model Name
XGBoost HR performance decision-support model using `no_salary_hike_no_attrition_no_department`.

## Intended Use
Research-grade decision support for auditing employee-performance prediction under leakage, fairness, explanation, calibration, and actionability constraints. Human review is required.

## Prohibited Use
This is not an autonomous employee evaluator. It should not be used for hiring, firing, compensation, promotion, disciplinary action, or individual employment decisions without independent validation, legal review, and governance approval.

## Dataset and Target
Public cross-sectional INX employee performance dataset. Target is ordinal `PerformanceRating` with classes 2, 3, and 4. Causal claims are not supported.

## Feature Exclusions and Leakage Policy
Age, Gender, MaritalStatus, EmpLastSalaryHikePercent, Attrition, EmpDepartment, EmpNumber, and PerformanceRating are excluded from the primary candidate input. Full-feature models are leakage-warning upper-bound baselines only.

## Model Family and Evaluation Protocol
XGBoost multiclass classifier with fold-safe one-hot preprocessing. Evidence uses config-backed CV/OOF predictions and final-candidate scripts under `src/experiments/`.

## Performance Summary
Macro-F1: 0.5988; balanced accuracy: 0.6261; QWK: 0.6376; ordinal MAE: 0.1525; severe error rate: 0.0017.

## Calibration Summary
Dashboard calibration method: `sigmoid`. Log loss: 0.4551; Brier: 0.2608; ECE: 0.0638. Probability bands and warnings are recommended.

## Fairness and Proxy-Risk Summary
EmpDepartment macro-F1 gap: 0.2689. Department proxy macro-F1: 0.9724. EmpJobRole remains present and may proxy department. Removing EmpDepartment does not prove fairness.

## SHAP and Explanation Summary
Top-10 grouped SHAP Jaccard: 0.7606; Spearman rank stability: 0.8717. SHAP is attribution, not causality.

## Counterfactual and Actionability Summary
Employee-only validity: 0.0000; employee+manager validity: 0.2500; organization-allowed validity: 0.2500. Counterfactuals are intervention hypotheses and may require manager or organisation action.

## Known Limitations
Public cross-sectional data, no external validation, possible organisational proxy effects, imperfect probability calibration, sparse class-4 support, and no causal identification.

## Ethical and Governance Warnings
Decision support only; no autonomous evaluation; no causal SHAP claims; no proof of fairness; proxy risk remains; external validation required before deployment.

## Artifact Paths
- `reports/model_selection/final_candidate_dashboard.csv`
- `reports/model_selection/final_recommendation.md`
- `reports/fairness/feature_set_sensitivity/bootstrap_disparity_ci.csv`
- `reports/calibration/final_candidates/calibration_summary.csv`
- `reports/xai/final_candidates/shap_stability_summary.csv`
- `reports/counterfactuals/final_candidates/actionability_summary.csv`
