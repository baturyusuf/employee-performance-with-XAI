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

## External Validation and Robustness Update
External evidence has been added under `reports/external_validation/`.

- HRDataset_v14 supports independent replication on a directly mappable external performance target (`PerformanceScore` mapped to 2/3/4).
- IBM HR Analytics supports schema-compatible robustness only because `PerformanceRating` is restricted to classes 3 and 4 in the audited data.
- IBM Attrition and Employee Turnover support related HR task-transfer robustness, not performance external validation.
- INX-to-HRDataset cross-dataset validation is reported as infeasible/too limited because only three department-free safe common features overlap.
- Real OpenAI governed explanation and OpenAI Agents SDK audits were run on 5 representative HRDataset_v14 cases, 5 IBM performance robustness cases, and 5 Employee Turnover task-transfer cases.

## Expanded LLM-Agent Evaluation Update
The config-driven expanded batch now includes a real OpenAI 40-case pilot followed by a real OpenAI 80-case final run. The final expanded run covers 40 INX primary cases and 40 HRDataset_v14 independent-replication cases using structured evidence, governed OpenAI explanations, deterministic faithfulness checks, chatbot guardrail checks, and deterministic batch governance agents. IBM performance, IBM attrition, and Employee Turnover LLM regeneration are second-stage robustness tasks and must not be presented as direct employee-performance validation.

## Known Limitations
Public cross-sectional data, public-mirror external dataset provenance requiring independent verification before publication, possible organisational proxy effects, imperfect probability calibration, sparse class-4 support, restricted IBM target space, limited cross-dataset feature overlap, and no causal identification.

## Ethical and Governance Warnings
Decision support only; no autonomous evaluation; no causal SHAP claims; no proof of fairness; proxy risk remains; external validation required before deployment.

## Artifact Paths
- `reports/model_selection/final_candidate_dashboard.csv`
- `reports/model_selection/final_recommendation.md`
- `reports/external_validation/external_validation_summary.md`
- `reports/manuscript_assets/external_validation_tables.md`
- `reports/governance_reports/external_validation_governance_summary.md`
- `reports/fairness/feature_set_sensitivity/bootstrap_disparity_ci.csv`
- `reports/calibration/final_candidates/calibration_summary.csv`
- `reports/xai/final_candidates/shap_stability_summary.csv`
- `reports/counterfactuals/final_candidates/actionability_summary.csv`
- `reports/llm_explanations/eval_case_manifest.csv`
- `reports/llm_explanations/governed_explanations.jsonl`
- `reports/llm_explanations/faithfulness_eval.csv`
- `reports/llm_explanations/llm_agent_eval_summary.md`
- `reports/agent_audits/agent_audit_results.csv`
- `reports/chatbot_eval/guardrail_evaluation_summary.md`
- `reports/governance_reports/gxair_component_dashboard.csv`
- `reports/governance_reports/final_governance_readiness_report.md`
