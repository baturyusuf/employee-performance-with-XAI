# LLM-Assisted Multi-Agent Governance Audit

## Architecture
XGBoost predicts -> XAI/audit tools create evidence -> deterministic agents audit evidence -> LLM agents synthesize governed findings -> supervisor aggregates readiness.

## Runtime
- Provider/runtime: `openai_agents_sdk`
- Model: `gpt-5.4-mini`

## Supervisor Readiness
- Overall status: `research_only`
- Human review required: `True`

## LLM Agent Syntheses
### LeakageAuditAgent
- Status: pass
- Risk level: high
- Summary: Leakage-risk features are excluded from the final candidate; full-feature results remain diagnostic only. The deterministic audit reports leakage_sensitivity_index 0.3399594430292904 and no missing required exclusions. The model card and evidence indicate research-only decision support with human review required, and full-feature models are leakage-warning upper-bound baselines only.
- Required warnings:
  - Full-feature models are leakage-warning upper-bound baselines only, not deployable models.

### FairnessProxyAuditAgent
- Status: pass_with_warnings
- Risk level: high
- Summary: The deterministic audit reports a high-risk fairness/proxy governance result with an EmpDepartment macro-F1 gap of 0.2688590531332467 and proxy warnings present. Low-support warnings were flagged for EmpDepartment (Data Science, n=20) and EducationBackground (Human Resources, n=21). The model must not be described as fair or unbiased. The evidence also states this is research-only decision support, not an autonomous HR decision system, and that human review is required.
- Required warnings:
  - Subgroup gaps require further investigation; they do not prove discrimination.
  - EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
  - Removing EmpDepartment does not prove fairness.
  - SHAP is attribution, not causality.
  - Prediction requires human review.
  - This model is decision support only and is not for autonomous HR decisions.
  - Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
  - Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
  - External validation is required before deployment.

### CalibrationAuditAgent
- Status: pass_with_warnings
- Risk level: medium
- Summary: Calibration audit passed with warnings. The evidence reports ECE=0.0638464082070905, Brier score=0.2608022278073826, and log loss=0.45509175349537. Probabilities should be treated as approximate confidence bands rather than calibrated truth. Use probability bands and retain human review for decisions.
- Required warnings:
  - Probability estimates should be interpreted as approximate confidence, not objective certainty.

### ShapStabilityAuditAgent
- Status: pass_with_warnings
- Risk level: low
- Summary: Grouped SHAP stability is acceptable for governance review: Spearman correlation is 0.8717293233082707 and top-10 Jaccard is 0.7606060606060606. The evidence supports using only stable grouped feature explanations, with SHAP treated as attribution rather than causality. This is research-grade decision support only and requires human review; it is not suitable for autonomous HR decisions.
- Required warnings:
  - SHAP is attribution, not causality.
  - Discuss only stable global feature groups.

### CounterfactualActionabilityAgent
- Status: pass_with_warnings
- Risk level: high
- Summary: The deterministic audit indicates the counterfactual is not employee-actionable. In employee-only mode, validity is 0.0 and no features changed, so there is no supported recourse path for the employee. The counterfactual should be treated as a model scenario only, not as an employee prescription. Governance context also requires human review, and the model card prohibits autonomous HR decisions.
- Required warnings:
  - Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
  - Do not say the employee should change a feature.
  - Prediction requires human review.
  - This model is decision support only and is not for autonomous HR decisions.

### ExplanationComplianceAgent
- Status: pass
- Risk level: low
- Summary: Explanation compliance audit passed with score 100. The provided explanation is faithful to the evidence JSON, with no forbidden claims, no unsupported claims, and no missing warnings detected by the deterministic audit tool. The explanation states that case 662 is predicted as class 3 with probability 0.984155535697937, while the recorded true class is 2. It includes SHAP attribution signals only, notes that SHAP is not causal, flags counterfactuals as not employee-actionable, reports calibration, fairness, proxy-risk, and leakage warnings, and indicates that human review is required before use.
- Required warnings:
  - SHAP is attribution, not causality.
  - Prediction requires human review.
  - This model is decision support only and is not for autonomous HR decisions.
  - Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
  - Removing EmpDepartment does not prove fairness.
  - Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
  - External validation is required before deployment.

## Governance Constraint
The LLM agents may interpret evidence but must not invent metrics, causal claims, fairness guarantees, or HR decisions.
