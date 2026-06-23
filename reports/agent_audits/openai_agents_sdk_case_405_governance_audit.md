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
- Summary: Leakage-risk features are excluded from the final candidate. The deterministic audit reports leakage_sensitivity_index 0.3399594430292904 and no missing required exclusions. Full-feature results are diagnostic only and not deployable.
- Required warnings:
  - Full-feature models are leakage-warning upper-bound baselines only, not deployable models.

### FairnessProxyAuditAgent
- Status: pass_with_warnings
- Risk level: high
- Summary: The deterministic audit reports a high-risk fairness/proxy governance result with an EmpDepartment macro-F1 gap of 0.2688590531332467 and proxy warnings present. Low-support subgroup warnings were also flagged for EmpDepartment/Data Science (n=20) and EducationBackground/Human Resources (n=21). The model must not be described as fair or unbiased. This is research-grade decision support only, with human review required.
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
  - Probability estimates should be interpreted as approximate confidence, not objective certainty.

### CalibrationAuditAgent
- Status: pass_with_warnings
- Risk level: medium
- Summary: Calibration audit passed with warnings. The evidence reports ECE=0.0638464082070905, Brier score=0.2608022278073826, and log loss=0.45509175349537. Probabilities should be treated as approximate confidence bands rather than calibrated truth, and human review remains required.
- Required warnings:
  - Probability estimates should be interpreted as approximate confidence, not objective certainty.
  - Prediction requires human review.
  - This model is decision support only and is not for autonomous HR decisions.

### ShapStabilityAuditAgent
- Status: pass_with_warnings
- Risk level: low
- Summary: Grouped SHAP stability is acceptable for governance review: Spearman correlation is 0.8717293233082707 and top-10 Jaccard is 0.7606060606060606. The evidence supports using SHAP only as attribution, not causality, and only discussing stable grouped features. Human review remains required under the model card and governance policy.
- Required warnings:
  - SHAP is attribution, not causality.
  - Discuss only stable global feature groups.

### CounterfactualActionabilityAgent
- Status: pass_with_warnings
- Risk level: high
- Summary: Counterfactual actionability audit indicates the counterfactual is not employee-actionable. The deterministic audit reports employee-only counterfactual validity is zero or unavailable, with no changed features, so no recourse recommendation can be made from this evidence. Counterfactuals must be treated as model-level scenarios, not employee prescriptions.
- Required warnings:
  - Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
  - Do not say the employee should change a feature.

### ExplanationComplianceAgent
- Status: pass
- Risk level: low
- Summary: Compliance audit passed with score 100. The explanation is faithful to the provided evidence, with no forbidden or unsupported claims detected and no missing warnings flagged by the deterministic audit tool. The case predicts class 2 for case 405, but the evidence requires human review and states the model is research-only decision support, not for autonomous HR decisions. SHAP is attribution only, calibration is approximate, counterfactuals are not employee-actionable, and leakage-safe evaluation must be used rather than full-feature baselines.
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
