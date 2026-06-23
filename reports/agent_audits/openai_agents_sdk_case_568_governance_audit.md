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
- Summary: The deterministic audit reports a high-risk fairness/proxy governance result. EmpDepartment macro-F1 gap is 0.2688590531332467, with low-support warnings for EmpDepartment/Data Science (n=20) and EducationBackground/Human Resources (n=21). Proxy-risk warnings are present, including that EmpJobRole may proxy EmpDepartment and that removing EmpDepartment does not prove fairness. The model is research-only decision support and must not be described as fair or unbiased. Human review is required; counterfactuals are not employee prescriptions; SHAP is attribution, not causality; and external validation is required.
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
- Summary: Calibration governance audit passed with warnings. The evidence reports ECE=0.0638464082070905, Brier score=0.2608022278073826, and log loss=0.45509175349537. Probabilities should be treated as approximate confidence bands rather than calibrated truth. Human review remains required; this is research-grade decision support only.
- Required warnings:
  - Probability estimates should be interpreted as approximate confidence, not objective certainty.
  - Prediction requires human review.
  - This model is decision support only and is not for autonomous HR decisions.

### ShapStabilityAuditAgent
- Status: pass_with_warnings
- Risk level: low
- Summary: Grouped SHAP stability audit passed with warnings. The deterministic audit reports Spearman correlation 0.8717293233082707 and top-10 Jaccard 0.7606060606060606, indicating stable grouped feature rankings. SHAP outputs should be treated as attribution only, not causality, and discussion should be limited to stable global feature groups. The evidence also indicates this is research-grade decision support, not an autonomous HR decision system.
- Required warnings:
  - SHAP is attribution, not causality.
  - Discuss only stable global feature groups.
  - Prediction requires human review.
  - This model is decision support only and is not for autonomous HR decisions.
  - Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
  - Removing EmpDepartment does not prove fairness.
  - Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
  - External validation is required before deployment.

### CounterfactualActionabilityAgent
- Status: pass_with_warnings
- Risk level: high
- Summary: The deterministic audit indicates the counterfactual is not employee-actionable. The employee-only counterfactual mode has zero or unavailable validity, no features changed, and no probability gain or proximity cost is available. The result should be treated as a model scenario only, not as employee recourse or instruction.
- Required warnings:
  - Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
  - Do not say the employee should change a feature.
  - This model is decision support only and is not for autonomous HR decisions.
  - Prediction requires human review.

### ExplanationComplianceAgent
- Status: pass
- Risk level: low
- Summary: The explanation compliance audit passed with score 100. The explanation is faithful to the provided evidence, with no forbidden claims, no unsupported claims, and no missing warnings detected by the deterministic audit tool. The evidence supports a leakage-safe XGBoost prediction of class 4 for case 568, with SHAP attributions presented as non-causal explanation signals and human review required.

## Governance Constraint
The LLM agents may interpret evidence but must not invent metrics, causal claims, fairness guarantees, or HR decisions.
