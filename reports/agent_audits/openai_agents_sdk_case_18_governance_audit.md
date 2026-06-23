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
- Summary: Leakage-risk features are excluded from the final candidate. The deterministic audit reports leakage_sensitivity_index = 0.3399594430292904 and no missing required exclusions. Full-feature results are diagnostic only and not deployable. The evidence also indicates the final candidate is leakage-safe under the stated feature policy, while the full-feature model remains a leakage-warning upper-bound baseline.
- Required warnings:
  - Full-feature models are leakage-warning upper-bound baselines only, not deployable models.

### FairnessProxyAuditAgent
- Status: pass_with_warnings
- Risk level: high
- Summary: The deterministic audit reports a high-risk fairness/proxy governance result with an EmpDepartment macro-F1 gap of 0.2688590531332467 and proxy warnings present. Low-support subgroup warnings were also reported for EmpDepartment/Data Science (n=20) and EducationBackground/Human Resources (n=21). The model must not be described as fair or unbiased. It is research-only decision support with strong warnings, not an autonomous HR decision system.
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
- Summary: Calibration audit passed with warnings. The evidence reports ECE=0.0638464082070905, Brier score=0.2608022278073826, and log loss=0.45509175349537. Probabilities should be treated as approximate confidence bands rather than calibrated truth. Human review remains required.
- Required warnings:
  - Probability estimates should be interpreted as approximate confidence, not objective certainty.

### ShapStabilityAuditAgent
- Status: pass_with_warnings
- Risk level: low
- Summary: Grouped SHAP stability audit passed with warnings. Deterministic stability metrics are Spearman 0.8717293233082707 and top-10 Jaccard 0.7606060606060606, indicating reasonably stable ranked explanations. SHAP outputs remain attribution only and should be discussed only as stable global feature groups, not as causal effects or employee prescriptions.
- Required warnings:
  - SHAP is attribution, not causality.
  - Discuss only stable global feature groups.
  - Prediction requires human review.
  - This model is decision support only and is not for autonomous HR decisions.
  - Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
  - External validation is required before deployment.

### CounterfactualActionabilityAgent
- Status: pass_with_warnings
- Risk level: high
- Summary: The deterministic counterfactual audit returned not_employee_actionable with employee_only validity 0.0 and no changed features. The counterfactual is therefore not employee-actionable, and it must be treated as a model scenario rather than employee-prescriptive recourse. The evidence also includes governance warnings that human review is required and autonomous HR decisions are prohibited.
- Required warnings:
  - Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
  - Do not say the employee should change a feature.
  - Prediction requires human review.
  - This model is decision support only and is not for autonomous HR decisions.

### ExplanationComplianceAgent
- Status: pass
- Risk level: low
- Summary: Compliance audit passed for case 18. The explanation is faithful to the provided evidence: it reports the predicted class 3, true class 4, and uses SHAP as attribution only, not causality. It also preserves the evidence-backed cautions on calibration, fairness, proxy risk, counterfactual non-actionability, leakage, human review, and non-autonomous HR use. No unsupported claims were detected by the deterministic audit.
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
