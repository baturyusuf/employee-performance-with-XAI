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
- Summary: Leakage-risk features are excluded from the final candidate. The deterministic audit reports leakage_sensitivity_index 0.3399594430292904 with no missing required exclusions. Full-feature results are diagnostic only and not deployable.
- Required warnings:
  - Full-feature models are leakage-warning upper-bound baselines only, not deployable models.

### FairnessProxyAuditAgent
- Status: pass_with_warnings
- Risk level: high
- Summary: Fairness/proxy audit indicates a high-risk result with an EmpDepartment macro-F1 gap of 0.2688590531332467 and proxy warnings present. Low-support subgroup warnings were reported for EmpDepartment=Data Science (n=20) and EducationBackground=Human Resources (n=21). The model must not be described as fair or unbiased. Deployment status remains research-only decision support with strong warnings.
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
- Summary: Calibration audit indicates approximate probability quality rather than calibrated truth. Reported metrics are ECE=0.0638464082070905, Brier=0.2608022278073826, and log loss=0.45509175349537. Probabilities should be treated as approximate confidence bands and used with human review.
- Required warnings:
  - Probability estimates should be interpreted as approximate confidence, not objective certainty.

### ShapStabilityAuditAgent
- Status: pass_with_warnings
- Risk level: low
- Summary: Grouped SHAP stability audit passed with warnings. The deterministic audit reports Spearman stability of 0.8717293233082707 and top-10 Jaccard stability of 0.7606060606060606. The evidence also states that SHAP is attribution, not causality, and that only stable grouped features should be discussed. This case is research-only decision support and requires human review; it is not for autonomous HR decisions.
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
- Summary: The counterfactual audit indicates the case is not employee-actionable. The counterfactual mode is employee_only, but validity is 0.0 and no features changed, so no actionable employee recourse can be supported. The result should be treated as a model scenario only, not as an employee prescription. Governance context remains research-only decision support with human review required.
- Required warnings:
  - Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
  - Do not say the employee should change a feature.
  - Prediction requires human review.
  - This model is decision support only and is not for autonomous HR decisions.

### ExplanationComplianceAgent
- Status: pass
- Risk level: low
- Summary: Explanation compliance audit passed with score 100. The explanation is faithful to the provided evidence, with no forbidden claims, no unsupported claims, and no missing warnings detected by the deterministic audit tool. The case predicts class 2 for case 528 with probability 0.9836298763239094, but this remains research-only decision support and requires human review. SHAP attributions are presented as attribution only, not causality. Counterfactuals are not employee-actionable. Calibration is approximate, and fairness/proxy warnings remain applicable from the evidence.
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
