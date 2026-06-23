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
- Summary: Leakage audit passed. The deterministic audit reports leakage_sensitivity_index 0.3399594430292904 with no missing required exclusions. Leakage-risk features are excluded from the final candidate, while full-feature results are retained only as diagnostic upper-bound baselines. The governed evidence also states the deployment status is research-only decision support with strong warnings, and the model card prohibits autonomous HR decisions. Prediction, SHAP, fairness, calibration, and counterfactual outputs remain subject to their stated warnings and human review requirements.
- Required warnings:
  - Full-feature models are leakage-warning upper-bound baselines only, not deployable models.

### FairnessProxyAuditAgent
- Status: pass_with_warnings
- Risk level: high
- Summary: Deterministic audit indicates an EmpDepartment macro-F1 gap of 0.2688590531332467 with proxy-risk warnings present. Low-support warnings were raised for EmpDepartment/Data Science (n=20) and EducationBackground/Human Resources (n=21). The model must not be described as fair or unbiased. Deployment status remains research-only decision support with strong warnings; human review is required. Counterfactuals are not employee prescriptions, and SHAP is attribution, not causality.
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
- Summary: Calibration audit passed with warnings. The evidence reports approximate probability quality with ECE=0.0638464082070905, Brier score=0.2608022278073826, and log loss=0.45509175349537. Probabilities should be treated as approximate confidence bands, not calibrated truth.
- Required warnings:
  - Probability estimates should be interpreted as approximate confidence, not objective certainty.

### ShapStabilityAuditAgent
- Status: pass_with_warnings
- Risk level: low
- Summary: Grouped SHAP stability audit passed with warnings. The deterministic audit reports Spearman rank stability of 0.8717293233082707 and top-10 Jaccard stability of 0.7606060606060606. The evidence also states that SHAP is attribution, not causality, and that only stable grouped features should be discussed. This is research-grade HR decision support only and requires human review; it is not for autonomous HR decisions.
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
- Summary: The counterfactual audit indicates the scenario is not employee-actionable. Employee-only counterfactual validity is zero or unavailable, no features changed, and the audit classifies the result as a model scenario rather than an employee prescription. Human review remains required, and the output should not be used to instruct an employee to change any feature.
- Required warnings:
  - Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
  - Do not say the employee should change a feature.

### ExplanationComplianceAgent
- Status: pass
- Risk level: low
- Summary: The explanation is compliant with the provided evidence. The model predicted class 4 for case 176 with probability 0.5255113840103149 and uncertainty_flag=true. The explanation correctly treats this probability as approximate confidence rather than calibrated truth, notes that SHAP values are attribution rather than causality, and states that counterfactuals are not employee-actionable. It also reflects the research-only decision-support status, human review requirement, leakage-safe framing, and fairness/proxy cautions present in the evidence. No unsupported claims were detected.

## Governance Constraint
The LLM agents may interpret evidence but must not invent metrics, causal claims, fairness guarantees, or HR decisions.
