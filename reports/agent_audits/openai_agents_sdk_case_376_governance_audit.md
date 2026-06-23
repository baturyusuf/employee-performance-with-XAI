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
- Summary: Leakage-risk features are excluded from the final candidate. The deterministic audit reports leakage_sensitivity_index 0.3399594430292904 and no missing required exclusions. Full-feature results are diagnostic only and not deployable. The final candidate is marked leakage-safe, but the overall governance context remains research-only decision support with strong warnings and human review required.
- Required warnings:
  - Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
  - SHAP is attribution, not causality.
  - Prediction requires human review.
  - This model is decision support only and is not for autonomous HR decisions.
  - Removing EmpDepartment does not prove fairness.
  - Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
  - External validation is required before deployment.

### FairnessProxyAuditAgent
- Status: pass_with_warnings
- Risk level: high
- Summary: The deterministic audit reports a high-risk fairness/proxy governance result. EmpDepartment macro-F1 gap is 0.2688590531332467, with low-support warnings for EmpDepartment/Data Science (n=20) and EducationBackground/Human Resources (n=21). Proxy-risk warnings are present, including that EmpJobRole may proxy EmpDepartment and that removing EmpDepartment does not prove fairness. The model is research-only decision support, not an autonomous HR decision system. Calibration is approximate, counterfactuals are not employee prescriptions, and SHAP is attribution not causality.
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
- Summary: Calibration audit indicates approximate probability quality rather than calibrated truth. Reported metrics are ECE=0.0638464082070905, Brier score=0.2608022278073826, and log loss=0.45509175349537. Probabilities should be treated as approximate confidence bands and used with human review.
- Required warnings:
  - Probability estimates should be interpreted as approximate confidence, not objective certainty.

### ShapStabilityAuditAgent
- Status: pass_with_warnings
- Risk level: low
- Summary: Grouped SHAP stability audit passed with warnings. Deterministic stability metrics are Spearman 0.8717293233082707 and top-10 Jaccard 0.7606060606060606. The evidence supports using SHAP only as attribution, not causality, and only for stable grouped features. Human review remains required under the governance context.
- Required warnings:
  - SHAP is attribution, not causality.
  - Discuss only stable global feature groups.

### CounterfactualActionabilityAgent
- Status: pass_with_warnings
- Risk level: high
- Summary: The counterfactual audit indicates the scenario is not employee-actionable. In employee-only mode, validity is 0.0 and no features changed, so no recourse path is available from the evidence. The result should be treated as a model scenario only, not as an employee prescription. Human review remains required.
- Required warnings:
  - Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
  - Do not say the employee should change a feature.
  - Prediction requires human review.
  - This model is decision support only and is not for autonomous HR decisions.

### ExplanationComplianceAgent
- Status: pass
- Risk level: low
- Summary: Compliance audit passed for case 376. The explanation is faithful to the provided evidence: the model predicted class 3 with probability 0.9948591941166247 and true class 3; calibration is described as approximate confidence rather than calibrated truth; SHAP is presented as attribution, not causality; counterfactuals are explicitly marked not employee-actionable; fairness and proxy-risk warnings are included; leakage-safe versus full-feature performance is distinguished; and human review is required before use. No forbidden or unsupported claims were detected.
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
