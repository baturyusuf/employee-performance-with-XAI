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
- Summary: The deterministic audit reports a high-risk fairness/proxy governance result with an EmpDepartment macro-F1 gap of 0.2688590531332467 and proxy warnings present. Low-support warnings were raised for EmpDepartment/Data Science (n=20) and EducationBackground/Human Resources (n=21). The model is research-only decision support, not an autonomous HR decision system. It must not be described as fair or unbiased. Department removal reduces direct use but does not eliminate proxy risk, and EmpJobRole may proxy EmpDepartment.
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
- Summary: Calibration audit passed with warnings. The evidence reports ECE=0.0638464082070905, Brier score=0.2608022278073826, and log loss=0.45509175349537. Probabilities should be treated as approximate confidence bands rather than calibrated truth, and probability estimates may be imperfectly calibrated.
- Required warnings:
  - Probability estimates should be interpreted as approximate confidence, not objective certainty.

### ShapStabilityAuditAgent
- Status: pass_with_warnings
- Risk level: low
- Summary: Grouped SHAP stability audit passed with warnings. The deterministic audit reports Spearman stability of 0.8717293233082707 and top-10 Jaccard stability of 0.7606060606060606. The evidence also indicates SHAP should be treated as attribution, not causality, and only stable grouped features should be discussed. Human review remains required for any HR use.
- Required warnings:
  - SHAP is attribution, not causality.
  - Discuss only stable global feature groups.

### CounterfactualActionabilityAgent
- Status: pass_with_warnings
- Risk level: high
- Summary: The counterfactual audit indicates the scenario is not employee-actionable. The employee-only counterfactual mode has zero or unavailable validity, no features changed, and no probability gain or proximity cost is provided. The governed interpretation is that this is a model scenario, not an employee prescription, and it requires human review under the stated HR governance constraints.
- Required warnings:
  - Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
  - Do not say the employee should change a feature.
  - Prediction requires human review.
  - This model is decision support only and is not for autonomous HR decisions.

### ExplanationComplianceAgent
- Status: pass
- Risk level: low
- Summary: Explanation compliance audit passed with score 100. No forbidden claims, unsupported claims, or missing warnings were detected by the deterministic audit tool. The explanation remains research-grade decision support only, with SHAP treated as attribution rather than causality, counterfactuals not employee-actionable, and human review required.
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
