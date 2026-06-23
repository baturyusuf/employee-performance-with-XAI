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
- Summary: Leakage-risk features are excluded from the final candidate. The deterministic audit reports a leakage_sensitivity_index of 0.3399594430292904 and no missing required exclusions. Full-feature results are diagnostic only and not deployable.
- Required warnings:
  - Full-feature models are leakage-warning upper-bound baselines only, not deployable models.

### FairnessProxyAuditAgent
- Status: pass_with_warnings
- Risk level: high
- Summary: The deterministic audit reports a high-risk fairness/proxy governance result with an EmpDepartment macro-F1 gap of 0.2688590531332467 and proxy warnings present. Low-support warnings were also reported for EmpDepartment/Data Science (n=20) and EducationBackground/Human Resources (n=21). The model must not be described as fair or unbiased. The evidence supports research-only decision support with strong warnings, not autonomous HR use.
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
- Summary: Calibration audit passed with warnings. Probability quality is approximate, with ECE=0.0638464082070905, Brier score=0.2608022278073826, and log loss=0.45509175349537. Probabilities should be treated as approximate confidence bands, not calibrated truth.
- Required warnings:
  - Probability estimates should be interpreted as approximate confidence, not objective certainty.

### ShapStabilityAuditAgent
- Status: pass_with_warnings
- Risk level: low
- Summary: Grouped SHAP stability audit passed with warnings. The deterministic audit reports Spearman stability of 0.8717293233082707 and top-10 Jaccard stability of 0.7606060606060606. The evidence also indicates SHAP should be treated as attribution, not causality, and only stable global feature groups should be discussed. The model card and governance evidence classify this as research-only decision support with human review required.
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
- Summary: The deterministic audit reports that the counterfactual is not employee-actionable. In employee-only mode, validity is zero or unavailable, no features changed, and no probability gain or proximity cost is available. The counterfactual should be treated as a model scenario only, not as employee guidance. Governance evidence also indicates research-only decision support, human review required, and counterfactuals may not be employee-actionable.
- Required warnings:
  - Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
  - Do not say the employee should change a feature.
  - Prediction requires human review.
  - This model is decision support only and is not for autonomous HR decisions.

### ExplanationComplianceAgent
- Status: pass
- Risk level: low
- Summary: The explanation complies with the provided evidence. The model predicts class 3 for case 906 with probability 0.5060936212539673 and uncertainty_flag=true. Calibration evidence says probabilities are approximate confidence bands, not calibrated truth. SHAP attributions identify positive and negative contributors, but SHAP must be treated as attribution, not causality. Fairness evidence includes an EmpDepartment macro F1 gap of 0.2688590531332467 with bootstrap CI [0.1848783532680399, 0.3492426064838457], plus low-support warnings for EmpDepartment/Data Science (n=20) and EducationBackground/Human Resources (n=21). Proxy-risk warnings indicate EmpJobRole may proxy EmpDepartment and that removing EmpDepartment does not prove fairness. Leakage evidence shows the full-feature score is much higher than the leakage-safe score, so full-feature models are upper-bound baselines only. Counterfactual evidence is not employee-actionable, with validity 0.0 and no changed features. Human review is required before use.
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
