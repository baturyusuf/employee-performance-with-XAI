# LLM-Assisted Multi-Agent Governance Audit

## Architecture
XGBoost predicts -> XAI/audit tools create evidence -> deterministic agents audit evidence -> LLM agents synthesize governed findings -> supervisor aggregates readiness.

## Runtime
- Provider/runtime: `offline`
- Model: `gpt-5.4-mini`

## Supervisor Readiness
- Overall status: `research_only`
- Human review required: `True`

## LLM Agent Syntheses
### LeakageAuditAgent
- Status: pass
- Risk level: high
- Summary: Leakage-risk features are excluded from the final candidate; full-feature results remain diagnostic only.
- Required warnings:
  - Full-feature models are leakage-warning upper-bound baselines only, not deployable models.

### FairnessProxyAuditAgent
- Status: pass_with_warnings
- Risk level: high
- Summary: EmpDepartment macro-F1 gap is 0.2688590531332467; proxy warnings are present. The model must not be described as fair or unbiased.
- Required warnings:
  - Subgroup gaps require further investigation; they do not prove discrimination.
  - Department removal reduces direct use but does not eliminate proxy risk.
  - EmpJobRole may proxy EmpDepartment.
  - Removing EmpDepartment does not prove fairness.

### CalibrationAuditAgent
- Status: pass_with_warnings
- Risk level: medium
- Summary: Probability quality is approximate: ECE=0.0638464082070905, Brier=0.2608022278073826. Use probability bands and calibration warnings.
- Required warnings:
  - Probability estimates are approximate and may be imperfectly calibrated.

### ShapStabilityAuditAgent
- Status: pass_with_warnings
- Risk level: low
- Summary: Grouped SHAP stability: top-10 Jaccard=0.7606060606060606, Spearman=0.8717293233082707. SHAP is attribution, not causality.
- Required warnings:
  - SHAP is attribution, not causality.
  - Discuss only stable global feature groups.

### CounterfactualActionabilityAgent
- Status: pass_with_warnings
- Risk level: high
- Summary: Employee-only counterfactual validity is zero or unavailable; recourse is not employee-actionable.
- Required warnings:
  - Counterfactuals are model-level scenarios, not employee instructions.
  - Do not say the employee should change a feature.

### ExplanationComplianceAgent
- Status: pass
- Risk level: low
- Summary: Explanation compliance score=100; forbidden=0, unsupported=0, missing_warnings=0.

## Governance Constraint
The LLM agents may interpret evidence but must not invent metrics, causal claims, fairness guarantees, or HR decisions.
