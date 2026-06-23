# Multi-Agent Governance Audit

## Architecture
XGBoost predicts -> XAI explains -> audit modules evaluate reliability -> deterministic LLM-style layer interprets structured evidence -> agents audit the explanation -> chatbot exposes governed answers.

## Agent Roles
- LeakageAuditAgent: checks leakage-risk feature exclusions and LSI context.
- FairnessProxyAuditAgent: checks subgroup gaps, low support, and proxy warnings.
- CalibrationAuditAgent: checks probability-quality risk.
- ShapStabilityAuditAgent: checks explanation stability and non-causal warning.
- CounterfactualActionabilityAgent: separates technical validity from practical actionability.
- ExplanationComplianceAgent: checks forbidden claims, unsupported claims, and missing warnings.
- SupervisorGovernanceAgent: aggregates readiness status and prohibited uses.

## Final Readiness Status
`research_only`

## Major Warnings
- Counterfactuals are model-level scenarios, not employee instructions.
- Department removal reduces direct use but does not eliminate proxy risk.
- Do not say the employee should change a feature.
- EmpJobRole may proxy EmpDepartment.
- Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- Removing EmpDepartment does not prove fairness.
- Subgroup gaps require further investigation; they do not prove discrimination.

## Agent Findings
### LeakageAuditAgent
- Status: pass
- Risk level: high
- Summary: Leakage-risk features are excluded from the final candidate; full-feature results remain diagnostic only.

### FairnessProxyAuditAgent
- Status: pass_with_warnings
- Risk level: high
- Summary: EmpDepartment macro-F1 gap is 0.2688590531332467; proxy warnings are present. The model must not be described as fair or unbiased.

### CalibrationAuditAgent
- Status: pass_with_warnings
- Risk level: medium
- Summary: Probability quality is approximate: ECE=0.0638464082070905, Brier=0.2608022278073826. Use probability bands and calibration warnings.

### ShapStabilityAuditAgent
- Status: pass_with_warnings
- Risk level: low
- Summary: Grouped SHAP stability: top-10 Jaccard=0.7606060606060606, Spearman=0.8717293233082707. SHAP is attribution, not causality.

### CounterfactualActionabilityAgent
- Status: pass_with_warnings
- Risk level: high
- Summary: Employee-only counterfactual validity is zero or unavailable; recourse is not employee-actionable.

### ExplanationComplianceAgent
- Status: pass
- Risk level: low
- Summary: Explanation compliance score=100; forbidden=0, unsupported=0, missing_warnings=0.

## Limitations
- The agents audit structured evidence; they do not retrain or validate the predictive model externally.
- Findings are research governance diagnostics, not legal determinations.
- The system must not be used for autonomous HR decisions.
