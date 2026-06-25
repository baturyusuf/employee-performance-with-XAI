# Readiness Explanation: not_ready

The final evidence package does not support a deployment-ready claim. The framework remains research-grade decision support only.

## Primary Blockers

- Proxy Risk Penalty: fail (high). Low score means high proxy risk; removing Department does not prove fairness.
- Counterfactual Actionability: fail (high). Counterfactuals are model scenarios, not employee prescriptions.

## Interpretation

The real 80-case OpenAI LLM-agent evaluation supports technical evidence-interpretation quality for the stated INX and HRDataset_v14 scope. It does not remove the underlying ML governance blockers. Proxy risk remains high because department can still be reconstructed from remaining variables, and counterfactual actionability remains weak because technically valid counterfactuals often depend on manager or organisation-controlled changes rather than employee-actionable changes.

## Claim Boundary

- No autonomous HR decision capability.
- No fairness guarantee.
- No causal SHAP or counterfactual claim.
- No deployment readiness without independent data provenance, human validation, legal review, and organisation-specific governance.

## Stub / Dry-Run Exclusion

Stub/dry-run outputs generated with offline_stub_llm or run_mode=dry_run are reproducibility and pipeline-validation artifacts only. They are excluded from the manuscript-grade evidence package and must not be cited as real LLM evidence.
