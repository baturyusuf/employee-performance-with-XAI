# Final LLM-Agent Research Summary

## Positioning
The LLM is not the predictive model. XGBoost remains the performance predictor, SHAP and audit modules produce structured evidence, and the LLM/agent layer interprets that evidence under governance constraints.

## Evidence Flow
XGBoost prediction -> grouped SHAP and reliability evidence -> structured evidence schema -> governed explanation -> multi-agent audit -> guardrailed chatbot.

## Agent Audit Functions
- Leakage: checks leakage-risk exclusions and labels full-feature baselines as diagnostic only.
- Fairness/proxy: checks subgroup gaps, low support, and JobRole/Department proxy risk.
- Calibration: warns that probabilities are approximate.
- SHAP stability: checks attribution stability and non-causal wording.
- Counterfactual actionability: separates technical validity from employee actionability.
- Explanation compliance: detects unsupported or forbidden explanation claims.

## Chatbot Guardrails
The chatbot refuses hiring, firing, promotion, compensation, disciplinary, autonomous decision, fairness-guarantee, sensitive-attribute justification, and employee-prescription prompts.

## Automatic Evaluation
- Faithfulness pass rate: 1.0
- Unsupported claim rate: 0.0
- Forbidden claim rate: 0.0
- Missing warning rate: 0.0
- Unsafe prompt refusal rate: 1.0

## G-XAIR Readiness
Overall readiness label: `research_only_proxy_risk_high`.

## Limitations
- No human-subject evaluation was performed.
- The offline LLM stub is deterministic and conservative; external LLM behavior must be separately validated.
- The evidence is based on a public cross-sectional dataset and does not support causal claims.
- The system is a research prototype, not an HR decision system.
