# LLM-Agent Extension Manuscript Support

## Contribution Statement
This extension proposes an LLM-assisted multi-agent XAI governance framework for employee performance prediction. The predictive model remains leakage-safe XGBoost. SHAP, calibration diagnostics, subgroup fairness/proxy analysis, leakage analysis, and counterfactual actionability modules generate structured evidence. OpenAI-backed LLM and agent components interpret this evidence under explicit governance constraints, while a guardrailed chatbot exposes the results without allowing autonomous HR decisions or unsupported claims.

## Research Questions
RQ1. How much does apparent employee performance prediction performance depend on leakage-risk variables?

RQ2. Can leakage-safe XAI evidence support a governance-aware model selection process?

RQ3. Can an LLM-assisted multi-agent governance layer generate faithful and compliant explanations from structured XAI evidence?

RQ4. Can the agent system detect and communicate leakage, proxy fairness, calibration, explanation stability, and counterfactual actionability risks?

RQ5. Can a guardrailed chatbot answer model-audit questions while refusing unsafe HR decision requests?

## LLM/Agent Methodology
The LLM receives constrained JSON evidence rather than raw uncontrolled data. Evidence includes prediction metadata, grouped SHAP attribution summaries, leakage status, fairness/proxy diagnostics, calibration metrics, counterfactual actionability summaries, and governance warnings. The real implementation uses OpenAI API structured outputs and OpenAI Agents SDK specialist agents. The deterministic offline stub is retained for tests and reproducibility.

## Real OpenAI Small-Batch Evaluation
- Cases: 10.
- Faithfulness pass rate: 1.0.
- Unsupported claim rate: 0.0.
- Forbidden claim rate: 0.0.
- Agent success rate: 1.0.
- Warning consistency rate: 0.829497.
- Unsafe prompt refusal rate: 1.0.

## Chatbot Guardrail Evaluation

| prompt_type | n | pass_rate |
| --- | --- | --- |
| adversarial | 10 | 1.0 |
| safe_control | 5 | 1.0 |
| unsafe | 7 | 1.0 |

## Limitations
- The LLM layer does not establish causality, fairness, or deployment readiness.
- The system is evaluated on a public cross-sectional dataset and requires external validation.
- Automatic checks are not a substitute for human-subject evaluation.
- Guardrails reduce unsafe behavior on tested prompts but do not prove robustness to all adversarial prompts.

## Suggested Tables and Figures
- Table A. Agent roles, inputs, outputs, and governance functions.
- Table B. Required warnings and forbidden claims in governed explanations.
- Table C. Real OpenAI faithfulness and guardrail compliance results.
- Table D. Unsafe/adversarial chatbot prompt refusal evaluation.
- Figure A. LLM-assisted multi-agent XAI governance architecture.
- Figure B. Evidence flow from XGBoost and SHAP to agents and chatbot.
- Figure C. Guardrailed chatbot workflow.
- Figure D. G-XAIR component dashboard including LLM governance compliance.
