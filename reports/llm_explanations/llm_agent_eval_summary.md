# LLM-Agent Evaluation Summary

Run ID: `llm_agent_eval_openai_final_80_2026-06-24T13:39:28+00:00`
Run mode: `real`
Real LLM used: `True`
LLM provider/model: `openai` / `gpt-5.4-mini`
Prompt version: `governed_evidence_v2`
Agent system version: `deterministic_agents_v2`

## Summary Metrics

- run_id: llm_agent_eval_openai_final_80_2026-06-24T13:39:28+00:00
- run_mode: real
- real_llm_used: True
- n_cases: 80
- n_datasets: 2
- llm_model_used: gpt-5.4-mini
- llm_provider: openai
- prompt_version: governed_evidence_v2
- agent_system_version: deterministic_agents_v2
- faithfulness_pass_rate: 1.0
- mean_faithfulness_score: 100.0
- unsupported_claim_rate: 0.0
- forbidden_claim_rate: 0.0
- missing_warning_rate: 0.0
- parsing_success_rate: 1.0
- chatbot_unsafe_refusal_rate: 1.0
- chatbot_safe_answer_rate: 1.0
- agent_compliance_pass_rate: 1.0

## Per-Dataset Breakdown

| dataset_name | n_cases | faithfulness_pass_rate | mean_faithfulness_score |
| --- | --- | --- | --- |
| hrdataset_v14 | 40 | 1.0 | 100.0 |
| inx_primary | 40 | 1.0 | 100.0 |

## Supervisor Readiness Distribution

| readiness_status | count |
| --- | --- |
| research_only | 50 |
| evidence_missing | 30 |

## Limitations

- This run used the real OpenAI-backed governed explanation path.
- Stub/dry-run outputs from `offline_stub_llm` or `run_mode=dry_run` are not manuscript-grade real LLM evidence and are excluded from the final evidence package.
- Automated LLM, faithfulness, agent, and chatbot checks do not replace human evaluation or legal/governance review.
- Future larger or second-stage real LLM batches still require explicit approval because they incur API cost.
