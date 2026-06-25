# Final Evidence Manifest

This package binds the final 80-case real OpenAI LLM-agent outputs and supporting governance artifacts for manuscript-support use.

Run ID: `llm_agent_eval_openai_final_80_2026-06-24T13:39:28+00:00`
Run mode: `real`
Real LLM used: `True`
LLM provider/model: `openai` / `gpt-5.4-mini`
Cases/datasets: `80` / `2`

## Manuscript-Grade Scope

- Included final scope: 40 INX primary cases and 40 HRDataset_v14 independent-replication cases.
- The LLM interprets structured ML/XAI/governance evidence; it is not the predictive model.
- IBM performance, IBM attrition, and Employee Turnover LLM regeneration are second-stage robustness work.
- Related-task datasets must not be described as direct employee-performance validation.

## Stub / Dry-Run Exclusion

Stub/dry-run outputs generated with offline_stub_llm or run_mode=dry_run are reproducibility and pipeline-validation artifacts only. They are excluded from the manuscript-grade evidence package and must not be cited as real LLM evidence.

## Readiness

Final readiness label: `not_ready`

The readiness label remains `not_ready` because two high-severity blockers remain:

- Proxy Risk Penalty: fail (high). Low score means high proxy risk; removing Department does not prove fairness.
- Counterfactual Actionability: fail (high). Counterfactuals are model scenarios, not employee prescriptions.

## Included Evidence and Traceability Files

| evidence_id | category | path | file_type | manuscript_grade_status | row_count | claim_role |
| --- | --- | --- | --- | --- | --- | --- |
| final_case_manifest | llm_agent_final_real | reports/llm_explanations/eval_case_manifest.csv | csv | primary_manuscript_grade | 80 | supports final 80-case manuscript-grade technical evidence |
| final_governed_explanations | llm_agent_final_real | reports/llm_explanations/governed_explanations.jsonl | jsonl | primary_manuscript_grade | 80 | supports final 80-case manuscript-grade technical evidence |
| final_faithfulness_eval | llm_agent_final_real | reports/llm_explanations/faithfulness_eval.csv | csv | primary_manuscript_grade | 80 | supports final 80-case manuscript-grade technical evidence |
| final_faithfulness_summary | llm_agent_final_real | reports/llm_explanations/faithfulness_eval_summary.md | markdown | primary_manuscript_grade |  | supports final 80-case manuscript-grade technical evidence |
| final_llm_agent_summary_csv | llm_agent_final_real | reports/llm_explanations/llm_agent_eval_summary.csv | csv | primary_manuscript_grade | 1 | supports final 80-case manuscript-grade technical evidence |
| final_llm_agent_summary_md | llm_agent_final_real | reports/llm_explanations/llm_agent_eval_summary.md | markdown | primary_manuscript_grade |  | supports final 80-case manuscript-grade technical evidence |
| final_agent_audit_csv | agent_audit_final | reports/agent_audits/agent_audit_results.csv | csv | primary_manuscript_grade | 560 | supports final 80-case manuscript-grade technical evidence |
| final_agent_audit_jsonl | agent_audit_final | reports/agent_audits/agent_audit_results.jsonl | jsonl | primary_manuscript_grade | 560 | supports final 80-case manuscript-grade technical evidence |
| final_agent_audit_md | agent_audit_final | reports/agent_audits/multi_agent_governance_audit.md | markdown | primary_manuscript_grade |  | supports final 80-case manuscript-grade technical evidence |
| chatbot_guardrail_eval_csv | chatbot_guardrails | reports/chatbot_eval/guardrail_evaluation.csv | csv | supporting_manuscript_grade | 75 | supports automated guardrail compliance evidence |
| chatbot_guardrail_summary | chatbot_guardrails | reports/chatbot_eval/guardrail_evaluation_summary.md | markdown | supporting_manuscript_grade |  | supports automated guardrail compliance evidence |
| unsafe_prompt_suite | chatbot_guardrails | reports/chatbot_eval/unsafe_prompt_suite.csv | csv | supporting_manuscript_grade | 50 | supports automated guardrail compliance evidence |
| safe_prompt_suite | chatbot_guardrails | reports/chatbot_eval/safe_prompt_suite.csv | csv | supporting_manuscript_grade | 25 | supports automated guardrail compliance evidence |
| gxair_dashboard_csv | readiness | reports/governance_reports/gxair_component_dashboard.csv | csv | primary_manuscript_grade | 10 | readiness interpretation and deployment-blocker evidence |
| gxair_dashboard_md | readiness | reports/governance_reports/gxair_component_dashboard.md | markdown | primary_manuscript_grade |  | readiness interpretation and deployment-blocker evidence |
| final_readiness_report | readiness | reports/governance_reports/final_governance_readiness_report.md | markdown | primary_manuscript_grade |  | readiness interpretation and deployment-blocker evidence |
| statistical_uncertainty_summary | statistical_reliability | reports/statistical_reliability/uncertainty_summary.md | markdown | supporting_manuscript_grade |  | supports final 80-case manuscript-grade technical evidence |
| llm_guardrail_ci | statistical_reliability | reports/statistical_reliability/llm_guardrail_ci.csv | csv | supporting_manuscript_grade | 4 | supports final 80-case manuscript-grade technical evidence |
| external_validation_summary | external_validation | reports/external_validation/external_validation_summary.md | markdown | supporting_manuscript_grade |  | supports external validation and related-task claim boundaries |
| external_validation_tables | external_validation | reports/manuscript_assets/external_validation_tables.md | markdown | supporting_manuscript_grade |  | supports external validation and related-task claim boundaries |
| external_governance_summary | external_validation | reports/governance_reports/external_validation_governance_summary.md | markdown | supporting_manuscript_grade |  | supports external validation and related-task claim boundaries |
| model_card | governance_documentation | reports/model_card/hr_xai_model_card.md | markdown | supporting_manuscript_grade |  | supports final 80-case manuscript-grade technical evidence |
| decision_log | research_traceability | RESEARCH_DECISION_LOG.md | markdown | supporting_traceability |  | supports final 80-case manuscript-grade technical evidence |
| handoff | research_traceability | PROJECT_CONTINUATION_HANDOFF.md | markdown | supporting_traceability |  | supports final 80-case manuscript-grade technical evidence |
| run_registry | research_traceability | reports/research_log/run_registry.csv | csv | supporting_traceability | 33 | supports final 80-case manuscript-grade technical evidence |
| llm_usage_log | cost_accounting | reports/llm_explanations/llm_usage_log.csv | csv | supporting_traceability | 668 | supports final 80-case manuscript-grade technical evidence |

## Excluded or Non-Final Files

| evidence_id | category | path | file_type | manuscript_grade_status | claim_role |
| --- | --- | --- | --- | --- | --- |
| dry_run_config | excluded_stub_dry_run | configs/llm_agent_eval.yaml | json | excluded_not_manuscript_grade | excluded; do not cite as real LLM evidence |
| offline_stub_client | excluded_stub_dry_run | src/llm/offline_stub_llm.py | python | excluded_not_manuscript_grade | excluded; do not cite as real LLM evidence |
| pilot_40_summary | pilot_quality_control | reports/llm_explanations/openai_pilot_40/llm_agent_eval_summary.csv | csv | pilot_not_final_manuscript_evidence | pilot quality-control evidence; not final 80-case result |

## Integrity

The CSV and JSON manifests include SHA-256 hashes and row counts for machine-readable traceability.
