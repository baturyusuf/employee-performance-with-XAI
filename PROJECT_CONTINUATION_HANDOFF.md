# Project Continuation Handoff

## Current Objective
Continue the HR XAI project as a leakage-aware, fairness/proxy-audited, calibration-aware, explanation-stable, actionability-constrained XAI framework with an OpenAI-backed LLM and multi-agent governance layer.

The LLM is not the predictor. XGBoost remains the predictive model. The LLM and agents consume structured XAI/governance evidence and produce governed explanations, audit summaries, chatbot responses, and manuscript-support reports.

## Current Recommended Model
- Primary research model: `no_salary_hike_no_attrition_no_department` + XGBoost.
- Main leakage-safe comparison baseline: `no_salary_hike_no_attrition` + XGBoost.
- Strict fairness/proxy sensitivity baseline: `no_salary_hike_no_attrition_no_department_no_job_role` + XGBoost.
- Full-feature models: leakage-warning upper-bound only, not deployable.

## Completed In Latest Phase
- Added `reports/research_log/code_completion_audit.md` after auditing README, decision logs, governance cards, configs, source modules, tests, and reports.
- Added config-driven LLM-agent evaluation pipeline: `src/llm/run_llm_agent_evaluation.py` with `configs/llm_agent_eval.yaml`.
- Added shared core utilities: `src/core/io_utils.py`, `src/core/reporting.py`, and `src/core/run_registry.py`.
- Added expanded chatbot guardrail runner: `src/chatbot/run_guardrail_eval.py` with `configs/chatbot_guardrail_eval.yaml`.
- Added component governance dashboard outputs through `src/governance/gxair_score.py --config configs/governance_dashboard.yaml`.
- Added statistical reliability reporting through `src/governance/statistical_reliability.py`.
- Added real OpenAI pilot/final configs: `configs/llm_agent_eval_openai_pilot_40.yaml` and `configs/llm_agent_eval_openai_final_80.yaml`.
- Added `requirements-dev.txt` with `pytest>=8.0.0` while preserving the existing unittest suite.
- Added final evidence package manifest generator: `src/governance/final_evidence_manifest.py`.
- Generated final evidence package under `reports/manuscript_assets/final_evidence_manifest/`.
- Generated expanded batch artifacts. The latest root-level LLM-agent outputs are real OpenAI results for INX + HRDataset_v14:
  - `reports/llm_explanations/eval_case_manifest.csv`
  - `reports/llm_explanations/governed_explanations.jsonl`
  - `reports/llm_explanations/faithfulness_eval.csv`
  - `reports/llm_explanations/faithfulness_eval_summary.md`
  - `reports/llm_explanations/llm_agent_eval_summary.csv`
  - `reports/llm_explanations/llm_agent_eval_summary.md`
  - `reports/agent_audits/agent_audit_results.jsonl`
  - `reports/agent_audits/agent_audit_results.csv`
  - `reports/chatbot_eval/unsafe_prompt_suite.csv`
  - `reports/chatbot_eval/safe_prompt_suite.csv`
  - `reports/chatbot_eval/guardrail_evaluation_summary.md`
  - `reports/governance_reports/gxair_component_dashboard.csv`
  - `reports/governance_reports/gxair_component_dashboard.md`
  - `reports/governance_reports/final_governance_readiness_report.md`
  - `reports/external_validation/external_dataset_roles.csv`
  - `reports/statistical_reliability/uncertainty_summary.md`
- Added external dataset infrastructure for HRDataset_v14, IBM HR Analytics, IBM Attrition transfer, and Employee Turnover.
- Generated dataset cards, schema mappings, target distributions, leakage/sensitive/proxy audits, missingness reports, and duplicate-ID reports under `reports/external_validation/`.
- Ran HRDataset_v14 independent replication using `PerformanceScore` mapped to 2/3/4.
- Reported INX-to-HRDataset cross-dataset validation as infeasible/too limited because only three department-free safe common features overlap.
- Ran IBM schema-compatible restricted target-space robustness and IBM attrition task-transfer robustness.
- Ran Employee Turnover task-transfer robustness with and without `last_evaluation`.
- Ran real OpenAI governed explanations and OpenAI Agents SDK audits on 5 HRDataset_v14 cases, 5 IBM performance robustness cases, and 5 Employee Turnover cases.
- Generated integrated external validation summary, manuscript tables, and governance summary.
- Added canonical governance warning taxonomy.
- Normalized governed explanation warnings and agent warning IDs.
- Expanded chatbot guardrail evaluation with adversarial and Turkish HR-decision prompts.
- Improved Streamlit `LLM Governance & Audit` tab so it remains usable even if the legacy CatBoost model is unavailable.
- Generated manuscript-ready LLM-agent tables and summary.
- Built 5 supplemental reason-code examples from existing XGBoost local grouped SHAP outputs, filtering final-policy forbidden features.
- Ran real OpenAI + OpenAI Agents SDK evaluation on 10 cases.
- Refreshed faithfulness and summary reports from saved OpenAI outputs after fixing a checker warning-variant false negative.

## Real OpenAI Evaluation Status
Expanded configurable batch:
- Pilot command: `python -m src.llm.run_llm_agent_evaluation --config configs/llm_agent_eval_openai_pilot_40.yaml`.
- Pilot result: 40 real OpenAI cases across INX and HRDataset_v14, faithfulness pass rate `1.0`, unsupported claim rate `0.0`, forbidden claim rate `0.0`, missing warning rate `0.0`, parsing success rate `1.0`.
- Final command: `python -m src.llm.run_llm_agent_evaluation --config configs/llm_agent_eval_openai_final_80.yaml`.
- Final result: 80 real OpenAI cases across INX and HRDataset_v14, faithfulness pass rate `1.0`, unsupported claim rate `0.0`, forbidden claim rate `0.0`, missing warning rate `0.0`, parsing success rate `1.0`, agent compliance pass rate `1.0`.
- Dataset scope: `inx_primary` 40 cases and `hrdataset_v14` 40 cases. IBM performance, IBM attrition, and Employee Turnover real LLM regeneration are deferred second-stage robustness tasks.
- Important claim boundary: related-task datasets must not be presented as direct employee-performance validation.

Final evidence manifest:
- `reports/manuscript_assets/final_evidence_manifest/final_evidence_manifest.csv`
- `reports/manuscript_assets/final_evidence_manifest/final_evidence_manifest.json`
- `reports/manuscript_assets/final_evidence_manifest/final_evidence_manifest.md`
- `reports/manuscript_assets/final_evidence_manifest/readiness_not_ready_explanation.md`
- Stub/dry-run outputs are indexed as excluded evidence and must not be cited as manuscript-grade real LLM evidence.

External validation batches:
- HRDataset_v14: 5 cases, real OpenAI governed explanations + OpenAI Agents SDK, faithfulness pass rate `1.0`, unsupported claim rate `0.0`, forbidden claim rate `0.0`, missing warning rate `0.0`, agent success rate `1.0`.
- IBM HR Analytics performance robustness: 5 cases, real OpenAI governed explanations + OpenAI Agents SDK, faithfulness pass rate `1.0`, unsupported claim rate `0.0`, forbidden claim rate `0.0`, missing warning rate `0.0`, agent success rate `1.0`.
- Employee Turnover task transfer: 5 cases, real OpenAI governed explanations + OpenAI Agents SDK, faithfulness pass rate `1.0`, unsupported claim rate `0.0`, forbidden claim rate `0.0`, missing warning rate `0.0`, agent success rate `1.0`.

Prior INX batch:
- Cases: `528;376;568;18;392;405;125;176;662;906`.
- Model: `gpt-5.4-mini`.
- Runtime: OpenAI API for governed explanations + OpenAI Agents SDK for governance audit.
- Faithfulness pass rate: `1.0`.
- Unsupported claim rate: `0.0`.
- Forbidden claim rate: `0.0`.
- Missing warning rate: `0.0`.
- Agent success rate: `1.0`.
- Warning consistency rate: `0.829497`.
- Unsafe/adversarial prompt refusal rate: `1.0`.
- Notes: the final summary was refreshed from saved OpenAI outputs after checker remediation; no additional API calls were made during refresh.

## Guardrail Evaluation Status
- Unsafe/adversarial prompts: 50, refusal success rate `1.0`.
- Safe audit-control prompts: 25, safe answer pass rate `1.0`.
- Guardrails refuse hiring, firing, promotion, compensation, disciplinary, autonomous HR decisions, sensitive-attribute justification, prompt injection, and employee prescriptions.

## Tests
Latest full test command:

```powershell
.\myenv\Scripts\python.exe -m unittest discover -s tests -v
```

Latest result:

```text
96 tests passed
```

Secret scan:

```text
text_secret_pattern_hits=0
```

## Important Files Added Or Updated
- `src/governance/warning_taxonomy.py`
- `src/governance/warning_taxonomy_report.py`
- `src/governance/manuscript_assets.py`
- `src/llm/build_supplemental_reason_codes.py`
- `src/llm/refresh_real_llm_reports.py`
- `src/llm/interpret_real_llm_evaluation.py`
- `src/llm/run_real_llm_evaluation.py`
- `src/llm/governed_explainer.py`
- `src/llm/faithfulness_checker.py`
- `src/agents/openai_agents_runtime.py`
- `src/agents/llm_agent_orchestrator.py`
- `src/chatbot/guardrails.py`
- `src/chatbot/evaluate_chatbot.py`
- `app/streamlit_app.py`
- `tests/test_warning_taxonomy.py`
- `tests/test_faithfulness_checker.py`
- `tests/test_chatbot_guardrails.py`

## Key Reports
- `reports/external_validation/external_validation_summary.md`
- `reports/external_validation/hrdataset_v14/external_experiment_interpretation.md`
- `reports/external_validation/hrdataset_v14/cross_dataset_inx_to_hrdataset/cross_dataset_validation.md`
- `reports/external_validation/ibm_hr_analytics/external_experiment_interpretation.md`
- `reports/external_validation/employee_turnover/external_experiment_interpretation.md`
- `reports/manuscript_assets/external_validation_tables.md`
- `reports/governance_reports/external_validation_governance_summary.md`
- `reports/llm_explanations/real_llm_eval_summary.csv`
- `reports/llm_explanations/real_llm_eval_summary.md`
- `reports/llm_explanations/real_llm_eval_interpretation.md`
- `reports/llm_explanations/governed_explanation_examples.md`
- `reports/agent_audits/openai_agents_sdk_case_*_governance_audit.md`
- `reports/chatbot_eval/guardrail_evaluation.md`
- `reports/governance_reports/warning_taxonomy.md`
- `reports/governance_reports/warning_taxonomy.csv`
- `reports/manuscript_assets/llm_agent_extension_tables.md`
- `reports/manuscript_assets/llm_agent_extension_summary.md`
- `reports/manuscript_assets/agent_roles_table.csv`
- `reports/manuscript_assets/warning_taxonomy_table.csv`
- `reports/manuscript_assets/llm_eval_metrics_table.csv`
- `reports/manuscript_assets/guardrail_eval_table.csv`

## How To Reproduce Current LLM-Agent Evidence
Use environment variables, not hardcoded secrets:

```powershell
$machineKey = [Environment]::GetEnvironmentVariable('OPENAI_API_KEY','Machine')
$userKey = [Environment]::GetEnvironmentVariable('OPENAI_API_KEY','User')
if ($machineKey) { $env:OPENAI_API_KEY = $machineKey } elseif ($userKey) { $env:OPENAI_API_KEY = $userKey }
$env:HR_XAI_LLM_PROVIDER = 'openai'
$env:HR_XAI_REQUIRE_REAL_LLM = '1'
$env:HR_XAI_OPENAI_MODEL = 'gpt-5.4-mini'
$env:HR_XAI_LLM_TEMPERATURE = '0'
```

Then:

```powershell
.\myenv\Scripts\python.exe -m src.llm.run_llm_agent_evaluation --config configs/llm_agent_eval.yaml
.\myenv\Scripts\python.exe -m src.agents.run_governance_audit --config configs/llm_agent_eval.yaml
.\myenv\Scripts\python.exe -m src.chatbot.run_guardrail_eval --config configs/chatbot_guardrail_eval.yaml
.\myenv\Scripts\python.exe -m src.governance.gxair_score --config configs/governance_dashboard.yaml
.\myenv\Scripts\python.exe -m src.governance.statistical_reliability
.\myenv\Scripts\python.exe -m src.governance.warning_taxonomy_report
.\myenv\Scripts\python.exe -m src.chatbot.evaluate_chatbot
.\myenv\Scripts\python.exe -m src.llm.build_supplemental_reason_codes
.\myenv\Scripts\python.exe -m src.llm.run_real_llm_evaluation --limit 10 --provider openai --agent-runtime openai-agents --require-real-llm
.\myenv\Scripts\python.exe -m src.llm.refresh_real_llm_reports --limit 10
.\myenv\Scripts\python.exe -m src.llm.interpret_real_llm_evaluation
.\myenv\Scripts\python.exe -m src.governance.manuscript_assets
.\myenv\Scripts\python.exe -m unittest discover -s tests -v
```

## Known Limitations
- External CSVs were retrieved from public mirrors and documented in dataset cards; source provenance and licensing should be independently verified before manuscript submission.
- HRDataset_v14 supports independent external performance replication, but it is small and cross-sectional.
- IBM performance robustness is restricted to target classes 3 and 4 and must not be described as direct 2/3/4 external validation.
- Employee Turnover and IBM Attrition are related task-transfer robustness checks, not performance validation.
- Cross-dataset INX-to-HRDataset validation is infeasible/too limited under the safe common feature set.
- The 10-case real OpenAI evaluation is still a small-batch engineering validation, not a human-subject study.
- Supplemental cases were built from existing XGBoost local grouped SHAP outputs and filtered for final-policy forbidden features; this is valid for prototype expansion but should be described transparently.
- Warning consistency is improved after taxonomy normalization but still depends on case evidence and specialist agent behavior.
- The chatbot guardrails are deterministic pattern-based and should be expanded with more adversarial prompts before strong deployment claims.
- The project remains a research prototype; external validation is required before operational use.

## Recommended Next Steps
1. Generate final manuscript tables/figures from the updated manuscript assets.
2. Add a visual architecture diagram for the LLM-agent governance layer.
3. If budget and evidence permit, expand to 20 real OpenAI cases only after generating additional case-level SHAP reason-code evidence.
4. Run a stronger prompt-injection benchmark for the chatbot.
5. Prepare a paper-ready methods section emphasizing that the LLM interprets evidence and never predicts employee performance.
