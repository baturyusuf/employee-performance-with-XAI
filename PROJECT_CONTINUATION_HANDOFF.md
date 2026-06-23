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
- Added canonical governance warning taxonomy.
- Normalized governed explanation warnings and agent warning IDs.
- Expanded chatbot guardrail evaluation with adversarial and Turkish HR-decision prompts.
- Improved Streamlit `LLM Governance & Audit` tab so it remains usable even if the legacy CatBoost model is unavailable.
- Generated manuscript-ready LLM-agent tables and summary.
- Built 5 supplemental reason-code examples from existing XGBoost local grouped SHAP outputs, filtering final-policy forbidden features.
- Ran real OpenAI + OpenAI Agents SDK evaluation on 10 cases.
- Refreshed faithfulness and summary reports from saved OpenAI outputs after fixing a checker warning-variant false negative.

## Real OpenAI Evaluation Status
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
- Unsafe prompts: 7, pass rate `1.0`.
- Adversarial prompts: 10, pass rate `1.0`.
- Safe audit-control prompts: 5, pass rate `1.0`.
- Guardrails refuse hiring, firing, promotion, compensation, disciplinary, autonomous HR decisions, sensitive-attribute justification, prompt injection, and employee prescriptions.

## Tests
Latest full test command:

```powershell
.\myenv\Scripts\python.exe -m unittest discover -s tests -v
```

Latest result:

```text
67 tests passed
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
