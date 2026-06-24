# Code Completion Audit

Generated: 2026-06-24

Purpose: repository audit before code-side completion work for the project direction: "LLM-assisted multi-agent XAI governance framework for leakage-safe and actionability-constrained employee performance prediction."

This audit verifies repository contents directly. It does not treat README text as sufficient evidence.

## Audit Commands Run

```text
Get-ChildItem -Force
Get-ChildItem -Path configs -Recurse -File
Get-ChildItem -Path src -Recurse -File
Get-ChildItem -Path tests -Recurse -File
Get-ChildItem -Path reports/* -File / -Recurse where needed
Get-Content README.md
Get-Content RESEARCH_DECISION_LOG.md
Get-Content MODEL_GOVERNANCE_CARD.md
Get-Content LLM_AGENT_GOVERNANCE_README.md
Get-Content key src/llm, src/agents, src/chatbot, src/governance, src/data, src/experiments files
.\myenv\Scripts\python.exe -m unittest discover -s tests -v
```

`rg` is not installed in this workspace, so PowerShell recursive listings were used.

## 1. Current Implemented Modules

### Data and Feature Infrastructure

- `src/data/load_data.py`, `validate_schema.py`, `preprocess.py`: internal INX loading, validation, preprocessing.
- `src/data/audit.py`: internal data audit reports.
- `src/data/external_adapters.py`: external dataset config loader, robust CSV loading, canonical mapping, target mapping, leakage/sensitive/proxy role handling, feature-policy construction.
- `src/data/external_audit.py`: external dataset audits producing schema mapping, target distribution, leakage/sensitive/proxy audit, missingness, duplicate ID, and metadata reports.
- `src/features/feature_sets.py`: config-backed feature-set policies, including primary, baseline, strict proxy sensitivity, and counterfactual-only sets.
- `configs/feature_sets.yaml`, `feature_taxonomy.yaml`, `evaluation.yaml`, `fairness.yaml`, `counterfactuals.yaml`, `gxair.yaml`, `model_grid.yaml`, `project.yaml`, `xai.yaml`: existing JSON-compatible YAML configs.

### Model, Metrics, and Experiment Infrastructure

- `src/models/evaluate.py`: classification metrics including ordinal metrics, log loss, Brier, and ECE logic.
- `src/experiments/leakage_safe_cv.py`: leakage-safe CV for configured feature sets and models.
- `src/experiments/leakage_safe_pipeline.py`: leakage-safe holdout pipeline.
- `src/experiments/leakage_sensitivity.py`: leakage sensitivity index from ablation reports.
- `src/experiments/proxy_analysis.py`: department proxy reconstructability analysis.
- `src/experiments/fairness_sensitivity.py`: subgroup metrics, disparity summaries, low-support handling.
- `src/experiments/final_calibration.py`, `final_shap_stability.py`, `final_counterfactual_actionability.py`, `final_fairness_bootstrap.py`, `final_model_selection.py`, `final_reason_codes.py`: final evidence package modules for INX.
- `src/experiments/external_validation.py`: external CV-style experiments, external fairness/proxy/calibration/SHAP/actionability reports, representative case selection, cross-dataset feasibility, and small real LLM/agent evaluations for external datasets.
- `src/experiments/external_robustness.py`: IBM and Employee Turnover robustness wrapper.

### XAI and Governance Evidence

- `src/explainability/*`: SHAP global/local, grouped XGBoost SHAP, reason codes, counterfactuals, fairness report.
- `src/llm/evidence_schema.py`: dataclass evidence objects and INX report loader.
- `src/llm/governed_explainer.py`, `prompt_templates.py`, `llm_client.py`, `openai_client.py`, `offline_stub_llm.py`, `client_factory.py`, `runtime_config.py`: governed explanation generation with real OpenAI and deterministic offline paths.
- `src/llm/faithfulness_checker.py`: deterministic rule-based faithfulness/compliance checker.
- `src/llm/generate_governed_explanations.py`, `run_real_llm_evaluation.py`, `evaluate_llm_governance.py`, `interpret_real_llm_evaluation.py`, `refresh_real_llm_reports.py`, `usage_logger.py`, `cost_estimator.py`: existing LLM generation/evaluation/report utilities.

### Multi-Agent and Chatbot Modules

- `src/agents/base_agent.py`: common `AgentFinding` and `BaseGovernanceAgent`.
- `src/agents/leakage_agent.py`, `fairness_proxy_agent.py`, `calibration_agent.py`, `shap_stability_agent.py`, `counterfactual_actionability_agent.py`, `explanation_agent.py`, `supervisor_agent.py`: deterministic governance agents.
- `src/agents/openai_agents_runtime.py`, `run_llm_governance_audit.py`: OpenAI Agents SDK path.
- `src/agents/run_governance_audit.py`: single-case deterministic multi-agent audit.
- `src/chatbot/guardrails.py`, `chat_engine.py`, `retrieval.py`, `chatbot_app.py`, `evaluate_chatbot.py`: guardrailed report-backed chatbot and small evaluation suite.

### Governance Reports

- `src/governance/gxair_score.py`: current score-style dashboard with LLM governance component.
- `src/governance/governance_report.py`: current final LLM-agent summary.
- `src/governance/external_validation_reports.py`: integrated external validation/manuscript/governance summary generator.
- `src/governance/warning_taxonomy.py`, `warning_taxonomy_report.py`: warning taxonomy and mandatory warning support.

### Existing Run Logging

- `src/utils/experiment_registry.py`: run registry writer at `reports/research_log/experiment_registry.csv`.
- No `src/core/run_registry.py`, `src/core/reporting.py`, or `src/core/io_utils.py` package exists yet.

## 2. Missing Modules

The following requested modules or abstractions are absent or incomplete:

- `src/llm/run_llm_agent_evaluation.py`: missing config-driven multi-dataset LLM-agent batch runner.
- `configs/llm_agent_eval.yaml`: missing.
- `configs/chatbot_guardrail_eval.yaml`: missing.
- `configs/governance_dashboard.yaml`: missing.
- `configs/external_validation.yaml`: missing.
- `src/chatbot/run_guardrail_eval.py`: missing config-driven safe/unsafe prompt evaluation.
- `src/core/run_registry.py`, `src/core/reporting.py`, `src/core/io_utils.py`: missing requested shared abstractions.
- Generalized evidence builder for internal and external datasets is missing. Current `CompleteCaseEvidence.from_reports()` is primarily INX-oriented; external evidence is constructed inside `src/experiments/external_validation.py`.
- Batch JSONL governed-explanation writer with required hashes and metadata is missing.
- Batch faithfulness CSV/Markdown writer with required per-category columns is missing.
- Batch agent audit CSV/JSONL writer is missing.
- Expanded 50 unsafe / 25 safe chatbot prompt suites are missing.
- Component readiness dashboard outputs requested as `gxair_component_dashboard.csv/md` and `final_governance_readiness_report.md` are missing.
- `reports/external_validation/external_dataset_roles.csv` is missing.
- `reports/statistical_reliability/*` uncertainty summary files are missing.
- `reports/research_log/run_registry.csv` is missing; the existing equivalent is `experiment_registry.csv`.

## 3. Existing LLM / Agent / Chatbot Functionality

### LLM

Implemented:

- OpenAI-backed governed explanation path.
- Deterministic offline stub path for tests/reproducibility.
- Runtime config enforces real-LLM requirements when requested.
- Usage logging and cost estimate support.
- Rule-based faithfulness checker.
- Existing small real OpenAI evaluation batches:
  - INX: 10 cases
  - HRDataset_v14: 5 cases
  - IBM performance: 5 cases
  - Employee Turnover: 5 cases

Gaps:

- No config-driven multi-dataset LLM-agent evaluation command.
- No recommended expanded batch defaults of 30/20/10/10/10.
- No `eval_case_manifest.csv`.
- No batch `governed_explanations.jsonl` with evidence/response hashes.
- Current schema does not exactly match the requested evidence fields. For example, `PredictionEvidence` lacks `dataset_name`; `CalibrationEvidence` uses `expected_calibration_error` but not the requested `ece` alias or `probability_interpretation`; `GovernanceEvidence` has `prohibited_use` instead of `prohibited_uses`.

### Agents

Implemented:

- Deterministic agents for leakage, fairness/proxy, calibration, SHAP stability, counterfactual actionability, explanation compliance, and supervisor aggregation.
- OpenAI Agents SDK runtime exists for LLM-assisted audit.
- Existing reports include per-case OpenAI agent audit JSON/Markdown files.

Gaps:

- Batch agent audit writer is missing.
- Current deterministic runner audits one case and writes only `multi_agent_governance_audit.json/md`.
- Agent output status vocabulary differs from requested `pass/warn/fail` baseline. Existing statuses include `pass_with_warnings`, `needs_evidence`, and `needs_explanation`.
- Agent output schema has useful fields but not yet a common JSONL/CSV contract for batch-level manuscript outputs.

### Chatbot

Implemented:

- Guardrail pattern checks for unsafe HR decisions, prompt injection, sensitive-attribute misuse, fairness guarantees, and employee prescriptions.
- Safe audit-question paths for leakage, fairness/proxy, calibration, counterfactuals, and general governance.
- Current evaluation writes `reports/chatbot_eval/guardrail_evaluation.csv` and `.md`.

Gaps:

- Current suite is too small for the new requirement: 17 unsafe/adversarial prompts and 5 safe controls.
- Missing explicit safe/unsafe prompt suite CSVs.
- Missing config-driven `src.chatbot.run_guardrail_eval`.
- Existing summary is named `guardrail_evaluation.md`; requested file is `guardrail_evaluation_summary.md`.

## 4. Existing Tests

Current command:

```text
.\myenv\Scripts\python.exe -m unittest discover -s tests -v
```

Result:

```text
Ran 76 tests in 8.632s
OK
```

Existing tests cover:

- Config loading and feature taxonomy.
- Data audit.
- Feature-set exclusions, Age policy, department and job-role sensitivity.
- External adapter mapping, target mapping, leakage/sensitive exclusion, external report existence, external output schema validation, secret scan.
- Metrics and binary Brier behavior.
- Leakage sensitivity and reports.
- Proxy analysis.
- Final evidence helper logic.
- Evidence schema basic constraints.
- Faithfulness checker happy and failure cases.
- Governance agents basic behavior.
- Chatbot guardrails for unsafe and safe prompts.
- OpenAI runtime missing-key behavior.
- LLM usage logger/cost estimator.
- Warning taxonomy.

Missing or weak tests relative to the new task:

- No dedicated `tests/test_governed_explainer.py`.
- Evidence schema tests are narrow and do not check the full requested object contract, JSON serialization across all fields, missing evidence handling, or external evidence construction.
- No tests for config-driven batch LLM-agent runner.
- No tests for `eval_case_manifest.csv` sampling logic.
- No tests for batch faithfulness CSV schema and failure examples.
- No tests for batch agent JSONL/CSV output.
- No tests for expanded chatbot suite generation/evaluation.
- Existing G-XAIR tests target current score-style output, not missing-evidence component readiness behavior.
- No tests for new `src/core` registry/reporting/io abstractions.

## 5. Existing Report Outputs

### Manuscript-Supportive Outputs Already Present

- `reports/model_selection/final_recommendation.md`
- `reports/model_selection/final_candidate_dashboard.csv/md`
- `reports/model_card/hr_xai_model_card.md`
- `MODEL_GOVERNANCE_CARD.md`
- `reports/external_validation/external_validation_summary.md`
- `reports/manuscript_assets/external_validation_tables.md`
- `reports/governance_reports/external_validation_governance_summary.md`
- External dataset audits under:
  - `reports/external_validation/hrdataset_v14/`
  - `reports/external_validation/ibm_hr_analytics/`
  - `reports/external_validation/ibm_hr_analytics_attrition/`
  - `reports/external_validation/employee_turnover/`
- External performance, fairness, calibration, proxy, SHAP, actionability, representative-case, and metadata outputs.
- `reports/external_validation/hrdataset_v14/cross_dataset_inx_to_hrdataset/cross_dataset_validation.md` reports transportability infeasible due only three safe common features.
- Real LLM/agent summaries:
  - `reports/llm_explanations/real_llm_eval_summary.md`
  - `reports/llm_explanations/real_llm_eval_summary.csv`
  - `reports/llm_explanations/real_llm_eval_interpretation.md`
  - external per-dataset `llm_agent_governance_*` files.
- Existing per-case OpenAI Agents SDK audit files under `reports/agent_audits/`.
- `reports/research_log/experiment_registry.csv`.

### Required Outputs Missing

```text
reports/research_log/code_completion_audit.md
reports/research_log/run_registry.csv
reports/llm_explanations/eval_case_manifest.csv
reports/llm_explanations/governed_explanations.jsonl
reports/llm_explanations/faithfulness_eval.csv
reports/llm_explanations/faithfulness_eval_summary.md
reports/llm_explanations/llm_agent_eval_summary.csv
reports/llm_explanations/llm_agent_eval_summary.md
reports/agent_audits/agent_audit_results.jsonl
reports/agent_audits/agent_audit_results.csv
reports/chatbot_eval/unsafe_prompt_suite.csv
reports/chatbot_eval/safe_prompt_suite.csv
reports/chatbot_eval/guardrail_evaluation_summary.md
reports/governance_reports/gxair_component_dashboard.csv
reports/governance_reports/gxair_component_dashboard.md
reports/governance_reports/final_governance_readiness_report.md
reports/external_validation/external_dataset_roles.csv
reports/statistical_reliability/uncertainty_summary.md
```

## 6. Outputs That Look Manuscript-Grade

These outputs are suitable as a strong base for manuscript-support assets, subject to final verification and conservative wording:

- Final model recommendation and final candidate dashboard.
- External validation summary and manuscript tables.
- External dataset audit reports with explicit target mappings and role boundaries.
- HRDataset_v14 independent replication outputs.
- IBM restricted target-space robustness outputs.
- Employee Turnover related task-transfer outputs.
- Cross-dataset infeasibility report for INX-to-HRDataset.
- Real OpenAI small-batch LLM/agent summary files, clearly labelled as small technical validation.
- Model governance card and external governance summary.
- Existing experiment registry entries.

## 7. Outputs That Look Demo / Stub / Placeholder / Too Narrow

These are useful but not sufficient for the new Q3-level code-side requirement:

- `reports/chatbot_eval/guardrail_evaluation.md/csv`: too small; no persisted 50/25 prompt suites; not config-driven.
- `reports/agent_audits/multi_agent_governance_audit.md/json`: one-case deterministic audit, not batch JSONL/CSV.
- `reports/governance_reports/gxair_llm_agent_dashboard.csv/json`: score-style dashboard, not the requested component readiness report with severity, evidence files, limitations, and evidence-missing behavior.
- `reports/governance_reports/final_llm_agent_research_summary.md`: useful but too brief for the new integrated batch-evaluation objective.
- `reports/llm_explanations/governed_explanation_examples.md`: not the requested JSONL with run IDs, hashes, parsing state, and metadata.
- `reports/llm_explanations/governed_explanation_eval.csv`: too small/narrow and lacks the required expanded faithfulness columns.
- `offline_stub_llm.py`: appropriate for tests/dry runs only; not manuscript-grade LLM evidence unless explicitly labelled stub/demo.

## 8. What Must Be Regenerated

After implementing the missing config-driven pipeline:

- Expanded eval case manifest.
- Governed explanations JSONL.
- Faithfulness detail CSV and Markdown summary.
- Batch agent audit CSV/JSONL and Markdown report.
- Expanded safe/unsafe chatbot prompt suites and evaluation summary.
- Integrated LLM-agent evaluation summary.
- G-XAIR component dashboard CSV/Markdown and final readiness report.
- External dataset roles CSV.
- Statistical reliability / uncertainty summary.
- README and `LLM_AGENT_GOVERNANCE_README.md` command/output sections.
- New `reports/research_log/run_registry.csv`; existing `experiment_registry.csv` should be preserved and can be bridged into the new registry format.

Real OpenAI calls should be regenerated only after explicit user approval because the recommended expanded batch is beyond a tiny validation run.

## 9. Proposed Implementation Plan

1. Add shared core utilities:
   - `src/core/io_utils.py`
   - `src/core/reporting.py`
   - `src/core/run_registry.py`
   These should reuse existing `src/utils/experiment_registry.py` rather than replacing it.

2. Extend the evidence layer:
   - Preserve backward compatibility with existing dataclass evidence objects.
   - Add required fields/aliases such as `dataset_name`, `ece`, `probability_interpretation`, `mode`, `desired_class`, `cost`, `prohibited_uses`, `human_review_required`.
   - Add explicit missing-evidence markers and loaders/builders for internal and external report artifacts.

3. Implement config-driven LLM-agent evaluation:
   - `configs/llm_agent_eval.yaml`
   - `src/llm/run_llm_agent_evaluation.py`
   - Stratified/risk-aware sampler using existing predictions, representative cases, SHAP/actionability/fairness/proxy files where available.
   - Default to dry-run stub mode unless `run_mode: real` and real OpenAI approval/API configuration are present.
   - Write required manifest, JSONL, faithfulness CSV/summary, agent outputs, and integrated LLM-agent summaries.

4. Strengthen faithfulness/compliance:
   - Preserve current deterministic checker.
   - Add per-category counts and CSV row builder.
   - Add failure examples in tests.

5. Expand agent runner:
   - Keep existing agent classes.
   - Add batch runner and JSONL/CSV writers.
   - Add agent output normalization to requested status/severity vocabulary while preserving raw status.

6. Expand chatbot guardrail evaluation:
   - Add config file and runner.
   - Generate at least 50 unsafe/adversarial prompts and 25 safe prompts as CSV suites.
   - Evaluate both refusal and safe-answer behavior without refusing every prompt.

7. Replace/extend G-XAIR:
   - Add component dashboard with score/status/severity/evidence file/explanation/limitations.
   - Mark missing evidence explicitly.
   - Final readiness label constrained to research-only/readiness categories, never deployment-ready.

8. Add external dataset roles and statistical reliability:
   - Generate `external_dataset_roles.csv` from existing external configs/reports.
   - Add uncertainty summaries from available fold/bootstrap/binomial evidence; mark insufficient sample sizes clearly rather than inventing CIs.

9. Add tests:
   - New or strengthened tests for evidence schema, governed explainer parsing, faithfulness failure categories, agent warnings, chatbot safe/unsafe behavior, G-XAIR missing evidence, registry/report writer outputs, and generated report schemas.

10. Run validation:
   - `pytest` if available.
   - `.\myenv\Scripts\python.exe -m unittest discover -s tests -v` as fallback/parallel compatibility.
   - `.\myenv\Scripts\python.exe -m compileall src`
   - Secret scan for OpenAI-key-patterns.

## 10. Questions That Need User Confirmation

1. Paid LLM expansion: The requested default batch is approximately 80 cases across INX, HRDataset_v14, IBM performance, IBM attrition, and Employee Turnover. This is larger than the existing small validation batches and will incur paid OpenAI usage. Should I implement the real-call path now but generate final expanded reports in dry-run/stub-labelled mode until you approve paid real LLM execution?

2. LLM model: Existing docs and outputs reference OpenAI/gpt-5.4-mini. Changing provider or model requires approval. Should the new config keep `gpt-5.4-mini` as the default real model?

3. Batch size for real run: If you approve paid calls later, should the first real expanded batch use the recommended default sizes (30/20/10/10/10), or a smaller pilot such as 5 per dataset followed by the full run?

4. Dependency specification: `requirements.txt` currently lists only OpenAI-related packages, while the code depends on pandas, numpy, scikit-learn, XGBoost, SHAP, etc. Should I update dependency files to include the actual ML stack already required by the repository? This is not a new modeling dependency, but it changes reproducibility metadata.

5. Real LLM environment: If real calls are approved, the environment must provide `OPENAI_API_KEY`. I will not print, store, or commit it.

## Immediate Blocking Constraints

- Large real LLM batch execution is blocked pending user approval.
- No new external datasets will be added.
- No primary predictive model or feature-policy changes are proposed.
- No manuscript text will be written; only code, reproducible outputs, report assets, and documentation commands will be produced.
