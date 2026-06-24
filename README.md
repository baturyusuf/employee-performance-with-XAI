# Leakage-Safe HR XAI Governance Framework

This repository is a research-grade HR analytics governance project. It started as an explainable employee performance prediction model, but now includes leakage-safe model selection, fairness and proxy auditing, calibration analysis, SHAP attribution, counterfactual actionability review, governed OpenAI explanations, OpenAI Agents SDK audits, and external validation or robustness experiments on additional HR datasets.

The system is decision-support research only. It must not be used to make autonomous hiring, firing, promotion, compensation, discipline, screening, or other individual employment decisions.

## Current Status

The primary internal research model is:

```text
no_salary_hike_no_attrition_no_department + XGBoost
```

Main comparison baseline:

```text
no_salary_hike_no_attrition + XGBoost
```

Strict fairness/proxy sensitivity baseline:

```text
no_salary_hike_no_attrition_no_department_no_job_role + XGBoost
```

Full-feature models are retained only as historical leakage-warning upper-bound baselines. They are not final or deployable models.

## What This Project Does

- Trains leakage-aware XGBoost models for employee performance research.
- Excludes or audits leakage-risk fields such as salary-hike and attrition variables.
- Audits subgroup performance, proxy risk, and small-group instability.
- Computes calibration diagnostics: log loss, Brier score, and expected calibration error.
- Generates SHAP explanations as model attribution, not causality.
- Classifies counterfactuals by actionability and avoids employee-prescription language.
- Uses real OpenAI-backed governed explanation generation for final LLM evidence.
- Uses the OpenAI Agents SDK governance runtime where supported by the project.
- Produces manuscript-ready external validation and governance reports.

## Evidence Snapshot

The integrated external evidence is summarized in:

```text
reports/external_validation/external_validation_summary.md
reports/manuscript_assets/external_validation_tables.md
reports/governance_reports/external_validation_governance_summary.md
```

### Dataset Roles

| Dataset | Role | Target | Claim Boundary |
| --- | --- | --- | --- |
| INX Future Inc Employee Performance | Internal primary benchmark | `PerformanceRating` 2/3/4 | Internal benchmark only |
| HRDataset_v14 | Independent external performance replication | `PerformanceScore` mapped to 2/3/4 | Direct external performance-target replication |
| IBM HR Analytics | Schema-compatible performance robustness | `PerformanceRating` restricted to 3/4 | Restricted target-space robustness, not full 2/3/4 validation |
| IBM HR Analytics Attrition | Related HR task transfer robustness | `Attrition` 0/1 | Attrition transfer only |
| Employee Turnover | Related HR task transfer robustness | `left` 0/1 | Turnover transfer only |
| Absenteeism at Work | Excluded | Absence hours | Not included due target mismatch and health/medical ethics complexity |

### Key Metrics

| Dataset / Task | Policy | Macro-F1 | Balanced Accuracy | QWK | Log Loss | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| INX primary model | `no_salary_hike_no_attrition_no_department` | 0.5988 | 0.6261 | 0.6376 | 0.4551 | 0.2608 | 0.0638 |
| HRDataset_v14 performance | `department_free` | 0.6389 | 0.6567 | 0.5945 | 0.5508 | 0.2565 | 0.0925 |
| IBM performance | `department_free` | 0.4603 | 0.4974 | -0.0085 | 0.4978 | 0.2869 | 0.0822 |
| IBM attrition | `department_free` | 0.6749 | 0.6388 | 0.3639 | 0.3560 | 0.2042 | 0.0388 |
| Employee Turnover | `without_last_evaluation` | 0.9666 | 0.9603 | 0.9331 | 0.0810 | 0.0395 | 0.0077 |

Interpret these metrics conservatively. HRDataset_v14 supports independent replication on a mappable performance target. IBM performance is restricted to classes 3 and 4. Employee Turnover and IBM Attrition are related task robustness checks, not performance validation.

### Cross-Dataset Validation

Train-on-INX/test-on-HRDataset transportability was audited and reported as infeasible or too limited. Only three department-free safe common features were available:

```text
EmpJobRole
EmpJobSatisfaction
ExperienceYearsAtThisCompany
```

Forcing the experiment would mainly measure schema mismatch rather than defensible model transportability. See:

```text
reports/external_validation/hrdataset_v14/cross_dataset_inx_to_hrdataset/cross_dataset_validation.md
```

### Real LLM and Agent Evaluation

Final LLM/agent evidence uses real OpenAI-backed code paths, not only offline stubs. The latest expanded real run prioritizes the primary INX benchmark and HRDataset_v14 independent replication; IBM and turnover related-task LLM regeneration remain a second-stage robustness task.

| Dataset | Cases | Faithfulness Pass Rate | Unsupported Claim Rate | Forbidden Claim Rate | Missing Warning Rate | Agent Success Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| INX primary | 40 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| HRDataset_v14 | 40 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |

Usage and estimated cost are logged in:

```text
reports/llm_explanations/llm_usage_log.csv
reports/external_validation/external_llm_usage_summary.csv
```

The cost logs are estimates for experiment accounting. Provider billing dashboards remain the source of truth.

## Project Structure

```text
app/                         Streamlit dashboard
configs/                     Project configuration files
data/
  raw/                       INX raw data
  interim/                   Validated/intermediate data
  processed/                 Processed model data
  external/
    hrdataset_v14/           Raw CSV, schema mapping, dataset card
    ibm_hr_analytics/        Raw CSV, schema mapping, dataset card
    employee_turnover/       Raw CSV, schema mapping, dataset card
models_artifacts/            Trained model artifacts
reports/
  external_validation/       Dataset audits, external metrics, LLM evidence
  governance_reports/        Governance summaries and dashboards
  manuscript_assets/         Manuscript-ready tables
  llm_explanations/          Governed explanation outputs and usage logs
  model_card/                Model card artifacts
src/
  agents/                    Governance agent runtime and audits
  chatbot/                   Guardrailed chatbot interface
  data/                      Data loaders, validation, external adapters
  experiments/               Leakage-safe, external, robustness experiments
  explainability/            SHAP, reason codes, counterfactuals, fairness reports
  features/                  Feature-set policies
  governance/                Integrated governance reports
  llm/                       Governed explanation generation and validation
  models/                    Training and evaluation utilities
tests/                       Unit and integration tests
```

## Data Layout

Primary INX data:

```text
data/raw/inx_employee_performance.csv
```

External datasets:

```text
data/external/hrdataset_v14/raw.csv
data/external/hrdataset_v14/schema_mapping.json
data/external/hrdataset_v14/dataset_card.md

data/external/ibm_hr_analytics/raw.csv
data/external/ibm_hr_analytics/schema_mapping.json
data/external/ibm_hr_analytics/dataset_card.md

data/external/employee_turnover/raw.csv
data/external/employee_turnover/schema_mapping.json
data/external/employee_turnover/dataset_card.md
```

The adapter layer standardizes external schemas, maps targets, marks leakage-risk columns, marks sensitive or audit-only columns, marks proxy-risk columns, and fails clearly when required columns are missing.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The project uses common Python ML/XAI packages including pandas, numpy, scikit-learn, XGBoost, SHAP, matplotlib, Streamlit, joblib, scipy, and the OpenAI SDK dependencies declared in `requirements.txt`.

## Reproducing Core Evidence

Run the existing internal pipeline commands as needed, then run the external evidence package.

### External Dataset Audits

```powershell
.\myenv\Scripts\python.exe -m src.data.external_audit --datasets all
.\myenv\Scripts\python.exe -m src.data.external_audit --datasets ibm_hr_analytics --target-kind attrition
```

Outputs are written under:

```text
reports/external_validation/<dataset_name>/
```

Each dataset audit includes:

- `dataset_audit.md`
- `schema_mapping.csv`
- `target_distribution.csv`
- `leakage_sensitive_proxy_audit.csv`
- `missingness_report.csv`
- `duplicate_id_report.csv`
- `metadata.json`

### HRDataset_v14 Independent Replication

```powershell
.\myenv\Scripts\python.exe -m src.experiments.external_validation --dataset hrdataset_v14
```

Cross-dataset feasibility audit:

```powershell
.\myenv\Scripts\python.exe -m src.experiments.external_validation --cross-dataset
```

### IBM and Turnover Robustness

```powershell
.\myenv\Scripts\python.exe -m src.experiments.external_robustness --task all
```

Run individual tasks:

```powershell
.\myenv\Scripts\python.exe -m src.experiments.external_robustness --task ibm_performance
.\myenv\Scripts\python.exe -m src.experiments.external_robustness --task ibm_attrition
.\myenv\Scripts\python.exe -m src.experiments.external_robustness --task employee_turnover
```

### Real OpenAI LLM and Agent Batches

Set the API key only in the environment. Do not print it, save it in reports, or commit it.

```powershell
$env:OPENAI_API_KEY = "..."
.\myenv\Scripts\python.exe -m src.llm.check_llm_setup
```

Run the config-driven LLM-agent evaluation pipeline:

```powershell
.\myenv\Scripts\python.exe -m src.llm.run_llm_agent_evaluation --config configs/llm_agent_eval.yaml
```

By default `configs/llm_agent_eval.yaml` uses `run_mode: dry_run` and the deterministic offline stub. Those outputs are useful for reproducibility and testing, but they are not manuscript-grade real LLM evidence. For real OpenAI execution, change the config to `run_mode: real`, set provider/model settings under `llm`, and provide `OPENAI_API_KEY`. Large real batches require explicit approval because they incur API cost.

Approved real OpenAI expanded evaluation uses a two-stage config:

```powershell
.\myenv\Scripts\python.exe -m src.llm.run_llm_agent_evaluation --config configs/llm_agent_eval_openai_pilot_40.yaml
.\myenv\Scripts\python.exe -m src.llm.run_llm_agent_evaluation --config configs/llm_agent_eval_openai_final_80.yaml
```

The pilot writes to `reports/llm_explanations/openai_pilot_40/`. The final 80-case run writes manuscript-support technical evidence to `reports/llm_explanations/` and currently includes only `inx_primary` and `hrdataset_v14`. IBM performance, IBM attrition, and Employee Turnover remain second-stage robustness regeneration targets and must not be presented as direct employee-performance validation.

Run the standalone batch agent audit from generated explanation JSONL:

```powershell
.\myenv\Scripts\python.exe -m src.agents.run_governance_audit --config configs/llm_agent_eval.yaml
```

Legacy small external governed explanation and agent audit commands:

```powershell
.\myenv\Scripts\python.exe -m src.experiments.external_validation --dataset hrdataset_v14 --run-llm --llm-policy department_free --llm-limit 5
.\myenv\Scripts\python.exe -m src.experiments.external_validation --dataset ibm_hr_analytics --run-llm --llm-policy department_free --llm-limit 5
.\myenv\Scripts\python.exe -m src.experiments.external_validation --dataset employee_turnover --run-llm --llm-policy without_last_evaluation --llm-limit 5
```

The deterministic offline path remains available for tests and reproducibility, but final evidence should use the real OpenAI-backed path.

Generated LLM-agent outputs:

```text
reports/llm_explanations/eval_case_manifest.csv
reports/llm_explanations/governed_explanations.jsonl
reports/llm_explanations/faithfulness_eval.csv
reports/llm_explanations/faithfulness_eval_summary.md
reports/llm_explanations/llm_agent_eval_summary.csv
reports/llm_explanations/llm_agent_eval_summary.md
reports/agent_audits/agent_audit_results.jsonl
reports/agent_audits/agent_audit_results.csv
reports/agent_audits/multi_agent_governance_audit.md
```

### Chatbot Guardrail Evaluation

```powershell
.\myenv\Scripts\python.exe -m src.chatbot.run_guardrail_eval --config configs/chatbot_guardrail_eval.yaml
```

This writes expanded prompt suites and evaluation outputs:

```text
reports/chatbot_eval/unsafe_prompt_suite.csv
reports/chatbot_eval/safe_prompt_suite.csv
reports/chatbot_eval/guardrail_evaluation.csv
reports/chatbot_eval/guardrail_evaluation_summary.md
```

### Integrated Reports

```powershell
.\myenv\Scripts\python.exe -m src.governance.external_validation_reports
.\myenv\Scripts\python.exe -m src.governance.gxair_score --config configs/governance_dashboard.yaml
.\myenv\Scripts\python.exe -m src.governance.statistical_reliability
```

This writes:

```text
reports/external_validation/external_validation_summary.md
reports/external_validation/external_dataset_roles.csv
reports/manuscript_assets/external_validation_tables.md
reports/governance_reports/external_validation_governance_summary.md
reports/governance_reports/gxair_component_dashboard.csv
reports/governance_reports/gxair_component_dashboard.md
reports/governance_reports/final_governance_readiness_report.md
reports/statistical_reliability/uncertainty_summary.md
```

## Internal Pipeline Entry Points

Important existing commands include:

```powershell
.\myenv\Scripts\python.exe -m src.data.load_data
.\myenv\Scripts\python.exe -m src.data.preprocess
.\myenv\Scripts\python.exe -m src.experiments.leakage_safe_cv --models xgboost --drop-sensitive
.\myenv\Scripts\python.exe -m src.experiments.leakage_safe_pipeline --task all --model all --feature-set no_salary_hike_no_attrition_no_department --drop-sensitive
.\myenv\Scripts\python.exe -m src.explainability.xgboost_grouped_shap --feature-set no_salary_hike_no_attrition_no_department --drop-sensitive
```

Run the Streamlit dashboard:

```powershell
streamlit run app/streamlit_app.py
```

## Governance Layer

The LLM is not the predictive model. The evidence flow is:

```text
XGBoost prediction
-> structured ML/XAI evidence
-> leakage, fairness/proxy, calibration, SHAP, actionability audits
-> governed LLM explanation
-> OpenAI Agents SDK governance audit
-> report-backed dashboard/chatbot outputs
```

The LLM and agents only interpret structured evidence generated by the pipeline. They must not invent evidence, infer causality, guarantee fairness, or recommend individual HR decisions.

Guardrail evaluations and summaries are stored under:

```text
reports/llm_explanations/
reports/agent_audits/
reports/chatbot_eval/
reports/governance_reports/
```

## Testing and Validation

Run the full test suite:

```powershell
.\myenv\Scripts\python.exe -m unittest discover -s tests -v
```

Compile modified Python files:

```powershell
.\myenv\Scripts\python.exe -m compileall src tests
```

Secret hygiene scan:

```powershell
rg -n "s[k]-[A-Za-z0-9_-]{20,}" .
```

Recent validation status:

```text
Unit tests: run with both `unittest discover` and `pytest`
Python compile checks: passed
Secret pattern scan: no OpenAI-key-pattern hits in text artifacts
```

## Scientific Claim Boundaries

Allowed claims:

- The project implements a leakage-aware, SHAP/XAI, fairness/proxy-audited, calibration-aware, actionability-constrained HR governance framework.
- HRDataset_v14 provides independent replication on a directly mappable external performance target.
- IBM HR Analytics provides schema-compatible robustness with restricted performance target space.
- IBM Attrition and Employee Turnover provide related HR risk prediction task transfer evidence, not direct employee-performance validation.
- Real OpenAI governed explanations and deterministic batch agent audits have been run on an expanded 80-case INX + HRDataset_v14 batch.

Not allowed:

- No autonomous HR decision capability.
- No hiring, firing, promotion, compensation, discipline, or applicant-screening recommendation.
- No causal claim from SHAP or counterfactuals.
- No fairness guarantee from removing sensitive or group variables.
- No direct performance external-validation claim for IBM or Employee Turnover.
- No deployment-readiness claim without data provenance review, organization-specific validation, legal review, calibration review, subgroup impact review, and human-centered evaluation.

## Key Reports

```text
reports/model_selection/final_recommendation.md
reports/model_card/hr_xai_model_card.md
MODEL_GOVERNANCE_CARD.md
reports/external_validation/external_validation_summary.md
reports/manuscript_assets/external_validation_tables.md
reports/governance_reports/external_validation_governance_summary.md
PROJECT_CONTINUATION_HANDOFF.md
RESEARCH_DECISION_LOG.md
```

## Citation Notes

If this repository is used in academic work, cite the original dataset providers and the main methods used, including XGBoost, SHAP, calibration metrics, counterfactual explanation methods, fairness/proxy auditing methods, and OpenAI-backed governed explanation/agent evaluation where applicable.

Before submission, independently verify external dataset provenance, licensing, and dataset-card metadata.
