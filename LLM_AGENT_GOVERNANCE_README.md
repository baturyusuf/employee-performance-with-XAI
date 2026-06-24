# LLM-Agent Governance README

## Architecture

```text
XGBoost / tabular ML predicts
  -> SHAP, calibration, fairness/proxy, leakage, and counterfactual modules generate evidence
  -> CompleteCaseEvidence JSON is passed to the LLM
  -> governed explanation JSON is generated
  -> deterministic and optional OpenAI Agents SDK governance agents audit the evidence/explanation
  -> guardrailed chatbot exposes report-backed evidence safely
  -> governance dashboard summarizes readiness and missing evidence
```

The LLM is not the predictive model. It only interprets structured ML/XAI/governance evidence.

## Main Modules

- `src.llm.evidence_schema`: structured evidence dataclasses, JSON serialization, report loaders, missing-evidence markers.
- `src.llm.run_llm_agent_evaluation`: config-driven multi-dataset LLM-agent evaluation runner.
- `src.llm.governed_explainer`: sends `CompleteCaseEvidence` JSON to the configured LLM client.
- `src.llm.offline_stub_llm`: deterministic dry-run/test client. Not manuscript-grade real LLM evidence.
- `src.llm.openai_client`: real OpenAI structured-output client.
- `src.llm.faithfulness_checker`: deterministic unsupported-claim, forbidden-claim, and missing-warning checks.
- `src.agents.*`: leakage, fairness/proxy, calibration, SHAP stability, counterfactual actionability, explanation compliance, and supervisor agents.
- `src.agents.run_governance_audit`: single-case or batch agent audit command.
- `src.chatbot.run_guardrail_eval`: expanded safe/unsafe prompt-suite evaluation.
- `src.governance.gxair_score`: component readiness dashboard, not a fake deployment score.
- `src.governance.statistical_reliability`: fold/bootstrap/binomial uncertainty summaries.

## Configs

```text
configs/llm_agent_eval.yaml
configs/chatbot_guardrail_eval.yaml
configs/governance_dashboard.yaml
configs/external_validation.yaml
```

`configs/llm_agent_eval.yaml` controls datasets, sample sizes, seed, sampling strategy, LLM provider/model, temperature, token limit, retry/rate-limit settings, output directory, chatbot evaluation, and dry-run versus real LLM mode.

Default mode is `dry_run` with `offline_stub_llm`. Real OpenAI execution requires:

```powershell
$env:OPENAI_API_KEY = "..."
```

and config changes:

```text
run_mode: real
llm.provider: openai
llm.require_real_llm: true
```

Do not store API keys in the repository.

## Commands

Run the full config-driven dry-run evaluation:

```powershell
.\myenv\Scripts\python.exe -m src.llm.run_llm_agent_evaluation --config configs/llm_agent_eval.yaml
```

Run the approved real OpenAI expanded path:

```powershell
.\myenv\Scripts\python.exe -m src.llm.run_llm_agent_evaluation --config configs/llm_agent_eval_openai_pilot_40.yaml
.\myenv\Scripts\python.exe -m src.llm.run_llm_agent_evaluation --config configs/llm_agent_eval_openai_final_80.yaml
```

The pilot writes to `reports/llm_explanations/openai_pilot_40/`. The final 80-case config writes to `reports/llm_explanations/` and currently prioritizes `inx_primary` plus `hrdataset_v14`. IBM performance, IBM attrition, and Employee Turnover are second-stage robustness regeneration targets and must not be framed as direct employee-performance validation.

Run the standalone agent audit from generated governed explanations:

```powershell
.\myenv\Scripts\python.exe -m src.agents.run_governance_audit --config configs/llm_agent_eval.yaml
```

Run chatbot guardrail evaluation:

```powershell
.\myenv\Scripts\python.exe -m src.chatbot.run_guardrail_eval --config configs/chatbot_guardrail_eval.yaml
```

Regenerate governance dashboards:

```powershell
.\myenv\Scripts\python.exe -m src.governance.gxair_score --config configs/governance_dashboard.yaml
.\myenv\Scripts\python.exe -m src.governance.statistical_reliability
```

Run tests:

```powershell
.\myenv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\myenv\Scripts\python.exe -m unittest discover -s tests -v
.\myenv\Scripts\python.exe -m pytest
.\myenv\Scripts\python.exe -m compileall src
```

## Output Files

LLM-agent evaluation:

```text
reports/llm_explanations/eval_case_manifest.csv
reports/llm_explanations/governed_explanations.jsonl
reports/llm_explanations/faithfulness_eval.csv
reports/llm_explanations/faithfulness_eval_summary.md
reports/llm_explanations/llm_agent_eval_summary.csv
reports/llm_explanations/llm_agent_eval_summary.md
```

Agent audits:

```text
reports/agent_audits/agent_audit_results.jsonl
reports/agent_audits/agent_audit_results.csv
reports/agent_audits/multi_agent_governance_audit.md
```

Chatbot evaluation:

```text
reports/chatbot_eval/unsafe_prompt_suite.csv
reports/chatbot_eval/safe_prompt_suite.csv
reports/chatbot_eval/guardrail_evaluation.csv
reports/chatbot_eval/guardrail_evaluation_summary.md
```

Governance dashboard:

```text
reports/governance_reports/gxair_component_dashboard.csv
reports/governance_reports/gxair_component_dashboard.md
reports/governance_reports/final_governance_readiness_report.md
```

Run registry:

```text
reports/research_log/run_registry.csv
reports/research_log/experiment_registry.csv
```

## Guardrails

The chatbot and LLM layer must not:

- make hiring, firing, promotion, compensation, discipline, or applicant-screening recommendations;
- rank employees for employment decisions;
- justify outcomes using sensitive attributes;
- claim the model is fair or unbiased;
- claim SHAP or counterfactuals are causal;
- give direct employee prescriptions;
- hide uncertainty, warnings, or human-review requirements;
- present full-feature leakage-risk models as deployable.

Safe behavior is to explain evidence, limitations, warnings, and allowed research interpretation.

## Current Limitations

- The default config remains dry-run/stub-labelled for reproducibility and must not be cited as manuscript-grade real LLM evidence.
- The latest expanded real OpenAI batch covers 80 INX + HRDataset_v14 cases. IBM and turnover related-task LLM regeneration remain second-stage robustness work.
- Automated faithfulness and guardrail tests do not replace human-subject evaluation.
- Component dashboard readiness can still be `not_ready` because proxy risk, actionability limits, and deployment validation blockers remain.
- The project remains research-grade decision support only, not a deployment-ready HR decision system.
