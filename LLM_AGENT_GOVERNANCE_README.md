# LLM-Agent Governance README

## Architecture

```text
XGBoost prediction
  -> grouped SHAP / fairness / calibration / leakage / counterfactual evidence
  -> structured evidence schema
  -> governed explanation generator
  -> multi-agent governance audit
  -> guardrailed chatbot
  -> governance reports and G-XAIR dashboard
```

The LLM is not used as a performance predictor. It only interprets structured evidence produced by the existing XAI and Responsible AI pipeline. The production path is OpenAI API-backed structured output; the offline stub is retained for tests and reproducibility.

## Modules

- `src.llm.evidence_schema`: dataclass-based structured evidence models and report loaders.
- `src.llm.openai_client`: OpenAI structured-output client for real LLM-backed governed explanations.
- `src.llm.check_llm_setup`: setup checker for SDK/API-key readiness.
- `src.llm.governed_explainer`: converts evidence into governed natural-language explanations.
- `src.llm.offline_stub_llm`: deterministic offline explanation generator for tests and reproducibility.
- `src.llm.faithfulness_checker`: rule-based checker for unsupported metrics, forbidden claims, missing warnings, and unsafe language.
- `src.agents.*`: leakage, fairness/proxy, calibration, SHAP stability, counterfactual actionability, explanation compliance, and supervisor audit agents.
- `src.chatbot.*`: report retrieval, guardrails, guarded response engine, and CLI/optional Streamlit entry point.
- `src.governance.gxair_score`: transparent multi-component G-XAIR dashboard with LLM governance compliance.
- `src.governance.governance_report`: final LLM-agent research summary generator.

## How To Run

Use the project Python environment. On this Windows workspace:

```bash
.\myenv\Scripts\pip.exe install -r requirements.txt
.\myenv\Scripts\python.exe -m src.llm.check_llm_setup
.\myenv\Scripts\python.exe -m src.llm.generate_governed_explanations --limit 5
.\myenv\Scripts\python.exe -m src.agents.run_governance_audit
.\myenv\Scripts\python.exe -m src.agents.run_llm_governance_audit --agent-runtime openai-agents --provider openai --require-real-llm
.\myenv\Scripts\python.exe -m src.llm.cost_estimator --write-report
.\myenv\Scripts\python.exe -m src.chatbot.evaluate_chatbot
.\myenv\Scripts\python.exe -m src.llm.evaluate_llm_governance
.\myenv\Scripts\python.exe -m src.governance.gxair_score
.\myenv\Scripts\python.exe -m src.governance.governance_report
```

For a real LLM run, create `.env` from `.env.example` or set these variables:

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:HR_XAI_LLM_PROVIDER = "openai"
$env:HR_XAI_REQUIRE_REAL_LLM = "1"
$env:HR_XAI_OPENAI_MODEL = "gpt-5.4-mini"
```

Generic command examples:

```bash
python -m src.llm.generate_governed_explanations --limit 5
python -m src.agents.run_governance_audit
python -m src.chatbot.chatbot_app --question "Why are full-feature models not deployable?"
python -m src.governance.gxair_score
```

## How To Evaluate

```bash
.\myenv\Scripts\python.exe -m src.llm.evaluate_llm_governance
.\myenv\Scripts\python.exe -m unittest discover -s tests -v
```

The automatic evaluation covers faithfulness, forbidden-claim detection, missing-warning behavior, unsafe-prompt refusal, deterministic consistency, rule-based agent agreement, and missing-evidence behavior. It is not a substitute for human-subject evaluation.

## Sample Outputs

- `reports/llm_explanations/governed_explanation_examples.md`
- `reports/agent_audits/multi_agent_governance_audit.md`
- `reports/chatbot_eval/guardrail_evaluation.md`
- `reports/governance_reports/gxair_llm_agent_dashboard.csv`
- `reports/governance_reports/final_llm_agent_research_summary.md`
- `reports/llm_explanations/llm_cost_estimate.md`

## Known Limitations

The offline LLM is deterministic and template-based and should not be treated as the production LLM. Real OpenAI-backed runs require `OPENAI_API_KEY` and must pass the same faithfulness and guardrail checks before being used in reported results. The chatbot retrieves local project evidence only and must not answer from general HR intuition. The evidence is based on a public cross-sectional dataset, so causal and deployment claims are not supported.
