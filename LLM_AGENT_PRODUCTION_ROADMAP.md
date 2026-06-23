# LLM-Agent Production Roadmap

## Current Objective
Move from deterministic research-only LLM stubs to a real, API-backed LLM and multi-agent governance layer while preserving the XGBoost/XAI pipeline as the predictive core.

## Non-Negotiable Architecture

```text
XGBoost predicts
  -> XAI/fairness/calibration/counterfactual modules produce evidence
  -> deterministic audit tools verify evidence
  -> LLM agents interpret bounded evidence
  -> supervisor combines governance findings
  -> chatbot exposes safe explanations
```

The LLM must not predict employee performance. It must only interpret structured evidence.

## User Setup Required

1. Install LLM dependencies:

```bash
.\\myenv\\Scripts\\pip.exe install -r requirements.txt
```

2. Set an OpenAI API key in PowerShell:

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:HR_XAI_LLM_PROVIDER = "openai"
$env:HR_XAI_REQUIRE_REAL_LLM = "1"
$env:HR_XAI_OPENAI_MODEL = "gpt-5.4-mini"
```

3. Verify setup:

```bash
.\\myenv\\Scripts\\python.exe -m src.llm.check_llm_setup
```

4. Run a real LLM explanation:

```bash
.\\myenv\\Scripts\\python.exe -m src.llm.generate_governed_explanations --provider openai --require-real-llm --limit 5
```

5. Run LLM-assisted agents:

```bash
.\\myenv\\Scripts\\python.exe -m src.agents.run_llm_governance_audit --provider openai --require-real-llm
```

Use `gpt-5.4-mini` for development and guardrail evaluation. Switch to `gpt-5.5` only for final manuscript examples or high-quality governance artifacts after reviewing token logs.

Agents SDK runtime:

```bash
.\\myenv\\Scripts\\python.exe -m src.agents.run_llm_governance_audit --agent-runtime openai-agents --provider openai --require-real-llm
```

Cost estimate:

```bash
.\\myenv\\Scripts\\python.exe -m src.llm.cost_estimator --write-report
```

## Technical Decision 1: LLM Provider

### Option A: OpenAI API via structured outputs
Pros:
- Strong structured-output support with strict JSON Schema.
- Direct integration with the current Python codebase.
- Easy to add model/version metadata to experiment registry.
- Good fit for governed explanation generation.

Cons:
- Requires API key and paid API access.
- Model availability/cost depends on the account.
- External LLM behavior must be evaluated and logged.

Recommendation: Use this as the first production path.

### Option B: Local LLM via Ollama/vLLM
Pros:
- No external API data transfer.
- Better for privacy-sensitive demos.
- Lower marginal cost after setup.

Cons:
- More infrastructure work.
- Structured output reliability may be weaker.
- Model quality and latency depend on local hardware.

Recommendation: Add later if privacy/offline deployment becomes a requirement.

### Option C: Multi-provider abstraction with LiteLLM
Pros:
- Can switch between OpenAI, Anthropic, Gemini, local models.
- Useful for comparative experiments.

Cons:
- Adds another dependency layer.
- More provider-specific edge cases.
- Harder to make a clean Q3 prototype quickly.

Recommendation: Defer until OpenAI path is stable.

## Technical Decision 2: Agent Runtime

### Option A: Custom orchestrator over deterministic tools plus LLM synthesis
Pros:
- Matches the current repo.
- Easy to test.
- Keeps evidence provenance clear.
- Minimal dependency risk.

Cons:
- Does not provide built-in tracing, handoffs, or sessions.
- More custom code if workflows become complex.

Current implementation: started.

### Option B: OpenAI Agents SDK
Pros:
- Built for agents with tools, guardrails, handoffs, sessions, and tracing.
- Better long-term architecture for a professional agent workflow.
- Official OpenAI path for managed agentic workflows.

Cons:
- Adds dependency and learning curve.
- Requires API key for meaningful runtime tests.
- May require refactoring current agent classes into tool functions.

Decision: Adopted after researcher approval. The current implementation wraps deterministic audit tools as OpenAI Agents SDK function tools and uses structured Pydantic outputs for specialist and supervisor agents.

### Option C: LangGraph
Pros:
- Strong graph workflow control.
- Good for complex state machines and retry paths.

Cons:
- Heavier framework.
- More boilerplate.
- Less direct alignment with OpenAI structured-output examples.

Recommendation: Use only if the governance workflow becomes a complex graph.

## Immediate Implementation Status

- OpenAI structured-output client added.
- Runtime provider config added.
- Real-LLM setup checker added.
- LLM-assisted multi-agent orchestrator added.
- OpenAI Agents SDK runtime added with specialist tools and supervisor synthesis.
- Cost estimator added.
- Offline fallback remains only for tests and reproducibility.

## Next Planned Steps

1. Install dependencies and run a real OpenAI smoke test after API key is available.
2. Run OpenAI Agents SDK smoke test with `--agent-runtime openai-agents`.
3. Add persistent trace/log artifacts for every real LLM call.
4. Add retry/backoff and hard budget limits.
5. Add end-to-end Streamlit switch between offline, custom OpenAI, and Agents SDK modes.
6. Run full LLM faithfulness/guardrail evaluation with the real model.
