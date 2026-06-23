# LLM Cost Estimate

Prices are approximate and based on OpenAI standard short-context text token pricing per 1M tokens at the time this file was generated.
Assumption: one governed explanation call, six specialist agent calls, and one supervisor call per case.

| scenario | model | cases | calls | avg input/call | avg output/call | estimated USD |
|---|---:|---:|---:|---:|---:|---:|
| small_eval | gpt-5.4-mini | 5 | 40 | 8000 | 600 | $0.3480 |
| small_eval | gpt-5.4 | 5 | 40 | 8000 | 600 | $1.1600 |
| small_eval | gpt-5.5 | 5 | 40 | 8000 | 600 | $2.3200 |
| paper_eval | gpt-5.4-mini | 50 | 400 | 8000 | 600 | $3.4800 |
| paper_eval | gpt-5.4 | 50 | 400 | 8000 | 600 | $11.6000 |
| paper_eval | gpt-5.5 | 50 | 400 | 8000 | 600 | $23.2000 |
| dashboard_demo | gpt-5.4-mini | 100 | 800 | 8000 | 600 | $6.9600 |
| dashboard_demo | gpt-5.4 | 100 | 800 | 8000 | 600 | $23.2000 |
| dashboard_demo | gpt-5.5 | 100 | 800 | 8000 | 600 | $46.4000 |

Cost controls:
- Start with `gpt-5.4-mini` for development and guardrail evaluation.
- Use `gpt-5.5` only for final high-quality examples or manuscript artifacts.
- Keep `--limit` small during iteration.
- Do not run full case sweeps until token logging is reviewed.
