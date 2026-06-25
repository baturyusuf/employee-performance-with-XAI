# Faithfulness Evaluation Summary

- n_cases: 40
- faithfulness_pass_rate: 1.0
- mean_faithfulness_score: 100.0
- unsupported_claim_rate: 0.0
- forbidden_claim_rate: 0.0
- missing_warning_rate: 0.0
- parsing_success_rate: 1.0

## Per-Dataset Breakdown

| dataset_name | n_cases | faithfulness_pass_rate | mean_faithfulness_score | missing_warning_rate |
| --- | --- | --- | --- | --- |
| hrdataset_v14 | 20 | 1.0 | 100.0 | 0.0 |
| inx_primary | 20 | 1.0 | 100.0 | 0.0 |

## Examples of Failures

No faithfulness failures detected.

## Evidence Boundary

This faithfulness summary is manuscript-grade real LLM evidence only when paired with a `run_mode=real` summary and `real_llm_used=True`. Stub/dry-run outputs from `offline_stub_llm` are reproducibility artifacts and must not be cited as real LLM evidence.
