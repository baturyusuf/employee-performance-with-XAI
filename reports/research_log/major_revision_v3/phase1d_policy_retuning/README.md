# Phase 1D Fixed-Schedule and Independently Retuned Policies — Compact Evidence

Source run: `phase1d_v3_20260904T063324Z_823c848`

This package contains aggregate policy evidence only. Employee-level OOF predictions, fold assignments, fold metrics, candidate-search rows, raw data, and fitted models are deliberately excluded.

## Headline results

Raw differences are independently retuned minus fixed primary-schedule values. For ordinal MAE, a negative raw difference is an improvement.

| Policy | Features | Fixed Macro-F1 | Retuned Macro-F1 | Δ Macro-F1 | Fixed QWK | Retuned QWK | Δ QWK | Fixed balanced accuracy | Retuned balanced accuracy | Δ balanced accuracy | Fixed ordinal MAE | Retuned ordinal MAE | Δ ordinal MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 INFORMATION_RICH_DIAGNOSTIC | 26 | 0.8914 | 0.8943 | +0.0029 | 0.8496 | 0.8529 | +0.0033 | 0.8982 | 0.8867 | -0.0115 | 0.0775 | 0.0733 | -0.0042 |
| P1 LEAKAGE_CONTROLLED | 24 | 0.6279 | 0.6396 | +0.0117 | 0.5814 | 0.5892 | +0.0078 | 0.6400 | 0.6520 | +0.0119 | 0.2308 | 0.2300 | -0.0008 |
| P2 GOVERNANCE_CONTROLLED | 21 | 0.6150 | 0.6335 | +0.0185 | 0.5569 | 0.5892 | +0.0324 | 0.6263 | 0.6437 | +0.0174 | 0.2483 | 0.2217 | -0.0267 |
| P3 PRIMARY_LEAKAGE_AWARE | 20 | 0.6210 | 0.6210 | +0.0000 | 0.5676 | 0.5676 | +0.0000 | 0.6360 | 0.6360 | +0.0000 | 0.2433 | 0.2433 | +0.0000 |
| P4 STRICT_PROSPECTIVE | 13 | 0.4117 | 0.4273 | +0.0157 | 0.2095 | 0.2368 | +0.0273 | 0.4605 | 0.4854 | +0.0249 | 0.4833 | 0.4833 | +0.0000 |
| P5 STRICT_PROXY | 6 | 0.3460 | 0.3547 | +0.0087 | 0.0945 | 0.0601 | -0.0344 | 0.3777 | 0.3706 | -0.0071 | 0.5958 | 0.5517 | -0.0442 |

## Bounded interpretation

- P3 is an exact replay control: its fixed and retuned values are identical because independent P3 tuning reproduces the canonical primary fold-specific candidate schedule and predictions.
- Retuning raises macro-F1 for P0, P1, P2, P4, and P5; the largest point difference is P2 (+0.0185). These are descriptive point differences, not confidence intervals or significance tests.
- The direction is not uniformly favorable across criteria. P0 balanced accuracy changes by -0.0115; P5 QWK changes by -0.0344 and balanced accuracy by -0.0071, even though its macro-F1 and ordinal MAE improve.
- P0 retains outcome-proximal/timing-risk fields and is an information-rich diagnostic upper bound, not a deployable policy. Its high scores cannot be interpreted as prospective performance.
- P4 is a prospective-plausibility sensitivity under timestamp-unverified cross-sectional data, not prospective validation. P5 removes declared organisational proxies but does not prove absence of residual proxies or fairness.
- Fixed-schedule contrasts isolate model/schedule control more tightly; retuned contrasts combine feature access with policy-specific model selection. Neither is a causal feature or retuning effect.
- No universally best policy, leakage-free system, fairness result, or deployment-ready HR decision system is identified.

## Files

- `aggregate_metrics.csv`: all 16 metrics for both estimands and six policies.
- `metric_comparison.csv`: fixed, retuned, raw-difference, and direction-aligned values for every metric.
- `headline_policy_comparison.csv`: macro-F1, QWK, balanced accuracy, and ordinal MAE summary.
- `selected_candidate_frequency.csv`: policy-specific retuned candidate frequencies without fold or employee rows.
- `provenance_receipt.json` and `manifest.json`: independent validation, immutable source identities, exclusions, and byte hashes.
