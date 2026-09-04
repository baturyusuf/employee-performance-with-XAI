# Phase 2A SHAP Stability and Model-Level Faithfulness — Compact Evidence

Source run: `phase2a_v3_20260904T073008Z_6e52de7`

This package contains only contract-approved aggregate evidence. Per-sample SHAP values, perturbation rows, fold/resample memberships, raw data, and fitted models are deliberately excluded.

## Ranking stability

| Comparison | Pairs | Top-5 Jaccard | Top-10 Jaccard | Top-15 Jaccard | All-feature Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| Canonical outer-fold pairs | 45 | 1.0000 | 0.7945 | 0.8250 | 0.9066 |
| Model seeds | 15 | 1.0000 | 0.9394 | 0.9083 | 0.9945 |
| 80% outer-train resamples | 10 | 1.0000 | 0.8909 | 0.9250 | 0.9847 |

The 45 canonical fold pairs, 15 model-seed pairs, and 10 resampling pairs share data and/or protocol components. They are not independent observations, so no confidence interval or significance claim is attached to these descriptive summaries.

## Deletion faithfulness

| Deleted features | Guided probability drop | Random-repetition mean | Guided − random |
| ---: | ---: | ---: | ---: |
| 1 | 0.2883 | 0.0208 | +0.2676 |
| 3 | 0.3097 | 0.0616 | +0.2481 |
| 5 | 0.3080 | 0.1016 | +0.2063 |

The mean probability-drop deletion AUC is 0.2720 for SHAP-guided deletion and 0.0512 across the 20 random-repetition means (difference +0.2208). These are descriptive model-level perturbation results, not inferential tests.

## Interpretation boundaries

- Global importance uses TreeSHAP raw margins, signed grouping from encoded columns to raw feature families, then absolute values and averaging across classes and exactly-once OOF samples.
- Stability and faithfulness are different properties. Ranking agreement alone does not establish faithfulness; deletion behavior does not establish explanation robustness.
- Median/mode masking can create out-of-distribution hybrid records. The deletion results therefore diagnose this fitted model under this masking intervention, not real employee outcomes.
- Model attribution is not a causal feature effect, prescriptive HR advice, fairness evidence, prospective validation, or evidence of human explanation usefulness.
- The evidence does not justify claims of a leakage-free or deployment-ready HR decision system.

## Files

- `aggregation_receipt.json`: SHAP library, output-space, axis, grouping, averaging, and additivity details.
- `stability_summary.csv`: top-5/10/15 Jaccard and all-feature Spearman summaries.
- `faithfulness_summary.csv`: probability/margin deletion summaries by method, repetition, and deletion count.
- `faithfulness_contrasts.csv`: guided-minus-random descriptive contrasts for probability and raw margin.
- `deletion_auc_summary.csv`: sample-aggregated probability-drop deletion AUC by method and repetition.
- `provenance_receipt.json` and `manifest.json`: independent validation, source identities, exclusions, and byte hashes.
