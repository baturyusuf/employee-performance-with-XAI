# Experiment Summary

## Core design

The primary estimand is ordinal classification of the recorded INX `PerformanceRating` under the P3 information policy. The table has 1,200 records and observed support 194/874/132 for labels 2/3/4. Feature and decision timestamps are unavailable, so every result is a cross-sectional sensitivity result.

## Phase-level evidence

| Phase | Analysis | Headline result | Required boundary |
| --- | --- | --- | --- |
| 1B | Nine-system 10×5 nested OOF benchmark | Macro-F1 leader cumulative-threshold XGBoost 0.6255; QWK leader Random Forest 0.6317 | Metric-specific, P3-conditional ranking |
| 1C | Five repeated 5×5 nested-CV designs | XGBoost/LightGBM mean macro-F1 0.6288/0.6249; Random Forest wins QWK 5/5 | Repetition ranges are not CIs |
| 1D | Fixed versus retuned P0–P5 | P2 macro-F1 +0.0185; P5 QWK −0.0344; P0 retuned macro-F1 0.8943 | Descriptive, noncausal policy estimands |
| 2A | Exact-fold SHAP stability and deletion | Seed top-5 Jaccard 1.0000; resample Spearman 0.9847; top-1 deletion contrast +0.2676 | Noncausal; dependent pairs; masking limits |
| 2B | Extended calibration | Log loss 0.5515→0.4556; top-label ECE worsens by 0.0044 | Event- and metric-specific |
| 2C | Subgroup and proxy diagnostics | 10.75% prediction changes without JobRole; reconstruction 0.9792→0.2908 | No fairness/discrimination/causal conclusion |
| 3A | HRDataset_v14 replication | Raw macro-F1 0.6531; sigmoid QWK 0.6044 | Independently trained mapped-target replication |
| 3B | Aggregate data quality | INX 1200 rows; HRDataset 215 effective missing cells and 3 rules/6 occurrences | Declared-rule audit, no source repair |
| 4A | Source-verified positioning | Complete shared contract absent from selected 25 works | Bounded set, not exhaustive novelty |
| 4B | Release and declarations audit | Raw rights, ethics, author fields, licence, history, release, DOI unresolved | No inference from public presence |

## Model comparison interpretation

No system leads every metric. Cumulative-threshold XGBoost leads macro-F1 and balanced accuracy, Random Forest leads QWK and ordinal MAE, LightGBM leads normalized RPS, and nominal XGBoost leads raw log loss. Nominal XGBoost remains the XAI/calibration reference because it was prespecified, not because it is universally superior.

## Replication interpretation

HRDataset_v14 repeats the protocol with different features, mapping, parameters, and population. No INX model is transported. The analysis therefore tests protocol portability under a mapped rating target, not locked-model external validity or construct equivalence.

