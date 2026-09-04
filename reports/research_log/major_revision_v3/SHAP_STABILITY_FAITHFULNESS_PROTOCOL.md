# SHAP Aggregation, Stability, and Faithfulness Protocol — v3

Date: 2026-09-04

Status: complete clean-commit execution, independent validation, and governed compact evidence

## Separated questions

Phase 2A preserves the canonical-v2 exact-model outer-fold SHAP analysis and separates three questions:

1. **Aggregation validity:** does the implemented multiclass/one-hot grouping preserve the additive raw-margin accounting that TreeSHAP provides?
2. **Ranking stability:** how similar are global grouped-SHAP rankings when only the XGBoost seed changes, or when each outer-training partition is reduced to a deterministic stratified 80% subsample?
3. **Model-level deletion faithfulness:** does masking the per-sample SHAP-ranked feature families change the exact prediction-producing model's output more than equally sized random feature sets?

Stability is not faithfulness. Neither is causal feature evidence, human-usefulness evidence, fairness evidence, or a deployment validation.

## Exact grouped multiclass SHAP algorithm

The reference model is the canonical-v2 P3 XGBoost model that produced each sample's held-out prediction. No full-data surrogate is used. The exact implementation is:

1. transform the outer-test rows with the fitted outer-training preprocessor;
2. call SHAP 0.51.0 `TreeExplainer` with no explicit background, `feature_perturbation="tree_path_dependent"`, `model_output="raw"`, and additivity checking enabled;
3. normalize the library output to `(sample, class, transformed feature)` for ordered classes 2/3/4;
4. sum **signed** transformed-column SHAP values once within each raw feature family;
5. verify per-sample/per-class transformed and grouped sums are equal;
6. verify expected value plus transformed SHAP sum reproduces the XGBoost raw class margin within absolute error `1e-5`;
7. only after grouping, take the absolute grouped value, average across classes, and then average across exactly-once OOF samples to obtain global importance.

Thus one-hot columns are never counted as independent raw features, absolute values are not summed before family grouping, and the explanation is in raw-margin rather than probability space. The no-background tree-path-dependent reference uses the fitted trees' path statistics; it is not an interventional or conditional feature distribution.

## Seed stability

The canonical seed-42 ten-fold model set is the reference and is not refitted. Five new model sets use seeds 1044, 2044, 3044, 4044, and 5044. Every run holds fixed the exact canonical outer assignments, full outer-training rows, P3 features, fold-specific selected XGBoost candidate, preprocessing scope, and model registry. Only the estimator seed changes. This requires 50 new fits.

Each seed run produces a 1,200-row exactly-once OOF grouped-SHAP ranking over the same 20 feature families. All 15 run pairs report top-5/top-10/top-15 Jaccard and all-feature Spearman values.

## Resampling stability

Five deterministic repetitions use base seeds 11042, 12042, 13042, 14042, and 15042. Within each canonical outer-training partition, a stratified 80% sample without replacement is drawn using `base seed + outer fold`; the canonical outer-test partition is unchanged. The fold-specific candidate schedule and estimator seed 42 are held fixed, so this sensitivity targets outer-training membership rather than re-tuning. It requires 50 new fits.

Every repetition again explains all 1,200 common outer-test samples exactly once. All 10 repetition pairs report the same Jaccard/Spearman grid. The existing 45 dependent outer-fold pairs remain reported as the historical reference. None of these overlapping pair values is treated as independent or converted to a confidence interval, and no strong “robust explanation” claim is authorized.

## Model-level deletion faithfulness

Faithfulness uses only the exact canonical-v2 prediction-producing fold models and their per-sample grouped SHAP values. For each OOF sample, feature families are ranked by absolute SHAP for the model's original predicted class. The top 1, 3, and 5 families are replaced by references learned from that sample's outer-training partition: median for numeric features and deterministic sorted-first mode for categorical features.

The primary response is the drop in the original predicted-class probability; raw-margin drop is secondary. A normalized probability-drop deletion AUC uses deleted-count points 0/1/3/5 at fractions 0/0.2/0.6/1.0. Twenty prespecified random repetitions select nested, equally sized feature sets without replacement per sample. Comparisons are guided means minus the distribution of random-repetition means; no p-value or significance claim is produced.

Median/mode masking is deterministic and leakage-isolated, but it can create unrealistic feature combinations outside the observed joint distribution. The analysis therefore supports only bounded model-level explanation faithfulness. It does not establish causal effects, realistic interventions, recommended personnel actions, human comprehension, or usefulness.

## Computational and publication boundaries

The complete run contains 100 new fits, 110 model/fold explanation evaluations including the reused reference, 3,600 guided sample perturbations, and 72,000 random sample perturbations. Scientific execution is offline, single-thread bounded, source/hash bound, and atomically published only from a clean unchanged Git state.

Fold feature rows, resampling-membership hashes, sample SHAP/deletion rows, fitted models, raw data, and employee identifiers remain under the ignored local run root. The governed compact export includes only the aggregation receipt, aggregate stability and faithfulness summaries/contrasts, deletion-AUC summary, and provenance.

## Verified implementation diagnostics

Fit-free preflight validated the 1,200-row/20-feature P3 frame, all ten exact canonical models, eight XGBoost candidates, and the 100-fit scope with zero actual fits/network/API calls. Re-explaining the exact canonical models reproduced the historical global importance to maximum absolute error `9.1e-17` and identical ranks; observed maximum raw-margin additivity error was `5.74e-6`, and grouped-family sum error was zero.

One full ten-fold seed-1044 diagnostic and one ten-fold 80% resample-11042 diagnostic completed with training counts 1,080 and 864 per fold. Their maximum raw-margin errors were `5.41e-6` and `5.57e-6`, with zero grouped-sum error. A noncanonical deletion diagnostic using two of the twenty random baselines generated 3,600 guided and 7,200 random perturbation records; it verified the output path only and is not admissible scientific evidence.

## Complete execution and bounded results

Run `phase2a_v3_20260904T073008Z_6e52de7` executed from clean pushed commit `6e52de76f7e486985cbc2b32a53b2554c1c6f6c1`. It completed all 100 new fits and 110 model/fold explanations, and generated 3,600 guided plus 72,000 random perturbation rows with zero network or paid-API calls. Maximum raw-margin additivity error was `6.5115e-6`; grouped-sum error was zero.

Mean top-5 Jaccard was 1.0000 for canonical outer-fold pairs, fixed-fold model seeds, and 80% outer-training resamples. Their mean top-10 Jaccard values were 0.7945, 0.9394, and 0.8909; mean top-15 values were 0.8250, 0.9083, and 0.9250; and mean all-feature Spearman values were 0.9066, 0.9945, and 0.9847. These 45/15/10 pairs are dependent descriptive comparisons, not independent observations or confidence-interval units.

SHAP-guided minus random-repetition-mean probability drops were +0.2676, +0.2481, and +0.2063 after deleting 1, 3, and 5 feature families. Mean probability-drop deletion AUC was 0.2720 for guided deletion and 0.0512 across the 20 random-repetition means, a descriptive difference of +0.2208. This supports only model-level perturbation faithfulness under the stated masking scheme; it does not validate realistic interventions, causal importance, fairness, human usefulness, or deployment.

The independent validator rehashed the 14-file local package, rebound the contract/sources/implementation to generation Git bytes, reconstructed all training memberships and rankings, replayed deterministic random orders, and recomputed every stability, faithfulness, frequency, contrast, and AUC table. The publication-safe aggregate package is `phase2a_shap_stability_faithfulness/`; its manifest SHA-256 is `4cd7bcace03d556e2bd27eea4dca87143745914ddd56136476edc6c662e44481`.
