# Reviewer Response Draft

This draft is organized around the major scientific concerns addressed by the revision. Replace the concern labels with the reviewers' verbatim comments and add final page/line references after journal-template typesetting.

## General response

We thank the reviewers for asking us to clarify the estimand, strengthen model comparison and uncertainty analysis, separate leakage control from feature ablation, and constrain the interpretation of explanations, calibration, subgroup results, and external evidence. We rewrote the manuscript from title through declarations. The revised study is now an ordinal XAI audit protocol governed by a frozen, hash-bound claim matrix; the previous explanation-system narrative has been removed from the scientific contribution.

## Concern 1 — The task and prediction time are unclear

**Response.** We agree that the previous wording implied more temporal certainty than the data support. Section 3.1 now defines a cross-sectional sensitivity estimand. Section 3.3 presents six prespecified information policies and states that feature and decision timestamps are unavailable. P3 is described as leakage-aware, while P4 is only prospective-plausibility sensitivity. We make no prospective performance claim.

## Concern 2 — The model comparison is too narrow

**Response.** We expanded the benchmark to nine systems evaluated on the same exactly-once OOF sample under P3. Section 4.1 and Table 3 report classification, ordinal, and probability metrics. The results are metric-specific: cumulative-threshold XGBoost leads macro-F1 and balanced accuracy, Random Forest leads QWK and ordinal MAE, LightGBM leads normalized RPS, and nominal XGBoost leads raw log loss. We no longer label one system universally best.

## Concern 3 — A single CV run does not show stability

**Response.** Section 3.5 now describes five fixed 5×5 nested-CV repetitions in which every trained system is refitted. Section 4.2 reports means, sample SDs, winner frequencies, and rank correlations. We explicitly label repetition ranges as descriptive rather than confidence intervals.

## Concern 4 — Feature-policy effects are confounded with retuning

**Response.** Section 3.6 separates a matched fixed-schedule estimand from independent policy-specific retuning. Table 5 reports both. We state that retuned contrasts combine information access with model selection and that neither estimand is causal. The mixed P2 and P5 changes are reported rather than presenting a monotonic policy story.

## Concern 5 — SHAP plots alone do not validate explanations

**Response.** Section 3.7 now requires exact prediction–fold–model identity, grouped raw-margin TreeSHAP, ranking stability across folds/seeds/resamples, and deletion comparisons against 20 random repetitions. Section 4.4 reports both stability and deletion behavior. We state that pairs are dependent, masking may be out of distribution, and results are noncausal model diagnostics.

## Concern 6 — Probability calibration is incompletely assessed

**Response.** Section 3.8 now describes training-only cross-fitted sigmoid calibration. Section 4.5 reports log loss, Brier, top-label ECE, classwise ECE, cumulative ECE, and RPS. We highlight the adverse result: top-label ECE worsened by 0.0044 even while several other point estimates improved. The conclusion is metric- and event-specific.

## Concern 7 — Removing direct group variables does not resolve proxy risk

**Response.** Section 3.9 and Section 4.6 distinguish support-aware subgroup gaps, P3 refitting without JobRole, marginal/department-conditional JobRole perturbations, and separate department reconstruction. We explicitly avoid fairness, discrimination, causal, and legal conclusions. The 0.9792 versus 0.2908 reconstruction contrast is not treated as proof of department use by the performance model.

## Concern 8 — External validation is overstated

**Response.** We narrowed the core external claim to HRDataset_v14 and renamed it an independently trained mapped-target protocol replication. Section 3.10 states that no INX model was transported and that feature space, target semantics, parameters, and population differ. Section 4.7 reports the retained mapping and five-repetition results without asserting construct equivalence or unqualified external validity.

## Concern 9 — Data quality and provenance are insufficiently transparent

**Response.** Section 3.2 and Section 4.7 add aggregate missingness, duplicate, identifier, temporal, consistency, schema, and target-support audits. No source value was silently repaired. The Data Availability Statement now excludes raw redistribution because authoritative source-to-byte and rights chains remain incomplete.

## Concern 10 — Novelty and reproducibility claims are too broad

**Response.** Section 2.5 and Section 5.5 bound positioning to a frozen, source-verified 25-work set. We do not claim an exhaustive review or universal novelty. The contribution is the shared evidence contract, and every reported number is tied to an approved claim ID, source selector, source hash, and mandatory qualifier. The exact claim-set digest is reported in Section 3.11.

## Concern 11 — Ethics and author declarations are incomplete

**Response.** We did not infer declarations from the public or apparently secondary nature of the data. The manuscript now carries visible placeholders for institutional review, consent, contributions, funding, conflicts, and AI-use wording. These fields require institution- or author-approved text before submission.

