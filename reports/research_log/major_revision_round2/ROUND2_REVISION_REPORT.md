# Round 2 Scientific Revision Report

Status: **scientific implementation complete; exact claim digest approved for rewrite**

This report synthesizes the prespecified Round 2 implementation and scientific experiments. The user explicitly approved the exact Round 2 claim digest on 2026-09-09 for manuscript, bibliography, and reviewer-response rewriting; release and author/institution blockers remain separate.

## What the new evidence changes

Round 2 does not support one universally best model. It sharpens that conclusion in four ways: the selection objective changes many candidate choices and several metric leaders; aggregate ordinal metrics can conceal complete extreme-class failure; timing/information restrictions produce large performance changes under both fixed and retuned estimands; and HR conclusions depend on target construction, cross-validation design, and the metric being emphasized.

The resulting manuscript story should therefore prioritize conditional rankings and information contracts rather than a single performance winner. No Round 2 result establishes causality, prospective validity, fairness, target equivalence, or deployment readiness.

## R2-E1 — Selection-objective sensitivity

Macro-F1 selection and QWK selection used the same P3 feature policy, candidate registries, outer and inner folds, preprocessing, seeds, and 0.001 inclusive tie pool. Only the primary and tie-break selection metrics were reversed. The QWK arm required 60 outer refits and no new inner fits because the complete candidate evidence already retained both selection scores.

- Selected candidates changed in 37 of 60 model-by-fold decisions.
- The metric leader changed for 6 of 9 metrics.
- The exact six-model ordering changed for all 9 metrics.
- These are separate descriptive facts. They are not compressed into one binary “ranking depends” conclusion, and a lower-order permutation alone is not interpreted as material dependence.

The effects were model- and metric-specific. For cumulative-threshold XGBoost, QWK selection changed QWK from 0.554974 to 0.631887 and ordinal MAE from 0.314167 to 0.153333, while macro-F1 changed from 0.625459 to 0.591213 and rating-4 recall from 0.416667 to 0.000000. Nominal XGBoost became the QWK leader at 0.641778 and the ordinal-MAE leader at 0.152500, but its rating-4 recall was only 0.007576. Random Forest retained zero rating-4 recall in both regimes.

Authoritative evidence: [`SELECTION_OBJECTIVE_SENSITIVITY.md`](SELECTION_OBJECTIVE_SENSITIVITY.md) and [`selection_objective_sensitivity/`](selection_objective_sensitivity/).

## R2-E2 — Probability baseline and extreme-class reporting

The training-only empirical-prior comparator produced log loss 0.768298, multiclass Brier 0.431305, normalized RPS 0.116719, and confidence ECE 0.000000. The zero ECE is a known consequence of the nearly constant majority-class confidence matching aggregate accuracy; it is not evidence of perfect individual calibration. Log loss, Brier, and RPS remain necessary complementary views.

The expanded per-class table makes clear why aggregate ordinal metrics are insufficient. Under the canonical macro-F1-selected benchmark, Random Forest combined QWK 0.631678 and ordinal MAE 0.158333 with rating-4 precision, recall, and F1 all equal to zero. The supported conclusion is about metric behavior, not an untested causal explanation for class-4 failure.

Authoritative evidence: [`PROBABILITY_BASELINE_REPORT.md`](PROBABILITY_BASELINE_REPORT.md), [`PER_CLASS_EXTREME_CLASS_REPORT.md`](PER_CLASS_EXTREME_CLASS_REPORT.md), and [`selection_objective_sensitivity/`](selection_objective_sensitivity/).

## Timing and information sensitivity

The P3→P4 contrast is now a central result rather than a peripheral ablation. Under the fixed schedule, macro-F1 changed by -0.209352 and QWK by -0.358104. Under policy-specific retuning, the corresponding changes were -0.193673 and -0.330799. Ordinal MAE increased by 0.240000 under both estimands.

The P4→P5 step was smaller for macro-F1 but not uniformly smaller across ordinal criteria. Fixed-schedule macro-F1 changed by -0.065688 and QWK by -0.114989; retuned macro-F1 changed by -0.072697 and QWK by -0.176686. The retuned macro-F1 sequence was 0.894274 / 0.621021 / 0.427348 / 0.354651 for P0 / P3 / P4 / P5.

The manuscript-facing policy labels are frozen as Information-Rich Diagnostic, Leakage-Risk-Controlled, Direct-Sensitive-Attribute-Excluded, Primary Leakage-Aware, Prospective-Plausibility, and Strict Proxy-Reduced Prospective-Plausibility. P4 and P5 remain timestamp-unverified sensitivities; P3 is not labelled confirmed leakage, and no policy is called leakage-free.

Authoritative evidence: [`headline_policy_comparison.csv`](../major_revision_v3/phase1d_policy_retuning/headline_policy_comparison.csv), [`ROUND2_PLAN.md`](ROUND2_PLAN.md), and [`METHOD_REPRODUCIBILITY_TABLES/`](METHOD_REPRODUCIBILITY_TABLES/).

## Subgroup reporting

All 18 prespecified P3 gap rows are publication-ready: six attributes by macro-F1, QWK, and ordinal MAE, using the n≥30 support rule and retaining endpoints, eligible-group counts, declared-group counts, gap status, and exploratory interval fields. No cell was selected according to favorability.

Department had the largest reported gaps: macro-F1 0.217907, QWK 0.438824, and ordinal MAE 0.119283. Only 3 of 6 declared department groups were eligible for the QWK gap. These are support-dependent descriptive diagnostics, not fairness, discrimination, protected-class, or legal-compliance findings.

Authoritative evidence: [`SUBGROUP_MANUSCRIPT_SUMMARY.csv`](SUBGROUP_MANUSCRIPT_SUMMARY.csv).

## HR target construction and CV-design sensitivity

The retained three-class support is 31 / 243 / 37; the raw-order four-class support is 13 / 18 / 243 / 37. The formulations are different estimands, so scores are presented side by side and are not subtracted or called improvements.

For the three-class formulation, raw XGBoost mean macro-F1/QWK was 0.6531/0.5339 and sigmoid XGBoost was 0.6274/0.6044. For the four-class formulation, raw XGBoost was 0.5847/0.6288 and sigmoid XGBoost was 0.5481/0.6621. Within each formulation, calibration effects remained metric-specific.

Across the 14 retained-mapping CV-design comparisons, 11 canonical 10×5 point estimates fell inside the observed repeated-5×5 ranges and 3 fell outside: raw-XGBoost macro-F1 and sigmoid-XGBoost Brier and RPS. The ranges are descriptive and are not confidence intervals or equivalence tests.

Authoritative evidence: [`HR_MAPPING_SENSITIVITY_REPORT.md`](HR_MAPPING_SENSITIVITY_REPORT.md) and the validated Phase 3A aggregate package.

## R2-E3 — HR target-alias matched-sample sensitivity

The historical 311-row canonical result remains unchanged. Existing canonical OOF predictions were restricted fit-free to the 309 rows without the two audited target-alias disagreements. The exclusion/refit arm was independently selected and fitted after excluding those rows and evaluated on the identical 309-row identities.

The primary 309↔309 comparison changed macro-F1 by +0.004210, balanced accuracy by -0.002102, QWK by -0.000339, ordinal MAE by +0.000000, RPS by +0.000024, log loss by +0.010803, Brier by +0.000668, and confidence ECE by -0.011197. Candidate selection changed in 2 of 5 outer folds. The mixed, small directions do not establish equivalence or target validity.

The separate fit-free 311→309 sample-removal contrast changed macro-F1 by +0.005357, QWK by -0.003035, ordinal MAE by -0.002071, and RPS by -0.000636. None of those changes is attributed to refitting.

Authoritative evidence: [`HR_MAPPING_SENSITIVITY_REPORT.md`](HR_MAPPING_SENSITIVITY_REPORT.md) and [`hr_target_alias_sensitivity/`](hr_target_alias_sensitivity/).

## Methods and literature readiness

Tables S1–S3 now enumerate all 28 governed features, preprocessing rules, six trained-model registries, candidate grids, selection/tie/failure rules, CV identities, seed identities, OOF coverage, and reuse/refit boundaries. The Methods notes explicitly describe proportional-odds assumptions and cumulative-threshold XGBoost's independent binary threshold fits, rowwise nonincreasing PAVA projection, and ordered-probability reconstruction.

The additive Literature v4 package verifies nine core method references against primary publisher or proceedings sources. The novelty boundary is fixed: “The contribution is not the novelty of any single component, but their operational integration into a traceable evaluation contract.” The package does not support a world-first or exhaustive-review claim.

Authoritative evidence: [`METHOD_REPRODUCIBILITY_TABLES/`](METHOD_REPRODUCIBILITY_TABLES/) and [`LITERATURE_V4/`](LITERATURE_V4/).

## Claim freeze and remaining gate

The Round 2 matrix preserves all 45 historical claims and classifies every row as retained, modified, superseded, new, or prohibited. It contains 129 review rows, including 99 exact numerical rows; 117 rows are active candidates for later drafting. Every numerical row resolves to one exact aggregate source row/value/hash, and every narrative or prohibited row resolves to a hash-bound text anchor.

The cross-source audit revalidated the historical Phase 5A digest, matched all 32 historical numerical claims to the Phase 5B comparison table, rehashed all 62 historical Phase 5B manifest-bound files, and independently resolved every active Round 2 numerical claim. Historical Phase 5A and Phase 5B bytes remain unchanged.

Approved claim-set SHA-256: `751208d036597461606bd02d68bfa9e642a4aa5ecf5df2df3fd7e46b99084aea`

Review artifacts: [`ROUND2_CLAIM_MATRIX.md`](ROUND2_CLAIM_MATRIX.md), [`RESULTS_COMPARISON_ROUND2.csv`](RESULTS_COMPARISON_ROUND2.csv), [`ROUND2_SOURCE_REGISTER.csv`](ROUND2_SOURCE_REGISTER.csv), and [`ROUND2_CROSS_SOURCE_AUDIT.md`](ROUND2_CROSS_SOURCE_AUDIT.md).

The explicit digest-specific decision authorizes editing `manuscript/mdpi_information/main.md`, regenerating `main.tex`, updating `references.bib`, and drafting the Round 2 reviewer response under this sole scientific boundary. It does not authorize release, tag, DOI, dataset redistribution, a software-licence decision, ethics/IRB wording, or author declarations.
