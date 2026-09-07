# Subgroup Sensitivity and Proxy-Use Diagnostics — v3

Date: 2026-09-04

Status: frozen design, exact clean-commit execution, and independent validation complete; governed compact publication pending

## Scope and source evidence

Phase 2C extends the canonical v2 subgroup/proxy package without changing it. It consumes the exact 1,200-row raw, uncalibrated OOF predictions for three already-fitted policy systems: the department-inclusive leakage-controlled comparator, the P3 primary leakage-aware system, and the exact canonical P3-minus-`EmpJobRole` comparator. The last system is not v3 P5 because it does not apply P4’s timing-uncertain exclusions or P5’s full proxy exclusions.

The stage also loads the exact ten persisted P3 XGBoost outer-fold models. It does not refit a performance model, a department-reconstruction model, or a calibrator. Every input table, fold contract, model index, transformed-feature lineage, and existing reconstruction interval is bound by SHA-256. Execution is offline and makes no paid API calls.

## Complete subgroup grid

The audit covers `Age`, `Gender`, `MaritalStatus`, `BusinessTravelFrequency`, `EmpDepartment`, and `EducationBackground` for all three systems. Age uses fixed right-closed bins `18–29`, `30–39`, `40–49`, `50–59`, and `60+`, with declared edges `[17, 29, 39, 49, 59, 200]`. The complete table repeats the analysis at minimum group sizes 20, 30, and 50; unsupported rows remain visible with missing estimates and an explicit reason.

Each supported group reports macro-F1, balanced accuracy, quadratic-weighted kappa, ordinal MAE, recall for classes 2/3/4, multiclass Brier score, and log loss. Macro-F1 uses the three declared labels, assigning zero to an undefined class-specific F1 component. Balanced accuracy and QWK require at least ten true observations from every declared class. Each class-recall row requires at least ten true observations for its class. The other metrics require the declared total group-size threshold. These rules are fixed before calculating gaps.

For every system, threshold, attribute, and metric, the reported disparity is the maximum minus the minimum among eligible groups. All 486 declared cells are retained, including non-estimable cells. The analysis never promotes only the largest observed gap: the identity of the minimum and maximum groups is descriptive and affected by selection over many groups and metrics.

## Multiplicity-aware exploratory uncertainty

The P3 primary gap family receives 5,000 deterministic employee-level bootstrap repetitions stratified jointly by outer fold and true class. Group eligibility is fixed from the complete OOF data before resampling. Each cell receives a percentile 95% interval. A second exploratory simultaneous interval uses the 95th percentile of the maximum absolute standardized bootstrap deviation across the complete estimable 162-cell P3 family, including all threshold-sensitivity rows.

This construction addresses the declared familywise multiplicity more directly than selecting one pointwise interval. It remains exploratory: repetitions condition on the observed employees, fitted models, fold assignments, support rules, and selected feature policies. They do not include model-training instability and do not turn the audit into a confirmatory fairness test.

## Reconstructability versus performance-model dependence

The existing department reconstruction results answer whether department information can be recovered from a feature space. They do not show that the performance model used department information. Phase 2C therefore preserves the six reconstruction intervals and three paired differences as a distinct evidence table with this limitation attached.

Performance dependence is examined in two separate ways:

1. Exact paired P3 versus P3-minus-`EmpJobRole` OOF predictions are compared overall and within every department. The diagnostics include signed class-probability changes, total variation, maximum absolute probability change, true-class probability change, argmax changes, ordinal prediction shifts, the top-one-minus-top-two probability margin, and the probability-derived ordinal margin `log(p4/p2)`. These are effects of a separately refitted feature policy, not isolated causal effects of JobRole.
2. The exact P3 outer-fold models receive 20 outcome-blind JobRole perturbations under two schemes: marginal shuffling within each outer-test fold and shuffling within each outer-test-fold-by-department cell. The stage records probability change, prediction change, ordinal shift, and the drop in the original predicted class’s XGBoost raw margin. Marginal shuffling can create implausible feature combinations; department-conditional shuffling preserves only the JobRole–department relationship and is not a fully conditional permutation test.

Permutation repetitions describe perturbation variability, not independent sampling uncertainty or confidence intervals. Neither analysis identifies a causal JobRole effect, discrimination, disparate treatment, or a deployment-ready HR rule.

## Claim boundaries

The permitted conclusion is narrow: subgroup performance and model dependence vary descriptively under the prespecified OOF, support, feature-policy, and perturbation contracts. The package cannot certify fairness, prove absence or presence of discrimination, infer department use merely from reconstructability, label P3-minus-JobRole as P5, infer causal feature effects, or support autonomous employment decisions.

The complete local package will retain employee-level paired and permutation rows for reproducibility. The later governed compact package will exclude those rows and publish only support-aware aggregate grids, gap intervals, department summaries, perturbation summaries, reconstruction aggregates, and provenance receipts.

## Implementation validation

Clean-commit run `phase2c_v3_20260907T070905Z_e314bb5` executed from pushed commit `e314bb5d1935da6a23e26b2085330dacb468fe32` and atomically produced the exact 12-file local package. It contains 2,025 subgroup metric rows, 486 gap-sensitivity rows, 162 P3 bootstrap interval rows, 1,200 paired policy-change rows, seven overall/department summaries, 48,000 perturbation rows, 40 repetition summaries, two scheme summaries, and the preserved six/three reconstruction rows. The 5,000 stratified draws all remained complete for the simultaneous family; their index hash is `d16aadb56f2dde124df62387447bb01ddf51236f7f9e9e01b6218e8f6265f646`. Runtime receipts record zero new model fits, network calls, or paid API calls.

An independent validator that imports no Phase 2C runner calculation rebound the generation Git blobs, all 22 source files, output hashes, offline receipt, canonical dataset, exact OOF alignment, persisted model set, and P3 feature order. It independently recalculated every group/support/metric/gap row, replayed the complete bootstrap family, reconstructed every paired probability/margin/prediction-change row, reran all 40 deterministic outer-fold perturbations, recomputed their aggregates, and verified byte-equivalence of the preserved reconstruction evidence. Validation passed for all 12 files and scientific-input SHA-256 `2a42827646702378fb92cf0e0b940fa2739334cc6b62fa1f2ff4f9af66787f1e`.

The exact P3-minus-JobRole versus P3 comparison has overall mean total variation 0.0906, a 10.75% prediction-change rate, and macro-F1 change −0.0330. Marginal JobRole permutations average total variation 0.1024, prediction-change rate 0.1305, and macro-F1 change −0.0432; department-conditional values are 0.0529, 0.0739, and −0.0034. These differences support only the bounded model-dependence interpretation above. Some department-specific policy differences improve while others worsen, and the simultaneous subgroup intervals are wide; no single maximum gap, fairness, discrimination, or causal claim is authorized.
