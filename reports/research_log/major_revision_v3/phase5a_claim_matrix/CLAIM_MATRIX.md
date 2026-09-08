# Phase 5A Sentence-Level Claim Matrix

Status: **APPROVED FOR PHASE 5B**. These sentences are the sole authorized claim boundary for manuscript and reviewer-response drafting.

## Methods — estimand and information contract

### C001 — procedural_boundary

The INX analysis is a cross-sectional sensitivity study under explicit feature-availability assumptions, not an observed prospective prediction exercise.

- Evidence: `reports/research_log/major_revision_v3/FEATURE_AVAILABILITY_GOVERNANCE_CONTRACT.md` (text anchor; SHA-256 `0f19e17db84e17f0a5e3aad179efb8323a37d5b63b9b032e119deadd159a7201`).
- Required qualifier: Feature and decision timestamps are unobserved, and prospective availability is not verified.
- Prohibited overclaim: Do not describe any policy as leakage-free, prospective validation, or deployment evidence.

## Results — ordinal benchmark

### C002 — bounded_synthesis

The benchmark identifies metric-specific leaders rather than one universally superior model.

- Evidence: `reports/research_log/major_revision_v3/phase1b_ordinal_benchmark/README.md` (text anchor; SHA-256 `f6ca7b8e8a48addaf6541595e0038d54eab8fb655c54d18c7b734844e74fb64c`).
- Required qualifier: Leadership is conditional on the metric, P3 feature policy, and frozen cross-validation protocol.
- Prohibited overclaim: Do not name any system as universally best or deployment-ready.

### C101 — direct_exact

Cumulative-threshold XGBoost achieved the highest full-OOF macro-F1, 0.6255.

- Evidence: `reports/research_log/major_revision_v3/phase1b_ordinal_benchmark/aggregate_metrics.csv` (row 4, `value`; SHA-256 `b7dcd0fa01f40c6d11450bb4afd32b1e768f7b301ca804674d8ae589f5a58edf`).
- Required qualifier: Highest applies only to macro-F1 among the nine evaluated systems.
- Prohibited overclaim: Do not generalize this ranking to every metric or future population.

### C102 — direct_exact

Cumulative-threshold XGBoost achieved the highest full-OOF balanced accuracy, 0.6745.

- Evidence: `reports/research_log/major_revision_v3/phase1b_ordinal_benchmark/aggregate_metrics.csv` (row 3, `value`; SHA-256 `b7dcd0fa01f40c6d11450bb4afd32b1e768f7b301ca804674d8ae589f5a58edf`).
- Required qualifier: Highest applies only to balanced accuracy among the nine evaluated systems.
- Prohibited overclaim: Do not claim universal or statistically significant superiority.

### C103 — direct_exact

Random Forest had the highest full-OOF quadratic weighted kappa, 0.6317.

- Evidence: `reports/research_log/major_revision_v3/phase1b_ordinal_benchmark/aggregate_metrics.csv` (row 108, `value`; SHA-256 `b7dcd0fa01f40c6d11450bb4afd32b1e768f7b301ca804674d8ae589f5a58edf`).
- Required qualifier: The ordering criterion is QWK, not macro-F1 or probability quality.
- Prohibited overclaim: Do not call Random Forest the universally best model.

### C104 — direct_exact

Random Forest had the lowest full-OOF ordinal MAE, 0.1583.

- Evidence: `reports/research_log/major_revision_v3/phase1b_ordinal_benchmark/aggregate_metrics.csv` (row 106, `value`; SHA-256 `b7dcd0fa01f40c6d11450bb4afd32b1e768f7b301ca804674d8ae589f5a58edf`).
- Required qualifier: Lower is better, and this result does not determine other metric rankings.
- Prohibited overclaim: Do not equate low ordinal MAE with complete error safety.

### C105 — direct_exact

LightGBM had the lowest normalized ranked probability score, 0.0804.

- Evidence: `reports/research_log/major_revision_v3/phase1b_ordinal_benchmark/aggregate_metrics.csv` (row 33, `value`; SHA-256 `b7dcd0fa01f40c6d11450bb4afd32b1e768f7b301ca804674d8ae589f5a58edf`).
- Required qualifier: Lower is better and the claim applies to normalized RPS only.
- Prohibited overclaim: Do not infer overall model superiority from one probability metric.

### C106 — direct_exact

Nominal XGBoost had the lowest raw log loss, 0.5515.

- Evidence: `reports/research_log/major_revision_v3/phase1b_ordinal_benchmark/aggregate_metrics.csv` (row 141, `value`; SHA-256 `b7dcd0fa01f40c6d11450bb4afd32b1e768f7b301ca804674d8ae589f5a58edf`).
- Required qualifier: Lower is better and the comparison precedes the predeclared sigmoid evaluation.
- Prohibited overclaim: Do not present raw log-loss leadership as universal superiority.

## Results — training variability

### C003 — procedural_boundary

The five-repetition ranges describe training and split variability and are not confidence intervals.

- Evidence: `reports/research_log/major_revision_v3/phase1c_repeated_nested_cv/README.md` (text anchor; SHA-256 `3631b3508bcbbfa181b6503bd6ff7ff76ca9d9eb57f315ecd0200794047647c6`).
- Required qualifier: The summaries reflect only the five fixed repetition identities and their shared sample.
- Prohibited overclaim: Do not present empirical repetition ranges as sampling confidence intervals.

### C201 — direct_descriptive

Across five repeated nested-CV designs, XGBoost mean macro-F1 was 0.6288.

- Evidence: `reports/research_log/major_revision_v3/phase1c_repeated_nested_cv/variability_summary.csv` (row 35, `mean`; SHA-256 `f50bbd1bb89e479ccdeb3e15538ea5f4a5c2eeabad6ef3d6d40d5439384c2acf`).
- Required qualifier: The mean is descriptive across five repetitions and is not a confidence estimate.
- Prohibited overclaim: Do not claim stable universal leadership or population-level certainty.

### C202 — direct_descriptive

Across five repeated nested-CV designs, LightGBM mean macro-F1 was 0.6249.

- Evidence: `reports/research_log/major_revision_v3/phase1c_repeated_nested_cv/variability_summary.csv` (row 7, `mean`; SHA-256 `f50bbd1bb89e479ccdeb3e15538ea5f4a5c2eeabad6ef3d6d40d5439384c2acf`).
- Required qualifier: LightGBM and XGBoost split repetition-level macro-F1 wins.
- Prohibited overclaim: Do not infer a significant difference from the two descriptive means.

### C203 — direct_descriptive

Random Forest ranked first for QWK in 5 of 5 repetitions.

- Evidence: `reports/research_log/major_revision_v3/phase1c_repeated_nested_cv/model_rank_summary.csv` (row 24, `winner_count`; SHA-256 `dfd3c314b88ee8bb9113d11189ad8c1ae1e9538f9fa385f526dd261ae2eedfe1`).
- Required qualifier: The stable winner statement applies only to QWK under this design.
- Prohibited overclaim: Do not infer universal superiority or prospective stability.

### C204 — direct_descriptive

Cumulative-threshold XGBoost ranked first for balanced accuracy in 4 of 5 repetitions.

- Evidence: `reports/research_log/major_revision_v3/phase1c_repeated_nested_cv/model_rank_summary.csv` (row 2, `winner_count`; SHA-256 `dfd3c314b88ee8bb9113d11189ad8c1ae1e9538f9fa385f526dd261ae2eedfe1`).
- Required qualifier: One repetition was won by another model and no inferential test is attached.
- Prohibited overclaim: Do not describe the ordering as invariant or statistically proven.

### C205 — direct_descriptive

Mean pairwise macro-F1 rank Spearman correlation across repetition pairs was 0.926.

- Evidence: `reports/research_log/major_revision_v3/phase1c_repeated_nested_cv/ordering_stability.csv` (row 3, `mean_pairwise_rank_spearman`; SHA-256 `5b36d6eb639441408a5997a60aaddb5cfc3a9b8986b4e3a1be6b6eaf50c4ca22`).
- Required qualifier: Correlations are descriptive and repetition pairs are not independent samples.
- Prohibited overclaim: Do not label the ordering perfectly stable or confidence-bounded.

## Results — information-policy sensitivity

### C004 — procedural_boundary

Fixed-schedule and independently retuned policy contrasts answer different descriptive questions and neither estimates a causal feature-policy effect.

- Evidence: `reports/research_log/major_revision_v3/phase1d_policy_retuning/README.md` (text anchor; SHA-256 `6454f8eae651fee2b40d9b9e5834d623cbb69e6e381749446bbf079036ca41a1`).
- Required qualifier: Retuned contrasts combine information access with policy-specific model selection.
- Prohibited overclaim: Do not interpret policy differences as causal feature effects or real deployment performance.

### C301 — direct_descriptive

For P2, independent retuning changed macro-F1 by +0.0185 relative to the fixed schedule.

- Evidence: `reports/research_log/major_revision_v3/phase1d_policy_retuning/headline_policy_comparison.csv` (row 4, `raw_difference_macro_f1`; SHA-256 `3ea78efa8b710759c1db2fa53e782b0144951ac927f3c86f0e18871b3c3e95e4`).
- Required qualifier: This point difference has no confidence interval and combines policy-specific selection with access.
- Prohibited overclaim: Do not call the difference causal, significant, or deployment-valid.

### C302 — direct_descriptive

For P5, independent retuning changed QWK by -0.0344 despite improving some other metrics.

- Evidence: `reports/research_log/major_revision_v3/phase1d_policy_retuning/headline_policy_comparison.csv` (row 7, `raw_difference_quadratic_weighted_kappa`; SHA-256 `3ea78efa8b710759c1db2fa53e782b0144951ac927f3c86f0e18871b3c3e95e4`).
- Required qualifier: P5 removes declared strong proxies but cannot prove absence of residual proxies.
- Prohibited overclaim: Do not claim monotonic benefit, fairness proof, or causal proxy removal.

### C303 — direct_descriptive

The retuned P0 information-rich diagnostic reached macro-F1 0.8943.

- Evidence: `reports/research_log/major_revision_v3/phase1d_policy_retuning/headline_policy_comparison.csv` (row 2, `retuned_macro_f1`; SHA-256 `3ea78efa8b710759c1db2fa53e782b0144951ac927f3c86f0e18871b3c3e95e4`).
- Required qualifier: P0 is a diagnostic upper bound and is explicitly never deployable.
- Prohibited overclaim: Do not present P0 as prospective performance or an admissible HR policy.

## Results — explanation diagnostics

### C005 — procedural_boundary

SHAP stability and deletion behavior characterize the fitted model under the declared aggregation and masking interventions, not causal employee-level effects.

- Evidence: `reports/research_log/major_revision_v3/phase2a_shap_stability_faithfulness/README.md` (text anchor; SHA-256 `3cfef71caf62e974e312fcebe92e8290b09c18ea7db048a9b89d0d94ff9235f9`).
- Required qualifier: Masking can create out-of-distribution hybrid records, and comparison pairs are not independent.
- Prohibited overclaim: Do not infer causality, actionability, human usefulness, or employee advice from SHAP.

### C401 — direct_descriptive

The mean top-5 Jaccard agreement across model-seed pairs was 1.0000.

- Evidence: `reports/research_log/major_revision_v3/phase2a_shap_stability_faithfulness/stability_summary.csv` (row 5, `jaccard_mean`; SHA-256 `4329e361ae3d006474efa724b4ef4b82d361ce4a3b3fb667f0b7169e0fbc0496`).
- Required qualifier: Pairs share data and protocol components and are not independent observations.
- Prohibited overclaim: Do not call SHAP perfectly robust, causal, or confidence-bounded.

### C402 — direct_descriptive

Mean all-feature Spearman agreement across outer-training resample pairs was 0.9847.

- Evidence: `reports/research_log/major_revision_v3/phase2a_shap_stability_faithfulness/stability_summary.csv` (row 8, `spearman_mean`; SHA-256 `4329e361ae3d006474efa724b4ef4b82d361ce4a3b3fb667f0b7169e0fbc0496`).
- Required qualifier: The same Spearman summary is repeated across top-k rows and is descriptive.
- Prohibited overclaim: Do not infer sampling independence or explanation validity from rank agreement alone.

### C403 — direct_descriptive

Deleting the top-ranked feature produced a mean probability-drop contrast of 0.2676 versus random deletion.

- Evidence: `reports/research_log/major_revision_v3/phase2a_shap_stability_faithfulness/faithfulness_contrasts.csv` (row 2, `guided_minus_random_mean`; SHA-256 `c8fa92829a3c519eeee7718e42c8ab13b5380d780ded9ab3019ef3175d5c06a3`).
- Required qualifier: The contrast is descriptive and median/mode masking can be out of distribution.
- Prohibited overclaim: Do not interpret the perturbation as a causal or actionable employee effect.

## Results — calibration

### C006 — bounded_synthesis

Cross-fitted sigmoid calibration improved several probability-quality point estimates but worsened top-label ECE, so calibration conclusions remain metric-specific.

- Evidence: `reports/research_log/major_revision_v3/phase2b_calibration_diagnostics/README.md` (text anchor; SHA-256 `506fa99890bc3a915b83cbef440050db2824be590308a954ca22ee8f6b95a938`).
- Required qualifier: New diagnostic differences lack intervals, and ECE depends on fixed binning and support.
- Prohibited overclaim: Do not claim that calibration improved every metric or proves future calibration.

### C501 — direct_exact

Cross-fitted sigmoid calibration reduced log loss to 0.4556.

- Evidence: `reports/research_log/major_revision_v3/phase2b_calibration_diagnostics/method_comparison.csv` (row 2, `sigmoid_value`; SHA-256 `bac8344ec08129d92a38ffe5c968da576b57fe8bac48aa0f7b50e052e4f3bbb3`).
- Required qualifier: The method was predeclared and outer-test outcomes were evaluation-only.
- Prohibited overclaim: Do not claim prospective calibration or objective outcome probabilities.

### C502 — direct_exact

Top-label ECE changed by +0.0044 after sigmoid calibration, indicating a worse point estimate on that metric.

- Evidence: `reports/research_log/major_revision_v3/phase2b_calibration_diagnostics/method_comparison.csv` (row 4, `raw_difference_sigmoid_minus_raw`; SHA-256 `bac8344ec08129d92a38ffe5c968da576b57fe8bac48aa0f7b50e052e4f3bbb3`).
- Required qualifier: The paired interval spans zero and ECE is bin-dependent.
- Prohibited overclaim: Do not state that every calibration metric improved.

## Results — subgroup and proxy diagnostics

### C007 — procedural_boundary

Department reconstructability and JobRole-dependent performance-model outputs are separate proxy-risk diagnostics and do not establish discrimination or fairness.

- Evidence: `reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use/README.md` (text anchor; SHA-256 `15654d4f69a36ba82009f0d80a38c059a4c4ba2dd55faee673f7a0eae9894a98`).
- Required qualifier: All diagnostics condition on this sample, fitted models, support rules, folds, and perturbations.
- Prohibited overclaim: Do not claim fairness certification, discrimination, causal proxy effects, or legal compliance.

### C601 — direct_descriptive

Removing JobRole and refitting changed the argmax prediction for 10.75% of cases.

- Evidence: `reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use/proxy_prediction_change_by_department.csv` (row 2, `prediction_change_rate`; SHA-256 `79ba56a192c8baccd5098ea37bfc1ab3e7d64f02276fe710df458df068d5963f`).
- Required qualifier: The comparison combines feature removal with refitting and is not the broader P5 policy.
- Prohibited overclaim: Do not describe the change rate as a causal JobRole effect.

### C602 — direct_descriptive

The proxy-reduced-minus-primary macro-F1 difference was -0.0330.

- Evidence: `reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use/proxy_prediction_change_by_department.csv` (row 2, `delta_macro_f1`; SHA-256 `79ba56a192c8baccd5098ea37bfc1ab3e7d64f02276fe710df458df068d5963f`).
- Required qualifier: The point difference is descriptive and combines removal with model refitting.
- Prohibited overclaim: Do not infer discrimination, causality, or significance.

### C603 — direct_descriptive

Marginal within-fold JobRole permutation produced mean total variation 0.1024.

- Evidence: `reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use/jobrole_permutation_summary.csv` (row 2, `mean_total_variation_mean`; SHA-256 `8d0af59191a90fe7fd78b14f6055b3f06fbda2b74ac74016b4c78b96d016d5ca`).
- Required qualifier: Marginal shuffling can create out-of-distribution feature combinations.
- Prohibited overclaim: Do not treat the perturbation as a causal or inferential effect.

### C604 — direct_descriptive

Department-conditional within-fold JobRole permutation produced mean total variation 0.0529.

- Evidence: `reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use/jobrole_permutation_summary.csv` (row 3, `mean_total_variation_mean`; SHA-256 `8d0af59191a90fe7fd78b14f6055b3f06fbda2b74ac74016b4c78b96d016d5ca`).
- Required qualifier: Conditioning only on department is not a fully conditional permutation test.
- Prohibited overclaim: Do not call the difference a causal JobRole contribution.

### C605 — direct_descriptive

Department reconstruction accuracy with JobRole in the feature space was 0.9792.

- Evidence: `reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use/department_reconstructability_metrics.csv` (row 2, `point_estimate`; SHA-256 `e838ea0f0e447df3abc6e6b10af5147db2b65ba955e8eaf92952bb54a6a82b4c`).
- Required qualifier: This measures information in the feature space, not use by the performance model.
- Prohibited overclaim: Do not infer discrimination, causality, or performance-model dependence.

### C606 — direct_descriptive

Department reconstruction accuracy without JobRole was 0.2908.

- Evidence: `reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use/department_reconstructability_metrics.csv` (row 5, `point_estimate`; SHA-256 `e838ea0f0e447df3abc6e6b10af5147db2b65ba955e8eaf92952bb54a6a82b4c`).
- Required qualifier: This is a proxy-risk contrast, not a fairness or causal test.
- Prohibited overclaim: Do not infer absence of residual department information or legal compliance.

## Results — independent replication sensitivity

### C008 — procedural_boundary

HRDataset_v14 provides an independently trained mapped-target protocol replication rather than locked-model external validation.

- Evidence: `reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity/README.md` (text anchor; SHA-256 `0ed65fed2414c386c53ee3f938c24b7f505bc4254adce6953ed727691b6afb83`).
- Required qualifier: Feature space, target semantics, parameters, and sample population differ from INX.
- Prohibited overclaim: Do not call this locked transport, target equivalence, or unqualified external validation.

### C701 — direct_descriptive

Under the retained three-class HRDataset_v14 mapping, raw XGBoost mean macro-F1 was 0.6531.

- Evidence: `reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity/variability_summary.csv` (row 35, `mean`; SHA-256 `1725d5161c414f21fe353fe01cf685cffc59ef5dde1b5253e0bf494b47659cd2`).
- Required qualifier: The value is mapping- and dataset-specific and the repetition range is not a confidence interval.
- Prohibited overclaim: Do not call the result locked external validation or direct INX comparability.

### C702 — direct_descriptive

Under the retained three-class mapping, sigmoid-calibrated XGBoost mean QWK was 0.6044.

- Evidence: `reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity/variability_summary.csv` (row 49, `mean`; SHA-256 `1725d5161c414f21fe353fe01cf685cffc59ef5dde1b5253e0bf494b47659cd2`).
- Required qualifier: Sigmoid calibration lowered macro-F1 even while improving this ordinal criterion.
- Prohibited overclaim: Do not claim that calibration uniformly improved HRDataset_v14 performance.

## Methods and limitations — data quality

### C009 — procedural_boundary

The targets are recorded organizational ratings; available documentation does not validate them as objective capability, productivity, or future potential.

- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/DATA_QUALITY_REPORT.md` (text anchor; SHA-256 `7a7ff636b2803e399ec4b1ae30d9604769d15bd7e50b7cdb65a3ce8b81c26c2f`).
- Required qualifier: The audit applies only to declared rules and does not establish construct validity.
- Prohibited overclaim: Do not relabel organizational ratings as objective employee performance or capability.

### C801 — direct_exact

The audited INX table contains 1200 rows.

- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/dataset_summary.csv` (row 2, `raw_row_count`; SHA-256 `df7d21a23be7cad851161d821f7ac376c30d5c6e5c8783c45fdee227de148d0b`).
- Required qualifier: Row count does not establish representativeness, authenticity, or licensing.
- Prohibited overclaim: Do not infer population coverage or source authority from table size.

### C802 — direct_exact

HRDataset_v14 contains 215 effective missing cells under the declared audit rule.

- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/dataset_summary.csv` (row 3, `effective_missing_cell_count`; SHA-256 `df7d21a23be7cad851161d821f7ac376c30d5c6e5c8783c45fdee227de148d0b`).
- Required qualifier: Most missing termination dates are structurally aligned with non-terminated records.
- Prohibited overclaim: Do not state that all missingness is error or silently impute source values.

### C803 — direct_exact

A total of 3 declared HRDataset_v14 data-quality rules had findings.

- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/dataset_summary.csv` (row 3, `rules_with_findings`; SHA-256 `df7d21a23be7cad851161d821f7ac376c30d5c6e5c8783c45fdee227de148d0b`).
- Required qualifier: The count applies to declared rules and is not a complete error census.
- Prohibited overclaim: Do not claim the dataset is otherwise error-free or invalid.

### C804 — direct_exact

Those three HRDataset_v14 rules account for 6 recorded anomaly occurrences.

- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/dataset_summary.csv` (row 3, `summed_rule_anomaly_occurrences`; SHA-256 `df7d21a23be7cad851161d821f7ac376c30d5c6e5c8783c45fdee227de148d0b`).
- Required qualifier: Occurrences can concern distinct rules and were retained without silent source repair.
- Prohibited overclaim: Do not imply six unique erroneous employees without row-level overlap analysis.

## Discussion — novelty positioning

### C010 — bounded_synthesis

Within the frozen 25-work source-verified positioning set, no selected work reports the complete shared evidence contract used here.

- Evidence: `reports/research_log/major_revision_v3/phase4a_literature/NOVELTY_POSITIONING.md` (text anchor; SHA-256 `b32962d51746d31eba5122e6f4eb3e5c119a02558f0632f2a845f7d7dd435299`).
- Required qualifier: The statement is bounded to the frozen set and does not cover unobserved publications.
- Prohibited overclaim: Do not state world first, first ever, exhaustive review, or universal novelty.

## Data availability

### C011 — direct_exact

No raw dataset is currently approved for redistribution; public materials must remain limited to authorized compact evidence, hashes, schemas, and qualified upstream locators.

- Evidence: `reports/research_log/major_revision_v3/phase4b_release_readiness/DATA_PROVENANCE_LICENSE_REGISTER.csv` (text anchor; SHA-256 `9e5880bfcd44e5168ec373f3663a75cba53d0587f15d5b778933929953bcc5a4`).
- Required qualifier: Each dataset retains its distinct unresolved source-to-byte and rights-chain status.
- Prohibited overclaim: Do not infer authenticity, ownership, unrestricted licensing, or redistribution rights from public presence.

## Declarations — ethics and consent

### C012 — procedural_boundary

Institutional review and informed-consent wording remain pending an institution-approved determination and may not be inferred from synthetic-data descriptions alone.

- Evidence: `reports/research_log/major_revision_v3/phase4b_release_readiness/ETHICS_DECLARATIONS_REGISTER.csv` (text anchor; SHA-256 `c95a298194f42d1f77f8dfd9fbac5ba4a5f5dccfab607eb51f805acb43fc9f79`).
- Required qualifier: Institution, review unit, determination, reference, date, and consent wording remain missing.
- Prohibited overclaim: Do not invent ethics approval, exemption, waiver, or not-applicable wording.

## Approval gate

### C013 — procedural_boundary

The final v3 claim set requires an explicit digest-specific user decision before any manuscript or bibliography edit is authorized.

- Evidence: `reports/research_log/major_revision_v3/phase4b_release_readiness/BLOCKER_REGISTER.csv` (text anchor; SHA-256 `0d42241dc670f7c2773bc97830dcd8ec3aa4cf3396ad650d491bcdca87241954`).
- Required qualifier: A generic instruction to continue does not identify or approve the frozen digest.
- Prohibited overclaim: Do not edit the manuscript, bibliography, or reviewer response before explicit approval.

## Methods — independent replication target

### C703 — direct_exact

The retained mapped HRDataset_v14 class 2 contains 31 records.

- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/target_distribution.csv` (row 11, `count`; SHA-256 `4845da4f51bdd6651e7a54c1aaa279cbe0af1d0dd5fc8d0175bcdaf1cc0a2502`).
- Required qualifier: The mapping is a study estimand and does not establish equivalence with INX labels.
- Prohibited overclaim: Do not treat mapped classes as validated construct-equivalent outcomes.
