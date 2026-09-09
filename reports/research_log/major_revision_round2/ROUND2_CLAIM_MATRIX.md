# Round 2 Claim Matrix

Status: **APPROVED FOR ROUND 2 REWRITE**. These active claims are the sole scientific boundary for manuscript, bibliography, and reviewer-response drafting.

Claim-set SHA-256: `751208d036597461606bd02d68bfa9e642a4aa5ecf5df2df3fd7e46b99084aea`

Historical rows marked `modified` or `superseded` are retained for audit but are not active rewrite language. `prohibited` rows are explicit non-claims.

## Retained (38)

### C001

The INX analysis is a cross-sectional sensitivity study under explicit feature-availability assumptions, not an observed prospective prediction exercise.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/FEATURE_AVAILABILITY_GOVERNANCE_CONTRACT.md` (text anchor; SHA-256 `0f19e17db84e17f0a5e3aad179efb8323a37d5b63b9b032e119deadd159a7201`).
- Required qualifier: Feature and decision timestamps are unobserved, and prospective availability is not verified.
- Prohibited overclaim: Do not describe any policy as leakage-free, prospective validation, or deployment evidence.

### C003

The five-repetition ranges describe training and split variability and are not confidence intervals.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1c_repeated_nested_cv/README.md` (text anchor; SHA-256 `3631b3508bcbbfa181b6503bd6ff7ff76ca9d9eb57f315ecd0200794047647c6`).
- Required qualifier: The summaries reflect only the five fixed repetition identities and their shared sample.
- Prohibited overclaim: Do not present empirical repetition ranges as sampling confidence intervals.

### C005

SHAP stability and deletion behavior characterize the fitted model under the declared aggregation and masking interventions, not causal employee-level effects.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase2a_shap_stability_faithfulness/README.md` (text anchor; SHA-256 `3cfef71caf62e974e312fcebe92e8290b09c18ea7db048a9b89d0d94ff9235f9`).
- Required qualifier: Masking can create out-of-distribution hybrid records, and comparison pairs are not independent.
- Prohibited overclaim: Do not infer causality, actionability, human usefulness, or employee advice from SHAP.

### C009

The targets are recorded organizational ratings; available documentation does not validate them as objective capability, productivity, or future potential.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/DATA_QUALITY_REPORT.md` (text anchor; SHA-256 `7a7ff636b2803e399ec4b1ae30d9604769d15bd7e50b7cdb65a3ce8b81c26c2f`).
- Required qualifier: The audit applies only to declared rules and does not establish construct validity.
- Prohibited overclaim: Do not relabel organizational ratings as objective employee performance or capability.

### C011

No raw dataset is currently approved for redistribution; public materials must remain limited to authorized compact evidence, hashes, schemas, and qualified upstream locators.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase4b_release_readiness/DATA_PROVENANCE_LICENSE_REGISTER.csv` (text anchor; SHA-256 `9e5880bfcd44e5168ec373f3663a75cba53d0587f15d5b778933929953bcc5a4`).
- Required qualifier: Each dataset retains its distinct unresolved source-to-byte and rights-chain status.
- Prohibited overclaim: Do not infer authenticity, ownership, unrestricted licensing, or redistribution rights from public presence.

### C012

Institutional review and informed-consent wording remain pending an institution-approved determination and may not be inferred from synthetic-data descriptions alone.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase4b_release_readiness/ETHICS_DECLARATIONS_REGISTER.csv` (text anchor; SHA-256 `c95a298194f42d1f77f8dfd9fbac5ba4a5f5dccfab607eb51f805acb43fc9f79`).
- Required qualifier: Institution, review unit, determination, reference, date, and consent wording remain missing.
- Prohibited overclaim: Do not invent ethics approval, exemption, waiver, or not-applicable wording.

### C101

Cumulative-threshold XGBoost achieved the highest full-OOF macro-F1, 0.6255.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1b_ordinal_benchmark/aggregate_metrics.csv` (row 4, `value`; SHA-256 `b7dcd0fa01f40c6d11450bb4afd32b1e768f7b301ca804674d8ae589f5a58edf`).
- Required qualifier: Highest applies only to macro-F1 among the nine evaluated systems.
- Prohibited overclaim: Do not generalize this ranking to every metric or future population.

### C102

Cumulative-threshold XGBoost achieved the highest full-OOF balanced accuracy, 0.6745.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1b_ordinal_benchmark/aggregate_metrics.csv` (row 3, `value`; SHA-256 `b7dcd0fa01f40c6d11450bb4afd32b1e768f7b301ca804674d8ae589f5a58edf`).
- Required qualifier: Highest applies only to balanced accuracy among the nine evaluated systems.
- Prohibited overclaim: Do not claim universal or statistically significant superiority.

### C103

Random Forest had the highest full-OOF quadratic weighted kappa, 0.6317.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1b_ordinal_benchmark/aggregate_metrics.csv` (row 108, `value`; SHA-256 `b7dcd0fa01f40c6d11450bb4afd32b1e768f7b301ca804674d8ae589f5a58edf`).
- Required qualifier: The ordering criterion is QWK, not macro-F1 or probability quality.
- Prohibited overclaim: Do not call Random Forest the universally best model.

### C104

Random Forest had the lowest full-OOF ordinal MAE, 0.1583.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1b_ordinal_benchmark/aggregate_metrics.csv` (row 106, `value`; SHA-256 `b7dcd0fa01f40c6d11450bb4afd32b1e768f7b301ca804674d8ae589f5a58edf`).
- Required qualifier: Lower is better, and this result does not determine other metric rankings.
- Prohibited overclaim: Do not equate low ordinal MAE with complete error safety.

### C105

LightGBM had the lowest normalized ranked probability score, 0.0804.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1b_ordinal_benchmark/aggregate_metrics.csv` (row 33, `value`; SHA-256 `b7dcd0fa01f40c6d11450bb4afd32b1e768f7b301ca804674d8ae589f5a58edf`).
- Required qualifier: Lower is better and the claim applies to normalized RPS only.
- Prohibited overclaim: Do not infer overall model superiority from one probability metric.

### C106

Nominal XGBoost had the lowest raw log loss, 0.5515.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1b_ordinal_benchmark/aggregate_metrics.csv` (row 141, `value`; SHA-256 `b7dcd0fa01f40c6d11450bb4afd32b1e768f7b301ca804674d8ae589f5a58edf`).
- Required qualifier: Lower is better and the comparison precedes the predeclared sigmoid evaluation.
- Prohibited overclaim: Do not present raw log-loss leadership as universal superiority.

### C201

Across five repeated nested-CV designs, XGBoost mean macro-F1 was 0.6288.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1c_repeated_nested_cv/variability_summary.csv` (row 35, `mean`; SHA-256 `f50bbd1bb89e479ccdeb3e15538ea5f4a5c2eeabad6ef3d6d40d5439384c2acf`).
- Required qualifier: The mean is descriptive across five repetitions and is not a confidence estimate.
- Prohibited overclaim: Do not claim stable universal leadership or population-level certainty.

### C202

Across five repeated nested-CV designs, LightGBM mean macro-F1 was 0.6249.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1c_repeated_nested_cv/variability_summary.csv` (row 7, `mean`; SHA-256 `f50bbd1bb89e479ccdeb3e15538ea5f4a5c2eeabad6ef3d6d40d5439384c2acf`).
- Required qualifier: LightGBM and XGBoost split repetition-level macro-F1 wins.
- Prohibited overclaim: Do not infer a significant difference from the two descriptive means.

### C203

Random Forest ranked first for QWK in 5 of 5 repetitions.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1c_repeated_nested_cv/model_rank_summary.csv` (row 24, `winner_count`; SHA-256 `dfd3c314b88ee8bb9113d11189ad8c1ae1e9538f9fa385f526dd261ae2eedfe1`).
- Required qualifier: The stable winner statement applies only to QWK under this design.
- Prohibited overclaim: Do not infer universal superiority or prospective stability.

### C204

Cumulative-threshold XGBoost ranked first for balanced accuracy in 4 of 5 repetitions.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1c_repeated_nested_cv/model_rank_summary.csv` (row 2, `winner_count`; SHA-256 `dfd3c314b88ee8bb9113d11189ad8c1ae1e9538f9fa385f526dd261ae2eedfe1`).
- Required qualifier: One repetition was won by another model and no inferential test is attached.
- Prohibited overclaim: Do not describe the ordering as invariant or statistically proven.

### C205

Mean pairwise macro-F1 rank Spearman correlation across repetition pairs was 0.926.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1c_repeated_nested_cv/ordering_stability.csv` (row 3, `mean_pairwise_rank_spearman`; SHA-256 `5b36d6eb639441408a5997a60aaddb5cfc3a9b8986b4e3a1be6b6eaf50c4ca22`).
- Required qualifier: Correlations are descriptive and repetition pairs are not independent samples.
- Prohibited overclaim: Do not label the ordering perfectly stable or confidence-bounded.

### C301

For P2, independent retuning changed macro-F1 by +0.0185 relative to the fixed schedule.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1d_policy_retuning/headline_policy_comparison.csv` (row 4, `raw_difference_macro_f1`; SHA-256 `3ea78efa8b710759c1db2fa53e782b0144951ac927f3c86f0e18871b3c3e95e4`).
- Required qualifier: This point difference has no confidence interval and combines policy-specific selection with access.
- Prohibited overclaim: Do not call the difference causal, significant, or deployment-valid.

### C302

For P5, independent retuning changed QWK by -0.0344 despite improving some other metrics.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1d_policy_retuning/headline_policy_comparison.csv` (row 7, `raw_difference_quadratic_weighted_kappa`; SHA-256 `3ea78efa8b710759c1db2fa53e782b0144951ac927f3c86f0e18871b3c3e95e4`).
- Required qualifier: P5 removes declared strong proxies but cannot prove absence of residual proxies.
- Prohibited overclaim: Do not claim monotonic benefit, fairness proof, or causal proxy removal.

### C303

The retuned P0 information-rich diagnostic reached macro-F1 0.8943.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1d_policy_retuning/headline_policy_comparison.csv` (row 2, `retuned_macro_f1`; SHA-256 `3ea78efa8b710759c1db2fa53e782b0144951ac927f3c86f0e18871b3c3e95e4`).
- Required qualifier: P0 is a diagnostic upper bound and is explicitly never deployable.
- Prohibited overclaim: Do not present P0 as prospective performance or an admissible HR policy.

### C401

The mean top-5 Jaccard agreement across model-seed pairs was 1.0000.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase2a_shap_stability_faithfulness/stability_summary.csv` (row 5, `jaccard_mean`; SHA-256 `4329e361ae3d006474efa724b4ef4b82d361ce4a3b3fb667f0b7169e0fbc0496`).
- Required qualifier: Pairs share data and protocol components and are not independent observations.
- Prohibited overclaim: Do not call SHAP perfectly robust, causal, or confidence-bounded.

### C402

Mean all-feature Spearman agreement across outer-training resample pairs was 0.9847.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase2a_shap_stability_faithfulness/stability_summary.csv` (row 8, `spearman_mean`; SHA-256 `4329e361ae3d006474efa724b4ef4b82d361ce4a3b3fb667f0b7169e0fbc0496`).
- Required qualifier: The same Spearman summary is repeated across top-k rows and is descriptive.
- Prohibited overclaim: Do not infer sampling independence or explanation validity from rank agreement alone.

### C403

Deleting the top-ranked feature produced a mean probability-drop contrast of 0.2676 versus random deletion.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase2a_shap_stability_faithfulness/faithfulness_contrasts.csv` (row 2, `guided_minus_random_mean`; SHA-256 `c8fa92829a3c519eeee7718e42c8ab13b5380d780ded9ab3019ef3175d5c06a3`).
- Required qualifier: The contrast is descriptive and median/mode masking can be out of distribution.
- Prohibited overclaim: Do not interpret the perturbation as a causal or actionable employee effect.

### C501

Cross-fitted sigmoid calibration reduced log loss to 0.4556.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase2b_calibration_diagnostics/method_comparison.csv` (row 2, `sigmoid_value`; SHA-256 `bac8344ec08129d92a38ffe5c968da576b57fe8bac48aa0f7b50e052e4f3bbb3`).
- Required qualifier: The method was predeclared and outer-test outcomes were evaluation-only.
- Prohibited overclaim: Do not claim prospective calibration or objective outcome probabilities.

### C502

Top-label ECE changed by +0.0044 after sigmoid calibration, indicating a worse point estimate on that metric.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase2b_calibration_diagnostics/method_comparison.csv` (row 4, `raw_difference_sigmoid_minus_raw`; SHA-256 `bac8344ec08129d92a38ffe5c968da576b57fe8bac48aa0f7b50e052e4f3bbb3`).
- Required qualifier: The paired interval spans zero and ECE is bin-dependent.
- Prohibited overclaim: Do not state that every calibration metric improved.

### C601

Removing JobRole and refitting changed the argmax prediction for 10.75% of cases.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use/proxy_prediction_change_by_department.csv` (row 2, `prediction_change_rate`; SHA-256 `79ba56a192c8baccd5098ea37bfc1ab3e7d64f02276fe710df458df068d5963f`).
- Required qualifier: The comparison combines feature removal with refitting and is not the broader P5 policy.
- Prohibited overclaim: Do not describe the change rate as a causal JobRole effect.

### C602

The proxy-reduced-minus-primary macro-F1 difference was -0.0330.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use/proxy_prediction_change_by_department.csv` (row 2, `delta_macro_f1`; SHA-256 `79ba56a192c8baccd5098ea37bfc1ab3e7d64f02276fe710df458df068d5963f`).
- Required qualifier: The point difference is descriptive and combines removal with model refitting.
- Prohibited overclaim: Do not infer discrimination, causality, or significance.

### C603

Marginal within-fold JobRole permutation produced mean total variation 0.1024.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use/jobrole_permutation_summary.csv` (row 2, `mean_total_variation_mean`; SHA-256 `8d0af59191a90fe7fd78b14f6055b3f06fbda2b74ac74016b4c78b96d016d5ca`).
- Required qualifier: Marginal shuffling can create out-of-distribution feature combinations.
- Prohibited overclaim: Do not treat the perturbation as a causal or inferential effect.

### C604

Department-conditional within-fold JobRole permutation produced mean total variation 0.0529.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use/jobrole_permutation_summary.csv` (row 3, `mean_total_variation_mean`; SHA-256 `8d0af59191a90fe7fd78b14f6055b3f06fbda2b74ac74016b4c78b96d016d5ca`).
- Required qualifier: Conditioning only on department is not a fully conditional permutation test.
- Prohibited overclaim: Do not call the difference a causal JobRole contribution.

### C605

Department reconstruction accuracy with JobRole in the feature space was 0.9792.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use/department_reconstructability_metrics.csv` (row 2, `point_estimate`; SHA-256 `e838ea0f0e447df3abc6e6b10af5147db2b65ba955e8eaf92952bb54a6a82b4c`).
- Required qualifier: This measures information in the feature space, not use by the performance model.
- Prohibited overclaim: Do not infer discrimination, causality, or performance-model dependence.

### C606

Department reconstruction accuracy without JobRole was 0.2908.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use/department_reconstructability_metrics.csv` (row 5, `point_estimate`; SHA-256 `e838ea0f0e447df3abc6e6b10af5147db2b65ba955e8eaf92952bb54a6a82b4c`).
- Required qualifier: This is a proxy-risk contrast, not a fairness or causal test.
- Prohibited overclaim: Do not infer absence of residual department information or legal compliance.

### C701

Under the retained three-class HRDataset_v14 mapping, raw XGBoost mean macro-F1 was 0.6531.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity/variability_summary.csv` (row 35, `mean`; SHA-256 `1725d5161c414f21fe353fe01cf685cffc59ef5dde1b5253e0bf494b47659cd2`).
- Required qualifier: The value is mapping- and dataset-specific and the repetition range is not a confidence interval.
- Prohibited overclaim: Do not call the result locked external validation or direct INX comparability.

### C702

Under the retained three-class mapping, sigmoid-calibrated XGBoost mean QWK was 0.6044.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity/variability_summary.csv` (row 49, `mean`; SHA-256 `1725d5161c414f21fe353fe01cf685cffc59ef5dde1b5253e0bf494b47659cd2`).
- Required qualifier: Sigmoid calibration lowered macro-F1 even while improving this ordinal criterion.
- Prohibited overclaim: Do not claim that calibration uniformly improved HRDataset_v14 performance.

### C703

The retained mapped HRDataset_v14 class 2 contains 31 records.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/target_distribution.csv` (row 11, `count`; SHA-256 `4845da4f51bdd6651e7a54c1aaa279cbe0af1d0dd5fc8d0175bcdaf1cc0a2502`).
- Required qualifier: The mapping is a study estimand and does not establish equivalence with INX labels.
- Prohibited overclaim: Do not treat mapped classes as validated construct-equivalent outcomes.

### C801

The audited INX table contains 1200 rows.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/dataset_summary.csv` (row 2, `raw_row_count`; SHA-256 `df7d21a23be7cad851161d821f7ac376c30d5c6e5c8783c45fdee227de148d0b`).
- Required qualifier: Row count does not establish representativeness, authenticity, or licensing.
- Prohibited overclaim: Do not infer population coverage or source authority from table size.

### C802

HRDataset_v14 contains 215 effective missing cells under the declared audit rule.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/dataset_summary.csv` (row 3, `effective_missing_cell_count`; SHA-256 `df7d21a23be7cad851161d821f7ac376c30d5c6e5c8783c45fdee227de148d0b`).
- Required qualifier: Most missing termination dates are structurally aligned with non-terminated records.
- Prohibited overclaim: Do not state that all missingness is error or silently impute source values.

### C803

A total of 3 declared HRDataset_v14 data-quality rules had findings.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/dataset_summary.csv` (row 3, `rules_with_findings`; SHA-256 `df7d21a23be7cad851161d821f7ac376c30d5c6e5c8783c45fdee227de148d0b`).
- Required qualifier: The count applies to declared rules and is not a complete error census.
- Prohibited overclaim: Do not claim the dataset is otherwise error-free or invalid.

### C804

Those three HRDataset_v14 rules account for 6 recorded anomaly occurrences.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/dataset_summary.csv` (row 3, `summed_rule_anomaly_occurrences`; SHA-256 `df7d21a23be7cad851161d821f7ac376c30d5c6e5c8783c45fdee227de148d0b`).
- Required qualifier: Occurrences can concern distinct rules and were retained without silent source repair.
- Prohibited overclaim: Do not imply six unique erroneous employees without row-level overlap analysis.

## Modified (5)

### C002

The benchmark identifies metric-specific leaders rather than one universally superior model.

- Active for rewrite: `false`
- Evidence: `reports/research_log/major_revision_v3/phase1b_ordinal_benchmark/README.md` (text anchor; SHA-256 `f6ca7b8e8a48addaf6541595e0038d54eab8fb655c54d18c7b734844e74fb64c`).
- Required qualifier: Leadership is conditional on the metric, P3 feature policy, and frozen cross-validation protocol.
- Prohibited overclaim: Do not name any system as universally best or deployment-ready.

### C004

Fixed-schedule and independently retuned policy contrasts answer different descriptive questions and neither estimates a causal feature-policy effect.

- Active for rewrite: `false`
- Evidence: `reports/research_log/major_revision_v3/phase1d_policy_retuning/README.md` (text anchor; SHA-256 `6454f8eae651fee2b40d9b9e5834d623cbb69e6e381749446bbf079036ca41a1`).
- Required qualifier: Retuned contrasts combine information access with policy-specific model selection.
- Prohibited overclaim: Do not interpret policy differences as causal feature effects or real deployment performance.

### C006

Cross-fitted sigmoid calibration improved several probability-quality point estimates but worsened top-label ECE, so calibration conclusions remain metric-specific.

- Active for rewrite: `false`
- Evidence: `reports/research_log/major_revision_v3/phase2b_calibration_diagnostics/README.md` (text anchor; SHA-256 `506fa99890bc3a915b83cbef440050db2824be590308a954ca22ee8f6b95a938`).
- Required qualifier: New diagnostic differences lack intervals, and ECE depends on fixed binning and support.
- Prohibited overclaim: Do not claim that calibration improved every metric or proves future calibration.

### C007

Department reconstructability and JobRole-dependent performance-model outputs are separate proxy-risk diagnostics and do not establish discrimination or fairness.

- Active for rewrite: `false`
- Evidence: `reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use/README.md` (text anchor; SHA-256 `15654d4f69a36ba82009f0d80a38c059a4c4ba2dd55faee673f7a0eae9894a98`).
- Required qualifier: All diagnostics condition on this sample, fitted models, support rules, folds, and perturbations.
- Prohibited overclaim: Do not claim fairness certification, discrimination, causal proxy effects, or legal compliance.

### C008

HRDataset_v14 provides an independently trained mapped-target protocol replication rather than locked-model external validation.

- Active for rewrite: `false`
- Evidence: `reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity/README.md` (text anchor; SHA-256 `0ed65fed2414c386c53ee3f938c24b7f505bc4254adce6953ed727691b6afb83`).
- Required qualifier: Feature space, target semantics, parameters, and sample population differ from INX.
- Prohibited overclaim: Do not call this locked transport, target equivalence, or unqualified external validation.

## Superseded (2)

### C010

Within the frozen 25-work source-verified positioning set, no selected work reports the complete shared evidence contract used here.

- Active for rewrite: `false`
- Evidence: `reports/research_log/major_revision_v3/phase4a_literature/NOVELTY_POSITIONING.md` (text anchor; SHA-256 `b32962d51746d31eba5122e6f4eb3e5c119a02558f0632f2a845f7d7dd435299`).
- Required qualifier: The statement is bounded to the frozen set and does not cover unobserved publications.
- Prohibited overclaim: Do not state world first, first ever, exhaustive review, or universal novelty.

### C013

The final v3 claim set requires an explicit digest-specific user decision before any manuscript or bibliography edit is authorized.

- Active for rewrite: `false`
- Evidence: `reports/research_log/major_revision_v3/phase4b_release_readiness/BLOCKER_REGISTER.csv` (text anchor; SHA-256 `0d42241dc670f7c2773bc97830dcd8ec3aa4cf3396ad650d491bcdca87241954`).
- Required qualifier: A generic instruction to continue does not identify or approve the frozen digest.
- Prohibited overclaim: Do not edit the manuscript, bibliography, or reviewer response before explicit approval.

## New (79)

### R2N001

Selection sensitivity is reported through separate leader changes, full-ordering changes, selected-candidate changes, and metric-specific effect magnitudes; it is not collapsed into one binary material-dependence verdict.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SELECTION_OBJECTIVE_SENSITIVITY.md` (text anchor; SHA-256 `4c4e712aa1deb49345b02d29a8df8129045f4ac89e9037ceeaf3744adca4801f`).
- Required qualifier: Each diagnostic remains separate and descriptive.
- Prohibited overclaim: Do not convert the diagnostics into a single binary claim or treat a lower-order permutation alone as material dependence.

### R2N002

The outer-training empirical-prior comparator contextualizes probability metrics but does not establish calibration quality or deployment performance.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/PROBABILITY_BASELINE_REPORT.md` (text anchor; SHA-256 `389b191a1c50799eaeedc0736c09b9c6fdd6c5bece088635940c05f83497ed99`).
- Required qualifier: The comparator is naive, sample-conditional, and not selected against outer-test data.
- Prohibited overclaim: Do not interpret ECE zero as perfect general calibration or the baseline as a deployable model.

### R2N003

Strong aggregate ordinal scores can coexist with complete failure to identify rating 4, so per-class behavior must accompany aggregate rankings.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/PER_CLASS_EXTREME_CLASS_REPORT.md` (text anchor; SHA-256 `4437bf690f9d5ab902f1d5e4b7267ccd1649ce050cb3f1df3463b751f3185db0`).
- Required qualifier: This is observed metric behavior under one sample and protocol.
- Prohibited overclaim: Do not infer a causal explanation for the extreme-class failure.

### R2N004

The P3→P4 and P4→P5 contrasts are central timing/information sensitivities under fixed and retuned estimands; neither is a causal feature-removal effect or prospective validation.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/ROUND2_PLAN.md` (text anchor; SHA-256 `668543c1307ceb2ea93b022ab60e67f350db3ef1eae3159cc72e201b7b942c2f`).
- Required qualifier: P4/P5 are prospective-plausibility sensitivities because feature timestamps are unavailable.
- Prohibited overclaim: Do not call P3 confirmed leakage or P4/P5 leakage-free prospective evidence.

### R2N005

All six prespecified P3 subgroup attributes are reported at the support threshold with endpoints and exploratory intervals, including unfavorable cells.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/ROUND2_PLAN.md` (text anchor; SHA-256 `668543c1307ceb2ea93b022ab60e67f350db3ef1eae3159cc72e201b7b942c2f`).
- Required qualifier: The table is descriptive and support-dependent.
- Prohibited overclaim: Do not claim fairness, discrimination, protected-class effects, or legal compliance.

### R2N006

The retained three-class and raw-order four-class HR formulations are different estimands and are compared side by side without subtraction or improvement language.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/HR_MAPPING_SENSITIVITY_REPORT.md` (text anchor; SHA-256 `53b836dc250d60f1e14f4d7fc2217f062aa8cfe43cf78806f816d5d6c10117de`).
- Required qualifier: Target construction and metric choice jointly condition interpretation.
- Prohibited overclaim: Do not claim construct equivalence or cross-formulation improvement.

### R2N007

The 10×5 versus repeated-5×5 HR comparison is descriptive CV-design sensitivity, not an equivalence test or robustness certification.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/HR_MAPPING_SENSITIVITY_REPORT.md` (text anchor; SHA-256 `53b836dc250d60f1e14f4d7fc2217f062aa8cfe43cf78806f816d5d6c10117de`).
- Required qualifier: Repeated ranges are not confidence intervals.
- Prohibited overclaim: Do not describe the design comparison as statistical equivalence or unqualified robustness.

### R2N008

The primary target-alias comparison restricts canonical predictions and exclusion/refit predictions to the identical 309-row population; the separate 311→309 contrast is only a fit-free sample-removal effect.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/HR_MAPPING_SENSITIVITY_REPORT.md` (text anchor; SHA-256 `53b836dc250d60f1e14f4d7fc2217f062aa8cfe43cf78806f816d5d6c10117de`).
- Required qualifier: Only the matched 309↔309 contrast isolates training/data-rule sensitivity.
- Prohibited overclaim: Do not attribute the 311→309 change to refitting or claim equivalence.

### R2N009

The Methods contract fully enumerates feature governance, preprocessing, candidate grids, tie rules, fold identities, seed identities, and refit boundaries.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/METHOD_REPRODUCIBILITY_TABLES/METHODS_NOTES.md` (text anchor; SHA-256 `16b2ce99950f7ada253d558a444292baca3f3a2f17956921935ab1c6e0449ba2`).
- Required qualifier: The description does not add a new experiment or change the frozen estimands.
- Prohibited overclaim: Do not omit outer-test isolation, tie tolerance, or timestamp caveats.

### R2N010

The contribution is not the novelty of any single component, but their operational integration into a traceable evaluation contract.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/LITERATURE_V4/NOVELTY_BOUNDARY.md` (text anchor; SHA-256 `85e83a24ed6d12e9d9104503076494625f72e982a84f2ede81c4003aadee8851`).
- Required qualifier: This is a positioning statement, not an exhaustive literature claim.
- Prohibited overclaim: Do not state world-first, first-ever, or exhaustive novelty.

### R2N011

The abstract distinguishes single frozen canonical INX OOF values from HR means across five repetitions and does not imply that they share an estimand.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/ROUND2_PLAN.md` (text anchor; SHA-256 `668543c1307ceb2ea93b022ab60e67f350db3ef1eae3159cc72e201b7b942c2f`).
- Required qualifier: INX and HR values must carry their respective design qualifiers.
- Prohibited overclaim: Do not pool, subtract, or directly rank the two dataset estimands.

### R2N012

The complete Round 2 claim digest requires explicit digest-specific user approval before any manuscript, bibliography, or reviewer-response rewrite.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/ROUND2_PLAN.md` (text anchor; SHA-256 `668543c1307ceb2ea93b022ab60e67f350db3ef1eae3159cc72e201b7b942c2f`).
- Required qualifier: Approval must identify the exact new digest.
- Prohibited overclaim: Do not infer approval from a generic continue instruction.

### R2E001

Metric leaders changed for 6 of 9 reported metrics.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/round2_claim_matrix/DERIVED_SUMMARIES.csv` (row 2, `value`; SHA-256 `7b1600988d2f0e9539cf529a23e4c633f44f4f0f2540202e523eb14e96b7ddf5`).
- Required qualifier: Leader change is distinct from full-ordering change.
- Prohibited overclaim: Do not convert this count into a binary materiality verdict.

### R2E002

The full model ordering changed for 9 of 9 reported metrics.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/round2_claim_matrix/DERIVED_SUMMARIES.csv` (row 3, `value`; SHA-256 `7b1600988d2f0e9539cf529a23e4c633f44f4f0f2540202e523eb14e96b7ddf5`).
- Required qualifier: A full-ordering change may be confined to lower ranks.
- Prohibited overclaim: Do not equate every permutation with material dependence.

### R2E003

The selected candidate changed in 37 of 60 model-by-fold selections.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/round2_claim_matrix/DERIVED_SUMMARIES.csv` (row 4, `value`; SHA-256 `7b1600988d2f0e9539cf529a23e4c633f44f4f0f2540202e523eb14e96b7ddf5`).
- Required qualifier: This count concerns candidate identities, not independent test repetitions.
- Prohibited overclaim: Do not interpret it as a probability of future selection change.

### R2E004

With QWK selection, nominal XGBoost achieved QWK 0.6418.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/selection_objective_sensitivity/metric_effect_magnitudes.csv` (row 17, `qwk_selection_value`; SHA-256 `a78d36ec202748c8c6ad0c4568fba63f7743f9ca99dcb6c669ae045df1da0938`).
- Required qualifier: Leadership is metric-, regime-, and protocol-specific.
- Prohibited overclaim: Do not call XGBoost universally best.

### R2E005

For cumulative-threshold XGBoost, QWK changed by 0.0769 under QWK rather than macro-F1 selection.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/selection_objective_sensitivity/metric_effect_magnitudes.csv` (row 19, `qwk_minus_macro_f1_selection`; SHA-256 `a78d36ec202748c8c6ad0c4568fba63f7743f9ca99dcb6c669ae045df1da0938`).
- Required qualifier: This is a descriptive sensitivity without an uncertainty interval.
- Prohibited overclaim: Do not call the change statistically significant.

### R2E006

Under canonical macro-F1 selection, Random Forest rating-4 recall was 0.0000.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/selection_objective_sensitivity/per_class_metrics.csv` (row 16, `recall`; SHA-256 `7766cd76abb7e1dda22629889f968cacc99d84e91df3c740e8aa0f59eb39a4f7`).
- Required qualifier: Aggregate QWK and MAE remain separate metrics.
- Prohibited overclaim: Do not imply adequate extreme-class detection from aggregate ordinal scores.

### R2E007

Cumulative-threshold XGBoost rating-4 recall was 0.4167 under macro-F1 selection.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/selection_objective_sensitivity/per_class_metrics.csv` (row 4, `recall`; SHA-256 `7766cd76abb7e1dda22629889f968cacc99d84e91df3c740e8aa0f59eb39a4f7`).
- Required qualifier: This value is selection-regime specific.
- Prohibited overclaim: Do not attribute the value to a causal mechanism.

### R2E008

Cumulative-threshold XGBoost rating-4 recall was 0.0000 under QWK selection.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/selection_objective_sensitivity/per_class_metrics.csv` (row 22, `recall`; SHA-256 `7766cd76abb7e1dda22629889f968cacc99d84e91df3c740e8aa0f59eb39a4f7`).
- Required qualifier: This value is selection-regime specific.
- Prohibited overclaim: Do not treat aggregate QWK improvement as uniform classwise improvement.

### R2E009

The outer-training empirical-prior baseline had RPS 0.1167.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/selection_objective_sensitivity/empirical_prior_metrics.csv` (row 13, `value`; SHA-256 `c2df616438acde3541588e0b40cdd4f37d0bc59fe5bb9a2b57691139c7ad3132`).
- Required qualifier: Lower RPS is better; this baseline is descriptive.
- Prohibited overclaim: Do not call this a calibrated deployable model.

### R2E010

The outer-training empirical-prior baseline had log loss 0.7683.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/selection_objective_sensitivity/empirical_prior_metrics.csv` (row 10, `value`; SHA-256 `c2df616438acde3541588e0b40cdd4f37d0bc59fe5bb9a2b57691139c7ad3132`).
- Required qualifier: Lower log loss is better; this baseline is descriptive.
- Prohibited overclaim: Do not claim statistical superiority without an interval or test.

### R2E011

Under the fixed schedule, P3→P4 changed macro-F1 by -0.2094.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/round2_claim_matrix/DERIVED_SUMMARIES.csv` (row 8, `value`; SHA-256 `7b1600988d2f0e9539cf529a23e4c633f44f4f0f2540202e523eb14e96b7ddf5`).
- Required qualifier: Fixed and retuned estimands answer different descriptive questions.
- Prohibited overclaim: Do not interpret the contrast as causal or timestamp-verified prospective performance.

### R2E012

Under independent retuning, P3→P4 changed macro-F1 by -0.1937.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/round2_claim_matrix/DERIVED_SUMMARIES.csv` (row 11, `value`; SHA-256 `7b1600988d2f0e9539cf529a23e4c633f44f4f0f2540202e523eb14e96b7ddf5`).
- Required qualifier: Fixed and retuned estimands answer different descriptive questions.
- Prohibited overclaim: Do not interpret the contrast as causal or timestamp-verified prospective performance.

### R2E013

Under the fixed schedule, P3→P4 changed QWK by -0.3581.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/round2_claim_matrix/DERIVED_SUMMARIES.csv` (row 9, `value`; SHA-256 `7b1600988d2f0e9539cf529a23e4c633f44f4f0f2540202e523eb14e96b7ddf5`).
- Required qualifier: Fixed and retuned estimands answer different descriptive questions.
- Prohibited overclaim: Do not interpret the contrast as causal or timestamp-verified prospective performance.

### R2E014

Under independent retuning, P3→P4 changed QWK by -0.3308.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/round2_claim_matrix/DERIVED_SUMMARIES.csv` (row 12, `value`; SHA-256 `7b1600988d2f0e9539cf529a23e4c633f44f4f0f2540202e523eb14e96b7ddf5`).
- Required qualifier: Fixed and retuned estimands answer different descriptive questions.
- Prohibited overclaim: Do not interpret the contrast as causal or timestamp-verified prospective performance.

### R2E015

Under the fixed schedule, P4→P5 changed QWK by -0.1150.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/round2_claim_matrix/DERIVED_SUMMARIES.csv` (row 15, `value`; SHA-256 `7b1600988d2f0e9539cf529a23e4c633f44f4f0f2540202e523eb14e96b7ddf5`).
- Required qualifier: Fixed and retuned estimands answer different descriptive questions.
- Prohibited overclaim: Do not interpret the contrast as causal or timestamp-verified prospective performance.

### R2E016

Under independent retuning, P4→P5 changed QWK by -0.1767.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/round2_claim_matrix/DERIVED_SUMMARIES.csv` (row 18, `value`; SHA-256 `7b1600988d2f0e9539cf529a23e4c633f44f4f0f2540202e523eb14e96b7ddf5`).
- Required qualifier: Fixed and retuned estimands answer different descriptive questions.
- Prohibited overclaim: Do not interpret the contrast as causal or timestamp-verified prospective performance.

### R2E017

Retuned macro-F1 for Information-Rich Diagnostic was 0.8943.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1d_policy_retuning/headline_policy_comparison.csv` (row 2, `retuned_macro_f1`; SHA-256 `3ea78efa8b710759c1db2fa53e782b0144951ac927f3c86f0e18871b3c3e95e4`).
- Required qualifier: The label is manuscript-facing; timestamps remain unavailable.
- Prohibited overclaim: Do not describe P4/P5 as prospective validation.

### R2E018

Retuned macro-F1 for Primary Leakage-Aware was 0.6210.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1d_policy_retuning/headline_policy_comparison.csv` (row 5, `retuned_macro_f1`; SHA-256 `3ea78efa8b710759c1db2fa53e782b0144951ac927f3c86f0e18871b3c3e95e4`).
- Required qualifier: The label is manuscript-facing; timestamps remain unavailable.
- Prohibited overclaim: Do not describe P4/P5 as prospective validation.

### R2E019

Retuned macro-F1 for Prospective-Plausibility was 0.4273.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1d_policy_retuning/headline_policy_comparison.csv` (row 6, `retuned_macro_f1`; SHA-256 `3ea78efa8b710759c1db2fa53e782b0144951ac927f3c86f0e18871b3c3e95e4`).
- Required qualifier: The label is manuscript-facing; timestamps remain unavailable.
- Prohibited overclaim: Do not describe P4/P5 as prospective validation.

### R2E020

Retuned macro-F1 for Strict Proxy-Reduced Prospective-Plausibility was 0.3547.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase1d_policy_retuning/headline_policy_comparison.csv` (row 7, `retuned_macro_f1`; SHA-256 `3ea78efa8b710759c1db2fa53e782b0144951ac927f3c86f0e18871b3c3e95e4`).
- Required qualifier: The label is manuscript-facing; timestamps remain unavailable.
- Prohibited overclaim: Do not describe P4/P5 as prospective validation.

### R2E101

For Age, the P3 macro-F1 maximum-minus-minimum gap was 0.0301.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 2, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E102

For Age, the P3 QWK maximum-minus-minimum gap was 0.0653.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 3, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E103

For Age, the P3 ordinal MAE maximum-minus-minimum gap was 0.0770.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 4, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E104

For Gender, the P3 macro-F1 maximum-minus-minimum gap was 0.0363.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 5, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E105

For Gender, the P3 QWK maximum-minus-minimum gap was 0.0527.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 6, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E106

For Gender, the P3 ordinal MAE maximum-minus-minimum gap was 0.0508.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 7, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E107

For Marital Status, the P3 macro-F1 maximum-minus-minimum gap was 0.0101.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 8, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E108

For Marital Status, the P3 QWK maximum-minus-minimum gap was 0.0381.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 9, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E109

For Marital Status, the P3 ordinal MAE maximum-minus-minimum gap was 0.0291.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 10, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E110

For Business Travel, the P3 macro-F1 maximum-minus-minimum gap was 0.0528.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 11, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E111

For Business Travel, the P3 QWK maximum-minus-minimum gap was 0.0711.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 12, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E112

For Business Travel, the P3 ordinal MAE maximum-minus-minimum gap was 0.0613.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 13, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E113

For Department, the P3 macro-F1 maximum-minus-minimum gap was 0.2179.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 14, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E114

For Department, the P3 QWK maximum-minus-minimum gap was 0.4388.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 15, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E115

For Department, the P3 ordinal MAE maximum-minus-minimum gap was 0.1193.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 16, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E116

For Education Background, the P3 macro-F1 maximum-minus-minimum gap was 0.0413.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 17, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E117

For Education Background, the P3 QWK maximum-minus-minimum gap was 0.1190.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 18, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E118

For Education Background, the P3 ordinal MAE maximum-minus-minimum gap was 0.0891.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv` (row 19, `gap_max_minus_min`; SHA-256 `42b27483275e7de1f183b898764c9b6227670ab75d417d708025b1aec89d36c9`).
- Required qualifier: This is a descriptive support-dependent gap with exploratory intervals.
- Prohibited overclaim: Do not infer fairness, discrimination, or legal compliance.

### R2E201

The retained three-class HR label 2 support was 31.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/target_distribution.csv` (row 11, `count`; SHA-256 `4845da4f51bdd6651e7a54c1aaa279cbe0af1d0dd5fc8d0175bcdaf1cc0a2502`).
- Required qualifier: The mapping is a study estimand and does not establish equivalence with INX labels.
- Prohibited overclaim: Do not treat mapped categories as validated construct-equivalent outcomes.

### R2E202

The retained three-class HR label 3 support was 243.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/target_distribution.csv` (row 10, `count`; SHA-256 `4845da4f51bdd6651e7a54c1aaa279cbe0af1d0dd5fc8d0175bcdaf1cc0a2502`).
- Required qualifier: The mapping is a study estimand and does not establish equivalence with INX labels.
- Prohibited overclaim: Do not treat mapped categories as validated construct-equivalent outcomes.

### R2E203

The retained three-class HR label 4 support was 37.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/target_distribution.csv` (row 9, `count`; SHA-256 `4845da4f51bdd6651e7a54c1aaa279cbe0af1d0dd5fc8d0175bcdaf1cc0a2502`).
- Required qualifier: The mapping is a study estimand and does not establish equivalence with INX labels.
- Prohibited overclaim: Do not treat mapped categories as validated construct-equivalent outcomes.

### R2E204

The raw-order HR PIP support was 13.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/target_distribution.csv` (row 8, `count`; SHA-256 `4845da4f51bdd6651e7a54c1aaa279cbe0af1d0dd5fc8d0175bcdaf1cc0a2502`).
- Required qualifier: The mapping is a study estimand and does not establish equivalence with INX labels.
- Prohibited overclaim: Do not treat mapped categories as validated construct-equivalent outcomes.

### R2E205

The raw-order HR Needs Improvement support was 18.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/target_distribution.csv` (row 7, `count`; SHA-256 `4845da4f51bdd6651e7a54c1aaa279cbe0af1d0dd5fc8d0175bcdaf1cc0a2502`).
- Required qualifier: The mapping is a study estimand and does not establish equivalence with INX labels.
- Prohibited overclaim: Do not treat mapped categories as validated construct-equivalent outcomes.

### R2E206

The raw-order HR Fully Meets support was 243.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/target_distribution.csv` (row 6, `count`; SHA-256 `4845da4f51bdd6651e7a54c1aaa279cbe0af1d0dd5fc8d0175bcdaf1cc0a2502`).
- Required qualifier: The mapping is a study estimand and does not establish equivalence with INX labels.
- Prohibited overclaim: Do not treat mapped categories as validated construct-equivalent outcomes.

### R2E207

The raw-order HR Exceeds support was 37.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3b_data_quality/target_distribution.csv` (row 5, `count`; SHA-256 `4845da4f51bdd6651e7a54c1aaa279cbe0af1d0dd5fc8d0175bcdaf1cc0a2502`).
- Required qualifier: The mapping is a study estimand and does not establish equivalence with INX labels.
- Prohibited overclaim: Do not treat mapped categories as validated construct-equivalent outcomes.

### R2E208

Three-class raw-XGBoost mean macro-F1 was 0.6531.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity/variability_summary.csv` (row 35, `mean`; SHA-256 `1725d5161c414f21fe353fe01cf685cffc59ef5dde1b5253e0bf494b47659cd2`).
- Required qualifier: The value is formulation-, system-, and metric-specific; ranges are not confidence intervals.
- Prohibited overclaim: Do not subtract across formulations or call either target construct-equivalent.

### R2E209

Three-class raw-XGBoost mean QWK was 0.5339.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity/variability_summary.csv` (row 39, `mean`; SHA-256 `1725d5161c414f21fe353fe01cf685cffc59ef5dde1b5253e0bf494b47659cd2`).
- Required qualifier: The value is formulation-, system-, and metric-specific; ranges are not confidence intervals.
- Prohibited overclaim: Do not subtract across formulations or call either target construct-equivalent.

### R2E210

Three-class sigmoid-XGBoost mean macro-F1 was 0.6274.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity/variability_summary.csv` (row 45, `mean`; SHA-256 `1725d5161c414f21fe353fe01cf685cffc59ef5dde1b5253e0bf494b47659cd2`).
- Required qualifier: The value is formulation-, system-, and metric-specific; ranges are not confidence intervals.
- Prohibited overclaim: Do not subtract across formulations or call either target construct-equivalent.

### R2E211

Three-class sigmoid-XGBoost mean QWK was 0.6044.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity/variability_summary.csv` (row 49, `mean`; SHA-256 `1725d5161c414f21fe353fe01cf685cffc59ef5dde1b5253e0bf494b47659cd2`).
- Required qualifier: The value is formulation-, system-, and metric-specific; ranges are not confidence intervals.
- Prohibited overclaim: Do not subtract across formulations or call either target construct-equivalent.

### R2E212

Four-class raw-XGBoost mean macro-F1 was 0.5847.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity/variability_summary.csv` (row 85, `mean`; SHA-256 `1725d5161c414f21fe353fe01cf685cffc59ef5dde1b5253e0bf494b47659cd2`).
- Required qualifier: The value is formulation-, system-, and metric-specific; ranges are not confidence intervals.
- Prohibited overclaim: Do not subtract across formulations or call either target construct-equivalent.

### R2E213

Four-class raw-XGBoost mean QWK was 0.6288.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity/variability_summary.csv` (row 89, `mean`; SHA-256 `1725d5161c414f21fe353fe01cf685cffc59ef5dde1b5253e0bf494b47659cd2`).
- Required qualifier: The value is formulation-, system-, and metric-specific; ranges are not confidence intervals.
- Prohibited overclaim: Do not subtract across formulations or call either target construct-equivalent.

### R2E214

Four-class sigmoid-XGBoost mean macro-F1 was 0.5481.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity/variability_summary.csv` (row 95, `mean`; SHA-256 `1725d5161c414f21fe353fe01cf685cffc59ef5dde1b5253e0bf494b47659cd2`).
- Required qualifier: The value is formulation-, system-, and metric-specific; ranges are not confidence intervals.
- Prohibited overclaim: Do not subtract across formulations or call either target construct-equivalent.

### R2E215

Four-class sigmoid-XGBoost mean QWK was 0.6621.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity/variability_summary.csv` (row 99, `mean`; SHA-256 `1725d5161c414f21fe353fe01cf685cffc59ef5dde1b5253e0bf494b47659cd2`).
- Required qualifier: The value is formulation-, system-, and metric-specific; ranges are not confidence intervals.
- Prohibited overclaim: Do not subtract across formulations or call either target construct-equivalent.

### R2E216

Of the 14 HR CV-design comparisons, 11 fell inside the repeated-run ranges.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/round2_claim_matrix/DERIVED_SUMMARIES.csv` (row 5, `value`; SHA-256 `7b1600988d2f0e9539cf529a23e4c633f44f4f0f2540202e523eb14e96b7ddf5`).
- Required qualifier: Observed repetition ranges are not confidence intervals.
- Prohibited overclaim: Do not call the result an equivalence test.

### R2E217

Of the 14 HR CV-design comparisons, 3 fell outside the repeated-run ranges.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/round2_claim_matrix/DERIVED_SUMMARIES.csv` (row 6, `value`; SHA-256 `7b1600988d2f0e9539cf529a23e4c633f44f4f0f2540202e523eb14e96b7ddf5`).
- Required qualifier: Observed repetition ranges are not confidence intervals.
- Prohibited overclaim: Do not claim unqualified robustness.

### R2E218

The matched 309↔309 exclusion/refit macro-F1 change was 0.004210.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/hr_target_alias_sensitivity/comparison_deltas.csv` (row 6, `right_minus_left`; SHA-256 `ef0c6e9782af1215a542286405d497fe17da4d43a9d0f053b159dde829c540eb`).
- Required qualifier: Matched refit and fit-free sample-removal contrasts are distinct estimands.
- Prohibited overclaim: Do not attribute sample removal to refitting or claim equivalence.

### R2E219

The matched 309↔309 exclusion/refit QWK change was -0.000339.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/hr_target_alias_sensitivity/comparison_deltas.csv` (row 12, `right_minus_left`; SHA-256 `ef0c6e9782af1215a542286405d497fe17da4d43a9d0f053b159dde829c540eb`).
- Required qualifier: Matched refit and fit-free sample-removal contrasts are distinct estimands.
- Prohibited overclaim: Do not attribute sample removal to refitting or claim equivalence.

### R2E220

The matched 309↔309 exclusion/refit ordinal-MAE change was 0.000000.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/hr_target_alias_sensitivity/comparison_deltas.csv` (row 11, `right_minus_left`; SHA-256 `ef0c6e9782af1215a542286405d497fe17da4d43a9d0f053b159dde829c540eb`).
- Required qualifier: Matched refit and fit-free sample-removal contrasts are distinct estimands.
- Prohibited overclaim: Do not attribute sample removal to refitting or claim equivalence.

### R2E221

The matched 309↔309 exclusion/refit RPS change was 0.000024.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/hr_target_alias_sensitivity/comparison_deltas.csv` (row 13, `right_minus_left`; SHA-256 `ef0c6e9782af1215a542286405d497fe17da4d43a9d0f053b159dde829c540eb`).
- Required qualifier: Matched refit and fit-free sample-removal contrasts are distinct estimands.
- Prohibited overclaim: Do not attribute sample removal to refitting or claim equivalence.

### R2E222

The matched 309↔309 exclusion/refit log-loss change was 0.010803.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/hr_target_alias_sensitivity/comparison_deltas.csv` (row 10, `right_minus_left`; SHA-256 `ef0c6e9782af1215a542286405d497fe17da4d43a9d0f053b159dde829c540eb`).
- Required qualifier: Matched refit and fit-free sample-removal contrasts are distinct estimands.
- Prohibited overclaim: Do not attribute sample removal to refitting or claim equivalence.

### R2E223

The matched 309↔309 exclusion/refit Brier-score change was 0.000668.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/hr_target_alias_sensitivity/comparison_deltas.csv` (row 9, `right_minus_left`; SHA-256 `ef0c6e9782af1215a542286405d497fe17da4d43a9d0f053b159dde829c540eb`).
- Required qualifier: Matched refit and fit-free sample-removal contrasts are distinct estimands.
- Prohibited overclaim: Do not attribute sample removal to refitting or claim equivalence.

### R2E224

The matched 309↔309 exclusion/refit ECE change was -0.011197.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/hr_target_alias_sensitivity/comparison_deltas.csv` (row 5, `right_minus_left`; SHA-256 `ef0c6e9782af1215a542286405d497fe17da4d43a9d0f053b159dde829c540eb`).
- Required qualifier: Matched refit and fit-free sample-removal contrasts are distinct estimands.
- Prohibited overclaim: Do not attribute sample removal to refitting or claim equivalence.

### R2E225

The fit-free 311→309 sample-removal macro-F1 change was 0.005357.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/hr_target_alias_sensitivity/comparison_deltas.csv` (row 22, `right_minus_left`; SHA-256 `ef0c6e9782af1215a542286405d497fe17da4d43a9d0f053b159dde829c540eb`).
- Required qualifier: Matched refit and fit-free sample-removal contrasts are distinct estimands.
- Prohibited overclaim: Do not attribute sample removal to refitting or claim equivalence.

### R2E226

The fit-free 311→309 sample-removal QWK change was -0.003035.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/hr_target_alias_sensitivity/comparison_deltas.csv` (row 28, `right_minus_left`; SHA-256 `ef0c6e9782af1215a542286405d497fe17da4d43a9d0f053b159dde829c540eb`).
- Required qualifier: Matched refit and fit-free sample-removal contrasts are distinct estimands.
- Prohibited overclaim: Do not attribute sample removal to refitting or claim equivalence.

### R2E227

The fit-free 311→309 sample-removal ordinal-MAE change was -0.002071.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/hr_target_alias_sensitivity/comparison_deltas.csv` (row 27, `right_minus_left`; SHA-256 `ef0c6e9782af1215a542286405d497fe17da4d43a9d0f053b159dde829c540eb`).
- Required qualifier: Matched refit and fit-free sample-removal contrasts are distinct estimands.
- Prohibited overclaim: Do not attribute sample removal to refitting or claim equivalence.

### R2E228

The fit-free 311→309 sample-removal RPS change was -0.000636.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/hr_target_alias_sensitivity/comparison_deltas.csv` (row 29, `right_minus_left`; SHA-256 `ef0c6e9782af1215a542286405d497fe17da4d43a9d0f053b159dde829c540eb`).
- Required qualifier: Matched refit and fit-free sample-removal contrasts are distinct estimands.
- Prohibited overclaim: Do not attribute sample removal to refitting or claim equivalence.

### R2E229

Candidate selection changed in 2 of 5 HR exclusion/refit outer folds.

- Active for rewrite: `true`
- Evidence: `reports/research_log/major_revision_round2/round2_claim_matrix/DERIVED_SUMMARIES.csv` (row 7, `value`; SHA-256 `7b1600988d2f0e9539cf529a23e4c633f44f4f0f2540202e523eb14e96b7ddf5`).
- Required qualifier: The count is descriptive and fold-specific.
- Prohibited overclaim: Do not interpret it as population uncertainty.

## Prohibited (5)

### R2P001

PROHIBITED: Treat selection sensitivity as one binary claim that ranking materially depends on the objective.

- Active for rewrite: `false`
- Evidence: `reports/research_log/major_revision_round2/SELECTION_OBJECTIVE_SENSITIVITY.md` (text anchor; SHA-256 `4c4e712aa1deb49345b02d29a8df8129045f4ac89e9037ceeaf3744adca4801f`).
- Required qualifier: This language is excluded from every manuscript component.
- Prohibited overclaim: Treat selection sensitivity as one binary claim that ranking materially depends on the objective.

### R2P002

PROHIBITED: Attribute the HR 311→309 contrast to model refitting.

- Active for rewrite: `false`
- Evidence: `reports/research_log/major_revision_round2/HR_MAPPING_SENSITIVITY_REPORT.md` (text anchor; SHA-256 `53b836dc250d60f1e14f4d7fc2217f062aa8cfe43cf78806f816d5d6c10117de`).
- Required qualifier: This language is excluded from every manuscript component.
- Prohibited overclaim: Attribute the HR 311→309 contrast to model refitting.

### R2P003

PROHIBITED: Describe score differences between the three-class and four-class HR formulations as improvements.

- Active for rewrite: `false`
- Evidence: `reports/research_log/major_revision_round2/HR_MAPPING_SENSITIVITY_REPORT.md` (text anchor; SHA-256 `53b836dc250d60f1e14f4d7fc2217f062aa8cfe43cf78806f816d5d6c10117de`).
- Required qualifier: This language is excluded from every manuscript component.
- Prohibited overclaim: Describe score differences between the three-class and four-class HR formulations as improvements.

### R2P004

PROHIBITED: Describe P4 or P5 as timestamp-verified prospective validation.

- Active for rewrite: `false`
- Evidence: `reports/research_log/major_revision_round2/ROUND2_PLAN.md` (text anchor; SHA-256 `668543c1307ceb2ea93b022ab60e67f350db3ef1eae3159cc72e201b7b942c2f`).
- Required qualifier: This language is excluded from every manuscript component.
- Prohibited overclaim: Describe P4 or P5 as timestamp-verified prospective validation.

### R2P005

PROHIBITED: Treat subgroup gaps as proof of fairness or discrimination.

- Active for rewrite: `false`
- Evidence: `reports/research_log/major_revision_round2/ROUND2_PLAN.md` (text anchor; SHA-256 `668543c1307ceb2ea93b022ab60e67f350db3ef1eae3159cc72e201b7b942c2f`).
- Required qualifier: This language is excluded from every manuscript component.
- Prohibited overclaim: Treat subgroup gaps as proof of fairness or discrimination.
