# Current Status

## Major-revision v3 extension — 2026-09-04

- The user supplied a broader 42-part reviewer-remediation brief and requested periodic tested GitHub pushes.
- The canonical v2 package remains immutable and valid for its frozen protocol. It is not being overwritten or relabelled.
- A scope reconciliation found that v2 does not close the broader brief: 13 numbered sections are addressed, 13 partial, 11 missing, and 6 approval/final-authoring pending.
- The active additive work plan is `../major_revision_v3/PLAN.md`; the requirement-level evidence/gap audit is `../major_revision_v3/REQUIREMENT_COVERAGE_AUDIT.md`.
- The current Markdown and LaTeX manuscript remain historical v1 and have not been edited. Claim-matrix approval is still required before manuscript authoring.
- The v3-only information contract is implemented and the ordinal/naive benchmark extension is complete. Any eventual integrated canonical result must use a new `canonical_v3_*` identity and preserve all v2 evidence.
- Planning validation passed: compact canonical asset validation, 10 focused publication/CI tests, 45 README links, 43 exact requirement rows, status-count reconciliation, and `git diff --check`.
- Planning checkpoint `618a3936e040db119eee3d43f6887d3ed4f447ea` was pushed normally to `origin/finalization/leakage-aware-v2`.
- The INX Phase 1A contract is now implemented: all 28 pinned acquisition fields, eight separate risk types, six exactly nested P0–P5 policies, and a fail-closed Markdown generator. Retained counts are 26/24/21/20/13/6.
- Contract SHA-256 is `6dd5fdde534e379cceacfaa01e865d1551310fb632b691f5b937ef39394e93cf`; semantic SHA-256 is `1773f42a1b33c8e51c91955f9ab28c1b67dc32fdfeb381e9f52dab6bcf571755`.
- The scenario explicitly remains timestamp-unverified and conceptual; zero features are asserted timestamp-verified, P4 is not prospective deployment evidence, and no policy is called leakage-free.
- Phase 1A final focus passed 44 tests plus 7 subtests; all 46 README local links resolved. No model fit, real-data experiment, API/network call, canonical-v2 mutation, or manuscript edit occurred.
- Phase 1A checkpoint `a60a2f3b5c3f17398b80e3b2b3e2ebaf04787378` was pushed; local/remote divergence was 0/0.
- Phase 1B code implements two ordinal models, three training-only naive baselines, normalized RPS, the renamed two-level reversal rate, per-class metrics, complete confusion grids, exact v2 fold/OOF reuse, and clean-Git/source-hash atomic-run gates. Its implementation checkpoint `dc5cb8b96b096bb2efc6c242403b7e51f870a01b` is pushed.
- Complete run `phase1b_v3_20260903T130912Z_dc5cb8b` contains nine closed-world files and exactly 1,200 OOF predictions per system. Independent validation rehashed the run, bound generation-time Git blobs and immutable v2 sources, checked 10,800 combined OOF rows, and recomputed all 144 aggregate, 27 per-class, and 81 confusion records.
- Results are metric-specific: cumulative-threshold XGBoost leads macro-F1/balanced accuracy, Random Forest leads QWK/ordinal MAE, LightGBM leads RPS/Brier, and nominal XGBoost leads log loss. Proportional-odds logistic underperforms the nominal models; cumulative-threshold XGBoost's log loss is materially adverse. No universal winner or significance claim is authorized.
- The tracked compact Phase 1B package has nine files/71,654 bytes and excludes employee-level OOF rows, raw data, and fitted models. Phase 1C repeated nested-CV sensitivity is the next scientific unit.
- Phase 1B evidence checkpoint `10d88dc95f322ef620d8f5bbad9a1af8ec73dd77` passed the clean repository gate and is synchronized with GitHub.
- Phase 1C implementation checkpoints `f71077e98a464ce93351f82dc8d53b7659364096` and `78649c426e69fb5270f9d027b11ba6ba87d71a41` are pushed. Complete run `phase1c_v3_20260903T215015Z_78649c4` refitted all nine systems over five prespecified 5×5 repetitions and produced 54,000 exactly-once OOF rows with zero network/API attempts.
- Independent validation rebound the exact generation contract, six implementation blobs, and all source contracts; reconstructed all five outer/inner fold systems; and recomputed 225 fold metrics, 720 repetition metrics, variability, ranks, ordering stability, and selection frequencies. The tracked nine-file/255,182-byte compact package excludes employee/fold-level evidence, raw data, candidate rows, and fitted models.
- Phase 1C is metric-specific: LightGBM/XGBoost split macro-F1 wins 3/2; cumulative-threshold XGBoost wins balanced accuracy 4/5; Random Forest wins QWK and ordinal MAE 5/5. The ranges are descriptive, not confidence intervals, and no universal winner is authorized.
- Final Phase 1C evidence regression passed 840 tests, 2 skips, and 11 subtests; unittest passed 179 tests with 1 skip; compile, compact validation, 50-link README audit, and `git diff --check` passed.
- Final complete pre-checkpoint regression passed 809 tests, 2 skips, and 11 subtests in 119.88 seconds. The one stale additive-source guard found by the first full run was corrected without weakening frozen-v2 modification/deletion protection.
- Phase 1D policy-retuning implementation checkpoint `823c84866b461266c75f3224527f679a86ab670e` is pushed. Its hash-bound contract separates fixed primary-schedule feature-access sensitivity from independently retuned within-policy performance for all six P0–P5 policies on the exact canonical-v2 10×5 folds.
- Fixed P0–P3 OOF evidence is reused only through exact feature-set mappings; P4/P5 receive new fixed-schedule fits, and all six policies are independently retuned. Historical nonmatching v2 policy rows remain preserved but outside the v3 P0–P5 comparison.
- Fit-free real-data preflight verified 1,200 samples, target support 194/874/132, policy feature counts 26/24/21/20/13/6, eight XGBoost candidates, 4,800 reusable fixed OOF rows, and 2,480 planned new fits. A non-persisted P3/fold-1 diagnostic exactly replayed the primary candidate schedule, labels, and probabilities. Both recorded zero network/API activity.
- Complete run `phase1d_v3_20260904T063324Z_823c848` produced 14,400 exactly-once OOF rows with zero network/API activity. Independent validation checked the 12-file closed world and exact source blobs, replayed all 60 policy/fold selections, proved P0–P3 source reuse and zero-error P3 replay, and recomputed all 120 fold and 192 aggregate metric records.
- Retuning raises macro-F1 for five policies, but its criterion effects are mixed: P0 balanced accuracy falls, and P5 QWK/balanced accuracy fall despite improved macro-F1/ordinal MAE. The tracked seven-file/92,567-byte compact package excludes all employee/fold/candidate rows.
- Phase 2A implementation now separates grouped-SHAP aggregation validity, ranking stability, and model-level deletion faithfulness. It freezes the exact raw-margin grouping order, five fixed-fold seed refits, five stratified 80% outer-training resamples, and top-1/3/5 training-fold median/mode masking against 20 random baselines.
- Exact-reference replay reproduces the canonical global grouped importance within `9.1e-17` and identical ranks. One seed and one resample diagnostic completed all ten folds with raw-margin additivity errors below `1e-5` and zero grouping-sum error; a two-random-baseline deletion diagnostic exercised 10,800 perturbation rows. All diagnostics are noncanonical and unpersisted.
- Complete clean-commit run `phase2a_v3_20260904T073008Z_6e52de7` executed 100 new fits and 110 model/fold explanations and generated all 75,600 perturbation rows with zero network/API activity. Independent validation reconstructed memberships/rankings, replayed all random orders, and recomputed all stability/faithfulness/AUC outputs.
- Mean top-5 Jaccard is 1.0000 across fold, seed, and resampling comparisons; all-feature Spearman means are 0.9066/0.9945/0.9847. Guided-minus-random probability drops are +0.2676/+0.2481/+0.2063 for top 1/3/5 deletion, and deletion-AUC difference is +0.2208. These are dependent/descriptive model-level results under potentially OOD masking, not causal or human-usefulness evidence.
- The tracked eight-file/24,079-byte Phase 2A package has manifest SHA-256 `4cd7bcace03d556e2bd27eea4dca87143745914ddd56136476edc6c662e44481` and excludes sample/fold/membership/model data. Phase 2B extended calibration is next.
- The Phase 2A result checkpoint `c4e25e42fccb6402ca14eef5b1ccd232ca47fc6d` is pushed and synchronized with GitHub; its clean post-commit repository gate passed with 1,932 tracked files and 54 resolved README links.
- Phase 2B design and implementation are validated. The new contract binds the exact canonical raw and cross-fitted sigmoid OOF probabilities and makes the one-vs-rest Platt, renormalization/simplex, ten-bin boundary/empty-bin, and top-label/classwise/cumulative ECE definitions explicit.
- The fit-free Phase 2B runner adds normalized RPS/cumulative Brier, classwise and cumulative reliability diagrams, and descriptive pooled-OOF calibration intercept/slope without fitting a performance model or probability calibrator. It retains the adverse sigmoid top-label ECE point change and forbids method selection or an all-metrics-improved claim.
- Phase 2B focused validation passed 16 tests; complete repository regression passed 898 tests, 2 skips, and 11 subtests. Clean implementation checkpoint/push and exact-commit diagnostic execution are next; the manuscript remains untouched.
- Phase 2B implementation checkpoint `21d1aecb6e61511e95aee498ab81c54fe6e5a6ab` is pushed and synchronized 0/0. Its clean repository gate passed with 1,938 tracked files, 55 README links, and zero publication-hygiene findings.
- Clean run `phase2b_v3_20260904T120838Z_21d1aec` atomically produced 11 local files, 120 explicit reliability bins, and ten descriptive calibration regressions with zero new performance-model/calibrator fits and zero network/paid-API calls. Scientific-input SHA-256 is `97797e436e816623d8652772a5089d226f304b59a30ffcdee110c33e413eec86`.
- Independent validation rebound all generation blobs and canonical sources and recomputed every table. Raw/sigmoid values are 0.5515/0.4556 log loss, 0.3426/0.2634 Brier, 0.0375/0.0419 top-label ECE, 0.1070/0.0249 macro classwise ECE, 0.0779/0.0184 cumulative ECE, and 0.0860/0.0669 RPS. Twenty-four empty bins are retained.
- The reliability figures passed visual and structural inspection. The compact aggregate exporter is implemented and deterministic; validation/export checkpoint and tracked compact publication are next. The complete local run remains ignored and the manuscript remains untouched.

Date: 2026-07-16

- Branch: `finalization/leakage-aware-v2`.
- Canonical technical evidence: complete, strictly validated, and atomically promoted.
- Canonical run: `canonical_v2_20260714T221501Z_483f96f`.
- Clean generation commit: `483f96fdbaab16cb0f32d03d9dbe676a759af44a`.
- Config hash: `51415c2ce68c89d9ce2b042b0a7a811fe3e98180b72460006f9d0465f6bf49b7`.
- Source-tree hash: `f1e358e99914563305428cece1b1595bc76a58643184407ec5b222162d650332`.
- Core scientific-input hash: `06c507bee525ea1daca43b61249764007d4d8baaa05c9333f23446ea723ce160`.
- Supplementary scientific-input hash: `caffb945d15f990e3a789e9707f7a8a9115be31fecbbd705822994a10cfaf151`.
- Publication pointer: `reports/manuscript_final/latest/pointer.json`, the sole file under `latest`, SHA-256 `ecd3990187112891c3f1d3bb41bb7939392576e8df7904cae46baf2106397219`.
- Compact canonical receipt: `../finalization_v2/15_canonical_evidence_receipt.json`.

## GitHub publication assets

- The tracked manuscript-support package is `../../../manuscript/mdpi_information/assets/`: 109 files/10,338,351 bytes with closed-world manifest SHA-256 `fbe7355b956df01ad9817f27b42dc13c0f3e0e33e7f0e5c42a2477beb9d001e1`.
- It contains seven main and three supplementary figures as PNG/SVG, captions/alt text, eight main and three supplementary manuscript-ready tables, exact compact source data, 221 evidence-ledger rows, insertion maps, claim boundaries, and source hashes.
- Canonical-to-manuscript numbering is explicit: manuscript Figures 1-7 map to canonical Figures 1, 3, 2, 4, 5, 6, 7. Figure 1 is a presentation-only rendering of frozen protocol/claim contracts; the other main figures are byte-for-byte copies.
- Local source-parity validation checked 308 declared source bindings across 108 manifest records after independently revalidating all 545 canonical files. No scientific stage was rerun.
- The complete canonical root is again ignored as one unit. Its formerly tracked 50-file aggregate subset was removed from the current index with all 545 local files preserved; receipt 16 is retained as historical publication chronology.
- Raw/interim data, employee-level OOF/local-SHAP outputs, models, calibrator internals, bootstrap arrays, and training-partition records remain ignored and are not authorized for Git publication.
- Asset checkpoint `902d4041fce94139ba58cb7c982f9a62b8427611` passed the clean repository gate and was pushed normally to `origin/finalization/leakage-aware-v2`; GitHub authentication succeeded.

## Acceptance evidence

- Core build: exit 0 in 1,641.749 seconds; 351 closed-world manifest records, 354 total files, 224,171,937 bytes.
- Supplementary build: exit 0 in 1,926.755 seconds; 188 closed-world manifest records, 191 total files, 222,423,549 bytes.
- Promotion: exit 0 in 5.968 seconds.
- Strict post-promotion validation: exit 0 in 5.677 seconds.
- All ten core and four supplementary stage commands are complete with return code 0.
- Seven final figures exist as PNG/SVG pairs with seven source CSVs and seven captions.
- Fourteen source tables exist: eleven core and three supplementary.
- Complete package scans found zero machine-absolute paths, active `leakage-safe` terms, secret patterns, raw dataset files, or lock/partial/staging directories.
- Scientific runtime receipts record zero attempted/successful network operations and zero paid API calls.
- The journal manuscript was not modified.

## Unit 2G

The original stage-validation run `stage_validation_hrdataset_20260713T175045Z_5af0262e83a3` remains preserved and noncanonical. The canonical run regenerated HRDataset_v14 from the current frozen inputs and independently replayed all acceptance invariants:

- raw SHA-256 `cb19996755c93c0a8d6527f59da4701c80aef65eff854906546dce286249813c`;
- parsed-cell SHA-256 `e6d5bb365c3548e9185f4c24f20737210cde62c758d6e1d1ce3462846f4c0983`;
- schema mapping `65c40fecc75235f57209cff6231d7386e4279ebedca9390f53ad2bb67ae63788`;
- feature policy `e08625831afd057d1102d0fc42e06d56c9cd51a5632d04780aafe880888d78e5`;
- 311 exactly-once cases with mapped support 31/243/37 and 10 outer x 5 inner folds;
- 400 candidate fits, 50 persisted outer models, 50 calibration fits, and 5,000 bootstrap draws;
- zero maximum outer-model, sigmoid, or grouped-SHAP replay error;
- exact-fold grouped OOF SHAP with 6,531 rows and maximum additivity error `4.468303814064711e-06`;
- forbidden features absent from primary SHAP and reason-code outputs;
- support-aware subgroup/proxy and source-table contracts passed.

## Test state

- Compact-export focus: 17 passed, 1 skipped in 4.15 seconds (7.357 seconds wall).
- Complete pytest: 758 passed, 2 skipped, 11 subtests passed in 129.91 seconds (133.988 seconds wall).
- Unittest: 179 tests passed, 1 skipped in 8.214 seconds (12.246 seconds wall).
- Compileall over `src`, `tests`, and `tools`: exit 0 in 0.312 seconds.
- Compact-package validation: 109 files/10,338,351 bytes, seven main and three supplementary PNG/SVG pairs, eight main tables, manifest SHA-256 `fbe7355b...`; exit 0.
- Candidate hygiene and README-link gates: 43 valid local links; maximum asset 2,195,799 bytes; zero raw/model/row-level extension, staging, secret, machine-path, forbidden-primary-SHAP, active `leakage-safe`, scientific-source-diff, or manuscript-diff findings.
- Post-commit repository gate: exit 0 in 0.760 seconds; 1,849 tracked files, 33 issue rows, 43 README links, and zero raw/environment/large/secret/machine-path findings.

## Remaining blockers

There is no remaining executable P0/P1 scientific or engineering blocker for the technical evidence package. Submission remains blocked by manual/external items:

1. Ethics/IRB institution, unit, application/reference number, and date remain pending.
2. Dataset source authenticity, licence, citation, and redistribution approval require manual verification.
3. Public Git history still contains previously committed raw/noncanonical evidence; remediation requires a separately authorized history/publication strategy because force-push and history rewriting were prohibited.
4. Claim-matrix approval, manuscript writing/formatting, journal submission metadata, release/DOI publication, and any public evidence upload remain separate manual tasks.

Do not begin manuscript writing in this technical-finalization chat.
