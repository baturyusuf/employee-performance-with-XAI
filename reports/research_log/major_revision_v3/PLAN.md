# Major Revision v3 Plan

Date: 2026-09-03
Branch: `finalization/leakage-aware-v2`
Status: active implementation contract; Phase 0 and INX Phase 1A complete; Phase 1B, Phase 1C, and Phase 1D evidence published; Phase 2A implementation complete and full execution pending

## Purpose

Extend, rather than overwrite, the immutable v2 evidence package so the repository can address the complete 42-part reviewer-remediation brief. The accepted v2 run remains historical evidence under `canonical_v2_20260714T221501Z_483f96f`; every new scientific result must be generated under a new `canonical_v3_*` identity.

The intended scientific conclusion is conditional rather than promotional: employee-performance prediction evidence depends on the information contract, uncertain feature timing, ordinal criteria, organisational proxy structure, calibration, explanation reliability, and the reproducibility protocol.

## Implementation state — 2026-09-03

- Phase 0 was committed and pushed at `618a3936e040db119eee3d43f6887d3ed4f447ea`.
- The INX portion of Phase 1A is implemented in `configs/feature_availability_v3.json`, validated by `src/governance/feature_availability_contract.py`, and rendered in `FEATURE_AVAILABILITY_GOVERNANCE_CONTRACT.md`.
- The contract covers all 28 acquisition-schema fields and freezes six nested policies with retained-feature counts P0=26, P1=24, P2=21, P3=20, P4=13, and P5=6.
- The scenario remains a conceptual cross-sectional estimand: zero feature timestamps are verified, P4 is only a prospective-plausibility sensitivity, and no policy is described as eliminating all leakage.
- Phase 1B implementation was committed and pushed at `dc5cb8b96b096bb2efc6c242403b7e51f870a01b`; the complete exact-commit run `phase1b_v3_20260903T130912Z_dc5cb8b` then produced exactly-once OOF evidence for nine systems on the common 10×5 fold contract.
- An independent validator rehashed all nine local outputs, bound the generation-time implementation blobs and immutable v2 sources, verified 10,800 combined OOF rows, and recomputed every aggregate, per-class, and confusion result. The publication-safe compact package is `phase1b_ordinal_benchmark/`; employee-level rows remain ignored locally.
- Phase 1B has no universal winner: cumulative-threshold XGBoost leads macro-F1/balanced accuracy, Random Forest leads QWK/ordinal MAE, LightGBM leads RPS/Brier, and nominal XGBoost leads log loss. The next implementation unit is Phase 1C repeated nested-CV training-variability sensitivity.
- The Phase 1B compact evidence/independent-validator checkpoint was pushed at `10d88dc95f322ef620d8f5bbad9a1af8ec73dd77`.
- Phase 1C implementation was pushed at `f71077e98a464ce93351f82dc8d53b7659364096`; the Git-identity validation repair was pushed at `78649c426e69fb5270f9d027b11ba6ba87d71a41`. Complete run `phase1c_v3_20260903T215015Z_78649c4` refitted all nine systems across five prespecified 5×5 repetitions, produced 54,000 exactly-once OOF rows, and recorded zero network/API activity.
- Independent validation reconstructed all five outer and inner fold systems from the persisted contracts and OOF evidence, rebound all generation inputs and six implementation blobs to the exact commit, and recomputed fold, repetition, variability, rank, stability, and tuning-frequency outputs. The safe aggregate package is `phase1c_repeated_nested_cv/`; local row-level evidence remains ignored.
- Phase 1C finds metric-specific stability rather than a universal winner: LightGBM/XGBoost split macro-F1 wins 3/2, cumulative-threshold XGBoost wins balanced accuracy 4/5, and Random Forest wins QWK and ordinal MAE 5/5. Phase 1D independently retuned policy estimands are next.
- Phase 1D now has a hash-bound two-estimand contract and runner. The fixed-schedule analysis exactly reuses canonical-v2 P0–P3 OOF evidence and newly fits P4/P5 under the P3 schedule; the retuned analysis independently tunes all six policies inside the same persisted 10×5 folds. Fit-free preflight and a non-persisted P3/fold-1 replay diagnostic passed with zero network/API activity.
- Complete run `phase1d_v3_20260904T063324Z_823c848` produced 14,400 exactly-once OOF rows and passed independent closed-world, source, selection-lineage, OOF-reuse, P3-replay, metric-recomputation, and publication-safety validation. Retuning raises macro-F1 for P0/P1/P2/P4/P5, but P0 balanced accuracy and P5 QWK/balanced accuracy decline; P3 is an exact zero-difference replay. The compact aggregate package is `phase1d_policy_retuning/`. Phase 2A SHAP stability and faithfulness is next.
- Phase 2A now has a hash-bound aggregation/stability/faithfulness contract and runner. It formalizes raw-margin multiclass grouping, retains historical outer-fold pairs, adds five fixed-fold seed refits and five stratified 80% outer-training resamples, and tests top-1/3/5 SHAP-guided training-reference masking against 20 random baselines. Fit-free, exact-reference, seed, resampling, and bounded deletion diagnostics passed; the clean-commit 100-fit full execution and independent result validation are pending.

## Non-negotiable controls

- Do not overwrite, relabel, or use mutable aliases for the canonical v2 package.
- Do not edit `manuscript/mdpi_information/main.md` or `main.tex` until the v3 claim matrix is frozen and approved by the user.
- Never make paid API calls. Scientific execution must remain offline.
- Do not publish raw/row-level employee data, fitted models, local SHAP rows, bootstrap arrays, secrets, or unverified licensed material to Git.
- Preserve the distinction among leakage/timing risk, sensitive-feature governance, and organisational proxy risk.
- Treat SHAP as noncausal model attribution, subgroup analysis as descriptive diagnostics, counterfactual search as nonprescriptive, and HRDataset_v14 as independently trained mapped-target replication rather than locked transport.
- Report adverse, null, unstable, or lower-performing results without suppression.
- Commit and push tested, reviewable checkpoints after the planning audit and after each completed phase.

## Work plan

| Phase | Problem | Planned change | Experiment or audit | Acceptance criterion | Principal files |
| --- | --- | --- | --- | --- | --- |
| 0 — Scope reconciliation | The frozen v2 protocol is narrower than the current reviewer brief. | Preserve v2 and define an additive v3 scope with a requirement-by-requirement gap register. | Compare all 43 numbered sections (0–42) with source, tests, canonical artifacts, and manuscript state. | Every requirement is classified as addressed, partial, missing, or approval-pending with evidence and a next action. | `reports/research_log/major_revision_v3/PLAN.md`; `REQUIREMENT_COVERAGE_AUDIT.md` |
| 1A — Information contract | No single sufficiently bounded prediction-time scenario or full feature-availability contract is canonical. | Add a dataset-supported scenario; a complete raw-feature availability/governance contract; explicit P0–P5 semantic mappings; and separate leakage, sensitive, and proxy taxonomies. | Validate every raw INX/HR feature against the declared policy and evidence status; unverifiable timing remains `timing_uncertain`. | Complete feature coverage; no unsupported “available before prediction” or “all leakage eliminated” claim; exact policy feature lists hash-bound. | New v3 config/contract files under `configs/`; governance code/tests; v3 tables |
| 1B — Ordinal benchmark | V2 contains nominal multiclass models only and no naive baselines or RPS. | Add proportional-odds ordinal logistic regression and a nonlinear cumulative-threshold boosting model; add majority, stratified, and ordinal-median baselines; add RPS and explicit two-level reversal metrics. | All models use training-only preprocessing and the shared outer folds; exactly-once OOF predictions; valid probability simplex; per-class and confusion outputs; no test-set selection. | New model/evaluation modules and tests; v3 benchmark artifacts |
| 1C — Training uncertainty | V2 sample-level OOF bootstrap does not measure training/fold variability. | Add deterministic repeated 5×5 outer/inner nested-CV sensitivity, with a documented escalation to 10 repetitions only if runtime is proportionate. | Report mean, SD, median, range/percentile interval, and ordering stability for macro-F1, balanced accuracy, QWK, and ordinal MAE. | Repetition identities and seeds persisted; every repetition complete; ordering instability explicitly retained. | v3 repeated-CV runner/config/tests/tables |
| 1D — Policy retuning separation | V2 includes fixed-schedule policy sensitivity only. | Retain the matched fixed-HP analysis and add independently retuned performance for every governance policy. | Same fold assignments; outer-test isolation; fixed-versus-retuned differences reported without causal interpretation. | Both estimands available and clearly labelled; no policy chosen from outer-test results. | v3 policy runner/tests/tables |
| 2A — SHAP validity | V2 has exact-fold SHAP and fold-pair stability, but lacks seed/resampling stability and model-level faithfulness. | Preserve exact-model grouped SHAP; add seed and stratified 80% resampling stability; add training-fold median/mode deletion tests versus repeated random baselines for top 1/3/5 feature families. | Top-5/10/15 Jaccard and all-feature Spearman reported; deletion probability/margin effects and random contrasts reported; OOD masking limitation mandatory. | Exact model/fold lineage; additivity test retained; no human-usefulness or causal claim. | v3 SHAP modules/tests/tables/figures |
| 2B — Calibration | V2 reports cross-fitted sigmoid log loss, Brier, and top-label ECE only. | Add RPS/cumulative calibration diagnostics, class-wise ECE/reliability, and suitable intercept/slope diagnostics while preserving predeclared training-only sigmoid fitting. | Probability sums equal one; all calibration fits exclude outer-test data; empty-bin behavior explicit; metric-specific improvements and degradations reported. | Reproducible class-wise tables/figures and leakage-isolation tests pass. | v3 calibration modules/tests/tables/figures |
| 2C — Subgroup/proxy use | V2 subgroup and reconstructability evidence does not fully cover requested ordinal/probability metrics, threshold sensitivity, or performance-model proxy-use sensitivity. | Add supported-group macro-F1, balanced accuracy, QWK, ordinal MAE, class recalls, log loss/Brier; support thresholds 20/30/50; primary-versus-proxy-reduced prediction-change/probability-change analysis and grouped permutation sensitivity. | Unsupported cells remain N/A; maximum-gap selection is labelled exploratory; no fairness/discrimination conclusion; proxy reconstructability and proxy use remain distinct. | v3 subgroup/proxy modules/tests/tables |
| 3A — Replication sensitivities | HRDataset_v14 lacks target-mapping and repeated-5-fold CV sensitivity. | Add one semantically defensible alternative target formulation and repeated 5×5 nested-CV sensitivity; retain the existing primary mapping. | Mapping rationale explicit; small-class instability visible; no target equivalence or locked transport claim. | v3 external-replication modules/tests/tables |
| 3B — Data quality | No canonical manuscript-ready data-quality audit exists. | Generate dataset-level duplicates, missingness, constants/near-constants, cardinality, ID, range, target-support, schema-hash, and cleaned-schema evidence. | Every core dataset covered; row values excluded from publication summary; anomalies recorded rather than silently repaired. | `DATA_QUALITY_REPORT.md`; source table; tests |
| 4A — Literature/novelty | The current manuscript literature review is narrow and stale. | Verify 15–25 close studies using primary bibliographic/publisher sources and build a positioning table focused on the shared evidence contract. | Titles, venues, year, DOI/URL, and applicability are source-verified; no fabricated citation or checklist novelty claim. | literature evidence log/table; `references.bib` only after verification |
| 4B — Provenance, ethics, release | Source authenticity/licence/redistribution and ethics metadata remain manual blockers; no immutable v3 release exists. | Record verifiable public provenance, unresolved legal/ethics fields, history blocker, release notes, tag proposal, and DOI instructions without publishing a release. | Unknowns remain explicit; no invented licence/IRB/DOI; history is not rewritten without separate authorization. | provenance/limitations/release-prep reports |
| 5A — Claim freeze | Manuscript v1 is numerically and conceptually stale. | Build a sentence-level v3 claim matrix tied to immutable source rows and hashes; request one explicit user approval. | Every numerical claim resolves to a source row; every narrative claim has support level and prohibited overclaim; user approval recorded. | `CLAIM_MATRIX.csv`/`.md`; validation tests |
| 5B — Manuscript and response | The current Markdown/LaTeX manuscript contains obsolete LLM/agent/chatbot material and stale figures/results. | Rewrite title through declarations, regenerate tables/figures, produce reviewer response, revision report, result comparison, limitations register, reproducibility report, and final review simulation. | No stale number/asset; Markdown/LaTeX parity; citations verified; render/build checks pass; no unresolved placeholder hidden. | `manuscript/mdpi_information/`; final deliverable reports |

## Execution order and checkpoints

1. Freeze this plan and the coverage audit; run repository hygiene/link checks; commit and push.
2. Implement Phase 1 contracts and benchmark extensions with focused tests; run bounded diagnostics; checkpoint and push.
3. Execute/validate Phase 1 real-data experiments; checkpoint only compact, non-row-level evidence and push.
4. Implement and execute Phase 2; validate lineage, calibration isolation, and explanation tests; checkpoint and push.
5. Implement and execute Phase 3; checkpoint and push.
6. Complete verified literature/provenance work and release preparation; checkpoint and push.
7. Freeze the complete claim matrix and obtain user approval.
8. Rewrite and render the manuscript and all requested final deliverables; run full regression and final reviewer simulation; checkpoint and push.

## Global quality gates

- Existing v2 tests remain green or are explicitly version-isolated.
- New preprocessing, fold, calibration, policy, target-mapping, SHAP identity/aggregation/faithfulness, exactly-once OOF, hash, and manuscript-source tests pass.
- `git diff --check`, compile checks, repository hygiene, secret/path/raw-data scans, and manuscript-source consistency pass before every checkpoint.
- No checkpoint includes canonical internals that violate the established publication policy.
- A failed or interrupted run is retained as inadmissible forensic evidence and is never relabelled canonical.
