# Implementation Progress

## 2026-07-13 — Baseline Audit Complete

- Captured clean current Git state and confirmed origin synchronization.
- Independently audited the v1 canonical config, orchestrator, manifest, cache contract, data loaders, calibration, policy uncertainty, SHAP stability, fairness/proxy, counterfactual, external scope, requirements, tests, reports and dataset provenance.
- Re-ran baseline pytest, unittest and compileall with paid-service environment variables removed.
- Confirmed critical defects V2-001 through V2-024 in the issue register.
- Rejected `reports/manuscript_final/latest` as v2 canonical evidence.
- Created finalization and interruption-resilience records.

Scientific implementation changes: **none yet**.
Scientific artifacts generated: **none**.
Paid/network API calls: **zero**.

Next gate: obtain user decisions D1-D5 before implementing model benchmark, tuning, dataset distribution, ethics metadata and artifact-publication architecture.

## 2026-07-13 — Decisions Accepted and Finalization Branch Created

- Accepted D1: Logistic Regression, Random Forest and LightGBM baselines.
- Accepted D2: restrained nested tuning.
- Accepted D3: user-provided pinned datasets plus sanitized publication export; approved-manifest URL acquisition only when a local file is absent.
- Recorded D4 as ethics application pending with all identifying/reference fields still unknown.
- Accepted D5: small Git artifacts, full Release/Zenodo package, pointer-only latest.
- Created dedicated branch `finalization/leakage-aware-v2` from `1c7c343bda401629a3619f92267384916f0708d0`.
- Checkpoint commits are authorized on this branch only after relevant tests pass.

Current implementation unit: explicit actual-input and side-input binding, pinned acquisition/data verification contract, and fail-closed cache identity.

### Unit 1A — Explicit dataset loader and acquisition preflight

Problem: scientific stages discover raw/interim/external inputs through global paths and do not verify a pinned expected-data contract.

Root cause: convenience loaders own path selection; the canonical config is not passed through the data boundary; no approved acquisition manifest exists.

Intended files:

- `configs/data_acquisition.yaml`
- `data/README.md`
- `src/data/canonical_loader.py`
- `src/data/preprocess.py`
- `src/data/external_adapters.py` if required for explicit path injection
- primary `src/experiments/manuscript_*.py` consumers that currently call the implicit loader
- `tests/test_actual_input_hash_binding.py`
- `tests/test_no_implicit_interim_fallback.py`
- acquisition/schema verification tests

Acceptance criteria:

- configured path is the only path used;
- no interim-exists fallback remains in production scientific loading;
- loader returns actual path/hash/rows/columns/schema/target support;
- local bytes must match pinned hash/schema/row/target expectations;
- missing local data may use only an explicitly approved manifest URL;
- download mismatch is quarantined/reported and never admitted;
- relevant focused tests and real local-data preflight pass;
- no scientific artifact is generated in this unit.

### Unit 1B — Manifest, side-input and cache binding

Problem: the run manifest and stage cache identify configured dataset paths and only a subset of configuration files; they do not prove which verified bytes were consumed or invalidate reuse when an external schema mapping, feature taxonomy, acquisition contract, provenance record, metric/task schema or versioned search space changes.

Root cause: `create_run_manifest` hashes paths from the canonical config directly, while `_stage_cache_valid`, `_write_stage_metadata` and `_write_input_snapshots` compare or preserve only config/source/dataset summaries. There is no aggregate scientific-input identity spanning verified loader receipts and all side inputs.

Intended files:

- `configs/manuscript_final.yaml`
- `src/data/canonical_loader.py`
- `src/governance/manuscript_contract.py`
- `src/experiments/build_manuscript_evidence.py`
- `src/data/external_adapters.py` and `src/experiments/manuscript_external_evidence.py` if needed to remove fixed external input discovery
- `tests/test_artifact_run_manifest_consistency.py`
- `tests/test_actual_input_hash_binding.py`
- `tests/test_side_input_hash_binding.py`
- `tests/test_cache_invalidates_on_scientific_input_change.py`

Acceptance criteria:

- every logical dataset is verified through the canonical loader and its actual path/hash/schema/support receipt is persisted;
- all declared scientific side inputs are present, SHA-256 hashed and snapshotted under repo-relative paths;
- one deterministic aggregate scientific-input hash covers config, actual dataset receipts and side-input hashes;
- stage metadata/cache reuse requires the same commit, source tree, config, actual inputs, side inputs and aggregate identity;
- changing dataset bytes or a side input invalidates cache eligibility;
- external stages receive config-bound data/mapping paths rather than discovering fixed files;
- manifest validation fails on missing/mismatched inputs or snapshots;
- focused tests, full pytest/unittest, compileall, diff hygiene and a real-data manifest preflight pass;
- no scientific result artifact is generated by this unit.

Unit 1B production edits have not started at this checkpoint.

### Unit 1B Result — Passed implementation checkpoint

- Upgraded the run manifest contract to schema v2.
- The manifest now verifies all five logical datasets through the canonical loader, records complete actual-input receipts and derives dataset hash records from the bytes actually consumed.
- Declared seven scientific side inputs: the acquisition contract, provenance record, feature taxonomy, model search space and the three external schema mappings.
- Added a deterministic aggregate `scientific_input_hash` over canonical config identity, actual dataset records and side-input records.
- Stage contracts and cache reuse now require identical Git commit, source tree, config, actual inputs, side inputs and aggregate scientific identity.
- Run-input snapshots now preserve every declared scientific side input and revalidate both source and snapshot hashes before reuse.
- Canonical external evidence now receives verified frames and config-declared mappings; its transport gate uses the same verified INX and HRDataset frames. Fixed-path discovery remains available only to explicitly legacy/direct helper calls.
- Manifest paths are repository-relative and repo-external paths are rejected.
- A real-data manifest create/validate preflight passed with schema v2, five logical receipts and seven side inputs. Its aggregate scientific-input hash was `38dfd51794a837c09cfb67a16eac283d5dc568c94bfa7043a7f5cd14ad6f3b67` for the then-current dirty implementation tree.
- Focused manifest/cache/snapshot tests: 21 passed.
- Focused external explicit-input/claim-boundary tests: 23 passed.
- Full pytest: 218 passed plus 4 subtests.
- Full unittest: 161 passed.
- Compileall, diff hygiene and manuscript no-change checks passed.
- No scientific result artifact, network download, paid API call or manuscript edit occurred.

V2-001 and V2-002 are recorded as implementation-complete but remain unresolved until a clean, cache-disabled full v2 build and final artifact validation pass. V2-014 remains in progress because raw-data Git tracking and sanitized publication export are not yet implemented.

Checkpoint: commit `5c8b0cd215153befd52f39bcf12e0d582d9b1648` (`fix(provenance): bind manifests and caches to scientific inputs`) records the complete tested Unit 1B implementation. The first commit attempt was blocked by the staged whitespace gate; the corrected attempt passed. No push was performed.

## Unit 2A — Core/supplementary orchestration isolation (planned)

Problem: the canonical stage graph still mixes core XAI evidence with counterfactual, IBM/turnover, LLM and chatbot stages. A core build can therefore import or package out-of-scope modules and cannot demonstrate an offline, no-LLM evidence boundary.

Root cause: one global `STAGE_ORDER`, one external stage combining HRDataset and related-task evidence, and one final claim/package generator were inherited from the broader v1 manuscript scope.

Intended files (subject to read-only dependency audit before edits):

- `configs/manuscript_final.yaml`
- `src/experiments/build_manuscript_evidence.py`
- `src/experiments/manuscript_external_evidence.py` or a narrow core/supplementary wrapper reusing it
- manifest/build-scope validation in `src/governance/manuscript_contract.py` only if required
- core/supplementary contract tests, including no-LLM/chatbot/API and scoped dataset-input assertions
- finalization logs

Acceptance criteria:

- core and supplementary stage graphs are explicit and machine-readable;
- core contains only provenance/preflight, shared folds/model evidence, policy ablation, sigmoid calibration, XGBoost SHAP, subgroup/proxy diagnostics, HRDataset replication and core tables/figures/claims;
- core imports/executes/packages no LLM, chatbot, counterfactual, IBM attrition/performance or Turnover stage;
- supplementary keeps heuristic counterfactual and secondary/related-task evidence clearly separated;
- no paid API path is reachable from either accepted build;
- scoped manifests bind exactly the datasets/side inputs used by their build scope;
- historical v1 outputs are never cache inputs;
- focused and full regression gates pass;
- no scientific result is generated until the scope contract tests pass.

### Unit 2A Result — Passed implementation checkpoint

- Added immutable `core` and `supplementary` evidence-scope contracts to the canonical config.
- Core dataset inputs are exactly INX and HRDataset_v14; supplementary inputs are exactly INX, IBM performance/attrition and Employee Turnover. Side-input mappings are scoped the same way.
- Manifest schema v3 records `evidence_scope`, the exact scope contract and its hash. Dataset receipts, side-input hashes, snapshots, stage cache and aggregate scientific identity are scope-specific; cross-scope reuse fails.
- Replaced the v1 monolithic accepted graph with explicit planned core and supplementary stage registries. No accepted registry contains an LLM, chatbot, agent or paid-API stage.
- Removed the LLM/chatbot sections, seeds and package requirements from the canonical core config. Their legacy code remains outside accepted entrypoints.
- Added separate `build_core_paper_evidence` and `build_supplementary_evidence` entrypoints.
- Both entrypoints currently fail before creating an output directory because their honest `release_ready` gates are false until all declared scientific stages are rebuilt. This prevents partial or stale packages from being labelled complete.
- Split external evidence without duplicating the task executor: core runs HRDataset only and its transport-feasibility gate; supplementary runs IBM performance, IBM attrition and Turnover only. The prior zero-denominator placeholder actionability artifact was removed.
- Dataset-card generation now accepts and validates exact scope dataset keys.
- Core claim/package path contracts reject legacy, counterfactual, secondary-external, LLM, chatbot, agent-audit and historical output prefixes.
- The old v1 Figures 1-4 generator is explicitly legacy and fails before writing when invoked with the core config. No replacement scientific figure was fabricated.
- Real in-memory manifest validation passed for both scopes. Core bound two datasets/five side inputs; supplementary bound four logical datasets/six side inputs.
- Focused Unit 2A suite: 50 passed, 2 historical-latest skips.
- Full pytest: 250 passed, 2 skipped, plus 4 subtests.
- Full unittest: 162 passed, 2 skipped.
- Compileall, diff hygiene and manuscript no-change checks passed.
- No scientific experiment, dataset download, network/API call or manuscript edit occurred.

The first full pytest run exposed one legacy figure test that still treated the removed LLM config as canonical. The legacy generator was changed to preflight and fail before writing, and the test now verifies that boundary. The subsequent full suite passed.

Unit 2A deliberately does not set either scope `release_ready=true`: shared folds, nested benchmark/tuning, corrected calibration/uncertainty, new core figures/tables and supplementary heuristic-search freeze remain unfinished.

Checkpoint: commit `5d642865653fbb3f4570666a64c7559439fbf117` (`refactor(scope): isolate core and supplementary evidence contracts`) records the tested Unit 2A implementation. The staged whitespace gate blocked the first attempt; the corrected commit passed. No push was performed.

## Unit 2B — Shared folds and restrained nested model benchmark (planned)

Problem: XGBoost/policy/calibration/SHAP/fairness stages create splitters independently; there is no immutable fold assignment, no common OOF model comparison and no approved baseline implementation. Current fixed XGBoost parameters are not the accepted nested-tuning protocol.

Root cause: model experiments evolved as independent report generators rather than consumers of one hashed split/search/metric contract. The only legacy baseline is an incomparable holdout logistic model.

Intended files (finalized after read-only audit):

- canonical config and versioned model search-space config;
- a shared-fold assignment module/artifact;
- one nested benchmark stage for XGBoost, Multinomial Logistic Regression, Random Forest and LightGBM;
- task-aware OOF metric and paired 5,000-resample stratified bootstrap utilities where reusable;
- build-stage registration only after the scientific contract passes;
- shared-fold, target/ID exclusion, preprocessing, tuning-isolation, OOF-exactly-once and paired-bootstrap tests;
- finalization logs and real trial-run metadata.

Acceptance criteria:

- every model consumes exactly the same immutable 10-fold outer assignment;
- each outer-test sample receives exactly one prediction and is absent from all training/tuning partitions for that prediction;
- inner CV/tuning occurs only inside each outer-training partition with a versioned restrained search space;
- target, identifiers and all canonical primary-policy exclusions never enter preprocessing/model inputs;
- preprocessing is fit only within training partitions;
- all four models use the same primary policy, labels, folds, metric definitions and OOF sample population;
- primary uncertainty is paired sample-level stratified bootstrap with 5,000 predeclared resamples, not fold-mean t intervals;
- selected parameters, fold membership, OOF probabilities/predictions, metrics, bootstrap intervals and paired model differences are persisted with run/config/input identities;
- a real-data trial run is labelled noncanonical until the full scoped package is rebuilt;
- if a baseline-minus-XGBoost paired OOF interval excludes zero for the predeclared primary comparison metric, stop and request the user's XAI-reference decision;
- focused/full tests and compile/hygiene gates pass; no manuscript edit occurs.

### Unit 1A Result — Passed

- Added a pinned acquisition manifest for four physical datasets and five logical tasks.
- Added a fail-closed canonical loader that verifies exact bytes, ordered schema, dimensions and raw target support and returns an actual-input receipt.
- Existing local input is always preferred only when it matches the configured and acquisition-manifest path; no interim discovery exists.
- Missing files cannot use recorded/unverified mirrors. Approved download requires both an approved URL and an explicit enable flag, validates a temporary candidate, and moves it atomically only after every check passes.
- Download mismatch produces a machine-readable comparison report and leaves the configured destination absent.
- The legacy loader now always validates the declared raw INX path and never writes/reads a global interim cache.
- Primary manuscript stages now call the config-backed canonical loader directly.
- Real local preflight verified all five logical tasks with zero downloads.
- Focused Unit 1A tests: 11 passed.
- Full pytest: 199 passed plus 4 subtests.
- Unittest: 161 passed.
- Compileall and diff hygiene: passed.

V2-001 and V2-014 remain open until Unit 1B binds receipts to manifests/cache and a real-data v2 build validates the entire chain. No issue is marked resolved prematurely.

Checkpoint note: commit `f4e2dd7` captured the tested Unit 1A implementation. Its staged diff check reported five Markdown trailing-whitespace lines, but the semicolon-separated shell command did not stop the commit. No scientific file or result was affected. The lines are corrected in a separate follow-up commit; the checkpoint is not amended or rewritten.

### Unit 2B implementation checkpoint — metric decision pending

- Added a deterministic, immutable shared-fold contract: one 10-fold outer assignment and three inner folds within each outer-training partition. Only hashed sample identities are persisted; raw employee identifiers are not.
- Added exact four-family model factories for Multinomial Logistic Regression, Random Forest, LightGBM and XGBoost. Every family uses the same train-fitted median/scale plus most-frequent/dense-one-hot preprocessing contract.
- Added a restrained nested benchmark stage with fail-entire-stage candidate handling, deterministic tie-breaking, exact-once OOF predictions, aligned class probabilities, fold-selected parameter/model persistence and transformed-feature lineage.
- Added 5,000-resample paired sample-level bootstrap infrastructure stratified jointly by outer fold and true class. Metric domains and higher/lower directions are explicit; fold summaries are descriptive only.
- Added the user-authorized baseline stop gate to orchestration. If a baseline's predeclared paired improvement interval has lower bound above zero, the benchmark stage is retained but the overall run fails before policy, calibration or SHAP execution and requests a model-reference decision.
- Bound the shared-fold loader receipt and model-grid path/hash/size to the scoped run manifest at stage boundaries. The model-grid hash is checked before configuration/data access and again before output persistence.
- Hardened model contracts: every canonical policy exclusion is materialized and guarded, target/ID cannot enter inputs, probabilities must be finite/bounded/normalized with exact rows/classes, estimator paths and parameter keys are allowlisted, random seeds belong to orchestration, all-null or invalid feature schemas fail, and native fits run under `threadpoolctl(limits=1)`.
- LightGBM row subsampling is now explicit (`subsample_freq=1`); the previous `subsample=0.9` setting was otherwise inactive under the installed LightGBM default.
- Real-data in-memory fold preflight passed for 1,200 samples: 1,200 outer assignments, 10 outer folds, 10,800 inner assignments and exactly three inner folds per outer fold. It wrote no artifact and is not a canonical scientific run. The dirty/predecision fold-contract hash was `827b18ed233b9400d3c3deaac046de6868b1a85629b4576b45d01bdc2cdbd8d8`; it must not be cited or reused after config/commit changes.
- Focused Unit 2B integration suite: 83 passed. Full pytest: 314 passed, 2 skipped, plus 4 subtests. Full unittest: 162 passed, 2 skipped. Compileall, diff hygiene and manuscript no-change checks passed.
- No model was fitted to the real dataset, no scientific result artifact was written, no network/API call occurred, and the manuscript was not edited.

Unit 2B is not scientifically complete. `selection_metric` and `baseline_gate_metric` intentionally remain null pending the user's A/B/C metric decision, so real nested benchmarking fails closed. Policy ablation, calibration, SHAP and subgroup/proxy stages still instantiate independent folds/models and must be refactored to consume this fold/model contract before either scope becomes release-ready. The obsolete fixed XGBoost block remains a documented open conflict until that consumer refactor; it is not used by the new benchmark stage.

Checkpoint: commit `8e9b5b9b9f66815abf7f9a599535a36737ea1706` (`feat(protocol): add shared nested OOF benchmark contract`) records this tested, deliberately predecision/fail-closed implementation. No push was performed.

### Unit 2B correction and real benchmark — in progress

Problem: the checkpoint encoded three inner folds, while the user's approved D2 protocol is five inner folds. The selection metric was also intentionally null pending a decision.

Root cause: the initial D2 handoff was interpreted as restrained nested tuning without carrying forward the user's exact 10×5 fold count. The benchmark hard-coded three inner folds in its validator/loop and tests, so changing only config would have produced a contract failure rather than the approved run.

Intended files: canonical/model-grid configs, shared-fold default and tests, benchmark selection/validation/tests, build gate validation, persistent logs, and a new versioned real benchmark trial directory after all tests pass.

Acceptance criteria:

- production and canonical tests require exactly 10 outer × 5 inner folds;
- macro-F1 alone drives primary selection and the baseline stop gate;
- QWK is secondary and used only inside the predeclared `0.001` macro-F1 tie pool;
- gate requires a positive point difference and paired OOF 95% CI lower bound above zero;
- verified INX bytes and all side inputs match scoped manifest receipts/hashes;
- exactly-once aligned OOF predictions exist for all 1,200 samples and four models;
- 5,000 paired stratified bootstrap resamples complete;
- artifacts, hashes, commands, runtime, failures and results are persisted;
- a triggered gate stops before downstream XAI; otherwise the approved plan continues;
- no API/network/manuscript edit occurs.

Correction implementation result before the real run:

- Frozen model-grid schema v3 and protocol `restrained_nested_tuning_v2_10x5`.
- Macro-F1 is the only primary inner-selection and gate metric. Candidates within `0.001` of the best mean macro-F1 are tie-broken by mean QWK, then candidate index. Both per-inner metrics, tie-pool membership and selection rationale are persisted.
- Production benchmark requires canonical outer folds=10, inner folds=5 and exact manuscript/model-grid protocol alignment before reading fold/data inputs.
- Gate logic and persisted verifier require both positive baseline-minus-XGBoost macro-F1 point estimate and paired OOF bootstrap CI lower bound above zero; no secondary metric is gate-eligible.
- Added a dedicated clean-worktree, offline, noncanonical benchmark-trial entrypoint. It runs only `shared_folds` and `model_benchmarks`, writes under `reports/manuscript_final/trials/<run_id>/core`, never updates `latest`, records complete/failed command runtimes, and denies TCP/UDP/DNS operations in process.
- Final trial validation requires 1,200 outer samples, 10×5 folds, 300 candidate rows, 40 selected rows, 40 fold rows, 4,800 exact-once OOF rows with normalized probabilities, 36 model-summary rows, 27 paired rows, a consistent 5,000-resample hash, exact gate recomputation, and 40 model files matching their index hashes/sizes.
- Independent review caught a production filename mismatch (`outer_fold_assignments.csv` versus actual `fold_assignments.csv`) before execution. The verifier now imports the production filename constant and its success fixture uses the same name.
- Corrected focused suite: 104 passed. Full pytest: 343 passed, 2 skipped, plus 4 subtests. Unittest: 162 passed, 2 skipped. Compileall, diff and manuscript no-change gates passed.
- Verified real-INX in-memory preflight: target support 194/874/132; 1,200 outer rows; 10 outer folds; 10,800 inner rows; five inner folds per outer fold; every inner validation partition has 216 samples. No artifacts or models were written.

The preflight was intentionally dirty and its fold hash `87a2798724d761e6b3a109ee6c33079e7d4d405d2b31ad3c260f0c9261afa6b7` is diagnostic only. The real trial must begin from the forthcoming clean checkpoint and will receive a new run-bound fold hash.

### Unit 2B final pre-run hardening

- Independent audit found and closed three pre-run provenance/contract gaps: the `0.001` practical-tie threshold is now an immutable validator constant; the bootstrap is frozen to 5,000 paired stratified percentile draws at 95% confidence with `outer_fold+y_true` strata and linear quantiles; every real-INX fold-metric row must report exactly 1,080 train and 120 test cases.
- `joblib` and `threadpoolctl` are now direct declared dependencies and manifest-recorded package versions because fitted-model serialization and single-thread fit control depend on them.
- The trial manifest records the fold, tie and bootstrap contracts and its final verifier requires exact equality.
- Focused benchmark/trial tests passed 41. Full pytest passed 345 with 2 historical skips and 4 subtests; unittest passed 162 with 2 skips; compileall, diff, manuscript-no-change and high-entropy secret scans passed.
- A fresh exact-command in-memory preflight verified the pinned INX bytes, 1,200 rows, support 194/874/132, 10×5 folds, 10,800 inner assignments, validation size 216, `joblib 1.5.3` and `threadpoolctl 3.6.0`. It wrote no file/model.

The standalone real trial command has not executed, `reports/manuscript_final/trials/` does not exist at this checkpoint, and no predictive result or gate outcome exists. This remains an engineering-only checkpoint. Raw-byte source-tree hashing is known to be EOL-sensitive across different checkout policies; the local trial remains hash-complete, but portable release verification is an explicit later blocker.

### Unit 2B real four-model benchmark trial

Checkpoint `6a80074c1402c11331cafc27a3bb5c1d8a2ed4c3` left a clean worktree. The offline noncanonical entrypoint then completed from the verified local INX bytes in 725.2 shell seconds (722.522 manifest seconds), with no network/API call and no `latest` update. Run identity:

- run: `benchmark-10x5-20260713-6a80074`;
- config: `7e70bf6646a542ad32e10ab3718654aa8232a46e44e2083ed10e2cfe526da595`;
- scientific input: `8be7c5d79f2b39af3e04f1c8a14a0ae70d2180c48c425595f69f23ac2e76b34a`;
- INX: `b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a`;
- model grid: `d8fb0584d7106f8941ca8e1b0e0a9c58e13f39519c6c75b383ff333d92d41617`;
- fold contract: `9fd24f0c7d499f0f2055d4f850e6342325ab5046a378b796ebca68e12a06f023`;
- bootstrap resamples: `3528e4377f9907c2a7c1b51806abf09e0ec2d5513da630618b0a0cb9950a37d0`.
- completed manifest: `1b4c3381489f8b0bf7ae60d57280b3ddd5aa5344cb250b1df63fdaaa6cc7379c`.

The complete verifier passed after the run: 53 registered outputs plus the manifest (54 physical files, 91,820,515 bytes), 40 model hashes, 4,800 exactly-once OOF predictions, 300 candidate rows, 40 selected rows, 36 model-summary rows, 27 paired rows and 5,000 valid paired stratified draws.

Primary macro-F1 results (paired-bootstrap 95% sample-level intervals):

| Model | Macro-F1 | 95% CI |
| --- | ---: | ---: |
| XGBoost | 0.621021 | 0.597319–0.644690 |
| LightGBM | 0.605488 | 0.583315–0.629174 |
| Random Forest | 0.592340 | 0.579571–0.604757 |
| Logistic Regression | 0.506221 | 0.480283–0.531841 |

Baseline-minus-XGBoost macro-F1 differences were LightGBM `-0.015533` (`-0.038121–0.006382`), Random Forest `-0.028681` (`-0.049949–-0.008049`) and Logistic Regression `-0.114800` (`-0.147597–-0.083224`). The gate JSON and recomputation both report `gate_triggered=false`; the approved reference-model decision gate therefore does not pause downstream work.

Secondary QWK point estimates were XGBoost `0.567602`, LightGBM `0.588329`, Random Forest `0.631678` and Logistic Regression `0.371011`. Random Forest's better secondary ordinal score is retained and must not be hidden, but QWK is not gate-eligible under the predeclared macro-F1 protocol.

Warning audit: sklearn repeatedly warned that some XGBoost probabilities did not sum to one at its float64 warning tolerance, although the stage contract allowed absolute deviation up to `1e-6`. Persisted maximum deviation was `8.3819e-08`. Explicit float64 row renormalization changed no argmax and changed aggregate XGBoost log loss by `1.8097e-10`; macro-F1/QWK selection and the gate are exactly unaffected. LightGBM also emitted feature-name metadata warnings while using position-stable transformed arrays. Both warning paths must be cleaned and tested before canonical probability outputs; the completed trial remains immutable and noncanonical.

Independent replay regenerated the fold mapping and all three paired macro-F1 comparisons from the persisted OOF rows; differences matched the package within `1e-16`, and registered-file/path/hash mismatches were zero. `PyYAML`, `openpyxl` and `xlrd` are recorded as `not_installed`; none was used by this JSON-compatible-config/CSV two-stage trial, so the gate remains valid, while clean-install/lock readiness remains open.

### Unit 2C-0 warning-hygiene plan — recorded before modification

Problem/root cause: aligned XGBoost soft probabilities are converted to float64 but not renormalized; the repository accepts `1e-6` row-sum error while sklearn warns at a tighter float64 tolerance. The common transformer emits unnamed ndarrays, while LightGBM exposes synthetic fitted feature names and warns on every ndarray prediction. Inner selection computes full probability metrics even though both accepted selection metrics are label-only.

Intended files: `src/models/canonical_models.py`, `src/experiments/manuscript_model_benchmark.py`, `configs/manuscript_final.yaml`, focused canonical-model/benchmark tests, and persistent logs. No completed trial artifact will be edited or overwritten.

Acceptance criteria/tests:

- aligned output is finite, clipped only within the existing numerical bound, reordered, float64 row-normalized to machine precision and warning-free in sklearn log loss;
- normalization changes no argmax for the observed numerical-drift pattern and invalid row sums still fail closed;
- every canonical model receives the same dense named pandas transformation, with deterministic feature order/lineage; LightGBM predict/proba emits no feature-name warning;
- label-only inner selection does not call `predict_proba` or calculate unrelated probability metrics;
- focused/full tests, compileall/diff/manuscript gates pass;
- the completed noncanonical run stays immutable and is not rerun solely for numerically negligible warnings.

Unit 2C-0 implementation result:

- The common dense ColumnTransformer now emits pandas output with deterministic transformed feature names for all four model families. This removes LightGBM's synthetic-name/unnamed-array mismatch without suppressing warnings or changing transformed values.
- Probability alignment now converts to float64, enforces the existing finite/range/sum boundary, clips only accepted numerical spillover, normalizes each row, and verifies machine-precision simplex sums.
- Manuscript config declares and benchmark validation freezes `pandas_named_dense` and `global_label_order_float64_clip_then_row_normalize`.
- Frozen label-only inner metrics no longer invoke `predict_proba` or calculate unrelated log-loss/Brier/ECE values.
- Focused suite: 63 passed. Full pytest: 350 passed, 2 skipped, plus 4 subtests. Unittest: 162 passed, 2 skipped. Compileall/diff/manuscript/secret gates passed.
- Real outer-fold-1 replay refit the selected LightGBM and XGBoost candidates in memory. Both reproduced 120/120 stored labels with no warnings; LightGBM probability delta was `3.33e-16`, XGBoost delta `7.02e-08` from normalization, and maximum new row-sum deviation `2.22e-16`.

No scientific artifact was produced or altered. The previous trial remains bound to commit `6a80074`; later canonical evidence will be regenerated under the warning-clean commit.

### Unit 2C-A exact-model OOF SHAP plan — recorded before modification

Problem/root cause: the existing manuscript SHAP stage creates its own stratified folds and refits a legacy fixed-parameter XGBoost pipeline. Its transformer/group mapping predates the canonical `numeric`/`categorical` nested pipelines, and its 45 dependent fold pairs receive t-based confidence intervals. Consequently it cannot prove that any local attribution came from the exact model that produced that sample's nested OOF prediction.

Intended files: a new reusable benchmark artifact contract, a canonical fitted-transformer axis/group helper, `src/experiments/manuscript_shap_evidence.py`, core-builder stage wiring, mandatory exact-fold/no-refit/lineage/policy/identity tests, and persistent logs. The old trial may be read only for reader/replay validation; it cannot supply new canonical SHAP evidence after the code identity changes.

Acceptance criteria:

- all ten XGBoost joblib paths, sizes and SHA-256 values are verified before load, and run/config/scientific-input/fold identities agree across gate, folds, OOF rows, selections, model index and lineage;
- every sample is explained once by its assigned outer-fold pipeline; no estimator/preprocessor fit occurs in the SHAP stage;
- loaded pipelines replay persisted XGBoost labels/probabilities within an explicit tolerance and expose the exact canonical raw-feature order and transformed lineage;
- each transformed index maps exactly once to a raw feature family, with no gap/overlap; grouped and transformed SHAP sums agree for every sample/class;
- forbidden policy fields are absent from raw, transformed, global, local, case and reason-code evidence;
- stability retains all 45 fold pairs but reports descriptive mean/SD/median/range only, explicitly making no independent-pair confidence-interval claim;
- every output carries run/config/scientific-input/fold/model-set identity and per-fold outputs carry the exact model SHA/candidate index;
- tamper, path traversal, wrong-fold, wrong-class, prediction drift, feature-order/lineage drift and no-refit tests fail closed;
- focused/full pytest, unittest, compileall, diff, path/secret/manuscript checks pass before checkpoint; no API/network/manuscript edit occurs.

Unit 2C-A implementation result:

- Added a fail-closed reader that binds shared 10x5 folds, non-triggered macro-F1 gate, three paired baseline rows, selected candidates, OOF rows, joblib path/hash/size, exact pipeline parameters/classes, transformed lineage and deterministic model-set hash before exposing ten XGBoost models.
- Replay rechecks model bytes and reproduces every OOF label/probability; the canonical SHAP stage tightens replay tolerance to `1e-12` and contains no `.fit` or splitter path.
- Added a fitted canonical axis contract based on `transformers_`, `output_indices_`, nested one-hot categories and exact feature-name lineage. Every transformed index has one raw owner, forbidden names fail closed, multiclass layouts normalize explicitly and grouped sums are checked per sample/class.
- SHAP uses each sample's exact outer-fold pipeline, carries run/config/scientific-input/fold/model-set plus per-fold model/candidate identity, and publishes through a temporary directory atomically. The builder-owned empty stage-shell contract and late-failure cleanup are tested.
- All 45 fold pairs remain visible, but stability summaries now use mean/SD/median/min/max and explicitly mark confidence intervals inapplicable for dependent pairs.
- Historical reader/OOF replay passed at its documented tolerance, while its old ndarray preprocessing failed the new one-hot lineage contract as intended. A current-code real fold-1 refit produced warning-free SHAP with exact grouped-sum preservation; no result artifact was saved.
- Focused suite: 59 passed plus 4 subtests. Full pytest: 389 passed, 2 skipped, plus 4 subtests. Unittest: 164 passed, 2 skipped. Compileall/diff/manuscript/secret/path/large-file gates passed.

No canonical SHAP artifact was generated. The final package must regenerate benchmark and SHAP under one clean current commit/run/config identity; the 91.8 MB historical trial remains unstaged and immutable.

### Unit 2D shared-fold leakage-policy ablation plan — recorded before modification

Problem/root cause: the legacy policy stage independently creates `StratifiedKFold` splits, fits a fixed-parameter legacy XGBoost wrapper, writes a separate fold assignment, treats fold means as independent population samples through t intervals, and uses fold-level Wilcoxon tests. It therefore cannot establish same-sample paired policy differences under the accepted 10×5 benchmark/run contract.

Accepted scientific consequence: D2 explicitly requires that leakage policies are not independently tuned. Each non-primary policy will use the primary policy's selected XGBoost candidate for the corresponding outer fold, so folds, model capacity and hyperparameter-selection evidence are held fixed. The primary policy will reuse the exact benchmark OOF predictions rather than silently refitting a second version. This is a matched sensitivity analysis, not a separately optimized policy leaderboard.

Intended files: `src/experiments/manuscript_policy_ablation.py`, core-builder stage wiring, canonical configuration only where a missing contract must be frozen, policy/shared-fold/bootstrap/identity tests, and persistent logs. Existing historical artifacts will not be edited or copied into new evidence.

Acceptance criteria:

- consume the exact shared 10-fold assignment and compatible benchmark identity; instantiate no independent splitter;
- bind every policy/fold fit to that fold's primary-selected XGBoost fixed/candidate parameters, with target/ID exclusion and preprocessing fitted only on the outer-train partition;
- reuse exact benchmark OOF rows for the primary policy, and produce one prediction per sample for every other declared policy;
- preserve diagnostic/audit-only roles and exact exclusions; never present the full-feature upper bound as deployable evidence;
- compute primary uncertainty and pairwise policy differences from the accepted 5,000 paired, sample-level, outer-fold-plus-class-stratified bootstrap; keep fold metrics as descriptive variability only;
- encode metric direction/domain, denominators, resample identity and run/config/scientific-input/fold/model identities in machine-readable outputs;
- generate all required policy tables/figure source/PNG/SVG atomically with relative metadata paths and no stale fallback;
- fail on identity drift, missing/tampered benchmark inputs, fold mismatch, forbidden primary features, duplicate/missing OOF predictions, hyperparameter drift or invalid uncertainty;
- focused/full pytest, unittest, compileall, scientific-diff/path/secret/manuscript checks pass before checkpoint; no API/network/manuscript edit occurs.

Read-only cross-config audit addendum: `configs/feature_sets.yaml` still maps three policy names shared with the canonical config to narrower legacy exclusions (omitting Gender and MaritalStatus), and uses obsolete “leakage-safe” wording. This violates the one-name/one-policy acceptance rule even though the new core runner reads only `manuscript_final.yaml`. Unit 2D will align the shared names, preserve genuinely distinct legacy audit variants under distinct names, and add a repository-level cross-config regression check rather than weakening the canonical policy.

### Unit 2D implementation and independent-review result

- Replaced the canonical policy runner's independent splitter and legacy fixed XGBoost path with explicit current-run `shared_folds` and `model_benchmarks` inputs. Builder wiring verifies both upstream stage contracts before invocation.
- The primary policy copies the exact benchmark OOF rows only after all ten persisted models, parameters, hashes and identities are validated and their complete OOF predictions replay at `1e-12` tolerance. The five non-primary policies fit on the exact 1,080-row outer-training partitions with the same fold's primary-selected candidate, canonical named preprocessing, every policy exclusion supplied as forbidden input, and a one-thread fit limit.
- Frozen the six-policy contrast order: full information-rich comparator → no salary hike → sensitive-retaining no attrition → governed no attrition → department-free primary → job-role-free sensitivity. Each adjacent contrast has a defined interpretation; all pairwise intervals remain pointwise with no multiplicity-adjusted rejection claim.
- Added exactly-once policy OOF predictions, per-fold descriptive metrics, raw bootstrap metric intervals, wide manuscript summaries, paired policy differences, finite-domain-normalized and native-scale leakage sensitivity, exact feature contracts, hyperparameter schedules, fit receipts, interpretation, figure source/PNG/SVG and relative-path metadata. Publication is atomic and late failure leaves no partial target.
- One deterministic 5,000-draw paired OOF bootstrap is used for all policy systems. Its resample hash must exactly equal the benchmark gate's resample hash; every reported valid-draw denominator must be 5,000. Fold variability fields are mean/SD/min/max only.
- Aligned every shared policy name in `configs/feature_sets.yaml` to the canonical exclusions, labelled it a legacy compatibility projection, validated drift during direct and orchestrated runs, and added it to the scoped scientific side-input hash/snapshot chain. Active README/config/policy terminology is leakage-aware.
- Rewrote README status so the old dirty v1 `latest` package is clearly historical, LLM/chatbot are excluded from core, counterfactuals are supplementary-only, and the verified 10×5 trial is decision evidence rather than a canonical release.
- Independent review reported no critical statistical flaw. Its seven findings (raw interval persistence, late atomic failure, direct projection validation, primary replay, side-input hashing, valid-draw derivation and exact six-policy scope) were all implemented and regression-tested.
- Real-INX bounded diagnostic refit fold 1 for all five non-primary policies using the historical fold-1 selected parameter schedule under current code: every fit used 1,080 train/120 test rows, feature/parameter lineage passed, probability simplex error was at most `2.22e-16`, warnings were zero, network was process-blocked, files written were zero and the historical manifest hash was unchanged. This is implementation evidence only, not a manuscript result.
- Final gates: 92 focused tests passed; full pytest 403 passed plus 4 subtests with 2 historical skips; unittest 174 passed with 2 skips; compileall, diff, manuscript no-change, secret, absolute-path, 100 MB candidate, README-link and active leakage-aware terminology scans passed.

No canonical policy artifact was generated because the historical benchmark is intentionally ineligible for current `1e-12` replay/lineage contracts. Policy evidence must be regenerated together with the benchmark in the final clean all-stage run.

Checkpoint: commit `984db46` (`feat(policy): bind leakage ablation to shared OOF evidence`) records the fully tested Unit 2D implementation, README correction, issue/artifact maps and exact command/test history. The 91.8 MB local historical trial remained untracked and unstaged; no push was performed.

### Unit 2E calibration audit — no modification

The current calibration module is not v2-admissible: it regenerates folds (1,091/1,200 differ from shared assignments), uses legacy fixed XGBoost settings, ranks raw/sigmoid/isotonic with outer-test results, reports fold-t intervals (including a historical negative severe-error lower bound), writes absolute paths and is non-atomic. Builder wiring supplies no upstream identities.

The recommended correction is five-inner-fold OOF cross-fitted sigmoid training inside each outer-training partition, followed by application to the exact persisted full-outer-train XGBoost probabilities. It adds 50 fits and preserves one exact primary base model. A single 20% holdout would add 10 fits but evaluates a different base model trained on only 80% of outer training data. Warning-free bootstrap timing measured about 93 seconds for two methods × nine metrics × 5,000 draws; total estimates are 2–3 minutes versus 1.5–2 minutes. This material choice is awaiting the user; no calibration file was modified.

### Unit 2E option-A implementation plan — recorded before modification

User decision: five-inner-fold outer-training-only cross-fitted sigmoid calibration. The exact persisted benchmark XGBoost outer-fold model and its untouched outer-test probabilities must be preserved. No outer-test value may affect tuning, calibrator fitting, method selection or thresholds.

Root causes to remove: the legacy runner creates its own outer splits, uses one 20% holdout and a reduced-training base model, uses legacy fixed parameters, evaluates raw/sigmoid/isotonic and selects among them from outer-test metrics, computes fold-t intervals, lacks upstream identities, publishes absolute paths and is non-atomic.

Intended files: `configs/manuscript_final.yaml`, `src/governance/manuscript_contract.py`, `src/experiments/manuscript_calibration.py`, `src/experiments/build_manuscript_evidence.py`, calibration/stage-runner contract tests, and persistent logs. Benchmark models/folds and the manuscript are out of scope and must not change.

Acceptance criteria:

- consume only current-run shared 10×5 folds and exact benchmark XGBoost artifacts after `1e-12` OOF replay;
- run exactly 50 inner fits using each outer fold's selected fixed/candidate parameters and model seed, with preprocessing fitted only on inner development rows;
- cover every outer-training sample once in cross-fit predictions and exclude every corresponding outer-test sample;
- fit exactly ten one-vs-rest logit sigmoid calibrators only from cross-fit rows; fail on missing/degenerate class evidence;
- reuse exact benchmark raw outer-test rows, apply the calibrator without outer-model refit, and record calibrator-to-model hashes/denominators;
- keep sigmoid fixed regardless of outer-test metrics, exclude isotonic and threshold selection, and use declared argmax labels;
- persist exactly-once raw/sigmoid OOF rows, fold-descriptive metrics, 5,000-draw paired sample-level intervals/differences, complete denominators and the benchmark resample hash;
- generate 10-bin class reliability evidence, Figure 5 PNG/SVG/source, rationale/warnings and relative-path metadata atomically;
- fail before/without partial output on input, identity, fold, parameter, probability, denominator or publication errors;
- pass focused tests, bounded verified-INX diagnostics, independent review, full pytest/unittest/compileall and hygiene gates before checkpoint.

### Unit 2E option-A implementation result - 2026-07-13

- Replaced the legacy independent-split calibration path with the exact shared 10x5 contract. Each outer fold receives five selected-candidate inner-development fits and one cross-fitted probability row per outer-training sample; outer-test rows cannot enter fitting.
- Fits exactly one predeclared one-vs-rest logit sigmoid per outer fold and applies it only to the exact persisted benchmark XGBoost outer-test probabilities. The stage never refits the outer model, selects a method, selects a threshold or includes isotonic.
- Persists 50 fit receipts, 10 calibrator/model relationships, 30 class parameter rows, 10,800 cross-fit rows, 1,200 raw plus 1,200 sigmoid OOF rows, 20 descriptive fold rows, raw 5,000-draw intervals/differences, 60 reliability bins, complete denominators, protocol/rationale/validation metadata and PNG/SVG sources through atomic staging.
- Raw benchmark floats are never silently renormalized. Persisted values and calibrator parameters are re-read with round-trip float parsing and replayed exactly; calibrated values are explicitly row-normalized.
- Froze the paired 95% sample bootstrap strata to `(outer_fold, y_true)`, reuses the benchmark resample hash, removes fold-t inference and renders percentile interval endpoints even when a valid interval does not contain its observed point.
- Bounded scikit-learn to `>=1.8,<1.9` for its warning-free L2 API and limits both XGBoost and sigmoid numerical fits to one thread. All model/calibrator warnings fail closed and thread limits are recorded.
- Re-reads and replays the config, actual dataset hash, shared folds, benchmark files and all ten model bytes immediately before atomic publication, blocking any mid-run upstream mutation.
- Independent code review found no remaining material implementation defect. Its findings on float replay, bootstrap freezing, dependency API, solver determinism, upstream races and CI rendering were implemented and regression-tested.
- Expanded focused suite passed 85 tests plus 7 subtests; independent review passed 48 plus 7 subtests. A real INX fold-1 diagnostic passed five fits, 1,080 cross-fit rows, 120 untouched test rows, zero warnings and `2.22e-16` maximum calibrated simplex error in 1.082 fit seconds.
- No canonical calibration artifact was generated. Current config `d755ecc3...` correctly rejects historical benchmark config `7e70bf66...`; rerunning the expensive benchmark now would be invalidated by planned downstream core-config changes. Same-config real execution remains required after complete core input freeze.
- Full-suite collection exposed one supplementary dependency on the deleted legacy calibration `_fit_pipeline`. The counterfactual module now owns a clearly supplementary, one-thread canonical-model fit helper and a regression forbids private calibration imports. This restores module isolation without generating or admitting counterfactual evidence; the full suite then passed.
- A final all-ten-fold real-INX in-memory diagnostic exercised the complete 50-fit/calibrator binding loop against the immutable historical bundle: 10,800 cross-fit rows, 50 receipts, 30 class-parameter rows, ten distinct calibrator hashes, ten source model hashes, 2,400 evaluation rows and zero warnings in 12.303 seconds. It deliberately retained historical raw float32 probabilities and wrote no artifact; it is implementation evidence only.

Checkpoint: commit `0f820b3` (`feat(calibration): cross-fit sigmoid on benchmark folds`) records the fully tested Unit 2E implementation, dependency isolation fix, tests and traceability. The 91.8 MB local historical trial remained untracked and unstaged; no push was performed.

### Unit 2F support-aware subgroup/proxy plan - recorded before modification

Problem/root cause: the existing stage creates independent performance and proxy splitters, refits a legacy fixed XGBoost instead of consuming exact policy OOF evidence, publishes proxy fold-t intervals, lacks scientific/fold/model/upstream identities, can silently lower the bootstrap denominator or skip configured audit attributes, writes absolute paths non-atomically, and treats two identical post-safeguard proxy feature sets as independent systems. The v1 package is historical evidence only; its folds disagree with the verified shared assignment for 1,091/1,200 samples.

Intended files: `configs/manuscript_final.yaml`, `src/governance/manuscript_contract.py`, `src/experiments/manuscript_fairness_proxy.py`, `src/experiments/build_manuscript_evidence.py`, focused subgroup/proxy/upstream/publication tests, and persistent logs. Existing benchmark/policy/calibration/SHAP artifacts and the manuscript are out of scope and must not change.

Acceptance criteria:

- require compatible current-run `shared_folds`, `model_benchmarks` and `policy_ablation` stages plus run/config/scientific-input identity; reject historical discovery or drift;
- read the three declared performance-policy OOF systems exactly once per sample from `policy_ablation/oof_predictions.csv`; validate fold, target, probability, dataset, model-set, feature/schedule and upstream hashes; fit no performance estimator;
- join every configured sensitive/audit-only attribute by validated sample identity; missing attributes fail closed and no audit field enters model input;
- preserve visible group counts, metric-specific numerators/denominators, minimum support, sensitive-versus-operational category, insufficient/unstable/wide/supported status and non-fairness/non-causality limitations;
- use raw policy OOF probabilities consistently; regenerate the exact 5,000-draw `(outer_fold,y_true)` plan and require its hash to equal the policy/benchmark plan; compute all policy rows and any paired policy-gap sensitivity from the same draws;
- freeze eligible group sets from full OOF evidence, count mathematically invalid draws, and prevent insufficient/unstable/wide rows from entering unqualified headline summaries;
- use the exact shared outer-fold assignment for department reconstructability, training-only preprocessing and a frozen balanced logistic-regression contract; exclude target/ID from raw/transformed predictors and emit exactly-once proxy OOF rows;
- fit only two unique proxy predictor contracts; publish the department-including name as an explicit alias of the department-free post-safeguard feature hash and verify identical OOF evidence rather than refitting it;
- replace fold-t inference with one task-specific 5,000-draw paired sample-level percentile plan stratified by `(outer_fold, EmpDepartment)`; publish interval/difference denominators and its distinct resample hash; keep folds descriptive only;
- label full-sample watchlist associations as exploratory proxy-risk associations, never performance-model use, discrimination, causality or fairness proof;
- publish complete relative-path metadata, upstream receipts, OOF/support/interval/equivalence tables and interpretation through temporary staging; revalidate inputs before atomic publication and leave no partial target on failure;
- pass focused tests, a bounded verified-INX diagnostic, independent review, full pytest/unittest/compileall and path/secret/manuscript/large-file gates before an authorized checkpoint; no API/network/manuscript operation occurs.

Compute estimate: zero performance-model fits, 20 unique lightweight proxy logistic fits, two 5,000-draw plans and 1,200 samples. Expected current-laptop runtime is roughly 2-5 minutes and conservatively below 10 minutes, so the material compute decision gate is not triggered.

### Unit 2F implementation and validation

- `fairness.prediction_contract` now requires current-run `shared_folds`, exact `model_benchmarks` XGBoost provenance and exact raw `policy_ablation` OOF rows. The stage cannot fit or split a performance model.
- The policy reader revalidates run/config/scientific/fold/model/data identity, exact primary benchmark labels/probabilities, all three feature contracts, selected candidates/parameters, persisted source-model hashes, schedule receipts and the 5,000-draw performance resample hash shared by benchmark and policy evidence. CSV probability reads use round-trip precision.
- Complete-OOF support eligibility is frozen before resampling. Individual gaps use that fixed set; paired policy gaps are recomputed over the intersection of groups eligible in both policies for the same attribute/metric/class. Fewer than two common groups yields an explicit insufficient-support status and NaN point/interval rather than a mismatched estimand.
- The accepted 5,000-draw `(outer_fold,y_true)` plan is reconstructed centrally and must equal both upstream hashes. String groups are encoded to compact integer codes and processed deterministically in batches of 200. Outputs report requested/valid draws, group/denominator minima, common group sets, status, pointwise scope and no multiplicity adjustment. Paired rows are never headline eligible.
- Department reconstructability uses the exact shared outer folds, outer-train-only preprocessing and warning-free balanced logistic regression. Only two unique target-free predictor contracts are fitted (20 fits total); `no_salary_hike_no_attrition` is a no-fit alias of the department-free contract. Raw and transformed lineage must exclude EmpDepartment, PerformanceRating and EmpNumber.
- Proxy uncertainty uses a separate 5,000-draw plan stratified semantically by `(outer_fold,EmpDepartment)`. The central utility's required `y_true` name is governed by a hashed deterministic sorted-class-code adapter with `performance_target_used=false`. Metric draws are batched at 200 and paired differences reuse the identical plan.
- All intervals and observed rankings are pointwise descriptive; no familywise/FDR, causal, discrimination, fairness-guarantee or legal claim is permitted. Publication uses relative metadata, staging, late upstream revalidation and atomic replacement of only an absent/empty builder-owned target.

Focused suites passed 79/79 and then 84/84 after statistical hardening; the direct scientific-behavior subset passed 31/31. Dynamic tests cover common-group insufficiency, identical-system exact-zero/order invariance, batch equivalence, exact shared-fold proxy OOF, exactly 20 real sklearn fits, alias/no-fit receipt and raw/transformed target/ID exclusion. Compilation, canonical config validation and diff checks passed. The intermediate pre-review config hash was `be2b3f9f7e052df42ad9dc413d10e29bfc2ad6dd63a38513d21957aa9908523f`; it is superseded by the final contract below.

The intermediate pre-review verified real-INX in-memory diagnostic loaded 1,200 rows/28 columns with dataset hash `b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a`. Its regenerated approved outer mapping exactly matched the immutable trial. Twenty warning-free fits and a 5,000-draw target bootstrap completed in 5.233 measured seconds; 2,400 OOF rows were exactly once, RSS was 239.64 MiB before and 234.00 MiB after, and no file/network/API operation occurred. Diagnostic macro-F1 was `0.968543` (pointwise 95% CI `0.956635-0.980215`) with job role retained and `0.247368` (`0.226694-0.268709`) without it. These values are implementation diagnostics only and cannot enter the manuscript until the complete same-config clean run regenerates the stage and final manifest; the final post-review diagnostic is recorded below.

Independent review identified no P0 defect and four P1 contract gaps: boolean seeds, ambiguous proxy-task identity, identity-free proxy label mapping, and missing department-class support context. The implementation now rejects boolean seeds; uses a dedicated nominal multiclass proxy task schema throughout config/code/artifacts; identity-binds the mapping; reports overall and fold-specific class support/zero cells; sources the watchlist from canonical config; records the conditional inference scope; validates persisted feature counts; and joins audit attributes by explicit sample index. A follow-up independent review reran 102 focused tests and found no remaining P0/P1 issue.

Final validation under config hash `3c9588c1327ac563a85586835b19b30768860165dc26b61fcf7aafbce3bb1421` passed: 44 direct focused tests; 102 expanded integration tests; full pytest 467 passed with 2 historical skips and 11 subtests; unittest 178 tests with 2 skips; compileall, diff/manuscript, secret, portable-path, 100 MB, terminology, legacy-import, README-link, issue-register and dependency gates. The final verified-INX diagnostic retained 20 fits and 2,400 exactly-once OOF rows, completed 5,000 batched draws in 3.776 seconds, and now records minimum overall department support 20, minimum nonzero outer-test support 1, and two zero-support fold/class cells. File writes and network/API access were blocked.

No Unit 2F scientific artifact has been generated or admitted. The stage remains fail-closed until benchmark and policy upstreams are regenerated under the same final config/run/scientific identity.

Checkpoint: commit `a490d1e` (`feat(fairness): bind subgroup and proxy diagnostics to OOF evidence`) records the fully tested Unit 2F implementation, tests and traceability. Immediate tracked status was clean; the 91.8 MB local historical trial remained untracked/unstaged, and no push was performed.

### Unit 2G HRDataset_v14 external-replication audit and preimplementation contract

The verified local HRDataset file matches the acquisition contract exactly: SHA-256 `cb19996755c93c0a8d6527f59da4701c80aef65eff854906546dce286249813c`, 311 rows × 36 columns. The explicit mapping hash is `4988bde12fbd0198102f22f4078fd31ba20ea4285160363d9cf101610e9f19d0`; raw target counts Exceeds/Fully Meets/Needs Improvement/PIP are 37/243/18/13 and mapped class 2/3/4 support is 31/243/37 with no unmapped value. Ten outer × five inner folds are feasible. Source authenticity and licence remain manual-review items; no download was attempted.

The pre-modification audit rejects current external evidence for v2. The module runs static rather than nested XGBoost, uses fold-normal intervals rather than 5,000 sample-level paired resamples, lacks predeclared sigmoid replication, has incomplete OOF SHAP stability and support-aware subgroup/proxy uncertainty, imports legacy fairness/SHAP utilities, writes non-atomically and omits scientific/fold/model-grid/model identities. Its policy aliases are false (`DeptID` survives department removal and `PositionID` survives job-role removal), with additional sensitive/status/date aliases and unresolved watchlist-field treatment. Existing focused tests pass 19/19 but do not cover these defects. Historical external packages are stale, mixed-scope and inadmissible.

Intended files after the user policy decision: canonical external protocol/config and schema mapping; neutral external fold/model/calibration/SHAP/subgroup-proxy implementation or carefully reused current v2 primitives; core builder wiring; task/claim/provenance validators; external contract/scientific-behavior/atomic-publication tests; traceability. The manuscript and historical artifacts remain out of scope.

Acceptance criteria:

- freeze one exact HRDataset primary policy and diagnostic/strict variants in canonical config, removing all direct aliases and forbidding sensitive, ID, target, post-outcome and raw-date fields;
- generate/hash deterministic dataset-specific 10×5 folds; tune XGBoost within each outer train on macro-F1 with QWK tie-break; persist candidate/selection/model/lineage receipts and exactly-once OOF rows;
- fit the fixed sigmoid calibrator only on five-fold cross-fitted outer-training probabilities and apply it to the corresponding untouched outer-test model; no method/threshold selection;
- compute 5,000-draw target-stratified sample-level pointwise intervals and paired policy differences; keep fold variability descriptive and record conditional inference;
- compute exact prediction-model grouped OOF SHAP with class/global tables, fold rankings, descriptive stability and deterministic local cases; forbid removed aliases;
- use support-aware, denominator-aware external subgroup/proxy diagnostics with valid bootstrap counts/status/limitations and no fairness/causal claim; remove legacy fairness dependency;
- preserve the computed three-safe-feature INX transport infeasibility as a claim-boundary artifact, never a transported-model result;
- bind run/config/scientific-input/data/mapping/acquisition/model-grid/fold/model identities, late-revalidate every input, publish atomically with portable paths and reject all historical discovery;
- keep IBM performance/attrition and Turnover out of core output; no LLM/chatbot/counterfactual/API/network path;
- pass focused contracts, bounded real-input diagnostics, independent review, full test/compile/hygiene gates before checkpoint.

The pre-implementation estimate was approximately 480 small-data fits and 3–6 minutes. A later exact execution-path audit corrected the production count to 500 XGBoost fits (400 candidate search, 50 outer-policy, 50 calibration cross-fitting) plus 30 one-vs-rest sigmoid logistic fits; the 3–6 minute runtime estimate remains plausible. At this historical checkpoint the feature-policy choice was still awaiting the user; the following decision record documents its subsequent acceptance.

Policy decision: the user accepted option A. The primary is `conservative_primary`; `department_including_audit`, `job_role_free_audit`, `proxy_rich_audit` and `temporality_restricted_audit` are explicit non-primary sensitivity roles. Salary/State/Zip/RecruitmentSource and all identified aliases are forbidden from the primary. `proxy_rich_audit` may restore only scientifically declared proxy fields and remains non-primary; Zip remains forbidden as a quasi-identifier. The temporality sensitivity removes EngagementSurvey, EmpJobSatisfaction, SpecialProjectsCount, DaysLateLast30 and Absences. Mapping/support are unchanged and the read-only preflight indicates technical feasibility, so no stop condition is currently triggered.

### Unit 2G implementation checkpoint candidate

Implemented files include the canonical config/provenance/schema/card; deterministic external adaptation; `hrdataset_replication_core.py`; `hrdataset_replication_diagnostics.py`; the atomic `manuscript_hrdataset_replication.py` orchestrator; external contract validation; core builder routing; legacy supplementary compatibility; and focused config, nested-selection, calibration, bootstrap, SHAP, subgroup/proxy, stage and manifest tests.

The primary contract is exact: `EmpJobRole`, `EngagementSurvey`, `EmpJobSatisfaction`, `SpecialProjectsCount`, `DaysLateLast30`, `Absences`, and `ExperienceYearsAtThisCompany`. The adapter records two invalid negative tenure durations as missing and rejects any count drift. Five policies share deterministic outer folds and each OOF row binds the exact source model hash. Cross-fitted sigmoid probabilities use only outer-training inner-OOF rows. SHAP uses the same serialized fold model and checks replay/additivity/axis/lineage. The same 5,000 target-stratified resample plan serves estimates and paired policy comparisons.

Publication is fail-closed and atomic. Config, core scope, actual dataset receipts, exact six side inputs, composite scientific identity, Git HEAD, source tree and clean worktree are checked independently. Output paths are portable; core-scope scans prohibit LLM/chatbot/counterfactual/IBM/Turnover content. The Unit 2G stage exposes no network or paid-API path; the still-missing global core runtime network denial remains tracked as V2-020.

Test state at that pre-interruption checkpoint: the broad Unit 2G run completed with 120 passed; the exact stage/manifest focus passed 12 and the selected external suite passed 68. The then-current review reported no remaining P0/P1, but that conclusion was superseded by the later alias/probability and publication-trust audits documented below. No real scientific artifact had been written, so issue V2-028 remained `implementation_complete_pending_real_run`.

### Unit 2G interruption recovery and remaining resume gate

The usage-limit interruption left no running process or partial scientific output. All source/test/schema changes through exact output-root binding and fail-closed date derivation are present. Recovery found no staged file, lock or temporary stage; only the immutable historical trial remains untracked.

The first full checkpoint suite exposed 11 integration regressions (519 passed): external validation ran before legacy seed-shape errors, and temporary manifest projects resolved external side inputs against the real repository. Both root causes were fixed without weakening validation; focused tests passed 64. Independent review subsequently found and closed stage-level builder-output and broad-output-root gaps. The remaining P1 is orchestration resume: `build()` creates a provisional dirty manifest and rejects it before loading an explicit interrupted run's original clean manifest. Unit 2G is not checkpoint-complete until exact-run-root resume is identity-validated and tested.

### Unit 2G interruption recovery, production-path repair and checkpoint completion

The exact-run resume defect was fixed by selecting and identity-validating the original clean manifest before treating current-run output as dirty. Publication-contract review then expanded the repair from resume alone to a closed package state machine: verified sibling-scope exclusion, exact configured layout, closed-world run-input/stage/final/package inventories, semantic command and receipt checks, final-manifest row stage/type validation, exclusive no-takeover locks, lock-aware complete reuse/promotion, package-level identity comparison and pointer-only explicit promotion. `release_ready:false` now blocks strict completion reuse and promotion as well as build start.

Two independent scientific reviews exercised the real HR preflight and found defects missed by synthetic fixtures. The double spelling `DateofHire`/`DateOfHire` made the forbidden contract case-insensitively non-unique and stopped the production evaluator before fitting. The contract/config/schema now retain only verified raw `DateofHire`, with an explicit uniqueness regression. Separately, exact-fold SHAP replay used unnormalized direct model probabilities while OOF generation used canonical float64 normalization. Replay now calls the same `aligned_predict_proba` helper; the `1e-12` identity tolerance was preserved.

A reduced, in-memory, no-write/no-network real HR diagnostic then traversed the production evaluator with 10 outer x 5 inner structure, 150 test-budget fit receipts, 1,555 raw policy OOF rows, 311 sigmoid OOF rows, 6,531 grouped SHAP rows, 45 fold pairs, six deterministic representatives and 85 subgroup intervals. Prediction replay error was zero; the department proxy returned the predeclared insufficient-support status. These are diagnostic implementation facts only: the candidate grid and bootstrap count were reduced under the explicit test-only override, and no artifact may enter a canonical manifest.

Test chronology is preserved rather than overwritten. An initial full run had 11 integration failures (519 passes); later superseded runs were terminated after new review findings. After all trust/scientific fixes, focus passed 161 with one platform skip. Independent post-fix reviews passed 132/2 skips for trust and 96/1 skip for external science. A first release-gate focus exposed two incomplete synthetic fixtures, which were corrected without weakening validation; final trust focus passed 141/2 skips. The current full suite passes 648 with 3 skips and 11 subtests; unittest passes 178 with 2 skips; compileall and dependency checks pass.

Current config hash is `5af0262e83a3720f8dca0b4d6795bdffc6bb2cefedc901ae0a47f9262d07f305`; scope hashes remain `af80b8a7...` and `18bbb5cb...`. No production Unit 2G file exists, so V2-028 remains `implementation_complete_pending_real_run`. The tested checkpoint is ready for explicit staging review; scientific completion still requires the clean-commit eight-candidate/5,000-draw run and artifact validation.

### Unit 2G pre-commit hygiene completion

The synchronized checkpoint gate passed with 45 reviewed candidates, an empty index, no raw-data candidate, no file at or above 100 MB, no reparse point, no credential-pattern match, no active `leakage-safe` terminology, no manuscript diff, 28 valid issue rows, and all 17 README local-link occurrences resolving to 15 tracked targets. The historical 91,820,515-byte trial remains untracked and locally excluded. Four candidate path-like literals were classified as production sanitization regexes or intentional negative test values; none is an emitted machine path. Historical tracked path-bearing logs/artifacts remain explicitly open under V2-005.

Checkpoint `ae5cf5a8e57f8e9bf0bcf3f458391f2c42d58411` committed the reviewed Unit 2G and publication-trust implementation. The next scientific action is the clean-commit standalone production stage-validation run; the checkpoint itself does not close V2-028 or authorize any numerical manuscript claim.

## Unit 2G production stage, interruption recovery and independent validation - 2026-07-13/14

The full stage completed from clean source commit `17a3dcd36390291b8eab24b4b3a746092dacee77` as run `stage_validation_hrdataset_20260713T175045Z_5af0262e83a3`. It used canonical config `5af0262e83a3720f8dca0b4d6795bdffc6bb2cefedc901ae0a47f9262d07f305`, scientific input `71f1fc4699113876d70c6853abfdc8d3ec4f9419bb88897414fae28ae166422a`, the verified 311-row HRDataset input, ten outer/five inner folds, eight nested candidates, five declared policies, cross-fitted sigmoid and 5,000 paired draws. The wrapper returned 0 after 491.424 seconds and the stage contract records 477.803 seconds.

Interruption recovery proved that no duplicate Python/Git/test/model process, owned listener, lock, partial or staging directory survived. No process was terminated. Three independent read-only reviews plus a root audit verified every closed-world path/size/hash; exact replay of all 50 outer models and ten calibrators; exact OOF/fold/selection isolation; bootstrap bytes/order/bounds; 6,531 exact-model grouped SHAP rows; support-aware subgroup status; proxy fail-closed output; and transport/claim boundaries. The selected recovery suite passed 72 tests in 29.58 seconds. No P0/P1 defect requires a Unit 2G rerun.

Scientific stage-validation findings are deliberately not frozen manuscript claims. Raw conservative macro-F1 is `0.666355` (`0.628073`-`0.704690`) and QWK is `0.541220` (`0.485143`-`0.598791`). Sigmoid improves log loss and Brier but changes macro-F1 by `-0.041104` (`-0.074614`-`-0.008355`) and predicts no class-4 argmax cases. The temporality-restricted audit changes macro-F1 by `-0.366008` (`-0.412759`-`-0.319906`), making timing uncertainty a major limitation. Department/job-role/proxy-rich macro-F1 comparisons are inconclusive. SHAP is raw-margin attribution only; subgroup gaps are descriptive; department proxy reconstructability is not estimated; locked transport is infeasible.

The atomic stage is valid under its saved identity, but the outer input manifest is provisional (`status=running`, no end/commands), and there is no canonical run/final manifest or promotion. Commit `e25f403` accidentally pushed its full 65,412,766-byte package. D5 remediation preserves all 126 local files, removes them from the current Git tip with an index-only forward change, and ignores that exact run root. The already-pushed blobs remain in history because force-push/history rewrite is prohibited.

Two non-invalidating reporting defects remain before canonical execution: external subgroup tables must serialize `probability_method=raw` plus source-OOF identity, and global grouped SHAP tables must serialize raw-margin attribution units. They are tracked as V2-029.

Checkpoint `b7b2ad3074ff4b27f358fd3b9394b4ae2b1ad4a2` records the validated Unit 2G stage status and removes the full package from the current publication tip while preserving it locally. Its normal push timed out at Git Credential Manager; the remote ref remains `e25f403`. The tested checkpoint and following log synchronization are therefore local/unpushed. No additional retry, credential change or history operation was attempted.

## V2-029 external reporting metadata implementation - 2026-07-14

The production contract now distinguishes evidence semantics explicitly. Subgroup diagnostics require config-owned `probability_method=raw`, valid source fold-model and outer-test-probability hashes, and publish a deterministic semantic SHA-256 over the exact policy-scoped OOF rows consumed with scope `exact_consumed_policy_oof_rows`, algorithm `sha256_canonical_csv_utf8_float17g`, and ordered hash columns. This is deliberately not labelled as the byte hash of the complete raw-OOF CSV.

External grouped SHAP is config-bound to `xgboost_raw_margin_score` with additivity checked in `xgboost_raw_margin` space. Provider output must declare both fields; drift fails before artifact generation. Local/global/class/fold tables, fold receipts, metadata, JSON reason codes and Markdown labels carry the semantics. Dimensionless stability/case-selection tables do not pretend to contain measured SHAP units. The old Unit 2G stage remains unchanged and noncanonical.

Seven production/config/test files plus README/traceability were changed. Compilation passed; direct config/SHAP/subgroup tests passed 56/56; the expanded external and manifest gate passed 80/80. The new config hash is `ac32f7d80695e95adbad458ef31d9f1790b16e1eec306aaba57c5233f304e2f8`. V2-029 is implementation-complete but remains open at artifact level until the final current-config canonical rebuild emits and validates the fields.

An independent final diff review found no P0/P1 defect and passed 66 relevant tests in 6.01 seconds. The synchronized checkpoint gate reran 80 tests and compile/hygiene checks with zero prohibited findings. No validated Unit 2G artifact was modified.

The full checkpoint gate then passed 656 pytest tests with 3 skips and 11 subtests, plus 178 unittest tests with 2 skips. This supersedes the earlier 648-test repository count for current HEAD. V2-029 still remains pending canonical artifact generation rather than being marked scientifically resolved.

Checkpoint `9c603534268e7ba953cc1a05b23225b4fde488f5` (`feat(external): bind reporting semantics to OOF evidence`) records the 20-file V2-029 implementation, tests, README and traceability after exact staged review. The branch is three commits ahead of remote pending a normal push; no artifact or manuscript file was committed.

Documentation sync `1639e182f877839995799319b434e1c356d131c1` records the checkpoint hash. A normal non-force push then failed immediately because HTTPS credentials were unavailable in noninteractive mode. Public remote verification remained `e25f403`; no orphan process, credential change, retry, force option or ref movement occurred. Local work continues from the tested chain.

## V2-021a core figure-plan preimplementation audit - 2026-07-14

Interruption recovery reconfirmed the clean `finalization/leakage-aware-v2` branch at `8ce39e7939f00ee3269ca7ceaf829740cfc8130b`, no Python/test/model process or repository listener, no lock/partial/staging residue and a byte-valid untouched Unit 2G atomic stage. A read-only validator passed all 124 stage-contract and 122 artifact-manifest hash/size records and structural model/OOF/calibration/SHAP/subgroup/proxy counts. The enclosing manifest remains provisional, so no rerun or promotion is warranted.

The figure audit found that canonical config still contains the v1 LLM/agent/G-XAIR/local-reason-code set; `validate_manuscript_config` checks only mapping shape; the explicitly legacy figure module cannot produce v2 figures; and `core_figures` has no runner. Historical `latest` figures are inadmissible. The exact approved subjects remain study design, leakage-policy trade-off, four-model benchmark, sigmoid calibration, global grouped OOF SHAP, descriptive SHAP stability and HRDataset_v14 mapped-target replication.

Intended V2-021a files are `configs/manuscript_final.yaml`, a focused core-figure contract in governance code, canonical config validation wiring and `tests/test_core_figure_plan_contract.py`, followed by README and traceability synchronization. Acceptance requires exact keys/order/stems/sources/claim boundaries, portable unique source paths from stages preceding `core_figures`, recursive rejection of prohibited core topics and explicit release-blocking status. No runner, placeholder file, scientific result or release-ready change is permitted in this unit.

### V2-021a implementation and validation

Canonical config now contains the exact seven approved definitions and future output/source-data/caption identities. The new governance validator checks exact plan equality, figure 1-7 order, portable source paths whose declared stages precede final `core_figures`, prohibited core-scope tokens, identity fields and both figure/core release blocking. It is metadata-only: no runner, placeholder or output artifact was added.

The initial focus passed 36 tests with one skip. An expanded run exposed 18 temporary manifest fixtures that replaced the core graph; their shared helper now preserves the canonical graph and inserts its synthetic stage. The first full run then exposed nine analogous side-input fixtures after 662 passes; that helper now also retains the final figure stage and explicit blocking status. The validator was not weakened. Final results are 114 expanded tests with two skips, 58 fixture/integration tests, 671 full pytest tests with three skips/11 subtests, 178 unittest tests with two skips, plus compileall and dependency checks.

Independent review passed 25 tests with one skip, found no P0 and verified source filenames against production writers. It found two P1 integration blockers tracked as V2-032: old numbered SHAP Figure 6/7 previews and the legacy v1 stem validator. V2-021a is therefore a tested plan-contract checkpoint, not a generated/canonical figure package. Config hash is `eef3539be6470644cc1b3892e1aa6bb8c3186aeb9df0d61ec56635a64e978a44`; evidence-scope hashes remain unchanged.

Checkpoint `6da8273b458fd249d47d9bb5c75ebe9ff364617f` (`feat(figures): freeze leakage-aware core plan`) records the 16-file tested plan/config/validator/test/README/traceability unit after exact staged review. It contains no raw data, manuscript edit or generated evidence. Commit-hash synchronization and the authorized normal branch push follow separately.

## Unit 2G recovery and reusable acceptance checkpoint - 2026-07-14

Recovery began from clean, synchronized branch checkpoint `eab2b32150245fe7d406afcfb64827be67797752`. No interrupted task process, owned listener or partial/lock/staging residue existed. The ignored real-data stage `stage_validation_hrdataset_20260713T175045Z_5af0262e83a3` remained intact at 126 files/65,412,766 bytes and was not rerun.

`src/governance/unit2g_stage_validator.py` now turns the one-off read-only audit into a fail-closed production validation tool. It validates the closed-world inventory and every Unit 2G publication contract: raw/canonical/schema/policy/side-input/source/config/scientific identity, folds and candidate selection, persisted model and probability replay, exact cross-fitted sigmoid provenance/replay, 5,000-draw bootstrap, exact-fold grouped SHAP replay, forbidden features and reason codes, mapped-target and subgroup/proxy denominators, portability, claim boundaries and source tables. `tests/test_unit2g_stage_validator.py` protects portable-path rejection and the canonical parsed-content hash against BOM/newline drift. The atomic small receipt is `10_unit2g_checkpoint_summary.json`.

The validator passed in 12.6 seconds without fitting a model. Focused Unit 2G/manifest tests passed 126 with one skip; complete pytest passed 676 with three skips/11 subtests; unittest passed 178 with two skips; compileall passed. No code changed after these successful gates. The stage is accepted as validated real stage evidence, not as a canonical package: its enclosing manifest is still provisional, `canonical=false`, `promoted=false`, and the claim matrix remains unfrozen. V2-032 is the exact next independent unit and was not started here.

## V2-032 legacy figure collision recovery and implementation plan - 2026-07-14

Recovery confirmed a clean, synchronized branch at `fa6f4b1e203bd857e1adef69ace8aaaacfa2b889`, no interrupted task process or repository-owned listener, no lock/partial/staging residue, and an intact ignored Unit 2G stage. The obsolete numbered images are produced unconditionally by the OOF-SHAP stage, while the general seven-figure validator accepts only the superseded v1 stems. Historical copies under `reports/manuscript_final/latest` and the preserved v1 run are evidence to index, not inputs to delete or admit.

The bounded implementation removes numbered image generation from the scientific OOF-SHAP stage while preserving its verified tables and reason codes, isolates the v1 stem map as legacy-only, and introduces a fail-closed v2 package validator derived from the frozen seven-figure plan. Acceptance requires one manifest-bound current-run package with exactly seven PNG/SVG pairs, source-data and caption files; exact run/config/scientific-input/source-tree identity; verified upstream contract and source hashes; portable contained paths; non-empty valid images; forbidden-feature exclusion; and closed-world rejection of missing pairs, manual copies, historical paths and obsolete Figure 6/7 stems. No production figure or canonical claim is created in this unit.

### V2-032 implementation and validation result

The OOF-SHAP stage no longer imports matplotlib or creates publication-numbered files. Its exact-model OOF global/class/local/stability tables, metadata and supplementary local reason codes remain production outputs. The legacy module retains only its isolated v1 generator and explicitly named legacy stem map; its general seven-stem validator is gone.

`core_figure_package.py` now derives admission from the frozen plan and verifies canonical config hash, exact four-field identity, exact current-run stage root, complete closed-world stage/input/upstream contracts, all source file hashes, seven ordered figure records, 14 valid nonempty PNG/SVG files, identity-bearing source CSVs/captions, scoped INX/HRDataset primary-feature exclusion and rejection of historical/latest/obsolete/manual packages. Seven fixture tests exercise valid admission plus missing, stale, drifted, copied, forbidden and obsolete cases. Focus passes 24 tests; full pytest passes 682 with two skips/11 subtests; unittest passes 176 with one skip; compileall passes. No real figure, numerical result or manuscript file was created. V2-032 is implementation-complete and remains artifact-open until a production generator and clean canonical run use the contract.

Exact staged review passed for 18 small allowlisted files. Checkpoint `5cd144a757a1a88271e01dd46a738c59a22aef43` records the implementation, tests, README and traceability; this following receipt sync contains no scientific or manuscript artifact.

## V2-012 supplementary heuristic-search implementation plan - 2026-07-14

The existing supplementary stage is OOF and training-fold-scoped, but its public schema still uses `actionability`, `intervention` and `validity`; includes a `no_salary` mode identical to organisation scope because the primary policy already excludes salary; leaves `max_prototypes` implicit; does not run the configured budget sensitivity; refits without scientific/source-tree/model-fit receipts; and writes non-atomic legacy filenames. The module is supplementary-only, so no core result is invalidated.

The replacement freezes four distinct taxonomy-labelled candidate-feature scopes and three nested within-scope budgets. It generates the maximum candidate pool once per OOF case/scope and derives restricted, primary and expanded results by deterministic filtering, guaranteeing budget inclusion without claiming monotonicity across scopes. Acceptance requires all eligible OOF cases, outer-training-only models/prototypes/domains/scales, explicit fold/model-fit/dataset/config/scientific/source identities, search-success denominators, failure reasons, candidate counts, model-score gains, normalized input-space search costs, sparsity, Wilson and 5,000-draw bootstrap uncertainty, primary-versus-sensitivity labeling, independent scope interpretation, nonprescriptive terminology, closed output validation and no manuscript/core artifact.

### V2-012 implementation and validation result

`manuscript_counterfactual_search.py` replaces the legacy actionability module and is wired into the supplementary builder with run/config/scientific-input/source-tree identity. The exact config contract fixes four distinct scopes; restricted/primary/expanded budgets of 50x2, 100x3 and 250x3; one shared 750-candidate maximum pool; one selected scenario per case/scope/budget; complete eligible-case coverage; Wilson search-success intervals; and 5,000-draw percentile bootstrap summaries conditional on successful cases. Each evaluated fold records training/test index hashes, parameter/feature hashes and OOF probability hashes. Full execution requires ten fold receipts and exactly one OOF prediction per row; bounded diagnostics require every actually evaluated fold and cannot satisfy production completeness.

The stage writes a protocol, exact OOF predictions, fold-model receipts, scope mapping, case-level evidence, primary summary, budget sensitivity, failure reasons, uncertainty, representative primary-budget scenarios, a nonprescriptive interpretation and a closed hash/size inventory. Cross-scope results are independent. Within each case/scope, restricted candidates are a subset of primary candidates and primary candidates a subset of expanded candidates. All model-input changes are labelled heuristic search results, never causal recourse, employee advice, intervention evidence or real-world feasibility.

Development focus first reported 35 passed/1 failed because a test incorrectly required the restricted budget to succeed; the production contract only guarantees candidate inclusion and monotone success when a smaller budget succeeds. Correcting the assertion produced 36 passes. The first two-case real-INX diagnostic failed before publication because bounded diagnostics were incorrectly forced to emit ten receipts; production strictness was retained and the diagnostic rule was corrected to require every evaluated fold. The next diagnostic exited 0 in 3.818 seconds, wrote the complete 12-file schema in a temporary directory, validated all 11 inventory entries, 48 case rows, four primary summaries, 12 sensitivity rows, two eligible cases and one fold receipt, then was safely deleted. It is diagnostic evidence only.

Final focused coverage passed 37 tests; the wider config/scope selection passed 54. After correcting bootstrap terminology to `case_percentile_bootstrap_conditional_on_search_success`, the post-change focus passed 11 tests. Staged review then found that rows exposed a fit-contract hash but not the fitted state itself. The production stage now hashes the actual serialized preprocessing/model pipeline per fold, publishes an aggregate model-set hash and validates every OOF/case row against its fold hash. A fresh real-INX diagnostic passed those bindings and its inventory, then was removed. Post-fix focus passed 37 tests; complete pytest passed 687 with two skips/11 subtests in 118.36 seconds; unittest passed 176 with one skip in 7.389 seconds; compileall passed. Canonical config hash is `ff4afa35c0f48ecf052be78af2074a2498bfd5af3697e0f8d863de0cb8952b59`. No production supplementary artifact, manuscript file, network call, paid API call or Unit 2G scientific artifact was created or modified.

Exact staged review passed for 23 small allowlisted files and checkpoint `7226effd30835fc678b0bb21644f45ac0464dff6` records the V2-012 production contract, tests, README and traceability. No raw dataset, persisted model, scientific output or manuscript file is included.

Documentation receipt `08bc14e` and the implementation checkpoint were pushed normally to the required branch in one attempt. V2-012 is closed at implementation/checkpoint level and remains open only for future canonical supplementary execution. V2-013 is the next active unit.

## V2-013 task-bounded supplementary external evidence - 2026-07-14

The production `external_robustness` builder stage now imports only `manuscript_supplementary_external.py`; the mixed predecessor is explicitly historical/non-admitted. The config freezes exact tasks, conservative primary/audit policies, 10 outer x 5 inner folds, eight XGBoost candidates, primary-policy-only macro-F1 selection, balanced-accuracy tie-break, same-fold candidate reuse, raw-probability scope, paired 5,000-draw uncertainty, literal `N/A` applicability and false validation/comparability/transport flags.

The runner binds actual loader receipts and model-grid/acquisition/schema inputs to config/scientific/source/Git/scope identity. Parsed-cell content, schema mapping, exact policy and fold hashes are carried through the evidence. Every outer policy model is serialized, persisted, SHA-256 bound and replayed; every OOF row resolves to its model/fit/candidate/path. Candidate fits, selections, fold metrics, descriptive summaries, task intervals and policy differences retain their applicable lineage. Atomic task/stage closed-world inventories contain only portable paths.

The source outputs are deliberately separate: `ibm_restricted_target_performance_robustness.csv`, `ibm_attrition_task_transfer.csv` and `employee_turnover_task_transfer.csv`. The cross-task index contains denominators and claim boundaries but no combined score. A real-IBM reduced diagnostic passed all ten outer/five inner folds with 1,470 rows, 4,410 OOF rows, 30 models and zero replay error, then was removed. Final focused tests passed 53; expanded integration passed 196 with one skip; full pytest passed 690 with two skips and 11 subtests. Config hash is `cba48e107d3f95cc6412b7ff4f743ae50b78d04edd65258e3fbdda7759f12ced`.

V2-013 is implementation-complete and canonical-artifact-open. No production supplementary output or numerical claim exists. The next bounded executable unit is V2-015 workbook/CSV equivalence; V2-014 forward raw-tip sanitation follows without deleting local source files or rewriting history.

Checkpoint `906f6360a971833d4cec39fd0d19873b7c567169` records the V2-013 production contract, tests, README and traceability. A normal authenticated push succeeded in one attempt and advanced the required remote branch from `8f256ab` to `906f636`.

## V2-015 executable INX workbook/CSV equivalence - 2026-07-14

`src/governance/inx_workbook_equivalence.py` now implements the production provenance gate. It prefers `xlrd` when present and otherwise creates an isolated Windows Excel COM instance. Excel opens the workbook read-only with link updates, macros, events and alerts disabled, and only the created instance is closed. The comparator applies one declared normalization contract, verifies exact headers/shapes/cells and emits only hashes and bounded mismatch coordinates. The atomic JSON receipt binds the acquisition/provenance configs, source tree, validator source, Git commit, workbook/CSV byte identities, reader version, sheet order, normalized content and partial codebook.

The config pins the workbook's OLE2/BIFF8 bytes, two-sheet order, expected shapes and seven exact definition blocks. A full explicit mapping resolves the workbook label `RelationshipSatisfaction` to CSV column `EmpRelationshipSatisfaction`; all other mappings are identity mappings. The receipt cannot label the 7/28 codebook complete and preserves manual-review status for semantic authority, source authenticity, licence and citation.

Final focused tests pass 25, integration passes 77, final full pytest passes 698 with two skips/11 subtests, unittest passes 176 with one skip, and compileall passes. A live Excel 16 diagnostic reports exact 1201 x 28 equivalence, zero mismatches and normalized hash `b5caa2eec9a46ad184cc452e1d1df01abc80658db7fdd61e2cb8939943e23fbb`. Staged review removed the only swallowed atomic-cleanup exception. The first clean CLI attempt then exposed a Windows-default-separator defect before Excel/output; portable CLI defaults and a regression test now pass. The clean production receipt is deferred until the fix checkpoint is committed and pushed so its Git/source/config identity is exact.

Fix checkpoint `8f02e5569f1073b1dd3e0861e29d5f9189d79173` was pushed and the production receipt then passed from that clean exact commit. `11_inx_workbook_equivalence_receipt.json` is 6,331 bytes, canonical-eligible and independently valid; receipt SHA-256 is `90e75733d469b8576b884c8e6eb849b017a4398f75becbacf5388c37edd1f2a5`. V2-015 is engineering-complete. V2-014 is next.

## V2-014 forward raw-tip sanitation - 2026-07-14

The current-index change removes 14 exact source/interim/row-level/fitted working-data paths after verifying all 2,335,429 local bytes. `.gitignore` now protects raw, external raw, interim and processed paths while retaining data documentation, mappings and placeholders. Local files remain available to the real-data pipeline. The baseline documentation's literal machine-home path was sanitized without changing its finding.

`configs/publication_export.yaml` freezes the local-preservation inventory, required tracked documentation, forbidden globs, exact technical allowlist, size limits and claim boundary. `sanitized_publication_export.py` verifies a clean exact commit and rebuilds an ephemeral ZIP twice (generation and independent receipt validation), hashes every closed-world member and rejects raw paths, Git metadata, links, machine-local documentation, secret-like content and oversize files. It never publishes or retains the archive.

Focus passes 20, integration passes 84, full pytest passes 703 with two skips/11 subtests, unittest passes 176 with one skip, and compileall passes. Implementation commit `9342b0c9a02788ff9e9867b13f2f824662fd1cf3` was pushed normally. The clean compact receipt then passed production generation and independent archive reconstruction: 305 exact members, archive SHA-256 `1917059f...`, member-manifest SHA-256 `df89a3a6...`, all 14 local inputs preserved, zero prohibited findings and no retained ZIP. V2-014 is engineering-complete; prior-history and redistribution decisions remain manual blockers.

## V2-018 dependency isolation and lock - 2026-07-14

Canonical core, approved supplementary, legacy-optional and development dependency groups are now explicit. Core contains 13 bounded direct packages; supplementary adds none; OpenAI/Agents SDK and other retained legacy/UI packages cannot enter core through the compatibility entry point. A single 96-pin CPython 3.14 constraints file controls all profiles. The production validator and nine direct tests enforce file taxonomy, exact lock syntax/order/completeness, marker/platform handling, path safety, Python/environment identity, `pip check`, atomic receipt writing and independent receipt validation.

Run-input packaging snapshots every dependency group and the lock; source-tree identity includes them. The legacy Unit 2G source-tree validator selects the old dependency file set for commits predating this contract so its accepted stage identity remains reproducible. Final focus passes 80; final pytest passes 712 with two skips/11 subtests; unittest passes 176 with one skip; compileall and both resolver graphs pass. A fresh isolated core install contains 31 exact distributions and passes all imports plus production validation. Linux validation correctly skips the Windows-only COM package. The clean receipt is deferred until the implementation checkpoint is pushed.

Checkpoint `498e8ad59166f275d120f78ce133cce122961f13` was pushed normally. The same isolated core environment then generated the clean canonical-eligible receipt, which passed independent validation with source tree `a2b361b8...`, lock `482cbf32...`, inventory `a7ac622b...` and receipt SHA-256 `8b77e727...`. The temporary environment was removed safely. V2-018 is engineering-complete; V2-019 retains hosted cross-platform execution.
