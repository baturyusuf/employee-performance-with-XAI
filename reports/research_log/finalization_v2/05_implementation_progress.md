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
