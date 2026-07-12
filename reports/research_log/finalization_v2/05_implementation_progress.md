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
