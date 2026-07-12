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
