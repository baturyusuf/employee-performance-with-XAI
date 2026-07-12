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

Planned only after Unit 1A passes. It will bind loader receipts, schema mappings and all scientific side inputs into run manifests, snapshots and cache invalidation. No Unit 1B production edits have started.

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
