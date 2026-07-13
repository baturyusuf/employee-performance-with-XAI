# Persistent Command Log

Authoritative detailed baseline commands and results: `../finalization_v2/06_commands_and_tests.md`.

Last exact scientific validation commands:

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
```

No canonical v2 build has been run.

Unit 1A exact commands and results are recorded in `../finalization_v2/06_commands_and_tests.md`. The real-data preflight used local files only and passed for all five logical tasks. Full pytest now passes 199 tests plus 4 subtests.

Unit 1B exact commands and results are recorded in `../finalization_v2/06_commands_and_tests.md`. A schema-v2 in-memory manifest was created and validated against all five real logical datasets and seven side inputs; no scientific stage was run. Full pytest now passes 218 tests plus 4 subtests. No network/API call occurred.

Unit 2A exact commands and results are recorded in `../finalization_v2/06_commands_and_tests.md`. Scoped schema-v3 manifests validated real inputs for core and supplementary independently. Accepted entrypoints are fail-closed while release readiness is false. Full pytest now passes 250 tests plus 4 subtests with 2 historical skips; no scientific stage or network/API call ran.

Historical Unit 2B commands and results are recorded in `../finalization_v2/06_commands_and_tests.md`. Shared-fold/model/bootstrap contracts passed 83 focused tests; full pytest passed 314 tests plus 4 subtests with 2 skips; unittest passed 162 with 2 skips. A real-input in-memory 10×3 fold preflight passed for 1,200 rows without writing artifacts or fitting models. At that checkpoint the primary metric was pending; both the 10×3 protocol and pending-metric condition are superseded by the accepted correction below. No network/API call occurred.

The corrected Unit 2B 10×5 protocol/trial commands are recorded in `../finalization_v2/06_commands_and_tests.md`. Focused tests passed 104; full pytest passed 343 plus 4 subtests with 2 skips; unittest passed 162 with 2 skips. The approved 10×5 real-input preflight passed without output/model fitting. The earlier 10×3 record is superseded and cannot be reused.

The final pre-run hardening commands and exact in-memory preflight are recorded in `../finalization_v2/06_commands_and_tests.md`. Focused benchmark/trial tests passed 41; full pytest passed 345 plus 4 subtests with 2 skips; unittest passed 162 with 2 skips; compileall/diff/manuscript/secret checks passed. The real trial entrypoint has not yet been executed and no gate outcome exists.

The real offline trial command, manifest verifier, diagnostic commands, runtime, hashes and model results are recorded in `../finalization_v2/06_commands_and_tests.md`. Trial exit was 0 after 725.2 shell seconds; manifest elapsed time was 722.522 seconds. The macro-F1 baseline gate did not trigger. No API/network/manuscript operation occurred.

Unit 2C-0 exact warning-hygiene commands are recorded in `../finalization_v2/06_commands_and_tests.md`. Focused tests passed 63; full pytest passed 350 plus 4 subtests with 2 skips; unittest passed 162 with 2 skips. A two-model real fold-1 replay produced zero label mismatches and zero warnings. No artifact/API/network/manuscript operation occurred.

Unit 2C-A exact reader/axis/OOF-SHAP commands, the expected historical-lineage failure, current-code fold-1 SHAP diagnostic, and all gates are recorded in `../finalization_v2/06_commands_and_tests.md`. Focused tests passed 59 plus 4 subtests; full pytest passed 389 plus 4 subtests with 2 skips; unittest passed 164 with 2 skips. No scientific artifact/API/network/manuscript operation occurred.

Unit 2D exact policy/config/README commands, the failed nonexistent-test filename attempt, independent-review corrections, real-INX five-fit fold-1 diagnostic and all final gates are recorded in `../finalization_v2/06_commands_and_tests.md`. Final focused suite passed 92; full pytest passed 403 plus 4 subtests with 2 skips; unittest passed 174 with 2 skips. No canonical artifact/API/network/manuscript operation occurred.

Unit 2D tested implementation checkpoint: `984db46` (`feat(policy): bind leakage ablation to shared OOF evidence`). The untracked historical trial was not staged; tracked worktree verification passed immediately after commit.

The read-only Unit 2E calibration audit and timing diagnostics are also recorded there. They read only the local historical OOF CSV in memory; no file/network/API operation occurred.

Unit 2E option-A implementation commands, warning/API compatibility probes, focused test reruns, independent-review findings and the bounded real-INX fold-1 diagnostic are recorded in `../finalization_v2/06_commands_and_tests.md`. The final expanded focused suite passed 85 tests plus 7 subtests; independent read-only review passed 48 tests plus 7 subtests and found no remaining code-level blocker. No canonical calibration artifact was generated because current config `d755ecc3...` must reject historical benchmark config `7e70bf66...`.

Unit 2E tested implementation checkpoint: `0f820b3` (`feat(calibration): cross-fit sigmoid on benchmark folds`). The local 91.8 MB historical trial was not staged or modified; no push occurred.

## Unit 2F Read-only Audit - 2026-07-13

Read-only commands inspected `src/experiments/manuscript_fairness_proxy.py`, the builder runner, canonical/legacy fairness configs, support/proxy tests, policy/bootstrap contracts, v1 fairness metadata/tables and the immutable benchmark shared-fold assignment. `git status --short --branch` showed a clean tracked branch plus only `reports/manuscript_final/trials/` untracked.

The exact fold comparison loaded 1,200 rows from v1 `common_fold_assignment.csv` and 1,200 rows from the immutable trial `shared_folds/fold_assignments.csv`: 1,091 assignments differed and match rate was `0.0908333333`. No file, model, network/API service or manuscript was modified. Three parallel independent read-only reviews confirmed the same upstream, inference, provenance, support and proxy-equivalence defects. Detailed acceptance criteria are in `../finalization_v2/05_implementation_progress.md`.

## Unit 2F Implementation Commands - 2026-07-13

The first broad `rg` audit used a Windows wildcard argument (`tests/test_fairness*`) that PowerShell passed literally; `rg` exited 1 with an invalid filename-syntax error. It performed no write or scientific execution. Corrected commands named the test files explicitly.

Focused implementation validation:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_fairness_support_and_ci_fields.py tests/test_stage_runner_scientific_input_binding.py tests/test_fairness_proxy_config_contract.py tests/test_fairness_proxy_stage_contract.py tests/test_shared_fold_assignments_across_models_and_policies.py tests/test_manuscript_policy_ablation.py tests/test_benchmark_artifact_contract.py
.\myenv\Scripts\python.exe -m pytest -q tests/test_fairness_proxy_scientific_behavior.py tests/test_fairness_support_and_ci_fields.py tests/test_fairness_proxy_config_contract.py tests/test_fairness_proxy_stage_contract.py tests/test_stage_runner_scientific_input_binding.py tests/test_shared_fold_assignments_across_models_and_policies.py tests/test_manuscript_policy_ablation.py tests/test_benchmark_artifact_contract.py
.\myenv\Scripts\python.exe -m py_compile src/experiments/manuscript_fairness_proxy.py src/governance/manuscript_contract.py src/experiments/build_manuscript_evidence.py
git diff --check
```

Results: the first suite passed 79 in 21.38 seconds; the expanded suite passed 84 in 20.02 seconds; a post-hardening direct subset passed 31 in 4.29 seconds. Compilation/config validation/diff checks passed. The intermediate pre-review config hash was `be2b3f9f7e052df42ad9dc413d10e29bfc2ad6dd63a38513d21957aa9908523f`; it is not the final Unit 2F contract hash.

The intermediate pre-review real-INX diagnostic was piped to `python.exe -` and wrote no file. It loaded the verified canonical dataset, generated the approved deterministic 10x5 shared folds in memory, confirmed the outer mapping exactly matched the immutable historical trial, blocked socket/DNS connection paths, and invoked `generate_proxy_oof_evidence` with 5,000 resamples and batch size 200. Exit 0 after 9.7 shell seconds; measured evidence time 5.233 seconds. It produced 20 fold-fit receipts and 2,400 exactly-once OOF rows in memory. No network/API/manuscript/artifact operation occurred. Exact diagnostic numbers are recorded in the chronological Unit 2F log; they are not manuscript evidence and are superseded by the final diagnostic below.

## Unit 2F Final Review and Checkpoint Gates - 2026-07-13

Post-review fixes reject boolean seeds, assign department reconstructability the explicit `nominal_multiclass_proxy_diagnostic` task, bind identity into the proxy label mapping, expose overall/fold department-class support, propagate the conditional-inference scope, source the watchlist from canonical config, validate policy feature counts, and join audit fields by explicit sample index. A duplicate `row.task_type` serialization reference and then an omitted manuscript-table task field were each caught by the direct dynamic test and corrected before any scientific publication.

Final commands:

```powershell
.\myenv\Scripts\python.exe -m py_compile src/models/task_schema.py src/governance/manuscript_contract.py src/experiments/manuscript_fairness_proxy.py
.\myenv\Scripts\python.exe -m pytest -q tests/test_task_metric_applicability.py tests/test_fairness_proxy_config_contract.py tests/test_fairness_proxy_scientific_behavior.py tests/test_fairness_support_and_ci_fields.py tests/test_fairness_proxy_stage_contract.py
.\myenv\Scripts\python.exe -m pytest -q tests/test_task_metric_applicability.py tests/test_fairness_proxy_scientific_behavior.py tests/test_fairness_support_and_ci_fields.py tests/test_fairness_proxy_config_contract.py tests/test_fairness_proxy_stage_contract.py tests/test_stage_runner_scientific_input_binding.py tests/test_shared_fold_assignments_across_models_and_policies.py tests/test_manuscript_policy_ablation.py tests/test_benchmark_artifact_contract.py
$env:OPENAI_API_KEY=$null; $env:OPENAI_AGENTS_API_KEY=$null; $env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
git diff --exit-code -- manuscript/mdpi_information/main.md
.\myenv\Scripts\python.exe -m pip check
```

Exit 0 results: direct focus 44 passed; expanded integration focus 102 passed; full pytest 467 passed, 2 skipped and 11 subtests in 85.74 seconds; unittest 178 tests with 2 skips; compileall/diff/manuscript/dependency gates passed. Secret, scientific-diff absolute-path, candidate-file over 100 MB, active `leakage-safe`, legacy proxy-import, README local-link and issue-register audits passed: 17 links, 27 issue rows, zero prohibited matches/files. Independent follow-up review reran 102 tests and reported no remaining P0/P1 defect.

The final read-only INX diagnostic used the same no-write/no-network wrapper and current config hash `3c9588c1327ac563a85586835b19b30768860165dc26b61fcf7aafbce3bb1421`. Exit 0: 20 fits, 2,400 exactly-once OOF rows, 5,000 draws in batches of 200, exact outer-map match, minimum department support 20, minimum nonzero fold support 1, two zero-support fold/class cells and 3.776 measured seconds. Diagnostic macro-F1 values were unchanged; they remain noncanonical implementation evidence.

Non-scientific command failures recorded for reproducibility: an initial Windows wildcard `rg` invocation failed before inspection; the first post-review test exposed the duplicate task-field reference; the second exposed the absent manuscript-table task field; an issue-register audit initially assumed the older remediation column names and was rerun against the actual v2 schema; a later log search repeated the literal Windows wildcard mistake and was corrected with `rg -g '*.md'`. None executed or wrote a scientific stage.

Checkpoint command `git commit -m "feat(fairness): bind subgroup and proxy diagnostics to OOF evidence"` exited 0 and created `a490d1e` with 22 files changed. Immediate status verification showed no tracked change and only the deliberately untracked `reports/manuscript_final/trials/`; `git ls-files` returned no trial file. No push, merge, release or manuscript edit occurred.

## Unit 2G External Replication Read-only Audit - 2026-07-13

Read-only inspection covered `configs/manuscript_final.yaml`, `configs/data_acquisition.yaml`, HRDataset schema mapping, canonical loader/external adapter, `manuscript_external_evidence.py`, builder/scopes/task/claim code, existing tests and historical external packages. A no-network in-memory loader/adaptation command verified SHA-256 `cb199967...`, 311 × 36 rows/columns, raw support 37/243/18/13, mapped support 31/243/37, no unmapped target and the exact current policy feature lists. It wrote no file or model.

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_manuscript_external_evidence.py tests/test_external_scope_contract.py tests/test_external_explicit_input_binding.py tests/test_external_claim_boundaries.py
```

Exit 0: 19 passed in 6.46 seconds. These are baseline tests, not evidence that the current stage is canonical. Historical package counts were read only: `latest/external` has 53 files/24.8 MB under config `c664...` and old commit `1834748`; `reports/external_validation` has 192 files/68.3 MB. The old package combines core/supplementary tasks and actionability, so it is incompatible and cannot be reused.

One broad `rg` command again included literal Windows wildcard test arguments and returned exit 1 after the non-wildcard portions printed; the corrected focused command named four files explicitly and passed. No scientific execution or write occurred in the failed inspection command.
