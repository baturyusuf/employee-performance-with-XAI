# Persistent Command Log

## Major-revision v3 planning checkpoint - 2026-09-03

The required handoff files, complete 42-part reviewer brief, current Git state, recent history, canonical compact handoff, v2 scope decisions, source/test inventory, and stale v1 manuscript headings/claims were inspected read-only. The branch began clean and synchronized at `050cfa7f53cb2a55d2b5eff1ef99490d01202d14`.

Repository-wide searches found no admissible ordinal-logistic/cumulative-threshold model, repeated nested-CV sensitivity, independent policy retuning, model-level SHAP deletion/perturbation faithfulness experiment, RPS, naive baselines, HR target-mapping/CV sensitivity, or canonical data-quality output. They confirmed that v2 contains exact-fold grouped SHAP, matched fixed-schedule policy sensitivity, cross-fitted sigmoid calibration, descriptive subgroup/proxy diagnostics, and independently trained mapped-target HR replication.

Created `../major_revision_v3/PLAN.md` and `../major_revision_v3/REQUIREMENT_COVERAGE_AUDIT.md`, then linked both from `../../../README.md`. No scientific code/config/artifact, dataset, model, manuscript, or canonical v2 byte was modified.

Validation commands used the repository interpreter with paid-API credential variables cleared:

```powershell
.\myenv\Scripts\python.exe -m tools.canonical_manuscript_asset_export --validate-only
.\myenv\Scripts\python.exe -m pytest -q tests/test_ci_workflow_contract.py tests/test_canonical_manuscript_asset_export.py
git diff --check
```

Additional read-only PowerShell assertions checked all README links and the exact 0–42 coverage/status counts. Results are recorded in `TEST_LOG.md`.

## GitHub publication-assets checkpoint - 2026-07-16

Read-only inventory identified a 50-file/6,337,343-byte aggregate publication subset in the canonical run. It includes all final figure/table assets and claim boundaries while excluding row-level/model/full-package internals. The maximum file is 2,195,799 bytes. Header, raw-data, secret, path-portability, and GitHub-size preflights passed. `.gitignore` was narrowed to expose only these exact canonical directories/files, and README was rewritten to link every figure, source CSV, caption, and table directly.

The three production package validators exited 0 in 2.770 seconds. The focused figure/table/manifest/CI-contract pytest command passed 22 tests with one skip in 8.37 seconds (14.335 seconds wall). The final staged audit found 61 allowlisted files/6,612,984 bytes: exactly 50 canonical publication assets plus 11 documentation/traceability files, no unstaged change, no manuscript change, and zero raw/model/OOF/employee-identifier/secret/machine-path candidates. All 64 README local links resolve to indexed files. Canonical inventory receipts use a documented repository-relative canonical-JSON algorithm rather than an undocumented one-off digest.

## Final canonical checkpoint - 2026-07-14

Final run `canonical_v2_20260714T221501Z_483f96f` completed core and supplementary with `--no-reuse-compatible`, then passed atomic promotion and strict post-promotion validation. Exact commands, durations, hashes, candidate failures, and acceptance scans are recorded in `../finalization_v2/06_commands_and_tests.md` and `../finalization_v2/15_canonical_evidence_receipt.json`.

Git push authentication worked for implementation checkpoints `3e4ea47`, `cd7b5e7`, and `483f96f`. The final receipt/log checkpoint is the only remaining normal push in this task.

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

Historical note: no canonical v2 build had run at this early checkpoint; the final canonical completion section above supersedes this statement.

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

## Unit 2G Implementation and Independent Review - 2026-07-13

The implementation commands and full file-level record are mirrored in `../finalization_v2/06_commands_and_tests.md`. The completed stage implementation provides the accepted exact feature policies, 10 outer x 5 inner nested XGBoost selection, fixed cross-fitted sigmoid, common 5,000-draw paired bootstrap, exact-model OOF SHAP, support-aware diagnostics, three-feature transport infeasibility and atomic portable publication. No production stage execution is implied.

Validation commands:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_external_replication_config_contract.py tests/test_external_nested_selection_contract.py tests/test_external_sigmoid_isolation.py tests/test_external_oof_bootstrap_contract.py tests/test_external_shap_contract.py tests/test_external_subgroup_proxy_contract.py tests/test_hrdataset_replication_stage_contract.py tests/test_external_validation.py tests/test_manuscript_external_evidence.py
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
git diff --exit-code -- manuscript/mdpi_information/main.md
```

The interrupted broad run resumed and exited 0 with 120 passed in 33.61 seconds. After the final independent-review identity fix, the selected external suite exited 0 with 68 passed in 20.94 seconds; compileall and diff checks passed. A first attempted focused command used the Windows Store `python.exe` shim and failed before test execution; it was immediately rerun with the repository interpreter `.\myenv\Scripts\python.exe`. No scientific model, artifact, network/API operation or manuscript write occurred in that failed command.

No real Unit 2G stage artifact has yet been generated. The next recorded command must be the full gate and checkpoint, followed by the clean real-data stage.

## Unit 2G Interruption Recovery - 2026-07-13

Recovery read `AGENTS.md` and all required finalization/Unit 2G records, confirmed branch `finalization/leakage-aware-v2` at `e4c3f0d`, inspected unstaged/cached diffs and 15 commits, and checked Win32 process command lines. No Python, pytest, XGBoost or LightGBM process was running; no process was terminated. `git diff --cached` was empty. No `.tmp`, `.lock`, `.partial`, `.incomplete` or staging path was found outside excluded environments/trials. The only untracked research evidence remained the intact 91,820,515-byte historical benchmark trial; no file was deleted or reset.

The audit found that the prior interruption occurred before the planned builder-resume fix was applied. Current stage output binding/date-fallback fixes and their tests are present. A broad `rg` hygiene command again passed a literal Windows wildcard (`tests/test_external_*`) and exited 1 after completing the preceding read-only checks; the corrected audit uses explicit files or `-g`. No code, data or artifact was changed by that command.

## Unit 2G Publication-Contract Recovery and Final Checkpoint Gates - 2026-07-13

The resumed work repaired the exact-run resume path and then independently audited the complete-package trust boundary. The following repository-interpreter commands were run with OpenAI/Azure/Anthropic API variables removed:

```powershell
.\myenv\Scripts\python.exe -m py_compile src\experiments\build_manuscript_evidence.py src\governance\manuscript_contract.py tests\test_manifest_completion_contract.py
.\myenv\Scripts\python.exe -m pytest -q tests/test_manifest_completion_contract.py tests/test_builder_path_contract.py tests/test_builder_resume_contract.py tests/test_atomic_stage_resume_contract.py tests/test_latest_pointer_contract.py tests/test_artifact_run_manifest_consistency.py tests/test_final_evidence_manifest_hashes.py tests/test_scoped_run_manifest_inputs.py tests/test_side_input_hash_binding.py
.\myenv\Scripts\python.exe -m pytest -q tests/test_external_replication_config_contract.py tests/test_external_nested_selection_contract.py tests/test_external_sigmoid_isolation.py tests/test_external_oof_bootstrap_contract.py tests/test_external_shap_contract.py tests/test_external_subgroup_proxy_contract.py tests/test_hrdataset_replication_stage_contract.py tests/test_external_validation.py tests/test_manuscript_external_evidence.py
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
.\myenv\Scripts\python.exe -m pip check
git diff --check
git diff --exit-code -- manuscript/mdpi_information/main.md
```

Initial trust focus: 125 passed with 2 platform symlink skips. Initial external focus: 80 passed. Independent review then exposed and the implementation closed: incompatible same-run sibling cleanliness; arbitrary command/stage admission; non-closed package/run-input/stage inventories; unlocked promotion/reuse; stale-lock takeover race; non-semantic claim/status/final-manifest rows; missing release-ready promotion gate; and a promotion CLI scope ambiguity.

Scientific review exposed two real-path P0 defects before any production artifact was written. The exclusion contract contained both `DateofHire` and `DateOfHire`, which are duplicates under the evaluator's case-insensitive contract; it now retains the verified raw spelling only. Exact SHAP replay used direct XGBoost probabilities while OOF generation used the normalized canonical helper; diagnostics now call the identical `aligned_predict_proba` path. The real-input production preflight and reduced no-write/no-network diagnostic then reached exact-fold SHAP with replay error zero. It remains noncanonical because candidates/resamples were test-reduced and no file was published.

One 161-test focus first failed only because an existing test regex did not include the new explicit `hidden or cache directory` error; after updating the assertion, 161 passed with one platform skip. The first post-release-gate trust focus had two fixture failures (new timing fields absent from a synthetic receipt; synthetic release-ready scopes named table/figure stages without runners); fixtures were corrected without weakening production checks. Final trust focus passed 141 with 2 skips. Independent post-fix focuses passed 132 with 2 skips for trust and 96 with 1 skip for Unit 2G science.

The final current-code full gate passed 648 tests with 3 skips and 11 subtests in 109.16 seconds. Unittest then ran 178 tests in 7.573 seconds with 2 skips; compileall and `pip check` exited 0. Config validation produced `5af0262e83a3720f8dca0b4d6795bdffc6bb2cefedc901ae0a47f9262d07f305`; core/supplementary scope hashes remain `af80b8a7...` and `18bbb5cb...`.

A combined read-only hygiene wrapper printed all component results but returned exit 1 because internal `git ls-files --error-unmatch` probes left a native nonzero status. Its reported facts were retained, but it is not treated as the final hygiene command. It found 45 checkpoint candidates, zero raw candidates, zero >=100 MB candidates, zero high-entropy secret matches, no reparse points, 17/17 valid tracked README links, 28 issue rows, no manuscript diff and an empty index. A broad whole-file path scan reported historical/log and intentional negative-test literals; the final candidate-diff portability gate is recorded separately after documentation synchronization.

No paid API, dataset download, network acquisition, manuscript edit, production Unit 2G artifact or canonical release operation occurred.

## Unit 2G checkpoint hygiene - 2026-07-13

After README and traceability synchronization, the checkpoint was validated with `git diff --check`, `git diff --exit-code -- manuscript/mdpi_information/main.md`, `git diff --cached --name-only`, `git ls-files --others --exclude-standard`, `git check-ignore`, `git ls-files --error-unmatch`, `Import-Csv`, `Get-Item`, `Get-ChildItem`, and `rg --pcre2` checks. The checks covered staged state, raw-data candidates, 100 MB candidates, reparse points, high-entropy credential patterns, active `leakage-safe` terminology, the issue-register schema/count, README local-link existence/index membership, and preservation/exclusion of the historical trial.

The consolidated gate exited 0: 45 candidates (28 modified, 17 untracked), zero staged/raw/large/reparse/secret/active-terminology/manuscript-change failures, 28 well-formed issue rows, and 17 README local-link occurrences (15 unique targets), all present and Git-tracked. The local historical trial remained 54 files/91,820,515 bytes, untracked and locally excluded. A candidate path-literal scan found only two sanitization regexes and two intentional negative-test fixtures; no machine-specific path is serialized as data or metadata. V2-005 remains open for the separately tracked historical-tree cleanup and final real-package content scan.

The first staged-review wrapper used the PowerShell variable name `$home`, which is case-insensitively reserved as read-only `$HOME`; it exited 1 before producing a verdict and changed no file or index entry. The corrected wrapper used `$pathHits` and exited 0: exactly 45 staged files, zero unstaged/untracked candidates, and zero raw, 100 MB, secret, actual-user-home or manuscript paths.

Checkpoint commit created: `ae5cf5a8e57f8e9bf0bcf3f458391f2c42d58411` (`feat(external): bind HR replication to nested OOF evidence`). It contains the explicitly reviewed 45-file implementation/config/test/README/log set. No raw dataset, trial file, secret, manuscript file or scientific output was included.

An initial unstaged log draft expanded the short commit hash incorrectly; `git rev-parse HEAD` detected it before staging, and every draft occurrence was corrected to the exact hash above. No commit or scientific metadata ever contained the incorrect value.

The normal `git push origin finalization/leakage-aware-v2` produced no output and remained in Git HTTPS for about 90 seconds; the yielded wrapper was terminated and its exact orphaned Git process tree was then identified by creation time/command line and stopped. Upstream remained two commits behind. One retry with `GIT_TERMINAL_PROMPT=0` hit the 60-second tool timeout with the same Git HTTPS state; its proven process tree was also stopped. No credential/security setting was changed, no force flag was used, and no remote ref moved. A maximum of one later retry remains under the user's two-retry limit; local scientific work continues.

## Unit 2G production stage and post-interruption recovery - 2026-07-13/14

The full verified-real-data stage was invoked from clean source commit `17a3dcd36390291b8eab24b4b3a746092dacee77` with the repository interpreter, canonical config, explicit output, run/config/scientific identities and the saved input receipt:

```powershell
.\myenv\Scripts\python.exe -m src.experiments.manuscript_hrdataset_replication --config configs/manuscript_final.yaml --output-dir reports/manuscript_final/stage_validation_hrdataset_20260713T175045Z_5af0262e83a3/core/external_replication --run-id stage_validation_hrdataset_20260713T175045Z_5af0262e83a3 --config-hash 5af0262e83a3720f8dca0b4d6795bdffc6bb2cefedc901ae0a47f9262d07f305 --scientific-input-hash 71f1fc4699113876d70c6853abfdc8d3ec4f9419bb88897414fae28ae166422a --manifest-inputs reports/manuscript_final/stage_validation_hrdataset_20260713T175045Z_5af0262e83a3/core/stage_validation_input_manifest.json
```

The wrapper returned 0 after 491.424 seconds; the atomic stage contract records 477.803 seconds, `status=complete`, source tree `706690fcc28c8e308dcb667872035f72c8382bcc02f8585698a0968a8b8f873a`, zero network calls and zero paid API calls. No canonical run/final manifest or `latest` promotion was requested. The generated package was later committed and pushed as `e25f403`; that publication-policy error is remediated forward below.

The 2026-07-14 interruption audit used read-only Windows process and endpoint enumeration plus Git/artifact inspection:

```powershell
Get-CimInstance Win32_Process
Get-NetTCPConnection -State Listen
Get-NetUDPEndpoint
git status --short --branch
git diff
git diff --cached
git log -15 --oneline --decorate
Get-ChildItem reports/manuscript_final/stage_validation_hrdataset_20260713T175045Z_5af0262e83a3 -Recurse -Force
```

No command line or descendant referenced the repository/run after excluding the audit shell itself. No owned listener, duplicate model/test process, lock, partial, temporary or staging directory existed. No process was terminated and no file was deleted.

A read-only inline Python validator rehashed the closed-world artifact and stage inventories; replayed fold/model/OOF/calibration/SHAP relationships; decompressed and checked the 5,000-draw bootstrap array; checked subgroup/proxy/transport boundaries; and asserted absence of package promotion. Exit 0 results: 122 artifact rows, 124 stage-contract outputs, 125 stage files, 10 outer x 5 inner folds, 50 hashed outer models, 1,555 raw OOF rows, 311 sigmoid rows, 6,531 grouped-SHAP rows, 85 subgroup disparity rows and fail-closed proxy non-estimation.

The provisional input manifest was deliberately tested against current HEAD:

```powershell
@'
from pathlib import Path
from src.governance.manuscript_contract import RunManifestError, validate_run_manifest
p = Path('reports/manuscript_final/stage_validation_hrdataset_20260713T175045Z_5af0262e83a3/core/stage_validation_input_manifest.json')
try:
    validate_run_manifest(p, require_complete=False, verify_source_tree=True)
except RunManifestError as exc:
    print('EXPECTED_NONCANONICAL_REJECTION:', exc)
else:
    raise SystemExit('provisional manifest unexpectedly validated')
'@ | .\myenv\Scripts\python.exe -
```

Exit 0 with the expected diagnostic: `git commit mismatch: HEAD changed after the run manifest was created`. This confirms noncanonical status; it does not invalidate the separately verified atomic stage receipt.

Focused regression command:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_external_replication_config_contract.py tests/test_external_nested_selection_contract.py tests/test_external_sigmoid_isolation.py tests/test_external_oof_bootstrap_contract.py tests/test_external_shap_contract.py tests/test_external_subgroup_proxy_contract.py tests/test_hrdataset_replication_stage_contract.py tests/test_artifact_run_manifest_consistency.py
```

Exit 0: 72 passed in 29.58 seconds (34.3 seconds shell time).

D5 forward remediation preserved every local byte while removing the package from the current Git tip:

```powershell
git rm --cached -r -- reports/manuscript_final/stage_validation_hrdataset_20260713T175045Z_5af0262e83a3
```

Exit 0. Post-command verification found 126 local files/65,412,766 bytes, zero tracked paths under the run root, and the exact root ignored by `.gitignore`. `.idea/misc.xml` is excluded only through `.git/info/exclude`; it is not a checkpoint candidate.

The first consolidated hygiene wrapper on 2026-07-14 reached successful compileall, `pip check`, manuscript/diff, README and issue checks, then exited 1 before its final scans because it reused PowerShell's case-insensitive read-only `$HOME` name. It changed no file or index entry. The corrected wrapper used `$pathHits` and exited 0: 20 README links/16 unique targets, 30 well-formed issue rows, 141 candidate paths consisting of documentation plus intentional stage deletions, zero raw additions, zero files at or above 100 MB, zero secret/absolute-path/active-`leakage-safe` additions, zero tracked stage files, and all 126 local stage files/65,412,766 bytes preserved.

Final staged review used `git diff --cached --check`, manuscript-scoped diff, exact path allowlisting, deletion-only status validation for the run root, added-line secret/home-path scans, and a fresh SHA-256 pass over every local artifact/stage-contract record. Exit 0: 141 staged paths = 126 deletion-only stage entries + 15 small approved files; 0 unstaged; 0 untracked; 0 secret/home-path findings; 122/122 artifact and 124/124 contract hashes still valid.

Checkpoint commit exited 0:

```powershell
git commit -m "chore(publication): validate Unit 2G and untrack full evidence"
```

It created `b7b2ad3074ff4b27f358fd3b9394b4ae2b1ad4a2` with 141 paths: 126 current-tip deletions and 15 small documentation/ignore records. All 126 ignored local evidence files remained present.

The authorized normal push was attempted once:

```powershell
git push origin finalization/leakage-aware-v2
```

It produced no output and timed out after 184 seconds. Process inspection recorded the verified repository tree created at 2026-07-14 11:35:29: wrapper PID 27992; `git push` PIDs 18496/36788; `git remote-https` PID 28948; `git-remote-https` PID 41176; no listening TCP/UDP endpoints. Git Credential Manager PIDs 18728/17356 were observed but explicitly left untouched. Graceful `taskkill /PID 18496` returned 1 because force was required; the four verified push/remote PIDs were then individually terminated after the wrapper timeout. Follow-up found zero surviving verified push PIDs/listeners/endpoints while both credential-manager processes remained untouched.

A read-only unauthenticated GitHub ref request returned `e25f403d82082f97e34aa4f8174bfc001fced5d8`, confirming the remote did not move. No force option, credential/security change or additional push retry was used. `b7b2ad3` and this following log synchronization remain local.

## V2-029 external reporting metadata - 2026-07-14

No scientific stage was rerun. Read-only inspection confirmed the Unit 2G package remained local/ignored and no repository model/test process was active. Production changes pin the external SHAP unit in canonical config and its frozen contract; require the SHAP provider to declare matching raw-margin semantics; serialize the unit into numeric tables, receipts, metadata and reason codes; require subgroup diagnostics to consume only receipt-bound raw OOF rows; and publish a deterministic exact-consumed-row semantic digest with its scope, algorithm and columns.

```powershell
.\myenv\Scripts\python.exe -m compileall -q src/experiments/hrdataset_replication_diagnostics.py src/experiments/manuscript_hrdataset_replication.py src/governance/external_replication_contract.py tests/test_external_shap_contract.py tests/test_external_subgroup_proxy_contract.py tests/test_external_replication_config_contract.py
.\myenv\Scripts\python.exe -m pytest -q tests/test_external_shap_contract.py tests/test_external_subgroup_proxy_contract.py tests/test_external_replication_config_contract.py
.\myenv\Scripts\python.exe -m pytest -q tests/test_external_replication_config_contract.py tests/test_external_nested_selection_contract.py tests/test_external_sigmoid_isolation.py tests/test_external_oof_bootstrap_contract.py tests/test_external_shap_contract.py tests/test_external_subgroup_proxy_contract.py tests/test_hrdataset_replication_stage_contract.py tests/test_artifact_run_manifest_consistency.py
.\myenv\Scripts\python.exe -c "from src.governance.manuscript_contract import canonical_config_hash,evidence_scope_contract_hash,load_manuscript_config; p='configs/manuscript_final.yaml'; c=load_manuscript_config(p); print(canonical_config_hash(c)); print(evidence_scope_contract_hash(c,'core')); print(evidence_scope_contract_hash(c,'supplementary'))"
git diff --check
```

Results: compileall exit 0; 56 focused tests passed in 5.84 seconds; expanded 80-test external/manifest gate passed in 27.18 seconds; config hash `ac32f7d80695e95adbad458ef31d9f1790b16e1eec306aaba57c5233f304e2f8`; scope hashes `af80b8a7...` and `18bbb5cb...`; diff check passed. The first bare `python` compile attempt resolved to a broken Windows app-execution alias and exited before Python started; the repository interpreter then passed. No API/network, dataset acquisition, scientific artifact write, historical artifact rewrite or manuscript edit occurred.

The synchronized pre-checkpoint wrapper reran compileall, the expanded suite, config/scope hashes, diff/manuscript checks, issue CSV validation, README-link audit, and changed-file raw/100-MB/secret/absolute-path scans. Exit 0: 80 tests in 27.11 seconds, 30 unique complete issues, 21 valid README links, 20 changed small tracked files, and zero raw, large, secret, absolute-path or manuscript findings. Independent final review found no P0/P1 defect and passed 66 relevant tests plus diff check.

Full checkpoint regression was then run concurrently:

```powershell
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
```

Both exited 0. Pytest: 656 passed, 3 skipped and 11 subtests passed in 143.88 seconds. Unittest: 178 passed, 2 skipped in 10.895 seconds. A non-required `python -m ruff check ...` probe exited 1 because Ruff is not installed in the locked environment; no lint result was claimed and no dependency was installed or changed.

Final pre-stage hygiene reran `pip check`, compileall, diff/manuscript checks, issue uniqueness/completeness, README links, changed-file raw/100-MB/secret/absolute-path scans and repository-process enumeration. Exit 0: no broken requirements; 20 small tracked candidates; 30 issues; 21 README links; zero raw, large, secret, absolute-path, manuscript or active-process findings.

Explicit staged review covered exactly 20 files and found zero raw/manuscript/100-MB/secret/absolute-path candidates with no unstaged or untracked file. Commit command completed successfully:

```powershell
git commit -m "feat(external): bind reporting semantics to OOF evidence"
```

Checkpoint: `9c603534268e7ba953cc1a05b23225b4fde488f5` (444 insertions, 23 deletions across 20 small files). The branch became three commits ahead of remote; no artifact, raw dataset or manuscript file entered the commit.

Commit-hash synchronization `1639e182f877839995799319b434e1c356d131c1` (`docs(log): record V2-029 checkpoint`) followed after a 9-file documentation-only review. The clean branch was then four commits ahead. Normal non-force push command used only process-local noninteractive environment flags:

```powershell
$env:GIT_TERMINAL_PROMPT='0'
$env:GCM_INTERACTIVE='Never'
git push origin finalization/leakage-aware-v2
```

Exit 1 after 2 seconds: `fatal: Cannot prompt because user interactivity has been disabled` and `fatal: could not read Username for 'https://github.com': terminal prompts disabled`. No credential/security configuration was changed and no process remained. A public noninteractive `git ls-remote` returned remote SHA `e25f403d82082f97e34aa4f8174bfc001fced5d8`; local HEAD remained `1639e18`, ahead four. The failure is authentication-bound, so no retry was made.

## Post-Unit-2G interruption recovery and V2-021a audit - 2026-07-14

Required handoff files, all Unit 2G root-cause/decision/acceptance records, current Git state, stage contents and process/endpoint state were read before modification. Git was clean on `finalization/leakage-aware-v2` at `8ce39e7939f00ee3269ca7ceaf829740cfc8130b`, five commits ahead of remote. `git diff` and `git diff --cached` were empty.

Read-only process commands used `Get-CimInstance Win32_Process`, `Get-NetTCPConnection -State Listen`, `Get-NetUDPEndpoint` and PID/command-line filtering. No Python, pytest or model process was active; no listener belonged to the repository/run. Generic IDE Git metadata processes had no repository/run path in their arguments and ambiguous ownership, so none was terminated. No lock, partial, temporary, incomplete or staging path was found. The stage root still contains 126 files including its enclosing provisional manifest, totalling 65,412,766 bytes.

A separate read-only closed-world validator rehashed the Unit 2G atomic stage without running or refitting it. The external stage has 125 files/65,404,420 bytes; 124/124 stage-contract outputs and 122/122 artifact-manifest rows match path, size and SHA-256. CSV/JSON manifests agree; zero missing, extra, duplicate, unsafe or residual paths were found. Structural counts remain 50 models/receipts, 400 candidate fits, 1,555 raw OOF rows, 311 sigmoid OOF rows, 50 calibration fits, 2,799 calibration-training OOF rows, ten calibrator relationships, 6,531 local grouped-SHAP rows, 45 stability pairs, six representative cases, 391 subgroup metric rows and 85 disparity rows. The enclosing input manifest remains intentionally provisional, so no canonical completion or promotion is inferred.

The V2-021a read-only audit then inspected canonical figure config, core scope/stage order, legacy generator, current stage output names and figure tests. It confirmed that the config still names the obsolete v1 LLM/agent/G-XAIR/local-reason-code set, config validation checks only that `figures` is a mapping, and no v2 `core_figures` runner exists. Historical `latest` figures remain inadmissible. No file, model, API, network acquisition, scientific artifact or manuscript content was changed by these audits.

## V2-021a implementation and regression - 2026-07-14

The implementation added an exact core-figure plan/validator, wired it into canonical config validation, replaced the seven obsolete config definitions and added fail-closed tests. No figure runner or output was created and both release flags remain false.

Initial compile/focus command passed 36 tests with one historical skip. An expanded 114-test config/scope/manifest/figure command first exposed 18 temporary manifest fixtures that replaced the core graph with a single synthetic stage. The shared fixture was corrected to retain the canonical upstream graph and final `core_figures` stage while inserting its synthetic stage; 49 direct tests then passed and the expanded suite passed 114 with two skips. The validator was not weakened.

The first full pytest run then reached 662 passes but exposed nine analogous side-input fixture failures. That fixture was corrected to preserve the canonical figure graph and explicit non-release-ready core status. The combined fixture/figure focus passed 58. The second full pytest exited 0 with 671 passed, 3 skipped and 11 subtests in 111.10 seconds. Unittest exited 0 with 178 tests and 2 skips in 8.012 seconds; compileall and `pip check` passed.

Independent read-only review ran 25 tests with one skip, confirmed every declared source filename against production writers and found no P0. It recorded two P1 canonical-build blockers: `manuscript_shap_evidence.py` still publishes obsolete numbered Figure 6/7 previews, and the legacy `validate_all_seven_figures` still encodes v1 stems. These are explicitly open; V2-021a freezes metadata only.

One read-only PowerShell `rg` command used literal Windows wildcard path arguments and exited 1 before the corrected `-g '*.py'` search; another inspection command had an unmatched quote and exited before execution. Neither changed code, data or artifacts. No paid API, network acquisition, scientific stage, model fit, manuscript edit or Unit 2G rerun occurred.

After README, issue and artifact/claim-map synchronization, the final focused config/figure/fixture gate passed 66 tests in 9.84 seconds. Config validation reproduced `eef3539b...`; 32 unique complete issue rows and all 21 README link occurrences (16 unique targets) validated. Diff/manuscript checks and changed-file scans found zero raw data, file at or above 100 MB, secret-like value, machine-home path or deprecated terminology addition across 16 small checkpoint candidates.

Explicit index review covered exactly 16 allowlisted small files and found zero unstaged/untracked, raw, manuscript, 100-MB, secret, machine-home-path or deprecated-terminology candidates. The first staged `diff --check` found only one extra blank line at EOF in each new Python file; both were removed and the rerun passed. Commit command `git commit -m "feat(figures): freeze leakage-aware core plan"` exited 0 and created `6da8273b458fd249d47d9bb5c75ebe9ff364617f` (815 insertions, 32 deletions). No scientific output or dataset entered the commit.

## Unit 2G interruption recovery and checkpoint acceptance - 2026-07-14

Recovery followed `AGENTS.md` before any edit. `git rev-parse --show-toplevel`, `git branch --show-current`, `git status --short --branch`, both unstaged/staged diffs, `git log -15 --oneline --decorate`, `git rev-parse HEAD`, `git rev-parse @{upstream}` and divergence counting all exited 0. The root was correct, the required branch was clean at `eab2b32150245fe7d406afcfb64827be67797752`, and local HEAD equalled its upstream. `Get-CimInstance Win32_Process`, `Get-NetTCPConnection -State Listen`, `Get-NetUDPEndpoint` and repository/run-identity filtering found no task-owned Python, pytest, Git push, Node, shell, model or validation process and no owned listener. No PID was terminated. Lock/partial/temporary/staging/atomic-residue scans found no interrupted-run residue; the preserved ignored stage had 126 files and 65,412,766 bytes, with zero tracked paths.

The reusable validation command was:

```powershell
$env:OPENAI_API_KEY=''; $env:ANTHROPIC_API_KEY=''; $env:GOOGLE_API_KEY=''; $env:GEMINI_API_KEY=''; .\myenv\Scripts\python.exe -m src.governance.unit2g_stage_validator reports/manuscript_final/stage_validation_hrdataset_20260713T175045Z_5af0262e83a3/core/external_replication --output reports/research_log/finalization_v2/10_unit2g_checkpoint_summary.json
```

The final invocation exited 0 in 12.6 seconds. It read and replayed the existing evidence only: no model was fitted and no scientific artifact was modified. Development invocations of the same command had exited 1 on explicit assertions while the validator was being completed: the JSON-formatted YAML loader, exact `stage_relative` path-basis value, 40-character Git SHA format, normalized-Git-tree versus generation-working-tree provenance, exact prepublication status, round-trip CSV float parsing, current in-memory SHAP identity columns, `mapped` target-support key and support-status vocabulary. Each assertion was corrected in validation code; no production contract was weakened and none indicated stage corruption. Those intermediate shell timings were not retained as acceptance evidence.

The one focused acceptance cycle used:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_unit2g_stage_validator.py tests/test_external_replication_config_contract.py tests/test_external_nested_selection_contract.py tests/test_external_sigmoid_isolation.py tests/test_external_oof_bootstrap_contract.py tests/test_external_shap_contract.py tests/test_external_subgroup_proxy_contract.py tests/test_hrdataset_replication_stage_contract.py tests/test_artifact_run_manifest_consistency.py tests/test_manifest_completion_contract.py tests/test_final_evidence_manifest_hashes.py tests/test_latest_pointer_contract.py
```

Exit 0: 126 passed, 1 skipped in 22.09 seconds (25.7 seconds wall time). The one complete regression cycle used `.\myenv\Scripts\python.exe -m pytest -q`; exit 0: 676 passed, 3 skipped and 11 subtests passed in 100.44 seconds (104.4 seconds wall time). `.\myenv\Scripts\python.exe -m unittest discover -s tests -q` exited 0 with 178 tests and 2 skips in 6.608 seconds (10.6 seconds wall time). `.\myenv\Scripts\python.exe -m compileall -q src tests` exited 0 in 0.3 seconds. API-key variables were cleared for all scientific/test invocations; the validator receipt records zero network and zero paid-API calls. No code changed after these successful cycles.

A single inline PowerShell hygiene assertion gate then re-read the immutable receipt, counted the preserved stage, parsed conservative feature/SHAP/reason-code CSVs, parsed the issue register, resolved every README link, inspected the exact changed-file set, checked the manuscript diff and ran `git diff --check`. It exited 0 in 1.192 seconds: 125 stage files, 32 complete unique issue rows, 24 README link occurrences, 14 checkpoint candidates and zero forbidden-feature, missing-link, raw/model, 10-MB, secret-like, machine-absolute-path, manuscript-diff, network-import or paid-API findings. It did not rerun the unchanged scientific validator.

The exact staged review covered 14 allowlisted files, 1,282 insertions/46 deletions, maximum file size 96,467 bytes, 24 README links and receipt SHA-256 `b9d8500d371e18573f6066a1bed77b8795195b2155b6780a1b1988dc9420fddc`. It exited 0 in 0.535 seconds with zero unstaged, untracked, raw/model/environment, 10-MB, secret, absolute-path, manuscript, full-package or whitespace findings. `git commit -m "feat(external): finalize Unit 2G validation checkpoint"` exited 0 and created `0e3f50c91693b0a0f22502c2f006d516178b5d88` with the exact reviewed 14 files.

The first normal `git push origin finalization/leakage-aware-v2` attempt emitted no output and timed out after 124 seconds while waiting on Credential Manager. The timeout left the exact repository push chain at PIDs 12360, 28296, 28628 and 16988. Their command lines and parent chain proved task ownership; local/remote refs had not moved. Only those four Git transport PIDs were stopped. The credential-manager PIDs and all browser/IDE/system processes were left untouched. A follow-up audit found zero remaining push process.

The second and final allowed attempt used only process-local `$env:GIT_TERMINAL_PROMPT='0'`, `$env:GCM_INTERACTIVE='Never'` and `$env:GCM_GUI_PROMPT='0'` before the same normal push command. It exited 128 in 1.3 seconds: `fatal: Cannot prompt because user interactivity has been disabled` and `fatal: could not read Username for 'https://github.com': terminal prompts disabled`. Local scientific checkpoint `0e3f50c91693b0a0f22502c2f006d516178b5d88` remains unpushed; remote remains `eab2b32150245fe7d406afcfb64827be67797752`. No third attempt, force option, credential change, merge, history rewrite or release occurred.

## V2-032 legacy figure collision implementation - 2026-07-14

Recovery rechecked the required root/branch, required handoff files, unstaged/staged diffs, 15-commit history, HEAD/upstream/divergence, repository-related processes/listeners and lock/partial/staging paths. It began from clean synchronized `fa6f4b1e203bd857e1adef69ace8aaaacfa2b889`; no PID was terminated and the ignored Unit 2G stage was untouched.

Read-only source inspection located unconditional `figure_6_global_grouped_shap` and `figure_7_local_reason_code` writers in `manuscript_shap_evidence.py` and a v1-only `validate_all_seven_figures` stem check in the explicitly legacy governance module. The implementation removed matplotlib and numbered image generation from the OOF-SHAP stage, retained scientific tables/reason codes, made the v1 map explicitly legacy-only, removed the general v1 validator and added `core_figure_package.py`. The new validator checks the frozen config/hash, exact run/config/scientific-input/source-tree identity, exact `run_root/core_figures` containment, closed-world stage/input/upstream receipts, seven ordered manifest records, 14 PNG/SVG files with valid dimensions, seven source-data CSVs and captions, all configured source hashes, primary/external forbidden features and rejection of historical/latest/obsolete/manual packages.

The first shell invocation used the unavailable Windows Store `python.exe` shim and did not start pytest (`Sistem belirtilen yolu bulamiyor`, 0.201 seconds). The repository interpreter `myenv\\Scripts\\python.exe` was then used consistently. Development focus one exited 1: 21 passed/3 failed in 3.00 seconds (6.491 wall); two assertions expected narrower error text and one test used main-task `Salary` rather than the Figure 7 external-primary scope. After scoped production validation and test corrections, focus two exited 1: 23 passed/1 failed in 2.72 seconds (5.872 wall) due only to error-message regex wording. The final unchanged focused command passed 24/24 in 2.22 seconds. Complete `pytest -q` exited 0 with 682 passed, 2 skipped and 11 subtests in 99.13 seconds (102.888 wall). `unittest discover -s tests -p 'test_*.py' -q` exited 0 with 176 tests/1 skip in 6.587 seconds (10.354 wall). `compileall -q src tests` exited 0 in 0.122 seconds. No scientific run, network/API call, manuscript edit or figure artifact occurred.

The exact staged gate covered 18 allowlisted files, 926 insertions/168 deletions, no unstaged file, no whitespace error, and zero raw/data addition, manuscript, environment, secret, machine-absolute-path or file at/above 10 MB. `git commit -m "feat(figures): enforce v2 package identity contract"` exited 0 and created `5cd144a757a1a88271e01dd46a738c59a22aef43`.

The V2-032 documentation receipt commit `5f893ec1f5ec0a3bf0ed21fa9a5a0869f060633a` followed. A normal `git push -u origin finalization/leakage-aware-v2` succeeded and advanced origin from `fa6f4b1` to `5f893ec`; no retry, force option, merge or credential change was used.

## V2-012 supplementary heuristic-search implementation - 2026-07-14

Read-only inspection confirmed that the legacy supplementary module was OOF/training-fold scoped but still used actionability/validity/intervention output language, had an organisation/no-salary duplicate, left prototype budget implicit, ignored top-k, omitted budget sensitivity and carried only run/config identity. The replacement module, config contract, builder wiring and tests were implemented without touching a manuscript file or producing a retained scientific artifact.

The focused command over counterfactual, scope and fallback tests first exited 1 after 4.384 seconds wall time: 35 passed and `test_nested_budgets_share_one_candidate_pool_and_preserve_inclusion` failed because the test incorrectly required restricted-budget success. Candidate inclusion, not success at every budget, is the contract. The assertion was corrected; the next focus passed 36 in 0.92 seconds (4.067 wall). After adding the production-path schema test, focus passed 37 in 1.65 seconds (4.725 wall). A wider `rg -l`-selected config/scope set passed 54 in 1.07 seconds (4.147 wall).

The first bounded real-INX command invoked `python -m src.experiments.manuscript_counterfactual_search` with `--max-cases 2` and failed before publication because diagnostic mode was incorrectly forced to produce ten fold receipts. Production still requires exactly ten. The code was changed so bounded diagnostics require every actually evaluated fold, while full execution retains ten-fold completeness. The second invocation exited 0 in 3.818 seconds. It wrote 12 files to a uniquely named temporary directory: 48 case rows, four primary summary rows, 12 budget-sensitivity rows, two eligible cases, one fold receipt and 11 inventory entries. Every listed size/SHA-256 and the nonprescriptive terminology scan passed. The directory was then verified as the intended temporary target and removed; no diagnostic output remains.

Before the final wording change, complete pytest passed 687 with 2 skips/11 subtests in 99.65 seconds (103.588 wall); unittest passed 176 with 1 skip in 6.649 seconds (10.525 wall); compileall passed in 0.117 seconds. Review then corrected the uncertainty label from an overbroad paired-case phrase to `case_percentile_bootstrap_conditional_on_search_success`, matching the implemented successful-case resampling estimand. The direct post-change focus passed 11 in 0.32 seconds.

The final complete regression command `myenv\Scripts\python.exe -m pytest -q` yielded after ten seconds under cell 310 and was recovered rather than rerun. It exited 0 after 104.753 seconds wall time: 687 passed, 2 skipped and 11 subtests passed in 100.87 seconds. Final `unittest discover -s tests -p 'test_*.py' -q` exited 0 after 11.015 seconds wall time with 176 tests/1 skip in 7.188 seconds. Final `compileall -q src tests` exited 0 in 0.110 seconds. Config validation exited 0 and produced `ff4afa35c0f48ecf052be78af2074a2498bfd5af3697e0f8d863de0cb8952b59`. No network/API call, model production run, Unit 2G rerun, manuscript edit or retained diagnostic artifact occurred.

The consolidated pre-stage hygiene assertion exited 0 in 0.818 seconds. It validated `git diff --check`, a zero manuscript diff, empty index, 23 candidate paths, 32 unique complete issue rows, 24 README link occurrences/18 unique tracked targets and zero raw/model/environment, 10-MB, reparse, secret-like, machine-absolute-path, active `leakage-safe`, production network/API import or legacy normative-term finding.

The staged review then found one substantive provenance gap: case rows resolved to a deterministic fit-receipt hash, but the actual fitted pipeline state was not hashed, contrary to the final dossier's model-identity requirement. Production now serializes each fitted preprocessing/model pipeline in memory with joblib protocol 4, records its SHA-256, derives an ordered model-set SHA-256 and validates every OOF/case row against the exact fold hash. The focused suite passed 37 in 1.90 seconds (5.395 wall). A fresh bounded real-INX diagnostic exited 0 in 3.946 seconds with one actual fold-model hash, one model-set hash, 48 case rows, four OOF rows, 11 verified inventory entries and exact row/fold/model mapping; its temporary directory was safely removed.

The allowed post-fix complete cycle supersedes the prior final timing: pytest exited 0 after 122.461 seconds wall time with 687 passed, 2 skipped and 11 subtests in 118.36 seconds. Unittest exited 0 after 11.319 seconds wall time with 176 tests/1 skip in 7.389 seconds; compileall exited 0 in 0.114 seconds. API-key variables were process-locally cleared. No production artifact, network/API call, manuscript edit or Unit 2G mutation occurred.

## V2-013 task-bounded supplementary external implementation - 2026-07-14

Read-only preflight used the verified local canonical loader with `allow_download=False`. IBM PerformanceRating support is 3=1,244/4=226; IBM attrition is 0=1,233/1=237; Employee Turnover is 0=11,428/1=3,571. Raw SHA-256 values are `a5c31e38...` for the shared IBM file and `2510e274...` for Turnover. Every task supports exact ten-fold OOF, target mapping is complete, and mapped identifiers have zero duplicates.

The production builder was rewired from the mixed compatibility helper to `manuscript_supplementary_external.py` with run/config/scientific-input/source-tree/Git/scope and exact dataset/side-input receipts. Config, task schema, mapping terminology and tests were updated. The stage uses atomic sibling staging, exact 10x5 nested selection, conservative primary-policy selection, same-fold candidate reuse, persisted/replayed outer models, complete OOF coverage, task-valid metrics, explicit `N/A`, descriptive fold summaries, paired 5,000-draw OOF bootstrap and three separate source tables. The predecessor is explicitly historical/non-admitted.

Development focus one exited 1 after 21 passes because OOF rows lacked `model_artifact_path`; production rows were fixed. Focus two exited 1 because pandas' default reader interpreted the literal `N/A` CSV value as missing; the test now disables default NA coercion for that contract column. Final focus passed 52, and the later post-review focus passed 53. Expanded external/manifest integration passed 196 with one skip.

A process-locally API-key-cleared real-IBM diagnostic used one explicit candidate and 20 bootstrap draws under `canonical_eligible=false`. It retained exact 10x5 folds and all 1,470 rows/three policies, producing 4,410 OOF rows, 30 actual model files, zero maximum replay error and 53 closed-world inventory records in 19.837 seconds. Python `TemporaryDirectory` removed the diagnostic after validation. No network/API/manuscript operation or production evidence occurred.

The single complete regression command `myenv\Scripts\python.exe -m pytest -q` exited 0 after 121.8 seconds wall time: 690 passed, 2 skipped and 11 subtests in 117.72 seconds. Current config hash is `cba48e107d3f95cc6412b7ff4f743ae50b78d04edd65258e3fbdda7759f12ced`.

The post-pytest gate used `myenv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -q`, `-m compileall -q src tests`, and `-m pip check`. All exited 0: unittest ran 176 tests with one skip in 7.370 seconds (11.312 wall), compileall took 0.112 seconds, and pip reported no broken requirements in 0.631 seconds.

The first exact staged review covered 22 allowlisted Unit V2-013 files, 2,196 insertions/29 deletions and maximum file size 119,380 bytes. Config/protocol validation produced `cba48e107d3f95cc6412b7ff4f743ae50b78d04edd65258e3fbdda7759f12ced`; 32 issue rows and 24 README local-link occurrences across 18 unique targets validated. Zero staged raw/model/environment/large artifacts, secret patterns, machine-absolute paths, manuscript files, active `leakage-safe` additions or network/paid-API imports were found. Pre-existing tracked raw-distribution blockers remain outside this checkpoint.

Checkpoint `906f6360a971833d4cec39fd0d19873b7c567169` (`feat(supplement): finalize task-bounded external evidence`) recorded the 22-file V2-013 implementation. `git push origin finalization/leakage-aware-v2` exited 0 in one attempt and advanced origin from `8f256ab` to `906f636`; no retry, force option, merge or credential change was used.

The final staged review exited 0 in 0.617 seconds. It covered exactly 23 allowlisted files, 1,896 insertions/707 deletions, maximum file size 118,791 bytes, 32 complete unique issues and 24 README links/18 unique targets. Unstaged/untracked counts and raw/model/environment, 10-MB, secret, machine-path, manuscript, network/API and legacy-terminology findings were zero; 26 staged model-identity references were found and cached diff whitespace checks passed. `git commit -m "feat(supplement): finalize OOF heuristic search contract"` exited 0 and created `7226effd30835fc678b0bb21644f45ac0464dff6`.

Documentation receipt `08bc14e` recorded the checkpoint after a five-file/zero-prohibited-finding review. `git push -u origin finalization/leakage-aware-v2` exited 0 in 3.5 seconds and advanced the remote from `5f893ec` to `08bc14e`. No retry, force option, merge, release or credential change occurred.

## V2-015 executable INX workbook/CSV equivalence - 2026-07-14

The interrupted patch was recovered as one untracked production module plus two modified config files. `myenv\Scripts\python.exe -m py_compile src/governance/inx_workbook_equivalence.py` exited 0. A read-only Excel 16 COM inspection with macros/events/alerts disabled and link updates off confirmed the exact two sheets and shapes 1201 x 28 and 36 x 2. A direct normalized comparison exited 0 in 1.012 seconds: both data matrices hash to `b5caa2eec9a46ad184cc452e1d1df01abc80658db7fdd61e2cb8939943e23fbb`, with zero mismatches. Seven codebook blocks are exact; the explicit `RelationshipSatisfaction -> EmpRelationshipSatisfaction` alias binds every block to a data column while retaining `complete_data_dictionary=false`.

The first focused command exited 1 after 5.850 seconds wall time: 20 passed/3 failed because the validator rejected its own Windows `Path` default after stringification introduced backslashes. `_portable_path` now normalizes `Path` objects with `as_posix()` while continuing to reject unsafe string paths. The permitted post-fix focused cycle passed 23 in 2.01 seconds (5.176 wall). The provenance/manifest integration command passed 77 in 4.73 seconds (7.814 wall).

The one complete regression command started at `2026-07-14T15:20:36.8932559Z` under PowerShell PID 11364 and exited 0: 696 passed, 2 skipped and 11 subtests in 119.04 seconds (123.193 wall). Unittest discovery exited 0 with 176 tests/one skip in 7.276 seconds (11.102 wall); compileall exited 0 in 0.114 seconds. API-key variables were cleared process-locally. No network/API call, scientific refit, raw-data write, manuscript edit or retained diagnostic output occurred.

The staged review then found one no-swallowed-exception violation: atomic-write failure cleanup ignored an `os.unlink` error. Production now retains the primary exception and attaches the cleanup failure as an exception note; a negative test proves both are reported. The allowed post-review focus passed 24 in 1.87 seconds (4.903 wall). The final full pytest started at `2026-07-14T15:29:26.1755877Z` under PID 5332 and passed 697 with 2 skips/11 subtests in 117.32 seconds (121.237 wall). Final unittest passed 176/one skip in 7.196 seconds (11.059 wall); compileall passed in 0.122 seconds.

Implementation checkpoint `e524879d4e24cac39241a91814fc43878a2396e7` was pushed normally in 3.172 seconds and synchronized origin. The first clean production CLI attempt began at `2026-07-14T15:33:12.0832782Z` and exited 1 in 0.122 seconds before Excel opened or any output existed: argparse had stringified the default receipt `Path` with Windows backslashes, which the portable-path gate correctly rejected. The library path test had not exercised the CLI default. All three CLI defaults now use `.as_posix()`, and a Windows regression test freezes this behavior.

The post-failure focus passed 25 in 2.56 seconds (5.541 wall). Full pytest started at `2026-07-14T15:33:55.4136935Z` under PID 29440 and passed 698 with 2 skips/11 subtests in 117.97 seconds (121.926 wall). Unittest passed 176/one skip in 7.287 seconds (11.093 wall); compileall passed in 0.114 seconds. The failed CLI attempt left the worktree clean and created no partial receipt.

CLI-fix checkpoint `8f02e5569f1073b1dd3e0861e29d5f9189d79173` was pushed normally in 2.837 seconds and synchronized origin. The clean production command began at `2026-07-14T15:37:22.4690275Z` under PID 25828 and exited 0 in 1.408 seconds using Excel COM 16.0. It atomically wrote the 6,331-byte receipt with no temporary sibling. Independent `validate_receipt` exited 0 in 0.216 seconds. Receipt SHA-256 is `90e75733d469b8576b884c8e6eb849b017a4398f75becbacf5388c37edd1f2a5`; source-tree hash is `d2407e99153e66b3fdf9e29703b695f57fed8e3104a545e6eb6df5c9e4437624`.

## V2-014 forward raw-tip sanitation - 2026-07-14

Read-only inventory identified the exact 14 recorded raw-like paths: five source datasets/workbook, one interim table and eight row-level/fitted processed artifacts. Pre-untracking validation matched every configured size/SHA-256. `git rm --cached -- <14 exact paths>` removed only index entries; all 14 files remained locally present and ignored with unchanged hashes, totaling 2,335,429 bytes. No recursive removal, history rewrite or force operation occurred.

The production export contract/module and five tests were added. Focus one exited 1 after 19 passes because internal post-write validation treated an absolute Windows receipt `Path` as relative; archive creation itself had completed in the temporary fixture. Absolute-path containment and drive-letter rejection were fixed. Final focus passed 20 in 2.73 seconds (5.950 wall); expanded provenance/path/data integration passed 84 in 6.04 seconds (9.222 wall).

The complete pytest command began at `2026-07-14T15:45:52.0281898Z` under PID 20444 and exited 0 with 703 passed, 2 skipped and 11 subtests in 120.78 seconds (124.708 wall). Unittest passed 176/one skip in 7.336 seconds (11.177 wall); compileall passed in 0.114 seconds. API keys were cleared process-locally. No archive/receipt, raw-data write, network/API call, manuscript edit, history rewrite or local-file deletion occurred.

Staged review covered 33 files: exactly 14 configured index deletions, 1,003 insertions and 22,890 deletions. It reverified every local file hash/ignore rule, 11 required tracked documents, 32 complete issue rows, README links, a zero manuscript diff, prospective 305-member allowlist and zero raw/model/environment/secret/machine-path additions. Commit `9342b0c9a02788ff9e9867b13f2f824662fd1cf3` (`feat(publication): sanitize current data tip`) was pushed normally in 3.148 seconds; local and origin synchronized.

The production command `myenv\Scripts\python.exe -m src.governance.sanitized_publication_export` began at `2026-07-14T15:51:15.6622850Z` under PID 7152 and exited 0 in 2.24 seconds. It wrote only the 2,267-byte compact receipt, removed the temporary archive and recorded 305 archive members. Independent `validate_receipt(..., rebuild_archive=True)` exited 0 in 1.122 seconds and reproduced archive SHA-256 `1917059f11bcd732312964ce2e924342ad2ae28be2586729c8507ce369daecf5`. The receipt binds clean commit `9342b0c`, source tree `1dfee2ad...`, config `321bcd50...`, validator `6014b84d...`, 14 local files/2,335,429 bytes, 4,005,919 uncompressed member bytes and member-manifest SHA-256 `df89a3a6...`. Forbidden members, Git metadata, portable-content findings, secrets, symlinks, network calls, paid API calls and residual temporary siblings are all zero. Receipt SHA-256 is `3ddad1b7d2206a13b7bcfc63cae1ee02e5eaf6ba1c9dbb2e1632a71b7939d549`.

## V2-018 dependency isolation and lock - 2026-07-14

The audit found one mixed runtime file: canonical model/XAI libraries, CatBoost/UI packages and OpenAI/Agents SDK packages shared `requirements.txt`; no exact constraints existed; and PyYAML/xlrd/openpyxl were declared but absent locally. The implementation creates bounded core, supplementary, legacy-optional and development groups plus one sorted 96-pin CPython 3.14 constraints file. The canonical compatibility entry point includes core only. Source-tree hashing and run-input snapshots now bind every group, the lock and `environment.yml`; old Unit 2G commit hashing remains backward-compatible.

The production dependency validator passed statically with 22 unique direct packages and 96 exact pins. Initial focus passed 79 with one Python 3.14 invalid-escape warning in a negative-test regex; the raw regex correction changed no production behavior. The corrected focus passed 79 in 4.69 seconds (7.758 wall). Core pip resolver dry-run passed in 5.728 seconds and development resolver dry-run passed in 20.953 seconds; these index-resolution checks installed nothing and made no scientific/API call.

The first complete pytest wrapper was incorrectly given a five-second tool timeout and was terminated before pytest reported any result; process inspection found no surviving repository Python/pytest process. The unchanged suite was then run once with an adequate timeout and exited 0: 711 passed, 2 skipped and 11 subtests in 122.21 seconds (126.185 wall), starting at `2026-07-14T16:02:04.2230647Z` under PowerShell PID 29304. Unittest passed 176/one skip in 7.361 seconds (11.192 wall); compileall passed in 0.114 seconds.

A fresh CPython 3.14 virtual environment was created under the operating-system temporary directory. Its first wrapper likewise timed out while the uniquely identified pip process remained active; PID/command-line/ancestry inspection proved it belonged to this exact install, so it was monitored rather than killed or duplicated. After completion, `pip check`, all 13 direct imports and `validate_environment(profile='core')` passed. The isolated environment contains 31 locked non-bootstrap distributions and inventory SHA-256 `a7ac622b28e450da11890d6fa737a6f08af0c798d3f2bc382188728faf2bcb4b`. It is retained only for the clean-commit receipt and will then be removed by verified exact temp path.

Final review found that development-profile validation incorrectly treated marker-declared `pywin32` as mandatory on Linux. Requirement markers are now restricted to the exact Windows-only COM case and environment validation evaluates that marker. The added Linux regression raised the focused result to 80 passed in 5.07 seconds (8.245 wall). The required post-fix complete cycle passed 712 with 2 skips/11 subtests in 122.06 seconds (126.148 wall), starting at `2026-07-14T16:08:09.4861974Z` under PowerShell PID 5968. Post-fix unittest passed 176/one skip in 7.303 seconds (11.100 wall); compileall passed in 0.122 seconds.

Staged review covered exactly 31 files, 1,059 insertions/51 deletions, maximum 120,076 bytes, 32 complete issues and 26 README links. It found zero unstaged/untracked, raw/model/environment/large/secret/machine-path/manuscript/receipt or forbidden-core-dependency entries. Commit `498e8ad59166f275d120f78ce133cce122961f13` (`build(repro): isolate and lock Python dependencies`) was pushed normally in 2.935 seconds and synchronized origin.

The retained isolated core interpreter generated `13_dependency_lock_receipt.json` from the clean commit in 0.685 seconds. Independent receipt validation exited 0 in 0.218 seconds. The 2,476-byte receipt binds source tree `a2b361b83e9ced24ee8da9e35ad6c09e5f2ea82c34be2b331abce1bac5f03f23`, config `65519abf...`, constraints `482cbf32...`, 31 distributions, inventory `a7ac622b...`, 13 direct versions and zero missing/unlocked/mismatched/core-forbidden packages. Receipt SHA-256 is `8b77e72702865f3950b91092cd458c79149d3378bcb8b703b323aec0c4033c20`; no atomic temp sibling exists.

The first cleanup ownership scan matched its own PowerShell command line and safely refused removal without changing the environment. The corrected check excluded only its own PID, found no separate owner, reverified the exact resolved path under the operating-system temp root and exact run leaf, then removed that directory. Zero matching V2-018 temp directories remain.

## V2-020 global scientific-runtime network/API denial - 2026-07-14

Read-only audit confirmed the former hard denial existed only inside the isolated benchmark trial, while the complete builder could import the canonical loader's approved-download path and had no process-wide boundary. `offline_runtime.py` now patches socket DNS/connect/send/listener operations globally across threads, removes/restores six paid-API credential variables, poisons a run after caught attempts, rejects shell and every non-Git child process, and restricts Git to exact local read-only provenance subcommands. The public `build` wraps `_build_impl` from preflight through package completion. Package status writes/validates the exact zero-attempt policy.

Initial focus passed 92 in 5.41 seconds (8.595 wall). Review narrowed Git from executable-only to an exact read-only subcommand allowlist; corrected focus passed 93 in 4.65 seconds (7.677 wall). The first full suite reported one compatibility failure after 724 passes: an existing test inspected public `build` for stage-command ordering after it became the wrapper. The test now correctly inspects `_build_impl`; no production behavior changed. Post-fix focus passed 107 in 3.35 seconds (6.329 wall).

The allowed post-fix full cycle started at `2026-07-14T16:20:10.6693708Z` under PowerShell PID 17796 and exited 0: 725 passed, 2 skipped and 11 subtests in 122.38 seconds (126.377 wall). Unittest passed 176/one skip in 7.563 seconds (11.357 wall); compileall passed in 0.115 seconds. A direct real-config core invocation with OPENAI/ANTHROPIC credentials present exited through the expected `release_ready=false` gate in 0.139 seconds, restored both credentials/removal marker and created no run.

## V2-019 CI and read-only release validation - 2026-07-14

Recovery began clean and synchronized at `d74206504e593c11cfdf1f9328600351b73ea3ad`. Read-only inspection confirmed that no `.github` workflow existed. The implementation added hosted push/PR CI, a validation-only trusted Windows self-hosted workflow, `ci_repository_gate.py`, workflow-contract tests, and a read-only release-candidate validator separated from `latest` promotion. No workflow uploads, publishes, pushes, promotes or edits manuscript files.

The default `python` shim was unavailable and both installed interpreters were bare. A disposable CPython 3.14 environment was therefore created at the exact operating-system temporary leaf `hrxai-v2019-py314`; `pip install --requirement requirements-dev.txt` exited 0 in 86.8 seconds using the tracked lock. Workflow files then parsed successfully with PyYAML, `python -m src.governance.dependency_contract --profile development --validate-only` reported 22 direct packages/96 pins, and `pip check` reported no broken requirements.

The final focused CI/release command passed 35 tests in 1.42 seconds (4.570 wall). The first complete locked-environment run failed during config loading because PyYAML typed JSON exponent values such as `1e-12` as strings. A lexical config workaround passed 57 tests but failed JSON-dumped temporary fixtures. The root fix changed the shared loader to parse documented JSON-compatible configs with `json.loads` first and use PyYAML only for non-JSON YAML; original config bytes/values were restored. All 122 previously affected tests passed in 12.01 seconds (15.217 wall).

The final complete cycle exited 0: pytest passed 730 with two skips/11 subtests in 122.16 seconds (126.145 wall); unittest passed 177 with one skip in 8.178 seconds (11.861 wall); compileall passed in 0.104 seconds. No real-data rebuild, network/API scientific call, package promotion, manuscript modification or release occurred.

Checkpoint `714a1c36bcae548a863a5fb77cd0d8144d17917e` pushed normally and triggered hosted run `29350928894`. Checkout, Python setup, locked install, dependency validation, manuscript diff and focus passed; full pytest failed because 22 exact tests require intentionally ignored local datasets/workbook. Authenticated read-only job-log retrieval confirmed only missing approved local paths, not a cross-platform production defect.

The failure-driven fix adds exact file-presence skips to those integrations on data-free hosts and makes the trusted release workflow fail before tests unless all five local inputs exist. The affected local focus passed 94 with real data. Post-fix local pytest passed 730/2 skips/11 subtests in 123.44 seconds (127.411 wall); unittest passed 177/one skip in 7.945 seconds (11.733 wall); compileall passed in 0.103 seconds. Follow-up commit `700adb235e66927df0ea4bf6b78ec811c9ffc1ee` pushed normally.

Hosted run `29351557672` then passed every step. Exact log results: dependency contract 22 direct/96 pins, `pip check` clean, focus 35 in 0.67 seconds, pytest 709 passed/23 explicit skips/11 subtests in 104.65 seconds, unittest 177 in 6.516 seconds with 10 skips, compileall and repository gate passed. No raw input was uploaded or synthesized.

## V2-021 source-bound core-figure implementation - 2026-07-14

Recovery started from clean synchronized `ea34ed658c141ad9158cf1543c409718753d2464`. GitHub Actions receipt run `29352015443` was read-only checked and had completed successfully at that exact commit. The production implementation added `manuscript_core_figures.py`, registered atomic `core_figures` execution, added specialized package validation to cache/completion admission, updated the frozen pending-execution reason and added a rendered synthetic contract test. It did not touch manuscript files or generate a retained real artifact.

The first environment install referenced a nonexistent lock filename and printed `Could not open requirements file ... requirements-lock-py314.txt`; no package was installed by that command. The corrected exact-lock installation used `constraints/py314-lock.txt`, exited 0 after 83.4 seconds, and passed `pip check` and imports.

Focused chronology: 38 passed/one failed because Matplotlib SVG contained a prohibited `DOCTYPE`; after production sanitization, 39 passed. A real-writer schema audit then aligned SHAP stability and empty calibration-bin handling; the generator focus passed 3. Complete pytest one reported 732 passed/two skipped/11 subtests and one generic promotion-fixture failure. The fixture was isolated from the separately tested specialized package contract; the exact test passed. Final complete pytest passed 733/two skipped/11 subtests in 128.34 seconds (132.292 wall), unittest passed 177/one skip in 7.334 seconds (11.190 wall), and compileall passed in 0.121 seconds.

Config/source validation produced `a866bd6f9851fffce9af4238459d05fc1c6540f9e11d3668698a11c23811107d` and `fcae16c902fc759be087a9c42c451799bdc97fc49b6c396eb54260086a1e5987`. Unit 2G, the historical packages and the manuscript were unchanged. No network/API scientific call, promotion, release or retained figure output occurred.

The consolidated pre-stage hygiene gate exited 0 in 1.143 seconds. It checked 16 candidate paths, 32 complete unique issues, 29 README local-link occurrences/23 unique targets, zero manuscript diff and a 127,236-byte maximum candidate. Raw/model/environment/large/link/secret/machine-path/network-import/forbidden-external-feature/staging-orphan findings were all zero.
## V2-022 metric registry and source-table implementation - 2026-07-14

The task began from synchronized commit `5a0d926`. GitHub's public Actions API confirmed hosted run `29353799148` completed successfully at that exact SHA. No scientific stage was rerun.

The first disposable CPython 3.14 environment wrapper timed out after creating its exact task-owned venv and left no child process. Inspection confirmed the environment path and no owner; installation resumed in the same environment. `pip install -r requirements-dev.txt -c constraints/py314-lock.txt` completed in 82.9 seconds and `pip check` reported no broken requirements.

The first focused command covered the new table generator, metric applicability/definitions, OOF bootstrap domains, external bootstrap, scope and promotion contracts. It reported 43 passes, two failures and three setup errors in 12.74 seconds (18.635 wall). Every failure had one cause: `MappingProxyType` was still used for aligned OOF systems after its import moved with the registry. The import was restored. The exact repeated focus passed 48 in 17.75 seconds (20.906 wall).

Complete `pytest -q` initially passed 738 with two skips and 11 subtests in 134.66 seconds (138.527 wall). `python -m unittest discover -s tests -p 'test_*.py'` passed 177 with one skip in 8.174 seconds (12.000 wall). `compileall -q src tests` passed in 0.121 seconds.

The production repository gate was invoked before commit and failed only its intentional clean-worktree precondition, listing the V2-022 edits. It did not report a scientific, schema, raw-data, secret, portability or test defect. The gate must be repeated at the clean checkpoint. No manuscript file, real scientific artifact, raw data, paid API or scientific network operation was created or changed.

Staged review identified that receipt-count validation did not require unique source paths or bind each emitted row's source hash back to its receipt. The validator now enforces exact unique receipt coverage, exact table/contract columns and non-empty provenance, and a negative test tampers a duplicated receipt. Post-review focus passed 30 in 17.26 seconds (20.446 wall); the required final complete suite passed 739/2 skips/11 subtests in 132.92 seconds (136.855 wall); unittest passed 177/one skip in 7.964 seconds (11.710 wall); compileall passed in 0.117 seconds. Final config/source/metric hashes are `1bcd1386...`, `fba949d2...`, and `98ae57b6...`.
## V2-023 execution-readiness scope freeze - 2026-07-14

Read-only inventory proved zero missing runners across the ten-stage core and four-stage supplementary graphs. The only remaining execution gate was stale `release_ready=false` metadata in both scopes and the implemented core-figure plan. The config/figure contract now marks technical execution readiness true while retaining explicit canonical-execution/promotion-pending notes. Negative tests revoke each flag and confirm pre-execution failure.

The initial focused scope/figure/manifest/builder command passed 114 in 5.50 seconds (8.635 wall). Full cycle one reported 732 passes/two skips/11 subtests and nine failures in 131.34 seconds (135.344 wall); all nine came from one side-input fixture that deliberately reconstructed an old incomplete core scope. Updating that fixture to the current readiness contract produced a 31-test affected focus pass in 2.62 seconds (5.605 wall). The allowed final full cycle passed 741/2 skips/11 subtests in 132.28 seconds (136.238 wall). Unittest passed 177/one skip in 7.937 seconds (11.797 wall); compileall passed in 0.117 seconds. Config/source hashes are `51415c2ce68c89d9ce2b042b0a7a811fe3e98180b72460006f9d0465f6bf49b7` and `2aee9de9e01f95357dd35cce63a281e4d145c8ded5b5d3d74f97b9f11d29cad3` before commit.

No real stage, network/API operation, table/figure artifact, manuscript edit, promotion or release occurred during the freeze.
## First clean build failure and class-compatible offline-runtime fix - 2026-07-14

Configured-input preflight found all five exact local datasets present and no active repository model/Python process, lock or staging directory. Core run `canonical_v2_20260714T175804Z_4d08ca2` began from clean synchronized commit `4d08ca2` with cache reuse disabled. It exited 1 after 1.239 seconds in `shared_folds`: sklearn imported joblib/asyncio after the offline boundary had replaced `subprocess.Popen` with a function, and Python 3.14 could not subclass that replacement. No model fit occurred. The run manifest is `status=failed`; 16 files totaling 141,235 bytes are preserved locally and ignored exactly.

The runtime guard now substitutes a subclass of the original Popen and consults a process-wide active state. This preserves subclass semantics for stdlib/library late imports while still rejecting shell, non-Git, remote-Git and unauthorized executable requests in all threads. A subclass created during one boundary works normally after exit, and a fresh child process imports sklearn inside the boundary successfully.

The first fix focus failed one test because the test made a prohibited child-process attempt and then expected the intentionally poisoned boundary to exit cleanly. The corrected test avoids the prohibited attempt. Final focus passes 51 in 4.15 seconds (7.208 wall); full pytest passes 743/2 skips/11 subtests in 134.15 seconds (137.961 wall); unittest passes 177/one skip in 8.087 seconds (11.967 wall); compileall passes in 0.114 seconds. Source-tree hash is `868a69aac143fe852a5738dd5a720be5f217243dfcd410d69c4954ca617ca8f7` before commit.

## Second clean build failure and registry/report-subset repair - 2026-07-14

Core run `canonical_v2_20260714T180533Z_770c8d0` began from clean synchronized commit `770c8d0` with cache reuse disabled. `shared_folds` completed and the complete four-model benchmark ran for approximately 630 seconds. Policy-ablation setup then failed before policy fitting because `_configured_metrics` required the complete ordinal applicability registry to equal the narrower predeclared report order. The builder exited 1 after 635.282 seconds and recorded `status=failed`; 71 files totaling 92,337,983 bytes are preserved locally under an exact ignore rule. No artifact from this run is admissible for reuse, promotion or a scientific claim.

Read-only applicability-consumer audit found the same exact-set assumption in calibration and no other affected production consumer. Policy ablation and calibration now reject duplicate/invalid applicability, require every report metric to remain applicable and return only their fixed report subsets. The complete authoritative registry, metric definitions and config remain unchanged.

The default Windows Store `python` shim could not launch and bare `py` interpreters lacked pytest; no test started under those commands. The established repository interpreter `myenv/Scripts/python.exe` was then used. One focused invocation named two nonexistent test paths and collected zero tests; the corrected exact focused command passed 43 in 4.99 seconds (8.272 wall). The complete pytest command started at `2026-07-14T18:19:37.9870620Z` and passed 747 with two skips/11 subtests in 140.55 seconds (144.800 wall). Unittest passed 179/one skip in 7.552 seconds (11.531 wall); compileall passed in 0.125 seconds. Config/source/metric hashes are `51415c2c...`, `0285772e...`, and `98ae57b6...` before checkpoint.

## Valid pre-migration candidate and historical-latest migration - 2026-07-14

Checkpoint `35b121bc9e541cf2736cfe0fc2912e327d86dbd2` pushed normally. The clean repository gate passed with 1,947 tracked files, 32 issue rows, 30 README links and zero raw/environment/large/secret/machine-path findings. Core `canonical_v2_20260714T182539Z_35b121b` exited 0 in 1,614.358 seconds; supplementary under the same run ID exited 0 in 1,862.575 seconds. No cache reuse, network/API operation or reduced scientific setting was used.

Read-only `--validate-run-id canonical_v2_20260714T182539Z_35b121b` exited 0 in 5.673 seconds with `status=valid`. It bound commit `35b121b`, config `51415c2c...`, source `0285772e...`, core/supplementary scientific inputs `34a5e692...`/`e2b41f9f...`, and final-manifest hashes `c68d8fe3...`/`4720a149...`.

Promotion preflight found the tracked physical `latest` alias. Exact recursive SHA/size comparison proved 215 common files/166,264,643 bytes identical to the separately tracked named v1 run, with zero drift/reparse points. The only extra was `run_pointer.json` (139 bytes, SHA-256 `e9a96a5d...`) already naming that run. The redundant 216-file alias was removed with `git rm -r` only after exact containment and preservation checks; the named run remains 215 files/166,264,643 bytes. Receipt `../finalization_v2/14_latest_migration_receipt.json` records the inventories. No historical evidence byte or Git history was deleted.

Because strict package validation requires the generation commit to equal current HEAD, the valid `35b121b` candidate is preserved locally and ignored but cannot be promoted after the migration commit. This is a publication-identity requirement, not a scientific failure; one new same-source exact-commit two-scope build is required.

## Migration-commit atomic publication failure and repair - 2026-07-14

Clean repository validation passed at pushed migration commit `0c2868bd98d9f70d8711dc0e75849180d457712f`: 1,732 tracked files, 32 issue rows at that checkpoint, zero prohibited inventory findings, preserved named v1 count 215, `latest` absent, config `51415c2c...`, source `0285772e...`, and no active repository process.

Core run `canonical_v2_20260714T192934Z_0c2868b` exited 1 after 1,066.336 seconds at calibration publication. `Path.replace` returned `WinError 5` while moving `.sigmoid_calibration.gs9vypah`; immediate cleanup returned `WinError 32` on `calibration_training_oof_predictions.csv` and masked the first exception. The run manifest is failed; 93 files/105,400,103 bytes and the staging directory are preserved and ignored. No resume/reuse/promotion is allowed.

`atomic_publish.py` now supplies bounded absent-destination directory replacement and cleanup. It retries only WinError 5/32/33, at most 40 total attempts with 0.25-second spacing, rechecks staging/destination every time and immediately raises every other error. Exhausted cleanup is attached to the primary exception. All nine production directory publication sites use the helper.

The first focus reported two stale assertions after 145 passes/one skip; both required literal `os.replace` text. They now require the shared helper and preserve completion-contract ordering. Corrected focus passed 147/one skip in 19.22 seconds (22.625 wall). Full cycle one reported 750 passes/two skips/11 subtests and one stale `tracked_file_count >= 1900` assertion after the intentional 216-file migration. The corrected test requires at least 1,700 tracked files, no physical `latest` and exactly 215 named v1 files; its direct test passed. Final pytest passed 751/two skips/11 subtests in 135.02 seconds (139.048 wall). Unittest passed 179/one skip in 7.683 seconds (11.742 wall); compileall passed in 0.124 seconds. Config/source hashes are `51415c2c...` and `e5527c99...` before checkpoint.

Staged review found that a non-permission cleanup exception could still mask an existing primary publication error. The cleanup helper now catches ordinary exceptions, retries only the same three Windows lock codes and otherwise attaches the cleanup failure to the primary exception. The new negative test raised focus to 148/one skip in 19.37 seconds (22.846 wall). The final post-review pytest passed 752/two skips/11 subtests in 134.51 seconds (138.459 wall); unittest passed 179/one skip in 7.670 seconds (11.659 wall); compileall passed in 0.124 seconds.

## Compact manuscript-support asset export - 2026-07-16

Recovery verified branch `finalization/leakage-aware-v2`, synchronized upstream HEAD `7bb2317a9325fc2be1a96851ee4b5df0fb5aa18c`, clean canonical generation commit `483f96fdbaab16cb0f32d03d9dbe676a759af44a`, pointer-only `latest`, and the expected run/config/source/scientific identities. The attachment's expected `be0c154...` HEAD was stale; committed changes since generation affected publication documentation only, with zero scientific-source/config diff.

The strict current-HEAD canonical CLI was invoked read-only and refused the older generation commit as designed. Detached exact-context attempts were not accepted because Windows byte/source-tree context or absent raw inputs prevented a faithful current environment. No experiment was rerun. A direct fail-closed validator then independently rehashed both closed-world manifests, all 539 artifact records, all 545 physical files, receipt/pointer identities, source-tree bytes and scientific-input identities: core 351 records/354 files/224,171,937 bytes and supplementary 188 records/191 files/222,423,549 bytes, exit 0 in 1.643 seconds.

The first deterministic export attempt failed before publication on an unsupported SVG metadata key and left no staging residue. After using supported creator metadata, export succeeded; visual QA found a clipped Figure 1 box and missing mapped-support display in Table 8. The rendering/filter logic was corrected, only the exact untracked export directory was removed under manifest/run/tracking safety assertions, and atomic export reran in 7.343 seconds. No canonical byte or scientific source/config changed.

Final `--validate-only` passed with 109 files/10,338,351 bytes, seven main and three supplementary PNG/SVG pairs, eight main tables, and manifest SHA-256 `fbe7355b...`. Independent parity validation rehashed 308 source bindings across 108 manifest records and passed in 1.845 seconds. The full canonical root was restored to a single ignore rule; `git rm --cached` removed its 50 current-index entries while 545 local files remained present and ignored.

The new focus initially reported two incorrect test-schema assumptions after 15 passes/one skip; no production failure occurred. Corrected focus passed 17/one skip in 4.15 seconds (7.357 wall). Complete pytest passed 758/two skips/11 subtests in 129.91 seconds (133.988 wall); unittest passed 179/one skip in 8.214 seconds (12.246 wall); compileall passed in 0.312 seconds. Candidate hygiene, package validation, README links, tracking policy, manuscript immutability, source-tree freeze, size, secret/raw/path/staging/forbidden-feature and active-terminology gates passed. GitHub CLI was unavailable, so the requested normal native-Git push path will be used; no PR is in scope.

Exact staged review covered 137 Git change records with 109 compact asset paths, zero unstaged/untracked or unexpected paths, no file at or above 100 MiB, no raw/model/environment/secret/machine/manuscript/scientific path, zero tracked canonical-root files, and all 545 canonical files preserved locally. Byte-preserving attributes prevented Windows line-ending conversion; all 109 staged asset blobs matched their declared SHA-256 values and `git diff --cached --check` passed.

Checkpoint `902d4041fce94139ba58cb7c982f9a62b8427611` (`feat(manuscript-assets): publish canonical figures and result tables`) was created. The clean post-commit repository gate exited 0 in 0.760 seconds with 1,849 tracked files, 33 issues, 43 README links, and zero prohibited inventory findings; package validation remained exact. `git push origin finalization/leakage-aware-v2` exited 0 in 7.229 seconds: `7bb2317..902d404  finalization/leakage-aware-v2 -> finalization/leakage-aware-v2`. Authentication is working.

## Major-revision v3 INX information contract — 2026-09-03

The additive JSON contract and fail-closed renderer were implemented without changing v2 configuration, canonical evidence, or manuscript source. The generator wrote `FEATURE_AVAILABILITY_GOVERNANCE_CONTRACT.md` atomically and returned contract SHA-256 `6dd5fdde...`, semantic SHA-256 `1773f42a...`, 28 fields, six policies, zero timestamp-verified features, and retained counts 26/24/21/20/13/6. A first 11-test focus passed. After staged self-review tightened every policy from minimum category removal to an exact exclusion set and added exact contract-identity enforcement, the final expanded focus passed 44 tests and seven subtests. Compile, 46-local-link README validation, and `git diff --check` passed. No scientific fit or external call occurred.
