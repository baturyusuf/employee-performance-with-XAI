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
