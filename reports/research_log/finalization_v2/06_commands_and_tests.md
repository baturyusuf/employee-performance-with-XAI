# Commands and Tests

Date: 2026-07-13

## Git and Repository Baseline

```powershell
git branch --show-current
git rev-parse HEAD
git status --short
git remote -v
git rev-list --left-right --count HEAD...origin/main
git ls-files
git ls-tree -r -l HEAD
git count-objects -vH
```

Result: `main`; HEAD `1c7c343bda401629a3619f92267384916f0708d0`; origin synchronized; initial worktree clean; 1,850 tracked files; about 601 MiB tracked content.

## Baseline Pytest

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m pytest -q
```

Exit status: 0. Result: 188 passed and 4 subtests passed in 14.75 seconds.

## Baseline Unittest

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
```

Exit status: 0. Result: 161 tests passed in 4.764 seconds.

## Compile

```powershell
.\myenv\Scripts\python.exe -m compileall -q src tests
```

Exit status: 0.

## Actual Input Audit

The raw and interim INX files were loaded with the repository CSV parser, hashed with SHA-256, and compared. Raw: `b8deac0...`; interim: `afc97e8...`; direct frame equality false because the identifier name differs by BOM normalization. A normalized cell-level comparison found identical scientific cells. `load_validated_or_raw_data()` still selects interim because it exists, so the manifest/cache binding remains false.

A read-only Excel COM audit compared the first INX workbook sheet against the normalized CSV and found 1,200 x 28 identical cells and zero duplicate identifiers. This was not a repository script and is not accepted as the final reproducible equivalence test.

## Path and Artifact Audit

```powershell
rg -n 'C:(\\\\|\\|/)+Users|/home/' reports/manuscript_final/latest
```

Result: multiple absolute-user-path hits in run manifest, stage metadata, LLM evidence and chatbot suites.

## CI Audit

- `.github/workflows`: absent.
- GitHub connector query for workflow runs at HEAD: zero runs.
- Local `gh`: unavailable.

## Worktree Safety

The worktree remained clean after baseline tests. Only finalization documentation was added after the read-only audit completed. No manuscript or scientific code file was changed.

## Unit 1A — Pinned Real-Data Preflight

```powershell
.\myenv\Scripts\python.exe -m src.data.canonical_loader --config configs/manuscript_final.yaml --datasets all
```

Exit status: 0. Verified five logical tasks against four local physical files. Actual SHA-256, ordered schema, row/column counts and target distributions all matched. `acquisition_method=existing_local_file` for every task; automatic download remained false.

## Unit 1A — Focused Tests

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_actual_input_hash_binding.py tests/test_no_implicit_interim_fallback.py tests/test_data_acquisition_preflight.py
```

Exit status: 0. Result: 11 passed in 1.26 seconds.

Coverage includes configured-path precedence, actual receipt hash, config/acquisition path consistency, no implicit interim fallback, undeclared dataset failure, unapproved missing-data failure, approved temporary download validation, mismatch quarantine/reporting and ordered-schema drift.

## Unit 1A — Full Regression Gate

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
git diff --quiet -- manuscript/mdpi_information/main.md
```

Results: pytest 199 passed plus 4 subtests in 12.47 seconds; unittest 161 passed in 4.444 seconds; compileall exit 0; diff check exit 0; manuscript unchanged. No paid service or real dataset download was attempted.

## Unit 1A Checkpoint

```powershell
git commit -m "fix(data): bind canonical inputs to pinned acquisition contract"
```

Commit: `f4e2dd7`. The staged diff check printed five Markdown trailing-whitespace findings, but the command chain did not fail closed before commit. The defect was recorded and corrected without amend/history rewriting. Future commit gates must use explicit conditional execution rather than semicolon-separated commands.

## Unit 1B — Manifest, Side-Input and Cache Binding

Real dataset/side-input manifest preflight:

```powershell
.\myenv\Scripts\python.exe -c "from src.governance.manuscript_contract import create_run_manifest,validate_run_manifest; m=create_run_manifest('configs/manuscript_final.yaml',run_id='unit1b_real_preflight'); validate_run_manifest(m); print({'schema':m['manifest_schema_version'],'datasets':len(m['actual_input_receipts']),'side_inputs':len(m['side_input_hashes']),'scientific_input_hash':m['scientific_input_hash'],'dirty':m['git_worktree_dirty']})"
```

Exit status: 0. Result: manifest schema 2, five logical dataset receipts, seven side inputs, aggregate scientific-input hash `38dfd51794a837c09cfb67a16eac283d5dc568c94bfa7043a7f5cd14ad6f3b67`. `dirty=true` is expected for this implementation preflight and is not accepted as a final release run.

Focused manifest/cache/snapshot contract:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_artifact_run_manifest_consistency.py tests/test_side_input_hash_binding.py tests/test_cache_invalidates_on_scientific_input_change.py
```

Exit status: 0. Result: 21 passed in 3.09 seconds.

Focused external explicit-input contract:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_external_explicit_input_binding.py tests/test_manuscript_external_evidence.py tests/test_external_validation.py tests/test_external_claim_boundaries.py
```

Exit status: 0. Result: 23 passed in 7.40 seconds.

Full regression gates:

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
git status --porcelain -- manuscript/mdpi_information/main.md
```

Results: pytest 218 passed plus 4 subtests in 14.58 seconds; unittest 161 passed in 4.789 seconds; compileall exit 0; diff check exit 0; manuscript status empty.

One hygiene wrapper incorrectly used the PowerShell expression `if (git diff --quiet ...)`, which interpreted the command's empty stdout as false and reported a nonexistent manuscript change. Direct `git status --porcelain -- manuscript/mdpi_information/main.md` confirmed no change; the corrected fail-closed wrapper then passed. This was a command-wrapper defect, not a repository or manuscript failure.

Network/API result: all paid-service environment variables were cleared; no dataset was missing, no approved acquisition URL was invoked, and no API/network call occurred.

The first Unit 1B commit attempt was correctly blocked before commit by `git diff --cached --check` because a new test file had one blank line at EOF. The whitespace-only defect was removed with no history rewrite; the scientific code and test outcomes were unaffected.

## Unit 2A — Core/Supplementary Scope Isolation

Scoped real-input manifest preflight:

```powershell
.\myenv\Scripts\python.exe -c "from src.governance.manuscript_contract import create_run_manifest,validate_run_manifest; [(lambda m,s: (validate_run_manifest(m,expected_evidence_scope=s), print(s, sorted(m['actual_input_receipts']), sorted(m['side_input_hashes']), m['scope_contract_hash'], m['scientific_input_hash'])))(create_run_manifest('configs/manuscript_final.yaml',evidence_scope=s,run_id='unit2a_'+s),s) for s in ('core','supplementary')]"
```

Exit status: 0. Core receipts: `hrdataset_v14`, `inx_primary`; five scoped side inputs. Supplementary receipts: `employee_turnover`, `ibm_hr_analytics`, `ibm_hr_analytics_attrition`, `inx_primary`; six scoped side inputs.

Fail-closed core entrypoint check:

```powershell
.\myenv\Scripts\python.exe -m src.experiments.build_core_paper_evidence --config configs/manuscript_final.yaml
```

Expected exit status: 1. The command failed before output creation with `Evidence scope 'core' is not release-ready`; it did not execute or simulate a scientific stage.

Focused contracts:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_core_scope_contract.py tests/test_core_build_contains_no_llm_or_chatbot_stage.py tests/test_no_cross_scope_cache_reuse.py tests/test_scoped_run_manifest_inputs.py tests/test_external_scope_contract.py tests/test_external_explicit_input_binding.py tests/test_manuscript_external_evidence.py tests/test_dataset_card_required_fields.py tests/test_final_evidence_manifest_hashes.py tests/test_manuscript_figures_generated.py
```

Exit status: 0. Result: 50 passed, 2 skipped in 7.44 seconds. Skips refer only to the deliberately rejected historical unscoped `latest` package.

Full regression gates:

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
git status --porcelain -- manuscript/mdpi_information/main.md
```

Final results: pytest 250 passed, 2 skipped and 4 subtests passed in 13.99 seconds; unittest 162 passed with 2 skipped in 3.259 seconds; compileall/diff/manuscript gates passed.

An earlier full pytest attempt produced one failure because the legacy Figures 1-4 test still supplied the core config after LLM settings were removed. The legacy generator was made fail-closed before creating files and its test was corrected to assert exclusion from core. The next full run passed.

Network/API result: credentials were cleared; no network function, approved dataset URL or paid service was invoked. Scientific artifacts generated: none.

The first Unit 2A commit attempt was correctly blocked before commit because the staged diff contained one blank line at EOF in each of two new contract tests. Both whitespace-only defects were removed; no history was created or rewritten and scientific/test behavior was unchanged.

## Unit 2B — Shared Folds, Nested Benchmark and Paired OOF Bootstrap

Focused contract regressions after integration hardening:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_shared_fold_artifact_contract.py tests/test_shared_fold_assignments_across_models_and_policies.py tests/test_nested_search_outer_test_isolation.py tests/test_bootstrap_is_stratified_paired_and_deterministic.py tests/test_oof_bootstrap_intervals_are_domain_valid.py tests/test_paired_model_difference_bootstrap.py tests/test_paired_model_gate_contract.py tests/test_canonical_model_factory.py tests/test_manuscript_model_benchmark.py tests/test_stage_runner_scientific_input_binding.py tests/test_core_scope_contract.py tests/test_core_build_contains_no_llm_or_chatbot_stage.py
```

Exit status: 0. Result: 83 passed in 16.65 seconds.

Full regression gates:

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
git status --porcelain -- manuscript/mdpi_information/main.md
```

Results: pytest 314 passed, 2 skipped and 4 subtests passed in 31.23 seconds; unittest 162 passed with 2 skipped in 3.345 seconds; compileall/diff/manuscript gates passed.

Real-input fold-contract preflight used the canonical loader and an in-memory scoped manifest, then called `generate_shared_folds(... outer_splits=10, inner_splits=3, seed=42, inner_seed=43)` and `validate_shared_folds` without writing outputs. Exit status: 0. Result: 1,200/1,200 exact outer assignments, 10 outer folds, 10,800 inner assignments, exactly three inner folds for each outer fold, pinned INX SHA-256 `b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a`. The emitted predecision/dirty-tree fold hash is diagnostic only and must not be reused.

One initial focused-test command invoked the broken Windows Store `python.exe` shim and failed before Python started; `py -3.14` then proved that the system interpreter has no pytest. All authoritative tests use `.\myenv\Scripts\python.exe` and passed. During development, an unbound `inner_splits` variable, bootstrap sorting/direction defects and inactive LightGBM row subsampling were detected by focused checks/review and corrected before the full suite. No real model benchmark was executed because the selection/gate metric decision is intentionally pending.

Network/API result: credentials were cleared; no dataset acquisition, network/API call or paid service occurred. Scientific artifacts generated: none. Manuscript edits: none.

The first Unit 2B staged checkpoint gate was blocked by `git diff --cached --check` because two new bootstrap test files had one blank line at EOF. The whitespace-only defects were removed with `apply_patch`; no commit or history mutation occurred before correction.

Checkpoint command `git commit -m "feat(protocol): add shared nested OOF benchmark contract"` exited 0 and created `8e9b5b9b9f66815abf7f9a599535a36737ea1706`. No push, merge, release or history rewrite occurred.

## Unit 2B 10×5 Correction and Trial Preflight

Accepted contract: macro-F1 primary selection/gate; QWK secondary tie-break only within an inclusive absolute `0.001` macro-F1 pool; 10 outer × 5 inner folds; gate requires positive point estimate and paired OOF 95% CI lower bound above zero.

Focused corrected protocol/trial/input suite:

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m pytest -q tests/test_model_benchmark_trial_entrypoint.py tests/test_manuscript_model_benchmark.py tests/test_canonical_model_factory.py tests/test_shared_fold_artifact_contract.py tests/test_shared_fold_assignments_across_models_and_policies.py tests/test_nested_search_outer_test_isolation.py tests/test_bootstrap_is_stratified_paired_and_deterministic.py tests/test_oof_bootstrap_intervals_are_domain_valid.py tests/test_paired_model_difference_bootstrap.py tests/test_paired_model_gate_contract.py tests/test_stage_runner_scientific_input_binding.py tests/test_actual_input_hash_binding.py tests/test_side_input_hash_binding.py
```

Exit status: 0. Result: 104 passed in 47.36 seconds.

Full gates:

```powershell
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
git status --porcelain -- manuscript/mdpi_information/main.md
```

Results: pytest 343 passed, 2 skipped and 4 subtests in 64.84 seconds; unittest 162 passed with 2 skips in 3.334 seconds; compileall/diff/manuscript gates passed.

The verified-INX 10×5 in-memory preflight invoked the canonical loader, scoped manifest creation, `generate_shared_folds` using config counts/seeds and `validate_shared_folds` without writing. Exit 0: 1,200 rows; target 2/3/4 support 194/874/132; 10 outer folds; 10,800 nested rows; five inner folds per outer fold; inner validation size 216. INX SHA-256 `b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a`; config hash `87198ca980d867a4889ee402fd5979fdae0d6a5563c8390b863295fdd66161e5`; diagnostic dirty scientific-input hash `3434668a5a7778ce0e70095fe8cbf3c39718d82c44d4604f16f5c6deced8fc58`. No artifact/model was created.

During review, the semantic verifier initially referenced a nonexistent `outer_fold_assignments.csv`; production writes `fold_assignments.csv`. This was caught before the expensive run, replaced with `OUTER_ASSIGNMENT_FILENAME`, and covered by the real filename fixture. A process-local socket/DNS denial was then added and tested. No network/API call or manuscript edit occurred.

## Unit 2B Final Pre-Run Hardening

Focused command:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_model_benchmark_trial_entrypoint.py tests/test_manuscript_model_benchmark.py
```

Exit 0: 41 passed in 35.99 seconds.

Full gates:

```powershell
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
git status --short -- manuscript
rg --pcre2 -n '(?<![A-Za-z])sk-(?:proj-)?[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}' --glob '!myenv/**' --glob '!reports/manuscript_final/**' --glob '!*.ipynb' .
```

Results: pytest 345 passed, 2 skipped and 4 subtests in 64.70 seconds; unittest 162 passed with 2 skips in 3.244 seconds; compileall/diff/manuscript gates passed; the secret expression returned no match. The `rg` no-match exit is the expected passing condition.

Exact fresh real-input in-memory preflight command:

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
@'
import json
from src.data.canonical_loader import load_canonical_dataset
from src.experiments.shared_folds import generate_shared_folds, validate_shared_folds
from src.governance.manuscript_contract import create_run_manifest, load_manuscript_config, manuscript_settings
config_path = "configs/manuscript_final.yaml"
config = load_manuscript_config(config_path)
settings = manuscript_settings(config)
manifest = create_run_manifest(config_path, evidence_scope="core", run_id="dirty-preflight-10x5", allow_dataset_download=False)
loaded = load_canonical_dataset(config_path, "inx_primary")
receipt = manifest["actual_input_receipts"]["inx_primary"]
folds = generate_shared_folds(
    loaded.frame,
    target_column=settings["target"]["column"],
    id_column=settings["governance_fields"]["identifier_fields"][0],
    run_id=manifest["run_id"],
    config_hash=manifest["config_hash"],
    scientific_input_hash=manifest["scientific_input_hash"],
    dataset_key="inx_primary",
    dataset_sha256=receipt["actual_sha256"],
    outer_splits=settings["evaluation"]["cv"]["n_splits"],
    inner_splits=settings["model"]["nested_tuning"]["inner_splits"],
    seed=settings["seeds"][settings["evaluation"]["cv"]["seed"]],
    inner_seed=settings["seeds"][settings["model"]["nested_tuning"]["inner_seed"]],
)
validate_shared_folds(folds)
print(json.dumps({
    "artifact_written": False,
    "git_worktree_dirty": manifest["git_worktree_dirty"],
    "config_hash": manifest["config_hash"],
    "scientific_input_hash": manifest["scientific_input_hash"],
    "inx_sha256": receipt["actual_sha256"],
    "rows": len(loaded.frame),
    "target_support": receipt["target_distribution"],
    "outer_rows": len(folds.outer_assignments),
    "outer_folds": folds.contract["outer_splits"],
    "inner_rows": len(folds.inner_assignments),
    "inner_folds": folds.contract["inner_splits"],
    "inner_validation_sizes": sorted({row["n_inner_validation"] for row in folds.contract["inner_fold_summaries"]}),
    "fold_contract_hash": folds.contract["fold_contract_hash"],
    "joblib_version": manifest["code_package_versions"]["joblib"],
    "threadpoolctl_version": manifest["code_package_versions"]["threadpoolctl"],
}, sort_keys=True))
'@ | .\myenv\Scripts\python.exe -
```

Exit 0 in 3.5 seconds. Result: pinned INX SHA-256 `b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a`; 1,200 rows; support 194/874/132; 10 outer folds; 10,800 inner assignments; five inner folds; validation size 216; `joblib 1.5.3`; `threadpoolctl 3.6.0`. Dirty diagnostic config hash `7e70bf6646a542ad32e10ab3718654aa8232a46e44e2083ed10e2cfe526da595`, scientific-input hash `8be7c5d79f2b39af3e04f1c8a14a0ae70d2180c48c425595f69f23ac2e76b34a`, and fold hash `f133710d7c45283951f0b5f36e9f87e273bdeb1ed22a317c65fc34cc32a48373` are engineering diagnostics only and must not be cited/reused.

The command `python -m src.experiments.run_model_benchmark_trial` was **not** executed in this checkpoint. `reports/manuscript_final/trials/` did not exist; no real model, trial artifact, gate result, API call or network call occurred. Manuscript status remained empty.

## Real Four-Model Benchmark Trial

Checkpoint and clean-start verification:

```powershell
git commit -m "fix(protocol): freeze approved 10x5 benchmark trial"
git status --short --branch
git rev-parse HEAD
```

Commit exit 0: `6a80074c1402c11331cafc27a3bb5c1d8a2ed4c3`; worktree porcelain was empty before the trial.

Exact scientific trial command:

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m src.experiments.run_model_benchmark_trial --config configs/manuscript_final.yaml --run-id benchmark-10x5-20260713-6a80074
```

Exit 0 after 725.2 shell seconds. The manifest records start `2026-07-13T07:39:56+00:00`, end `2026-07-13T07:51:58+00:00`, elapsed `722.5215989999706`, clean start, commit `6a80074c1402c11331cafc27a3bb5c1d8a2ed4c3`, two completed stages, `canonical_release_eligible=false`, `latest_pointer_updated=false`, and `decision_required=false`.

Exact post-run manifest verification:

```powershell
@'
import json
from pathlib import Path
from src.experiments.run_model_benchmark_trial import verify_trial_manifest
path=Path('reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core/run_manifest.json')
m=verify_trial_manifest(path)
print(json.dumps({
 'verified': True,
 'status':m['status'],
 'git_commit':m['git_commit'],
 'git_worktree_dirty':m['git_worktree_dirty'],
 'config_hash':m['config_hash'],
 'scientific_input_hash':m['scientific_input_hash'],
 'source_tree_hash':m['source_tree_hash'],
 'dataset_sha256':m['actual_input_receipts']['inx_primary']['actual_sha256'],
 'model_grid_sha256':m['side_input_hashes']['model_search_space']['sha256'],
 'output_files':len(m['output_files']),
 'executed_stages':m['executed_stages'],
 'elapsed_seconds':m['elapsed_seconds'],
 'decision_required':m['decision_required'],
 'joblib':m['code_package_versions']['joblib'],
 'threadpoolctl':m['code_package_versions']['threadpoolctl'],
}, sort_keys=True))
'@ | .\myenv\Scripts\python.exe -
```

Exit 0 in 2.7 seconds. Verification result: 53 manifest outputs; commit/config/source/input/fold/model-grid/bootstrap identities consistent; all registered file hashes/sizes and 40 indexed model hashes valid. The physical package has 54 files including its manifest and totals 91,820,515 bytes.

Completed manifest SHA-256: `1b4c3381489f8b0bf7ae60d57280b3ddd5aa5344cb250b1df63fdaaa6cc7379c`. An independent replay of the fold mapping and all three 5,000-draw paired macro-F1 comparisons matched persisted values within `1e-16`; path/hash/size mismatches were zero.

Authoritative trial paths:

- `reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core/run_manifest.json`
- `reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core/model_benchmarks/model_summary.csv`
- `reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core/model_benchmarks/paired_model_differences.csv`
- `reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core/model_benchmarks/baseline_xgboost_gate.json`
- `reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core/model_benchmarks/selected_hyperparameters.csv`

Primary macro-F1: XGBoost `0.621021` (`0.597319–0.644690`), LightGBM `0.605488` (`0.583315–0.629174`), Random Forest `0.592340` (`0.579571–0.604757`), Logistic Regression `0.506221` (`0.480283–0.531841`). Baseline-minus-XGBoost paired differences: LightGBM `-0.015533` (`-0.038121–0.006382`), Random Forest `-0.028681` (`-0.049949–-0.008049`), Logistic Regression `-0.114800` (`-0.147597–-0.083224`). Gate `false`.

Secondary QWK: XGBoost `0.567602`, LightGBM `0.588329`, Random Forest `0.631678`, Logistic Regression `0.371011`. This does not change the predeclared macro-F1 gate.

The command produced repeated sklearn probability-sum warnings for XGBoost and feature-name warnings for LightGBM. A read-only diagnostic on persisted OOF rows found maximum XGBoost row-sum deviation `8.381903171539307e-08`; float64 renormalization produced zero argmax changes, reduced maximum sum deviation to `2.22e-16`, suppressed the probability warning, and changed aggregate log loss by `1.809657979023882e-10`. The macro-F1/QWK selection and gate are unaffected; canonical probability evidence will be regenerated after code/test cleanup.

Paid/API/network calls: zero. The process-local socket/DNS guard remained active. Manuscript edits: none. `latest` was not touched.

Package-provenance caveat: `PyYAML`, `openpyxl` and `xlrd` are recorded as `not_installed`. This trial parsed the JSON-compatible canonical config without PyYAML and consumed verified CSV only; workbook inspection was not a trial stage. The gate is unaffected, but dependency lock/clean-install release readiness remains open.

## Unit 2C-0 Probability and Feature Warning Hygiene

Focused tests:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_canonical_model_factory.py tests/test_manuscript_model_benchmark.py tests/test_nested_search_outer_test_isolation.py tests/test_model_benchmark_trial_entrypoint.py
```

Exit 0: 63 passed in 37.39 seconds.

Full gates:

```powershell
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
git status --short -- manuscript
rg --pcre2 -n '(?<![A-Za-z])sk-(?:proj-)?[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}' --glob '!myenv/**' --glob '!reports/manuscript_final/**' --glob '!*.ipynb' .
```

Results: pytest 350 passed, 2 skipped and 4 subtests in 64.88 seconds; unittest 162 passed with 2 skips in 3.339 seconds; compileall/diff/manuscript passed; high-entropy secret scan had no matches.

Exact read-only fold-1 warning-hygiene replay (rerun with an explicit process-local socket/DNS guard so the invocation is reproducible from this log):

```powershell
@'
import json
import socket
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.canonical_loader import load_canonical_dataset
from src.experiments.manuscript_model_benchmark import exact_primary_feature_frame
from src.governance.manuscript_contract import primary_excluded_features
from src.models.canonical_models import aligned_predict_proba, build_model_pipeline
from src.utils.config_loader import load_config

TRIAL = Path("reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core")
CONFIG = Path("configs/manuscript_final.yaml")
FOLD = 1
MODELS = ("lightgbm", "xgboost")

config = load_config(CONFIG)
settings = config["manuscript_final"]
frame = load_canonical_dataset(CONFIG, "inx_primary").frame
excluded = primary_excluded_features(config)
features = exact_primary_feature_frame(frame, excluded_features=excluded)
target = frame[settings["target"]["column"]].astype(int)
labels = [int(value) for value in settings["target"]["labels"]]
seed = int(settings["seeds"]["model"])
folds = pd.read_csv(TRIAL / "shared_folds" / "fold_assignments.csv")
selected = pd.read_csv(TRIAL / "model_benchmarks" / "selected_hyperparameters.csv")
oof = pd.read_csv(TRIAL / "model_benchmarks" / "oof_predictions.csv")
train_ids = folds.loc[folds["outer_fold"].ne(FOLD), "sample_index"].astype(int).to_numpy()
test_ids = folds.loc[folds["outer_fold"].eq(FOLD), "sample_index"].astype(int).to_numpy()

originals = {name: getattr(socket, name) for name in ("socket", "create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex")}
def deny_network(*args, **kwargs):
    raise RuntimeError("Network access denied during warning-hygiene replay")
for name in originals:
    setattr(socket, name, deny_network)

results = []
try:
    for model_name in MODELS:
        row = selected[(selected["outer_fold"].eq(FOLD)) & (selected["model"].eq(model_name))]
        if len(row) != 1:
            raise RuntimeError(f"Expected one selected row for {model_name}, found {len(row)}")
        record = row.iloc[0]
        pipeline = build_model_pipeline(
            model_name,
            features.loc[train_ids],
            fixed_parameters=json.loads(record["fixed_parameters_json"]),
            candidate_parameters=json.loads(record["selected_candidate_parameters_json"]),
            random_state=seed,
            forbidden_features=excluded,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(features.loc[train_ids], target.loc[train_ids])
            prediction = np.asarray(pipeline.predict(features.loc[test_ids]), dtype=int)
            probability = aligned_predict_proba(pipeline, features.loc[test_ids], labels=labels)
        persisted = oof[(oof["outer_fold"].eq(FOLD)) & (oof["model"].eq(model_name))].set_index("sample_index").loc[test_ids]
        persisted_probability = persisted[[f"prob_class_{label}" for label in labels]].to_numpy(dtype=np.float64)
        transformed = pipeline.named_steps["preprocessor"].transform(features.loc[test_ids])
        results.append({
            "model": model_name,
            "n_test": int(len(test_ids)),
            "label_mismatches": int(np.count_nonzero(prediction != persisted["y_pred"].to_numpy(dtype=int))),
            "max_probability_delta": float(np.max(np.abs(probability - persisted_probability))),
            "max_row_sum_deviation": float(np.max(np.abs(probability.sum(axis=1) - 1.0))),
            "transformed_container": type(transformed).__name__,
            "transformed_columns": int(transformed.shape[1]),
            "warnings": [str(item.message) for item in caught],
        })
finally:
    for name, value in originals.items():
        setattr(socket, name, value)

print(json.dumps({"artifact_written": False, "network_guard": "socket_and_dns_denied", "fold": FOLD, "results": results}, sort_keys=True))
'@ | .\myenv\Scripts\python.exe -
```

Exit 0 in 2.7 seconds. The replay loaded the verified trial's fold assignment and selected-parameter records, rebuilt only the selected LightGBM and XGBoost pipelines with the new preprocessing/probability contract, and compared their in-memory predictions against persisted OOF rows. It wrote no output:

- LightGBM: 0/120 label mismatches; maximum probability delta `3.3306690738754696e-16`; maximum row-sum deviation `2.220446049250313e-16`; 46-column pandas transformed output; zero warnings.
- XGBoost: 0/120 label mismatches; maximum probability delta `7.015278280508852e-08`; maximum row-sum deviation `1.1102230246251565e-16`; 46-column pandas transformed output; zero warnings.

The current code/config identity intentionally differs from the immutable completed trial. No trial file, scientific artifact, API/network resource or manuscript file was modified by this diagnostic.

## Unit 2C-A Exact-Model OOF SHAP

Focused and complete validation commands:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_benchmark_artifact_contract.py tests/test_canonical_shap_axis.py tests/test_shap_uses_exact_oof_fold_model.py tests/test_shap_outputs_match_primary_policy.py tests/test_stage_runner_scientific_input_binding.py tests/test_shared_fold_assignments_across_models_and_policies.py tests/test_forbidden_features_absent_from_primary_artifacts.py
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
git diff --exit-code -- manuscript/mdpi_information/main.md
```

Results: focused 59 passed plus 4 subtests in 15.58 seconds; full pytest 389 passed, 2 skipped, plus 4 subtests in 78.40 seconds; unittest 164 passed with 2 skips in 3.531 seconds; compileall/diff/manuscript passed.

Additional exact scans:

```powershell
$m = rg --pcre2 -n '(?<![A-Za-z])sk-(?:proj-)?[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}' --glob '!myenv/**' --glob '!reports/manuscript_final/**' --glob '!*.ipynb' .; if ($LASTEXITCODE -eq 0) { $m; exit 2 } elseif ($LASTEXITCODE -eq 1) { 'NO_SECRET_PATTERN_MATCHES'; exit 0 } else { exit $LASTEXITCODE }
$m = git diff -- . ':!reports/research_log/**' | Select-String -Pattern 'C:\\Users\\|/home/[^/]+/|file://' -CaseSensitive; if ($m) { $m; exit 2 } else { 'NO_ABSOLUTE_USER_PATHS_IN_SCIENTIFIC_DIFF'; exit 0 }
$tracked = @(git diff --name-only) + @(git ls-files --others --exclude-standard | Where-Object { $_ -notlike 'reports/manuscript_final/trials/*' }); $bad=@(); foreach($f in ($tracked | Sort-Object -Unique)){ if(Test-Path -LiteralPath $f -PathType Leaf){$n=(Get-Item -LiteralPath $f).Length; if($n -gt 100MB){$bad += "$f`t$n"}}}; if($bad){$bad; exit 2}else{'NO_100MB_CANDIDATE_FILES'; exit 0}
```

All three scans exited 0 with no matches/failures.

Exact historical reader/OOF replay (read-only, network denied):

```powershell
@'
import json
import socket
from pathlib import Path

from src.data.canonical_loader import load_canonical_dataset
from src.experiments.benchmark_artifact_contract import read_xgboost_oof_artifacts, validate_xgboost_oof_replay
from src.experiments.manuscript_model_benchmark import exact_primary_feature_frame
from src.governance.manuscript_contract import primary_excluded_features, sha256_file
from src.utils.config_loader import load_config

trial = Path("reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core")
config_path = Path("configs/manuscript_final.yaml")
manifest_before = sha256_file(trial / "run_manifest.json")
config = load_config(config_path)
settings = config["manuscript_final"]
canonical = load_canonical_dataset(config_path, "inx_primary")
features = exact_primary_feature_frame(canonical.frame, excluded_features=primary_excluded_features(config))
target = canonical.frame[settings["target"]["column"]].astype(int)
labels = [int(value) for value in settings["target"]["labels"]]
fold_contract = json.loads((trial / "shared_folds" / "fold_contract.json").read_text(encoding="utf-8"))
originals = {name: getattr(socket, name) for name in ("socket", "create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex")}
def deny_network(*args, **kwargs):
    raise RuntimeError("Network access denied during exact-model artifact replay")
for name in originals:
    setattr(socket, name, deny_network)
try:
    bundle = read_xgboost_oof_artifacts(
        trial / "shared_folds",
        trial / "model_benchmarks",
        expected_run_id=fold_contract["run_id"],
        expected_config_hash=fold_contract["config_hash"],
        expected_scientific_input_hash=fold_contract["scientific_input_hash"],
        expected_feature_columns=features.columns.tolist(),
        expected_labels=labels,
    )
    validate_xgboost_oof_replay(bundle, features, target, labels=labels)
finally:
    for name, value in originals.items():
        setattr(socket, name, value)
manifest_after = sha256_file(trial / "run_manifest.json")
print(json.dumps({
    "artifact_written": False,
    "network_guard": "socket_and_dns_denied",
    "historical_trial_only": True,
    "run_id": bundle.identity.run_id,
    "fold_contract_hash": bundle.identity.fold_contract_hash,
    "model_set_sha256": bundle.model_set_sha256,
    "n_xgboost_models": len(bundle.fold_models),
    "n_oof_rows": len(bundle.oof_predictions),
    "gate_triggered": bundle.baseline_gate["gate_triggered"],
    "manifest_sha256_before": manifest_before,
    "manifest_sha256_after": manifest_after,
    "manifest_unchanged": manifest_before == manifest_after,
}, sort_keys=True))
'@ | .\myenv\Scripts\python.exe -
```

Exit 0 in 3.0 seconds: ten XGBoost models; 1,200 exact-once OOF rows; gate false; model-set hash `492aa445efc0df9348b9c85714ec09a539d6195fd46c2151c102b6ba02a1c607`; completed manifest remained `1b4c3381489f8b0bf7ae60d57280b3ddd5aa5344cb250b1df63fdaaa6cc7379c`. This validates the reader only and does not make the historical run canonical.

Exact stale-lineage check:

```powershell
@'
import joblib
from pathlib import Path
from src.data.canonical_loader import load_canonical_dataset
from src.experiments.manuscript_model_benchmark import exact_primary_feature_frame
from src.explainability.canonical_shap_axis import build_canonical_shap_axis
from src.governance.manuscript_contract import primary_excluded_features
from src.utils.config_loader import load_config
config = load_config("configs/manuscript_final.yaml")
features = exact_primary_feature_frame(
    load_canonical_dataset("configs/manuscript_final.yaml", "inx_primary").frame,
    excluded_features=primary_excluded_features(config),
)
model = joblib.load(Path("reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core/model_benchmarks/models/xgboost/outer_fold_01.joblib"))
build_canonical_shap_axis(
    model.named_steps["preprocessor"],
    raw_feature_order=features.columns.tolist(),
    forbidden_features=primary_excluded_features(config),
)
'@ | .\myenv\Scripts\python.exe -
```

Expected exit 1 in 1.9 seconds: `CanonicalShapAxisError: one_hot feature_names_in_ must not be empty.` The historical pipeline was fitted before named pandas propagation, so its nested encoder lacks auditable input lineage. This is an intentional fail-closed incompatibility, not a reason to add a heuristic fallback.

Exact current-code real fold-1 axis and grouped-SHAP diagnostic (in memory, network denied, no output):

```powershell
@'
import json
import socket
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.data.canonical_loader import load_canonical_dataset
from src.experiments.manuscript_model_benchmark import exact_primary_feature_frame
from src.explainability.canonical_shap_axis import build_canonical_shap_axis, group_canonical_shap_values, normalize_multiclass_shap_values
from src.governance.manuscript_contract import primary_excluded_features, sha256_file
from src.models.canonical_models import aligned_predict_proba, build_model_pipeline
from src.utils.config_loader import load_config

trial = Path("reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core")
config_path = Path("configs/manuscript_final.yaml")
manifest_before = sha256_file(trial / "run_manifest.json")
config = load_config(config_path)
settings = config["manuscript_final"]
canonical = load_canonical_dataset(config_path, "inx_primary")
excluded = primary_excluded_features(config)
features = exact_primary_feature_frame(canonical.frame, excluded_features=excluded)
target = canonical.frame[settings["target"]["column"]].astype(int)
labels = [int(value) for value in settings["target"]["labels"]]
folds = pd.read_csv(trial / "shared_folds" / "fold_assignments.csv")
selected = pd.read_csv(trial / "model_benchmarks" / "selected_hyperparameters.csv")
persisted = pd.read_csv(trial / "model_benchmarks" / "oof_predictions.csv")
fold = 1
train_ids = folds.loc[folds["outer_fold"].ne(fold), "sample_index"].astype(int).to_numpy()
test_ids = folds.loc[folds["outer_fold"].eq(fold), "sample_index"].astype(int).to_numpy()
record = selected[(selected["outer_fold"].eq(fold)) & (selected["model"].eq("xgboost"))].iloc[0]
pipeline = build_model_pipeline(
    "xgboost",
    features.loc[train_ids],
    fixed_parameters=json.loads(record["fixed_parameters_json"]),
    candidate_parameters=json.loads(record["selected_candidate_parameters_json"]),
    random_state=int(settings["seeds"]["model"]),
    forbidden_features=excluded,
)
originals = {name: getattr(socket, name) for name in ("socket", "create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex")}
def deny_network(*args, **kwargs):
    raise RuntimeError("Network access denied during current-contract SHAP diagnostic")
for name in originals:
    setattr(socket, name, deny_network)
try:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(features.loc[train_ids], target.loc[train_ids])
        probability = aligned_predict_proba(pipeline, features.loc[test_ids], labels=labels)
        prediction = np.asarray(pipeline.predict(features.loc[test_ids]), dtype=int)
        preprocessor = pipeline.named_steps["preprocessor"]
        axis = build_canonical_shap_axis(preprocessor, raw_feature_order=features.columns.tolist(), forbidden_features=excluded)
        transformed = preprocessor.transform(features.loc[test_ids])
        axis.validate_transformed_matrix(transformed, n_samples=len(test_ids))
        raw_shap = shap.TreeExplainer(pipeline.named_steps["model"].model_).shap_values(transformed)
        normalized = normalize_multiclass_shap_values(raw_shap, n_samples=len(test_ids), n_classes=len(labels), n_transformed_features=axis.n_transformed_features)
        grouped = group_canonical_shap_values(normalized, axis)
finally:
    for name, value in originals.items():
        setattr(socket, name, value)
old = persisted[(persisted["outer_fold"].eq(fold)) & (persisted["model"].eq("xgboost"))].set_index("sample_index").loc[test_ids]
manifest_after = sha256_file(trial / "run_manifest.json")
print(json.dumps({
    "artifact_written": False,
    "network_guard": "socket_and_dns_denied",
    "diagnostic_only": True,
    "outer_fold": fold,
    "n_test": len(test_ids),
    "raw_features": axis.n_raw_features,
    "transformed_features": axis.n_transformed_features,
    "grouped_shape": list(grouped.shape),
    "max_group_sum_error": float(np.max(np.abs(grouped.sum(axis=2) - normalized.sum(axis=2)))),
    "one_hot_feature_names_in": preprocessor.named_transformers_["categorical"].named_steps["one_hot"].feature_names_in_.tolist(),
    "label_mismatches_vs_historical_trial": int(np.count_nonzero(prediction != old["y_pred"].to_numpy(dtype=int))),
    "max_probability_delta_vs_historical_trial": float(np.max(np.abs(probability - old[[f"prob_class_{label}" for label in labels]].to_numpy(dtype=float)))),
    "max_probability_sum_deviation": float(np.max(np.abs(probability.sum(axis=1) - 1.0))),
    "warnings": [str(item.message) for item in caught],
    "manifest_unchanged": manifest_before == manifest_after,
}, sort_keys=True))
'@ | .\myenv\Scripts\python.exe -
```

Exit 0 in 3.3 seconds: 120 test rows; 46 transformed features; 20 raw families; grouped shape `(120,3,20)`; maximum grouped/transformed sum error `0`; zero warnings; zero label mismatches; maximum normalized probability delta from the immutable pre-cleanup trial `7.015278280508852e-08`; manifest unchanged. These are diagnostic compatibility checks, not manuscript results.

No canonical SHAP package or manuscript file was generated or changed. Paid API/network calls were zero.

### Unit 2D initial builder and cross-config policy binding — 2026-07-13

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_canonical_feature_policy_consistency.py tests/test_stage_runner_scientific_input_binding.py
```

Exit 0: 22 passed in 1.64 seconds (3.6 shell seconds). This check covers fail-closed builder delivery of current-run shared-fold/benchmark identities and exact canonical-to-legacy-projection policy exclusions. The main policy OOF/bootstrap implementation is still in progress, so this is not a scientific completion result. No artifact, API/network call, or manuscript edit occurred.

### Unit 2D complete policy implementation and review closure — 2026-07-13

All Python commands below ran with `OPENAI_API_KEY`, `OPENAI_AGENTS_API_KEY` and `AZURE_OPENAI_API_KEY` removed from the subprocess environment. No paid API or network call was made.

The first broad focused command contained a nonexistent filename:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_manuscript_policy_ablation.py tests/test_stage_runner_scientific_input_binding.py tests/test_canonical_feature_policy_consistency.py tests/test_shared_fold_assignments_across_models_and_policies.py tests/test_oof_bootstrap_uncertainty.py tests/test_benchmark_artifact_contract.py tests/test_canonical_model_factory.py
```

Exit 1 in 1.6 seconds before collection: `ERROR: file or directory not found: tests/test_oof_bootstrap_uncertainty.py`. No test, model fit or scientific stage ran. The corrected repository test modules were then used:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_manuscript_policy_ablation.py tests/test_stage_runner_scientific_input_binding.py tests/test_canonical_feature_policy_consistency.py tests/test_shared_fold_artifact_contract.py tests/test_shared_fold_assignments_across_models_and_policies.py tests/test_bootstrap_is_stratified_paired_and_deterministic.py tests/test_oof_bootstrap_intervals_are_domain_valid.py tests/test_paired_model_difference_bootstrap.py tests/test_benchmark_artifact_contract.py
```

Exit 0: 75 passed in 22.15 seconds.

Independent review found seven gaps: raw interval persistence, a late atomic-publication failure check, direct cross-config projection validation, exact primary OOF replay, independent feature-projection side-input hashing, valid-bootstrap-denominator derivation, and exact six-policy scope. After all seven were corrected:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_manuscript_policy_ablation.py tests/test_canonical_feature_policy_consistency.py tests/test_side_input_hash_binding.py tests/test_scoped_run_manifest_inputs.py
```

Exit 0: 35 passed in 7.82 seconds (9.8 shell seconds).

Final expanded focused command:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_manuscript_policy_ablation.py tests/test_stage_runner_scientific_input_binding.py tests/test_canonical_feature_policy_consistency.py tests/test_shared_fold_artifact_contract.py tests/test_shared_fold_assignments_across_models_and_policies.py tests/test_bootstrap_is_stratified_paired_and_deterministic.py tests/test_oof_bootstrap_intervals_are_domain_valid.py tests/test_paired_model_difference_bootstrap.py tests/test_benchmark_artifact_contract.py tests/test_side_input_hash_binding.py tests/test_scoped_run_manifest_inputs.py
```

Exit 0: 92 passed in 26.85 seconds (about 29 shell seconds).

Complete regression and compilation gates:

```powershell
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
git diff --exit-code -- manuscript/mdpi_information/main.md
```

Results: pytest 403 passed, 2 skipped, plus 4 subtests in 83.35 seconds; unittest 174 passed with 2 skips in 6.889 seconds; compileall, diff hygiene and manuscript no-change checks exited 0.

Exact hygiene scans:

```powershell
$m = rg --pcre2 -n '(?<![A-Za-z])sk-(?:proj-)?[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}' --glob '!myenv/**' --glob '!reports/manuscript_final/**' --glob '!*.ipynb' .; if ($LASTEXITCODE -eq 0) { $m; exit 2 } elseif ($LASTEXITCODE -eq 1) { 'NO_SECRET_PATTERN_MATCHES'; exit 0 } else { exit $LASTEXITCODE }
$m = git diff -- . ':!reports/research_log/**' | Select-String -Pattern 'C:\\Users\\|/home/[^/]+/|file://' -CaseSensitive; if ($m) { $m; exit 2 } else { 'NO_ABSOLUTE_USER_PATHS_IN_SCIENTIFIC_DIFF'; exit 0 }
$tracked = @(git diff --name-only) + @(git ls-files --others --exclude-standard | Where-Object { $_ -notlike 'reports/manuscript_final/trials/*' }); $bad=@(); foreach($f in ($tracked | Sort-Object -Unique)){ if(Test-Path -LiteralPath $f -PathType Leaf){$n=(Get-Item -LiteralPath $f).Length; if($n -gt 100MB){$bad += "$f`t$n"}}}; if($bad){$bad; exit 2}else{'NO_100MB_CANDIDATE_FILES'; exit 0}
$m = rg -n -i 'leakage[- ]safe' README.md configs/feature_sets.yaml configs/manuscript_final.yaml src/experiments/manuscript_policy_ablation.py; if ($LASTEXITCODE -eq 0) { $m; exit 2 } elseif ($LASTEXITCODE -eq 1) { 'NO_ACTIVE_LEAKAGE_SAFE_TERMINOLOGY'; exit 0 } else { exit $LASTEXITCODE }
```

All exited 0 with the corresponding no-match/no-large-file markers. The 91.8 MB untracked historical trial was deliberately excluded from the candidate commit scan under accepted D5.

The original exact real-INX bounded diagnostic was:

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
@'
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from src.data.canonical_loader import load_canonical_dataset
from src.experiments.benchmark_artifact_contract import read_xgboost_oof_artifacts
from src.experiments.manuscript_policy_ablation import (
    _json_mapping,
    _policy_definitions,
    _selected_policies,
    _validate_nonprimary_fitted_pipeline,
    exact_policy_frame,
)
from src.experiments.run_model_benchmark_trial import _deny_network_connections
from src.models.canonical_models import aligned_predict_proba, build_model_pipeline
from src.utils.config_loader import load_config

root = Path.cwd()
trial = root / 'reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core'
manifest = trial / 'run_manifest.json'
before = hashlib.sha256(manifest.read_bytes()).hexdigest()
config_path = root / 'configs/manuscript_final.yaml'
raw = load_config(config_path)
settings = raw['manuscript_final']
loaded = load_canonical_dataset(config_path, 'inx_primary')
data = loaded.frame
target = data['PerformanceRating'].astype(int)
definitions = _policy_definitions(settings)
policies = _selected_policies(definitions)
primary = settings['feature_policies']['primary_policy']
primary_features, _ = exact_policy_frame(
    data, primary, definitions[primary], target_column='PerformanceRating', id_column='EmpNumber'
)
bundle = read_xgboost_oof_artifacts(
    trial / 'shared_folds',
    trial / 'model_benchmarks',
    expected_run_id='benchmark-10x5-20260713-6a80074',
    expected_config_hash='7e70bf6646a542ad32e10ab3718654aa8232a46e44e2083ed10e2cfe526da595',
    expected_scientific_input_hash='8be7c5d79f2b39af3e04f1c8a14a0ae70d2180c48c425595f69f23ac2e76b34a',
    expected_feature_columns=primary_features.columns,
    expected_labels=(2, 3, 4),
)
selected = bundle.selected_hyperparameters.set_index('outer_fold').loc[1]
fixed = _json_mapping(selected['fixed_parameters_json'], context='diagnostic fixed')
candidate = _json_mapping(selected['selected_candidate_parameters_json'], context='diagnostic candidate')
outer = bundle.folds.outer_assignments
train_ids = outer.loc[outer['outer_fold'].astype(int).ne(1), 'sample_index'].astype(int).tolist()
test_ids = outer.loc[outer['outer_fold'].astype(int).eq(1), 'sample_index'].astype(int).tolist()
records = []
with _deny_network_connections(), warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    for policy in policies:
        if policy == primary:
            continue
        features, excluded = exact_policy_frame(
            data, policy, definitions[policy], target_column='PerformanceRating', id_column='EmpNumber'
        )
        pipeline = build_model_pipeline(
            'xgboost',
            features.loc[train_ids],
            fixed_parameters=fixed,
            candidate_parameters=candidate,
            random_state=int(settings['seeds']['model']),
            forbidden_features=tuple(excluded),
        )
        with threadpool_limits(limits=1):
            pipeline.fit(features.loc[train_ids], target.loc[train_ids])
        _validate_nonprimary_fitted_pipeline(
            pipeline,
            feature_columns=features.columns,
            fixed_parameters=fixed,
            candidate_parameters=candidate,
            policy=policy,
            outer_fold=1,
        )
        prediction = np.asarray(pipeline.predict(features.loc[test_ids]), dtype=int)
        probability = aligned_predict_proba(pipeline, features.loc[test_ids], labels=(2, 3, 4))
        assert prediction.shape == (120,)
        assert probability.shape == (120, 3)
        assert np.isfinite(probability).all()
        assert np.max(np.abs(probability.sum(axis=1) - 1.0)) <= np.finfo(np.float64).eps * 6
        assert set(excluded).isdisjoint(features.columns)
        records.append({
            'policy': policy,
            'n_train': len(train_ids),
            'n_test': len(test_ids),
            'n_raw_features': features.shape[1],
            'n_transformed_features': int(pipeline.named_steps['preprocessor'].transform(features.loc[test_ids]).shape[1]),
            'selected_candidate_index': int(selected['selected_candidate_index']),
            'probability_simplex_max_error': float(np.max(np.abs(probability.sum(axis=1) - 1.0))),
        })
after = hashlib.sha256(manifest.read_bytes()).hexdigest()
print(json.dumps({
    'status': 'passed_real_data_fold1_noncanonical_diagnostic_only',
    'records': records,
    'warnings': [str(item.message) for item in caught],
    'trial_manifest_unchanged': before == after,
    'files_written': 0,
    'network_guard': 'connect_connect_ex_sendto_create_connection_getaddrinfo_blocked',
}, sort_keys=True))
'@ | .\myenv\Scripts\python.exe -
```

Exit 0 in 4.4 seconds. Five non-primary current-code fits each used 1,080 train and 120 test rows with selected candidate index 3. Raw/transformed feature counts were 26/61, 25/60, 24/58, 21/52 and 19/27 in policy order excluding the primary exact-reuse row. Maximum probability simplex error was `2.22e-16`; warnings were empty; files written were zero; the network guard was active; the historical manifest was unchanged.

A checkpoint recheck on 2026-07-13 repeated the same five current-code fold-1 fits using a direct socket/DNS guard and additionally verified the trial file count stayed 54 and manifest SHA-256 stayed `1b4c3381489f8b0bf7ae60d57280b3ddd5aa5344cb250b1df63fdaaa6cc7379c`. The first recheck attempt exited 1 before model construction because it incorrectly read nonexistent `target.id_column`; the corrected command used `governance_fields.identifier_fields[0]` and exited 0 in 3.3 seconds. Neither attempt wrote a file or contacted a network/API.

These diagnostics are implementation evidence only. The primary policy was deliberately not replayed because the historical trial predates the exact current `1e-12` lineage/probability contract. No policy estimate is admitted until a same-current-commit benchmark and policy stage run together.

Post-change identities:

```powershell
.\myenv\Scripts\python.exe -c "from src.governance.manuscript_contract import canonical_config_hash, load_manuscript_config, sha256_file; load_manuscript_config('configs/manuscript_final.yaml'); print(canonical_config_hash('configs/manuscript_final.yaml')); print(sha256_file('configs/feature_sets.yaml'))"
git status --short --branch
git diff --stat
git diff --name-only
```

Exit 0. Config hash: `57da6ae89ef43c3cb25783d7d5eb36a394287c0d647ee1e51bfca2dc431c8f74`. Feature-policy projection SHA-256: `4b270ad6f0b07e2d51ce2d3d3b38f193ad5e15925420e0a9293a361c805d8fc3`.

### Unit 2E calibration read-only audit and timing — 2026-07-13

No calibration code or artifact was modified. Static/read-only inspection established that the historical stage selects methods with outer-test outcomes, generates a fold mapping that differs on 1,091 of 1,200 samples from the current shared-fold contract, uses legacy fixed XGBoost parameters, uses fold-t intervals, lacks builder-supplied run/fold/model/scientific identities, emits absolute paths, and publishes non-atomically. Historical calibration results are therefore inadmissible for v2.

The exact common-bootstrap timing command was:

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
@'
import time
import pandas as pd
from src.models.oof_bootstrap import BootstrapProtocol, ComparisonSpec, compute_paired_oof_bootstrap
path = r"reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core/model_benchmarks/oof_predictions.csv"
frame = pd.read_csv(path)
xgb = frame[frame['model'].astype(str).eq('xgboost')].copy()
raw = xgb.copy(); raw['system_id'] = 'raw'
sig = xgb.copy(); sig['system_id'] = 'sigmoid'
pred = pd.concat([raw, sig], ignore_index=True)
metrics = ('accuracy','balanced_accuracy','macro_f1','quadratic_weighted_kappa','ordinal_mae','severe_error_rate','nll_log_loss','multiclass_brier','ece_confidence')
start = time.perf_counter()
result = compute_paired_oof_bootstrap(
    pred,
    labels=(2,3,4),
    task_type='ordinal_multiclass_performance',
    metrics=metrics,
    comparisons=(ComparisonSpec('sigmoid_minus_raw','sigmoid','raw'),),
    protocol=BootstrapProtocol(n_resamples=5000, confidence_level=0.95, seed=42),
    n_bins=10,
)
elapsed = time.perf_counter() - start
print(f'bootstrap_runtime_seconds={elapsed:.3f}')
print(f'interval_rows={len(result.metric_intervals)} paired_rows={len(result.paired_differences)} resamples={result.metadata["n_resamples"]}')
'@ | .\myenv\Scripts\python.exe -
```

Exit 0 in 95.1 shell seconds: `bootstrap_runtime_seconds=93.032`, `interval_rows=18`, `paired_rows=9`, `resamples=5000`. The immutable historical probabilities produced repeated sklearn probability-sum warnings (9,199 output lines) because they predate current normalization. This is timing only, not calibration or scientific evidence.

The warning-free normalized projection check was:

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
@'
import time, warnings
import numpy as np
import pandas as pd
from src.models.oof_bootstrap import BootstrapProtocol, ComparisonSpec, compute_paired_oof_bootstrap
path = r"reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core/model_benchmarks/oof_predictions.csv"
frame = pd.read_csv(path)
xgb = frame[frame['model'].astype(str).eq('xgboost')].copy()
cols = ['prob_class_2','prob_class_3','prob_class_4']
p = np.clip(xgb[cols].to_numpy(dtype=np.float64), 0.0, 1.0)
p /= p.sum(axis=1, keepdims=True)
xgb.loc[:, cols] = p
raw = xgb.copy(); raw['system_id'] = 'raw'
sig = xgb.copy(); sig['system_id'] = 'sigmoid'
pred = pd.concat([raw, sig], ignore_index=True)
metrics = ('accuracy','balanced_accuracy','macro_f1','quadratic_weighted_kappa','ordinal_mae','severe_error_rate','nll_log_loss','multiclass_brier','ece_confidence')
start = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter('error')
    result = compute_paired_oof_bootstrap(
        pred, labels=(2,3,4), task_type='ordinal_multiclass_performance', metrics=metrics,
        comparisons=(ComparisonSpec('sigmoid_minus_raw','sigmoid','raw'),),
        protocol=BootstrapProtocol(n_resamples=500, confidence_level=0.95, seed=42), n_bins=10,
    )
elapsed = time.perf_counter() - start
print(f'normalized_500_runtime_seconds={elapsed:.3f}')
print(f'linear_5000_projection_seconds={elapsed * 10:.1f}')
'@ | .\myenv\Scripts\python.exe -
```

Exit 0 in 10.9 shell seconds: `normalized_500_runtime_seconds=9.252`; linear 5,000-draw projection `92.5` seconds; warnings-as-errors passed. Both timing commands duplicate the same raw OOF probabilities under two system IDs solely to measure the common paired-bootstrap workload. They read one local CSV in memory, wrote no file, cleared API variables and made no network/API call.

Using the observed benchmark duration, 1,540 fits / 722.522 seconds = about 0.469 seconds per fit. The recommended five-inner-fold cross-fitted sigmoid design adds 50 fits (about 23.5 fit seconds) plus bootstrap/reporting; a single 20% holdout adds 10 fits (about 4.7 fit seconds) but changes the base model. Total conservative local estimates are about 2–3 minutes versus 1.5–2 minutes. This is the pending material scientific decision; no implementation was started.

### Unit 2D checkpoint revalidation after documentation repair — 2026-07-13

The complete pytest command was rerun after the README, issue register, artifact map and readiness-status corrections:

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m pytest -q
```

Exit 0: 403 passed, 2 skipped, plus 4 subtests in 83.35 seconds (86.2 shell seconds). Unittest was also rerun and reported 174 tests, 2 skipped, in 6.904 seconds. Compileall, `git diff --check`, manuscript no-change, secret, scientific-diff absolute-path, 100 MB candidate-file and active-terminology checks all exited 0.

The earlier parallel checkpoint wrapper initially called nonexistent `manuscript_contract.config_hash`, so that wrapper exited 1 before its results were collected. The corrected identity command uses `canonical_config_hash`; this was a command-name error, not a configuration/test failure. The corrected command loaded the full manuscript contract and reproduced config hash `57da6ae89ef43c3cb25783d7d5eb36a394287c0d647ee1e51bfca2dc431c8f74` and feature-policy SHA-256 `4b270ad6f0b07e2d51ce2d3d3b38f193ad5e15925420e0a9293a361c805d8fc3`.

The README clean-checkout link and issue-register audit used:

```powershell
@'
import csv
import re
import subprocess
from pathlib import Path

root = Path.cwd()
text = (root / 'README.md').read_text(encoding='utf-8')
targets = []
for raw in re.findall(r'\[[^\]]+\]\(([^)]+)\)', text):
    target = raw.strip().strip('<>')
    if target.startswith(('http://', 'https://', 'mailto:', '#')):
        continue
    target = target.split('#', 1)[0]
    if target:
        targets.append(target)
problems = []
for target in targets:
    path = root / target
    if not path.exists():
        problems.append(f'missing:{target}')
        continue
    tracked = subprocess.run(
        ['git', 'ls-files', '--', target], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    if not tracked:
        problems.append(f'untracked:{target}')
if problems:
    raise SystemExit('README_LINK_FAILURES\n' + '\n'.join(problems))
with (root / 'reports/research_log/finalization_v2/02_issue_register.csv').open(
    encoding='utf-8', newline=''
) as handle:
    rows = list(csv.DictReader(handle))
if not rows or any(None in row for row in rows):
    raise SystemExit('ISSUE_REGISTER_CSV_INVALID')
print(f'README_TRACKED_LOCAL_LINKS_OK count={len(targets)}')
print(f'ISSUE_REGISTER_CSV_OK rows={len(rows)}')
'@ | .\myenv\Scripts\python.exe -
```

Exit 0: `README_TRACKED_LOCAL_LINKS_OK count=17`; `ISSUE_REGISTER_CSV_OK rows=26`. This stronger check verifies that each local README target exists **and** is represented in the Git index; the untracked trial is deliberately plain code text rather than a link.

## Unit 2E Option-A Calibration Implementation - 2026-07-13

Builder/config focused command:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_stage_runner_scientific_input_binding.py tests/test_canonical_feature_policy_consistency.py
```

The first run exited 1 with 1 failed, 28 passed and 6 subtests because the new test used obsolete field `outer_test_evaluation_only`. After binding the test to the actual frozen fields, the rerun exited 0 with 29 passed and 7 subtests in 1.86 seconds. This was a test-field error before scientific execution.

Calibration-only command:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_manuscript_calibration.py tests/test_calibration_is_predeclared_sigmoid.py tests/test_calibration_outer_test_not_used_for_selection.py
```

The initial implementation passed 12 tests but exposed nine scikit-learn 1.8 `FutureWarning` records from explicit `penalty='l2'`. After moving the same L2 contract to warning-free `l1_ratio=0.0`, freezing one thread and bounding scikit-learn to `>=1.8,<1.9`, the command exited 0 with 13 passed in 1.76 seconds and no warning summary.

Repeated source checks:

```powershell
.\myenv\Scripts\python.exe -m compileall -q src\experiments\manuscript_calibration.py src\experiments\build_manuscript_evidence.py src\governance\manuscript_contract.py
git diff --check
```

Exit 0. Git printed only LF-to-CRLF notices; no whitespace error occurred.

Final expanded focused command before the full regression gate:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_manuscript_calibration.py tests/test_calibration_is_predeclared_sigmoid.py tests/test_calibration_outer_test_not_used_for_selection.py tests/test_stage_runner_scientific_input_binding.py tests/test_canonical_feature_policy_consistency.py tests/test_benchmark_artifact_contract.py tests/test_shared_fold_artifact_contract.py tests/test_bootstrap_is_stratified_paired_and_deterministic.py tests/test_oof_bootstrap_intervals_are_domain_valid.py tests/test_paired_model_difference_bootstrap.py
```

Exit 0: 85 passed plus 7 subtests in 19.77 seconds. An earlier pre-review run passed 79 plus 7 subtests; the increase is added regression coverage.

The warning-free sigmoid smoke exited 0 on scikit-learn 1.8.0: zero warnings, parameter SHA-256 `8d11d5ccabf19409275f5f26a9bcca3019622e44f65f8a4057e0ec4c2f853117`, and maximum row-sum error 0. A direct API probe confirmed `l1_ratio=0.0` is warning-free while explicit `penalty='l2'` is deprecated in 1.8.

The bounded real-INX fold-1 diagnostic loaded the verified local dataset and immutable historical fold/model files through `load_canonical_dataset`, `read_xgboost_oof_artifacts`, `cross_fit_outer_training`, `fit_sigmoid_calibrator` and `apply_sigmoid_calibrator`; it did not call the canonical stage runner or write output. Exit 0: 5 fits, 1,080 cross-fit rows, 120 untouched test rows, zero warnings, 1.082 fit seconds, calibrator SHA-256 `ee7a5bc816f5b6ed41db6a54b229ca12c28bfb502d14a2854e178900925e856c`, and maximum simplex error `2.22e-16`.

Exact diagnostic invocation:

```powershell
@'
import json, time
from pathlib import Path
from src.data.canonical_loader import load_canonical_dataset
from src.governance.manuscript_contract import load_manuscript_config, manuscript_settings
from src.experiments.manuscript_policy_ablation import exact_policy_frame
from src.experiments.benchmark_artifact_contract import read_xgboost_oof_artifacts
from src.experiments.manuscript_calibration import cross_fit_outer_training, fit_sigmoid_calibrator, apply_sigmoid_calibrator
root=Path('reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core')
meta=json.loads((root/'model_benchmarks/stage_metadata.json').read_text(encoding='utf-8')); gate=meta['baseline_gate']
cfg=load_manuscript_config('configs/manuscript_final.yaml'); s=manuscript_settings(cfg)
loaded=load_canonical_dataset('configs/manuscript_final.yaml','inx_primary'); data=loaded.frame
policy=s['feature_policies']['primary_policy']; definition=s['feature_policies']['definitions'][policy]
X,excluded=exact_policy_frame(data,policy,definition,target_column=s['target']['column'],id_column=s['governance_fields']['identifier_fields'][0]); y=data[s['target']['column']].astype(int)
bundle=read_xgboost_oof_artifacts(root/'shared_folds',root/'model_benchmarks',expected_run_id=gate['run_id'],expected_config_hash=gate['config_hash'],expected_scientific_input_hash=gate['scientific_input_hash'],expected_feature_columns=X.columns.tolist(),expected_labels=(2,3,4))
identity={'run_id':gate['run_id'],'config_hash':gate['config_hash'],'scientific_input_hash':gate['scientific_input_hash'],'fold_contract_hash':bundle.identity.fold_contract_hash,'xgboost_model_set_sha256':bundle.model_set_sha256,'dataset_sha256':loaded.receipt['actual_sha256'],'calibration_protocol_sha256':'0'*64}
started=time.perf_counter(); pred,receipts=cross_fit_outer_training(features=X,target=y,bundle=bundle,outer_fold=1,forbidden_features=excluded,model_seed=s['seeds']['model'],identity=identity); elapsed=time.perf_counter()-started
cal=fit_sigmoid_calibrator(pred[['prob_class_2','prob_class_3','prob_class_4']].to_numpy(float),pred.y_true,(2,3,4),seed=s['seeds']['calibration'],settings=s['calibration']['sigmoid'])
raw=bundle.oof_predictions[bundle.oof_predictions.outer_fold==1].sort_values('sample_index')[['prob_class_2','prob_class_3','prob_class_4']].to_numpy(float); out=apply_sigmoid_calibrator(cal,raw)
print({'inner_fits':len(receipts),'training_rows':len(pred),'outer_test_rows':len(raw),'warnings':int(receipts.warning_count.sum()),'elapsed_seconds':round(elapsed,3),'calibrator_sha256':cal.parameter_sha256,'max_sigmoid_simplex_error':float(abs(out.sum(axis=1)-1).max())})
'@ | .\myenv\Scripts\python.exe -
```

Independent read-only final review reran 48 tests plus 7 subtests and found no material code-level calibration blocker. Current config `d755ecc39e516cab51269b314a7736ee4cea66bf500fa407d44fa81021ea0d18` correctly rejects historical benchmark config `7e70bf6646a542ad32e10ab3718654aa8232a46e44e2083ed10e2cfe526da595`. No hash was patched or relabelled; the same-config real run is deferred until all remaining core inputs freeze.

Two exploratory shell probes exited 1 without scientific execution: one printed XGBoost simplex residuals before a final lookup used a stale DataFrame index; another guessed obsolete artifact names before listing and using the actual `stage_metadata.json` and `fold_contract.json`. Both were corrected immediately and changed no file.

### Unit 2E full regression and hygiene gate

The first full run:

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m pytest -q
```

exited 1 during collection after 5.7 shell seconds because supplementary `manuscript_counterfactual_actionability.py` imported private legacy calibration helper `_fit_pipeline`, which the option-A replacement correctly removed. No test or scientific stage executed. The supplementary module was decoupled from calibration internals and now builds its noncanonical heuristic model through a local one-thread canonical-model helper. A first attempted focused command also named two not-yet-existing counterfactual tests and exited 1 before collection; the corrected exact repository command was:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_counterfactual_denominators_reported.py tests/test_counterfactual_protocol_is_oof.py
```

Exit 0: 5 passed in 1.38 seconds. After adding the private-import regression, the combined calibration/counterfactual focus passed 25 tests in 2.14 seconds.

The corrected complete commands were:

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
```

Results: pytest 425 passed, 2 skipped and 11 subtests in 83.64 seconds; unittest ran 173 tests in 7.089 seconds with 2 skips and exited 0.

Final hygiene command set:

```powershell
git diff --exit-code -- manuscript/mdpi_information/main.md
$m = rg --pcre2 -n '(?<![A-Za-z])sk-(?:proj-)?[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}' --glob '!myenv/**' --glob '!reports/manuscript_final/**' --glob '!*.ipynb' .
$m = git diff -- . ':!reports/research_log/**' | Select-String -Pattern 'C:\\Users\\|/home/[^/]+/|file://' -CaseSensitive
$tracked = @(git diff --name-only) + @(git ls-files --others --exclude-standard | Where-Object { $_ -notlike 'reports/manuscript_final/trials/*' })
$m = rg -n -i 'leakage[- ]safe' README.md configs/feature_sets.yaml configs/manuscript_final.yaml src/experiments/manuscript_policy_ablation.py src/experiments/manuscript_calibration.py
.\myenv\Scripts\python.exe -m pip check
```

All gates exited 0 after applying the documented no-match/100 MB conditions: no manuscript change, secret match, scientific-diff home path, large candidate or active leakage-safe term; pip reported no broken requirements. The README/index and issue-register audit exited 0 with 17 tracked local links and 26 well-formed issue rows.

Final post-regression rerun after adding the supplementary private-import guard:

```powershell
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
```

All exited 0: pytest 426 passed, 2 skipped and 11 subtests in 88.59 seconds; unittest 174 passed with 2 skips in 7.128 seconds; compileall and diff check passed.

Final real-INX all-fold diagnostic used the same loader/bundle construction shown in the exact fold-1 script above, then invoked the complete internal evidence loop exactly as follows:

```python
training, receipts, params, relations, predictions = _calibration_evidence(
    features=X,
    target=y,
    bundle=bundle,
    forbidden_features=excluded,
    model_seed=s["seeds"]["model"],
    calibration_seed=s["seeds"]["calibration"],
    sigmoid_settings=s["calibration"]["sigmoid"],
    primary_policy=policy,
    identity=identity,
)
```

The PowerShell wrapper was the same `@' ... '@ | .\myenv\Scripts\python.exe -` form and wrote no file. Exit 0 in 15.4 shell seconds (12.303 measured fit/evidence seconds): 50 receipts over ten folds; 10,800 cross-fit rows; 30 parameter rows; ten distinct sigmoid hashes; ten relationships/source model hashes; 1,200 raw and 1,200 sigmoid rows; zero warnings. The combined maximum probability-sum residual `8.381903171539307e-08` is the unchanged historical float32 raw comparator, not a sigmoid normalization failure.

Checkpoint command:

```powershell
git commit -m "feat(calibration): cross-fit sigmoid on benchmark folds"
```

Exit 0: commit `0f820b3`, 22 files changed, 3,175 insertions and 400 deletions. The local historical trial was not staged; no push, merge, release or manuscript edit occurred.

## Unit 2F - Focused Implementation and Real-INX Diagnostic

One initial read-only `rg` command used the literal Windows wildcard `tests/test_fairness*` and exited 1 with a filename-syntax error. It performed no write or scientific execution. The corrected commands named files explicitly.

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_fairness_support_and_ci_fields.py tests/test_stage_runner_scientific_input_binding.py tests/test_fairness_proxy_config_contract.py tests/test_fairness_proxy_stage_contract.py tests/test_shared_fold_assignments_across_models_and_policies.py tests/test_manuscript_policy_ablation.py tests/test_benchmark_artifact_contract.py
.\myenv\Scripts\python.exe -m pytest -q tests/test_fairness_proxy_scientific_behavior.py tests/test_fairness_support_and_ci_fields.py tests/test_fairness_proxy_config_contract.py tests/test_fairness_proxy_stage_contract.py tests/test_stage_runner_scientific_input_binding.py tests/test_shared_fold_assignments_across_models_and_policies.py tests/test_manuscript_policy_ablation.py tests/test_benchmark_artifact_contract.py
.\myenv\Scripts\python.exe -m py_compile src/experiments/manuscript_fairness_proxy.py src/governance/manuscript_contract.py src/experiments/build_manuscript_evidence.py
.\myenv\Scripts\python.exe -c "from pathlib import Path; from src.governance.manuscript_contract import load_manuscript_config,canonical_config_hash; load_manuscript_config(Path('configs/manuscript_final.yaml')); print(canonical_config_hash(Path('configs/manuscript_final.yaml')))"
git diff --check
```

All exited 0. Initial focus: 79 passed in 21.38 seconds. Expanded scientific-behavior focus: 84 passed in 20.02 seconds. Post-hardening direct subset: 31 passed in 4.29 seconds. Intermediate pre-review config hash: `be2b3f9f7e052df42ad9dc413d10e29bfc2ad6dd63a38513d21957aa9908523f`; this is superseded by the final hash below.

The intermediate pre-review real diagnostic used a PowerShell here-string piped to `python.exe -`. The script installed process-local `socket`, DNS and connection denial guards, loaded `inx_primary` through the canonical loader, generated 10 outer x 5 inner shared folds in memory using configured seeds, compared the outer map to the immutable historical trial, and invoked `generate_proxy_oof_evidence`. It created no output path and wrote no file.

Exit 0 after 9.7 shell seconds; measured evidence runtime 5.233 seconds. Dataset: 1,200x28, SHA-256 `b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a`; historical outer map exact match `true`; 20 fold-fit rows; 2,400 exactly-once proxy OOF rows; 5,000 resamples; batch size 200; proxy resample SHA-256 `ceca117f41907c8c39965fa168241e4004381a33a1cf7a66cfab1a260beec558`; adapter SHA-256 `ab0464e823b773cc79cf512bba2b5bb294e83c73b3a415775983878534ab549f`; RSS before/after 239.64/234.00 MiB. Diagnostic macro-F1 was 0.968543 (pointwise 95% CI 0.956635-0.980215) for the job-role-retained system and 0.247368 (0.226694-0.268709) for the job-role-removed system. The alias fit flag was false. These values are noncanonical implementation diagnostics and are not admitted manuscript evidence.

At this intermediate point, independent final review and the full repository gates remained pending. The completed final results follow.

### Unit 2F final review, regression and hygiene gate

The first post-review focused run failed one dynamic proxy test because a duplicate dictionary field referenced nonexistent `row.task_type`; after removal, the rerun failed one assertion because the manuscript proxy row omitted its nominal task field. Both serialization defects were fixed and no artifact was written. The corrected commands were:

```powershell
.\myenv\Scripts\python.exe -m py_compile src/models/task_schema.py src/governance/manuscript_contract.py src/experiments/manuscript_fairness_proxy.py
.\myenv\Scripts\python.exe -m pytest -q tests/test_task_metric_applicability.py tests/test_fairness_proxy_config_contract.py tests/test_fairness_proxy_scientific_behavior.py tests/test_fairness_support_and_ci_fields.py tests/test_fairness_proxy_stage_contract.py
.\myenv\Scripts\python.exe -m pytest -q tests/test_task_metric_applicability.py tests/test_fairness_proxy_scientific_behavior.py tests/test_fairness_support_and_ci_fields.py tests/test_fairness_proxy_config_contract.py tests/test_fairness_proxy_stage_contract.py tests/test_stage_runner_scientific_input_binding.py tests/test_shared_fold_assignments_across_models_and_policies.py tests/test_manuscript_policy_ablation.py tests/test_benchmark_artifact_contract.py
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
git diff --exit-code -- manuscript/mdpi_information/main.md
.\myenv\Scripts\python.exe -m pip check
```

Exit 0: direct focus 44 passed in 3.83 seconds; expanded focus 102 passed in 20.30 seconds; full pytest 467 passed, 2 skipped and 11 subtests in 85.74 seconds; unittest ran 178 tests with 2 skips in 7.045 seconds; compileall, diff, manuscript and dependency gates passed. Independent follow-up review reran 102 tests in 21.01 seconds, compiled the changed modules, passed diff checks and found no remaining P0/P1 issue.

The final hygiene wrapper found zero high-entropy secret matches, zero home/file-URI strings in the scientific diff, zero candidate files over 100 MB, zero active leakage-safe terms and zero canonical imports from legacy `proxy_analysis.py`. It verified 17 README local links and 27 issue-register rows. Its first issue-register attempt expected the older remediation column names and exited before completing the wrapper; the corrected audit used the actual v2 schema. A later read-only log search repeated the documented Windows literal-wildcard error and was corrected with `rg -g '*.md'`. Neither command changed files or executed science.

The final here-string INX diagnostic used current config hash `3c9588c1327ac563a85586835b19b30768860165dc26b61fcf7aafbce3bb1421`, the verified `b8deac...` dataset and process-local write/socket/DNS denials. Exit 0 after 6.9 shell seconds and 3.776 measured evidence seconds: exact historical outer-map match; 20 fits; 2,400 exactly-once OOF rows; nominal proxy task only; 5,000 draws in batches of 200; resample hash `ceca117f41907c8c39965fa168241e4004381a33a1cf7a66cfab1a260beec558`; adapter hash `ab0464e823b773cc79cf512bba2b5bb294e83c73b3a415775983878534ab549f`; minimum overall department support 20; minimum nonzero fold support 1; two zero-support fold/class cells. Macro-F1 estimates/intervals were unchanged from the intermediate diagnostic and remain noncanonical implementation evidence.

Checkpoint command `git commit -m "feat(fairness): bind subgroup and proxy diagnostics to OOF evidence"` exited 0: commit `a490d1e`, 22 files, 4,052 insertions and 658 deletions. Immediate `git status --short --branch` showed only the excluded untracked historical trial; `git ls-files reports/manuscript_final/trials` was empty. No push, merge, release or manuscript edit occurred.

## Unit 2G - Required Read-only External Replication Audit

The audit used `Get-Content`/`rg`/`git` and a process-local no-network Python here-string to inspect canonical config/acquisition/mapping, loader/adapters, the external module, tests and historical packages. The in-memory loader returned the exact expected HRDataset receipt and adapter mapping; it wrote no file and fitted no model. The corrected baseline test command was:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_manuscript_external_evidence.py tests/test_external_scope_contract.py tests/test_external_explicit_input_binding.py tests/test_external_claim_boundaries.py
```

Exit 0: 19 passed in 6.46 seconds. Input evidence: 311 × 36; dataset SHA-256 `cb19996755c93c0a8d6527f59da4701c80aef65eff854906546dce286249813c`; mapping SHA-256 `4988bde12fbd0198102f22f4078fd31ba20ea4285160363d9cf101610e9f19d0`; raw target support 37/243/18/13; mapped 2/3/4 support 31/243/37; zero unmapped. The raw file is Git-tracked and not ignored; source/licence fields remain manual review.

Historical read-only inventory: `reports/manuscript_final/latest/external` contains 53 files and 24,821,361 bytes under stale config `c664...`/commit `1834748`; `reports/external_validation` contains 192 files and 68,261,087 bytes. The former includes HR/IBM/Turnover plus actionability in one scope. Neither may feed v2.

One broad audit command passed literal Windows wildcard test names to `rg` and exited 1 after printing its valid non-wildcard inspection output. The explicit four-file pytest command above passed. No science/write/network/API/manuscript operation occurred.

## Unit 2G Implementation and Final Review

The resumed pre-review test runner completed with 120 passed in 33.61 seconds. Its exact invocation was started before the usage-limit interruption and is not reconstructed from memory; only the captured exit/output is reported here. The post-review commands were executed exactly as follows:

```powershell
.\myenv\Scripts\python.exe -m pytest tests/test_hrdataset_replication_stage_contract.py tests/test_artifact_run_manifest_consistency.py -q
.\myenv\Scripts\python.exe -m compileall -q src/governance/manuscript_contract.py src/experiments/manuscript_hrdataset_replication.py
.\myenv\Scripts\python.exe -m pytest -q tests/test_hrdataset_replication_stage_contract.py tests/test_external_replication_config_contract.py tests/test_external_nested_selection_contract.py tests/test_external_sigmoid_isolation.py tests/test_external_oof_bootstrap_contract.py tests/test_external_shap_contract.py tests/test_external_subgroup_proxy_contract.py tests/test_artifact_run_manifest_consistency.py
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
.\myenv\Scripts\python.exe -c "from src.governance.manuscript_contract import load_manuscript_config,canonical_config_hash,evidence_scope_contract_hash; p='configs/manuscript_final.yaml'; c=load_manuscript_config(p); print(canonical_config_hash(c)); print(evidence_scope_contract_hash(c,'core'))"
git diff --exit-code -- manuscript/mdpi_information/main.md
```

Exit 0 results: 12 passed in 2.10 seconds; then 68 passed in 20.94 seconds; both compile commands, diff check and manuscript no-change check passed. Config hash is `b7188f3d883c5ea90563ceb458c2ec4e721299659ddb0be62ab0dd27cabb7ed4`; core scope hash is `af80b8a7c355a57b3ad8cd39775132fcc6c87e19983df901bf5e503b422c90dc`.

One attempted focused invocation used bare `python`, which resolves on this machine to the unavailable Windows Store shim; it failed before collection with `Program 'python.exe' failed to run`. The repository interpreter command above replaced it. No scientific output, network/API operation or manuscript write occurred.

Independent read-only review reran 67 focused tests, compileall and diff checks and reported no remaining P0/P1 defect. It confirmed that the department proxy will be `not_estimated` because of singleton-class outer-training support. The real full stage remains pending and no Unit 2G number is yet an authoritative manuscript result.

### Interruption recovery and full-gate failures

The first checkpoint full-suite command exited 1 after 103.20 seconds: 519 passed, 2 skipped and 11 subtests passed; 11 tests failed. Four failures expected the pre-existing subgroup/proxy seed error boundaries, while seven temporary-project manifest tests revealed external side-input validation incorrectly rooted at the real repository and used incomplete semantic fixtures. The implementation now performs generic seed-shape validation before exact external-seed equality, resolves semantic side inputs against the supplied/inferred project root, and uses complete schema/provenance fixture bytes. The corrected focus passed 64.

Two later full-suite commands were intentionally terminated after independent review delivered new integration findings; they were not allowed to continue under already-superseded code. No result count is claimed for either. Recovery process inspection found no surviving child process or partial artifact.

Exact recovery inspection included branch/HEAD, all persistent records, `git status --short`, unstaged/cached diffs, 15 commits, Win32 process command lines, artifact roots, and temp/lock/partial searches. One read-only `rg` tail used a literal Windows wildcard and exited 1; no mutation or scientific execution occurred. The latest exact focus command before recovery passed 96 in 10.54 seconds and compile/diff checks passed. Builder resume remains the only open P1 before the next full gate.

## Unit 2G Final Recovery, Trust Boundary and Checkpoint Gate

The principal exact invocations after interruption were:

```powershell
.\myenv\Scripts\python.exe -m py_compile src\experiments\build_manuscript_evidence.py src\governance\manuscript_contract.py tests\test_manifest_completion_contract.py
.\myenv\Scripts\python.exe -m pytest -q tests/test_manifest_completion_contract.py tests/test_builder_path_contract.py tests/test_builder_resume_contract.py tests/test_atomic_stage_resume_contract.py tests/test_latest_pointer_contract.py tests/test_artifact_run_manifest_consistency.py tests/test_final_evidence_manifest_hashes.py tests/test_scoped_run_manifest_inputs.py tests/test_side_input_hash_binding.py
.\myenv\Scripts\python.exe -m pytest -q tests/test_external_replication_config_contract.py tests/test_external_nested_selection_contract.py tests/test_external_sigmoid_isolation.py tests/test_external_oof_bootstrap_contract.py tests/test_external_shap_contract.py tests/test_external_subgroup_proxy_contract.py tests/test_hrdataset_replication_stage_contract.py tests/test_external_validation.py tests/test_manuscript_external_evidence.py
.\myenv\Scripts\python.exe -m pytest -q tests/test_manifest_completion_contract.py tests/test_latest_pointer_contract.py tests/test_builder_resume_contract.py tests/test_atomic_stage_resume_contract.py tests/test_builder_path_contract.py tests/test_artifact_run_manifest_consistency.py tests/test_final_evidence_manifest_hashes.py tests/test_scoped_run_manifest_inputs.py tests/test_side_input_hash_binding.py tests/test_core_scope_contract.py
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
.\myenv\Scripts\python.exe -m pip check
.\myenv\Scripts\python.exe -c "from src.governance.manuscript_contract import load_manuscript_config,canonical_config_hash,evidence_scope_contract_hash; p='configs/manuscript_final.yaml'; c=load_manuscript_config(p); print(canonical_config_hash(c)); print(evidence_scope_contract_hash(c,'core')); print(evidence_scope_contract_hash(c,'supplementary'))"
git diff --check
git diff --exit-code -- manuscript/mdpi_information/main.md
```

Results: first trust focus 125 passed/2 skipped; external focus 80 passed; integrated post-fix focus 161 passed/1 skipped; independent trust 132 passed/2 skipped; independent external 96 passed/1 skipped; final release/trust focus 141 passed/2 skipped. The current full run passed 648 tests, 3 skips and 11 subtests in 109.16 seconds. Unittest ran 178 tests with 2 skips in 7.573 seconds. Compileall and `pip check` exited 0.

Failure record: the first integrated 161-test run had one assertion regex mismatch after production began emitting an explicit hidden/cache-directory error; the assertion was updated. The first final trust focus had two synthetic fixture failures: missing timing fields in a hand-built stage receipt and table/figure stage names without registered fixture runners. Both fixtures were corrected while production timing and release checks remained strict. Earlier 519-pass/11-failure and deliberately terminated superseded runs remain recorded above.

Scientific reviews used only the verified local HR file and process-local no-write/no-network diagnostics. They found and closed the case-insensitive date-alias contradiction and the direct-versus-normalized probability replay mismatch. The reduced diagnostic used an explicit test-only execution budget and produced no file; it is not an artifact or result source. No production eight-candidate/5,000-draw stage was executed.

Final identity pre-check: config `5af0262e83a3720f8dca0b4d6795bdffc6bb2cefedc901ae0a47f9262d07f305`; core scope `af80b8a7c355a57b3ad8cd39775132fcc6c87e19983df901bf5e503b422c90dc`; supplementary scope `18bbb5cb3dae267eac18f3cc60082e5023843b01bd565f8912555fc9c83aa329`. No API, acquisition-network, manuscript or scientific-artifact write occurred.

### Final Unit 2G checkpoint hygiene

The final post-documentation checks used `git diff --check`, `git diff --exit-code -- manuscript/mdpi_information/main.md`, `git diff --cached --name-only`, `git diff --name-only`, `git ls-files --others --exclude-standard`, `git ls-files --error-unmatch`, `git check-ignore -q`, `Import-Csv`, `Get-Item`, `Get-ChildItem` and `rg --pcre2` credential/terminology scans. An additional zero-context diff plus untracked-file `Select-String` scan classified all home-path-shaped literals.

The consolidated command exited 0: 45 candidates; zero staged, raw, 100 MB, reparse, secret, active `leakage-safe`, missing README target, untracked README target or manuscript-diff findings; 28 valid issue rows; and 17 local README link occurrences across 15 unique tracked targets. The immutable trial remains 54 files/91,820,515 bytes, untracked and locally excluded. The path scan returned only the two production sanitization regexes and the two negative-test fixture paths. V2-005 was narrowed honestly to new-package implementation complete with historical cleanup and real-package validation still pending.

## Unit 2G real stage and recovery validation - 2026-07-13/14

Production invocation, identity and runtime are recorded verbatim in `../finalization/COMMAND_LOG.md`. Exit 0: wrapper 491.424 seconds; atomic stage 477.803 seconds; run `stage_validation_hrdataset_20260713T175045Z_5af0262e83a3`; source commit `17a3dcd`; config `5af0262e...`; scientific input `71f1fc46...`; source tree `706690fc...`; zero network/API calls.

Recovery inspected Win32 processes, parent/child command lines, TCP listeners, UDP endpoints, Git state/diffs/15 commits, every run file, and lock/partial/temp/staging paths. No repository/run process or listener existed, so nothing was terminated. No duplicate stage was started.

The independent closed-world validator exited 0 with 122 artifact rows, 124 stage outputs and 125 stage files; all paths, sizes and SHA-256 values match. It verified 311 exactly-once outer rows; 2,799 inner rows; 400 candidate fits; 50 persisted models; 1,555 raw and 311 sigmoid OOF rows; 50 calibration fits; ten model-calibrator relationships; a `(5000,311)` bootstrap matrix; 6,531 local SHAP rows; 85 disparity rows; fail-closed proxy non-estimation; and transport infeasibility. Independent model and calibrator reconstruction produced maximum probability replay error zero.

Focused command:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_external_replication_config_contract.py tests/test_external_nested_selection_contract.py tests/test_external_sigmoid_isolation.py tests/test_external_oof_bootstrap_contract.py tests/test_external_shap_contract.py tests/test_external_subgroup_proxy_contract.py tests/test_hrdataset_replication_stage_contract.py tests/test_artifact_run_manifest_consistency.py
```

Exit 0: 72 passed in 29.58 seconds. Current-HEAD validation of the provisional input manifest produced the expected noncanonical `git commit mismatch`; it was caught and asserted, so the diagnostic command exited 0. No canonical manifest/promotion exists.

`git rm --cached -r -- reports/manuscript_final/stage_validation_hrdataset_20260713T175045Z_5af0262e83a3` exited 0. Verification found all 126 files/65,412,766 bytes preserved locally, zero tracked paths under the run root and the exact path ignored. This is D5 tip cleanup, not scientific deletion or history rewrite.

Commit `b7b2ad3074ff4b27f358fd3b9394b4ae2b1ad4a2` exited 0 after exact staged review. The subsequent normal `git push origin finalization/leakage-aware-v2` timed out after 184 seconds while Git Credential Manager awaited a response. The verified push/remote PID tree was bounded and removed after a graceful close failed; Credential Manager was not terminated. An unauthenticated GitHub ref query confirmed remote SHA `e25f403`. No second retry, force option, credential change, merge, release or history rewrite occurred.
