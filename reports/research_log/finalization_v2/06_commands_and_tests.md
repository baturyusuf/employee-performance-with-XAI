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
