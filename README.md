# Employee Performance with XAI

This repository is being finalized as a **leakage-aware XAI audit protocol** for three-class employee `PerformanceRating` prediction (classes 2/3/4). It is a research and reproducibility package, not an autonomous or deployment-ready HR system.

> **Research use only.** Nothing in this repository authorizes hiring, firing, promotion, compensation, discipline, ranking, screening, or other individual employment decisions. The study does not establish causal effects, legal compliance, fairness, human usefulness, or deployment readiness.

## Current v2 status

There is **no current canonical v2 manuscript release**. Finalization is active on `finalization/leakage-aware-v2`; the authoritative live status is [`CURRENT_STATUS.md`](reports/research_log/finalization/CURRENT_STATUS.md).

The existing [`reports/manuscript_final/latest/`](reports/manuscript_final/latest/) directory and the run `manuscript_final_20260712T181754Z_c664ef152ff3` are **historical v1 evidence**. They were produced from an old commit and dirty worktree, contain incompatible core scope and methods, and must not supply v2 manuscript numbers or claims. Earlier [`manuscript_remediation`](reports/research_log/manuscript_remediation/) records are historical context, not proof that a v2 issue is resolved.

At this checkpoint:

- Actual dataset and side-input binding, scoped core/supplementary orchestration, shared 10-fold assignments, the four-model benchmark contract, paired OOF bootstrap, warning-clean model preprocessing, exact prediction-model-to-OOF-SHAP binding, and shared-fold leakage-policy ablation have implementation/test checkpoints.
- The policy stage now consumes the shared folds and each fold's primary-selected XGBoost parameters; final scientific policy numbers still require regeneration with the benchmark in one clean current-commit run. Calibration, subgroup/proxy diagnostics, external replication, replacement figures/tables, dependency locking, CI, sanitized publication packaging, and a clean full rebuild remain unfinished.
- The core and supplementary entry points exist but deliberately fail closed because both scopes have `release_ready: false` in [`configs/manuscript_final.yaml`](configs/manuscript_final.yaml). No full current-commit canonical build or verified release manifest exists yet.
- The manuscript has not been edited. A claim matrix must be technically frozen and approved before manuscript changes.

Open engineering and scientific issues are tracked in [`02_issue_register.csv`](reports/research_log/finalization_v2/02_issue_register.csv). Do not infer readiness from green unit tests alone.

## Fixed study scope

The v2 core paper scope is:

1. Verified INX input/provenance preflight and one shared outer-fold assignment.
2. XGBoost compared with multinomial Logistic Regression, Random Forest, and LightGBM.
3. XGBoost feature-policy/leakage ablation on matched folds and parameters.
4. Predeclared sigmoid calibration evaluated only on outer test folds.
5. Exact-fold out-of-fold grouped SHAP and descriptive stability analysis.
6. Support-aware subgroup and proxy-risk diagnostics.
7. HRDataset_v14 independent mapped-target external replication.
8. Run-bound tables, figures, claim matrix, and evidence manifest.

The approved nested benchmark uses **10 outer folds x 5 inner folds**. Macro-F1 is the sole primary inner-selection and baseline-gate metric. QWK is secondary and is used only as the predeclared tie-breaker inside the inclusive `0.001` macro-F1 tie pool. Detailed XAI remains attached to XGBoost unless a baseline has a positive baseline-minus-XGBoost macro-F1 estimate and a paired OOF bootstrap 95% CI lower bound above zero.

The primary feature policy is `no_salary_hike_no_attrition_no_department`. It excludes:

```text
Age
Gender
MaritalStatus
EmpDepartment
EmpLastSalaryHikePercent
Attrition
EmpNumber
PerformanceRating  # target; never a model input
```

The full-feature policy (whose legacy name contains `upper_bound`) is an information-rich diagnostic comparator, not a mathematically guaranteed or separately optimized upper bound. Sensitive-retaining policies are audit-only. SHAP is attribution, not causality; removing sensitive fields does not establish fairness or eliminate proxy risk.

### Core exclusions

- LLM, chatbot, and agent evaluations are excluded from the core pipeline, results, tables, figures, and claims. Preserved code/results are legacy or experimental only and are not v2 scientific evidence. Core builds must be offline and make no paid API calls.
- Counterfactual analysis is supplementary-only if retained and may report only **heuristic search success**. It must not claim employee actionability, causal recourse, feasible intervention, or advice.
- IBM performance is supplementary restricted-target robustness. IBM attrition and Employee Turnover are supplementary related binary tasks; none is employee-performance external validation.
- HRDataset_v14 is an independently trained mapped-target replication, not locked INX-model transport.

The complete fixed scope and prohibited claims are recorded in [`00_scope_and_fixed_decisions.md`](reports/research_log/finalization_v2/00_scope_and_fixed_decisions.md).

## Verified four-model trial (decision evidence, not a canonical release)

The real offline trial `reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core/` completed 1,540 fits on the verified 1,200-row INX input and used 5,000 paired OOF bootstrap resamples. Its 91.8 MB package is intentionally local/untracked under D5, so the path is not a GitHub link. Its manifest explicitly records `canonical_release_eligible=false`; it is retained only as verified model-reference decision evidence and must be regenerated with downstream stages under the final clean commit before manuscript numbers are frozen.

| Model | Macro-F1 (paired OOF bootstrap 95% CI) | QWK (secondary) |
| --- | ---: | ---: |
| XGBoost | 0.621021 (0.597319 to 0.644690) | 0.567602 |
| LightGBM | 0.605488 (0.583315 to 0.629174) | 0.588329 |
| Random Forest | 0.592340 (0.579571 to 0.604757) | 0.631678 |
| Logistic Regression | 0.506221 (0.480283 to 0.531841) | 0.371011 |

Baseline-minus-XGBoost macro-F1 differences were:

- LightGBM: `-0.015533` (95% CI `-0.038121` to `0.006382`)
- Random Forest: `-0.028681` (95% CI `-0.049949` to `-0.008049`)
- Logistic Regression: `-0.114800` (95% CI `-0.147597` to `-0.083224`)

The predeclared superiority gate did not trigger, so XGBoost remains the XAI reference. Random Forest's higher secondary QWK does not override the macro-F1 gate.

Authoritative local trial records (the 91.8 MB package is intentionally untracked under D5 and is not expected to resolve from a GitHub source checkout):

- `reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core/model_benchmarks/model_summary.csv`
- `reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core/model_benchmarks/paired_model_differences.csv`
- `reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core/model_benchmarks/baseline_xgboost_gate.json`
- `reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core/run_manifest.json`, SHA-256 `1b4c3381489f8b0bf7ae60d57280b3ddd5aa5344cb250b1df63fdaaa6cc7379c`
- Exact command, runtime, input/config/fold/bootstrap hashes, and validation results: [`06_commands_and_tests.md`](reports/research_log/finalization_v2/06_commands_and_tests.md)

The trial's old ndarray preprocessing lacks the fitted feature-name lineage required by the current fail-closed grouped-SHAP contract. It therefore cannot be reused as the upstream source of final SHAP artifacts; benchmark and SHAP must be regenerated together.

## Data acquisition and redistribution policy

[`configs/data_acquisition.yaml`](configs/data_acquisition.yaml) pins each local dataset's SHA-256, schema, row/column counts, and target distribution. Scientific stages must consume the configured verified file; they may not silently fall back to an interim file or unapproved mirror.

Current acquisition records classify all datasets as user-provided local files and all licence/source authenticity fields as requiring manual review. The recorded mirror URLs are not approved download URLs. Consequently, if a local file is absent, the current configuration fails closed. Automatic download becomes permissible only after an explicit URL is approved in the acquisition manifest, and downloaded bytes must match every pinned hash/schema/support check. A mismatch must not be used.

Raw datasets with unverified redistribution rights will not be included in the sanitized publication repository. Local raw files must not be deleted. This repository-hygiene step is still open.

## Ethics and publication status

Ethics/IRB confirmation is pending. The previously supplied institution, unit, reference, and date strings were placeholders and are not verified facts. Engineering may continue, but submission readiness remains blocked until authentic institutional metadata is supplied.

The approved publication plan is:

- code plus small source tables/figures/manifests in Git;
- the full evidence package prepared for GitHub Release or Zenodo;
- `reports/manuscript_final/latest` reduced to a small pointer rather than a physical duplicate;
- no release, push, merge, or publication without explicit user approval.

Dataset licence/source verification and ethics confirmation remain manual submission blockers. See [`08_manual_submission_blockers.md`](reports/research_log/finalization_v2/08_manual_submission_blockers.md).

## Build entry points

The intended clean core command is:

```powershell
.\myenv\Scripts\python.exe -m src.experiments.build_manuscript_evidence --config configs/manuscript_final.yaml --scope core --no-reuse-compatible
```

The supplementary scope uses `--scope supplementary`. **These are not release-ready commands yet:** the current config intentionally blocks both before scientific stage execution while their `release_ready` flags are false. The flags must be enabled only after every declared stage, contract test, and claim boundary is complete; then the final release requires a clean current-commit, cache-disabled rebuild and manifest verification.

Routine validation commands are:

```powershell
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
```

Do not treat the historical v1 test count or package-completion report as current v2 readiness evidence. Current commands and outcomes are recorded in the persistent logs below.

## Reviewer and agent resumption order

Start with [`AGENTS.md`](AGENTS.md), then read:

1. [`CURRENT_STATUS.md`](reports/research_log/finalization/CURRENT_STATUS.md)
2. [`DECISION_LOG.md`](reports/research_log/finalization/DECISION_LOG.md)
3. [`NEXT_ACTIONS.md`](reports/research_log/finalization/NEXT_ACTIONS.md)
4. [`COMMAND_LOG.md`](reports/research_log/finalization/COMMAND_LOG.md)
5. [`TEST_LOG.md`](reports/research_log/finalization/TEST_LOG.md)
6. [`02_issue_register.csv`](reports/research_log/finalization_v2/02_issue_register.csv)
7. Current branch, HEAD, `git status`, and `git diff`

Detailed v2 records are in [`reports/research_log/finalization_v2/`](reports/research_log/finalization_v2/): scope, baseline audit, issue register, root-cause analysis, decisions, implementation progress, exact commands/tests, artifact/claim map, manual blockers, and readiness report. These files must reflect real progress; an issue is not resolved until implementation, real-data execution, tests, and artifact validation all pass.

## Repository areas

```text
configs/                           Canonical scope, acquisition, model, and policy contracts
data/                              Local scientific inputs and schema mappings; publication policy pending
src/experiments/                   Core/supplementary orchestration and scientific stages
src/explainability/                Grouped-SHAP lineage and supporting explainability code
src/governance/                    Run, provenance, scope, claim, and manifest contracts
src/llm/, src/chatbot/, src/agents Legacy/experimental modules; forbidden from core evidence
reports/manuscript_final/          Historical v1 packages and noncanonical v2 trials until final rebuild
reports/research_log/finalization/ Persistent interruption-safe handoff state
reports/research_log/finalization_v2/ Detailed v2 audit and implementation evidence
tests/                             Unit, contract, and integration checks
manuscript/                        Author-owned manuscript; not edited during technical finalization
```

The governing principle is simple: green tests are necessary but insufficient. Every manuscript claim must trace to verified real input, an approved protocol, correct uncertainty and claim boundaries, and artifacts from one clean run/config/commit identity.
