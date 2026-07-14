# Employee Performance with XAI

This repository is being finalized as a **leakage-aware XAI audit protocol** for three-class employee `PerformanceRating` prediction (classes 2/3/4). It is a research and reproducibility package, not an autonomous or deployment-ready HR system.

> **Research use only.** Nothing in this repository authorizes hiring, firing, promotion, compensation, discipline, ranking, screening, or other individual employment decisions. The study does not establish causal effects, legal compliance, fairness, human usefulness, or deployment readiness.

## Current v2 status

There is **no current canonical v2 manuscript release**. Finalization is active on `finalization/leakage-aware-v2`; the authoritative live status is [`CURRENT_STATUS.md`](reports/research_log/finalization/CURRENT_STATUS.md).

The named run `manuscript_final_20260712T181754Z_c664ef152ff3` is **historical v1 evidence**. It was produced from an old commit and dirty worktree, contains incompatible core scope and methods, and must not supply v2 manuscript numbers or claims. Its former tracked `latest` physical duplicate was verified byte-for-byte and migrated under the compact [`migration receipt`](reports/research_log/finalization_v2/14_latest_migration_receipt.json), leaving `latest` reserved for the pointer-only v2 contract. Earlier [`manuscript_remediation`](reports/research_log/manuscript_remediation/) records are historical context, not proof that a v2 issue is resolved.

At this checkpoint:

- Unit 2G implementation is recorded by `ae5cf5a`; the verified-real-data stage was generated from clean source commit `17a3dcd`. Commit `e25f403` then accidentally pushed the complete 65.4 MB noncanonical evidence package. Forward cleanup `b7b2ad3` preserves that package locally and removes it from the current Git tip under D5 without rewriting history.
- V2-021 implementation checkpoint `5a0d926` is synchronized with `origin/finalization/leakage-aware-v2`; hosted CI run `29353799148` passed at that exact commit. Unit 2G, V2-032, dependency, offline-runtime, and CI checkpoints were also sent by normal authenticated pushes. The full noncanonical evidence blobs remain in public history, which is still a publication-hygiene blocker under the no-history-rewrite rule.
- Actual dataset and side-input binding, scoped core/supplementary orchestration, shared 10-fold assignments, the four-model benchmark contract, paired OOF bootstrap, warning-clean model preprocessing, exact prediction-model-to-OOF-SHAP binding, shared-fold leakage-policy ablation, cross-fitted sigmoid calibration, support-aware subgroup/proxy diagnostics, and conservative HRDataset_v14 replication now have implementation/test checkpoints.
- These downstream stages are implemented but have **not** yet been regenerated together as one canonical scientific package. Final benchmark, policy, calibration, SHAP, subgroup/proxy, and external numbers still require one clean current-commit run. The replacement seven-figure plan, both scope-bound source-table generators, fail-closed package validators, controlled Python dependency split/lock, and hosted CI are implemented. Real current-run table/figure artifacts and the complete two-scope rebuild remain unfinished.
- Unit 2G uses 10 outer x 5 inner nested XGBoost, cross-fitted sigmoid, 5,000 paired/bootstrap draws, exact-model OOF grouped SHAP, support-aware diagnostics, and atomic provenance-bound output. The reusable direct validator and its small [`checkpoint receipt`](reports/research_log/finalization_v2/10_unit2g_checkpoint_summary.json) independently replay all 50 models, ten calibrators, bootstrap indices, and ten-fold SHAP values with zero numerical drift. This validates the stage only: its outer input manifest is provisional, no complete package manifest exists, and no numerical v2 manuscript claim is frozen.
- The post-Unit-2G reporting contract requires future subgroup tables/metadata to identify `probability_method=raw` and hash the exact policy-scoped OOF rows consumed. Future SHAP tables, fold receipts, local reason codes and metadata must declare XGBoost raw-margin-score units. The current full repository gate passes 747 tests with 2 skips and 11 subtests; the older validated stage was not rewritten or promoted and does not contain these new fields.
- The approved core figure plan is exact and source-bound: study design, leakage-policy trade-off, four-model comparison, predeclared sigmoid calibration, global grouped OOF SHAP, descriptive SHAP stability, and HRDataset_v14 mapped-target replication. V2-021 now renders the seven PNG/SVG pairs, machine-readable source tables, technical captions, and manifest only from exact current-run upstream receipts, using an atomic `core_figures` publication directory. V2-032 validates the closed-world package and rejects historical/manual/obsolete artifacts. No real v2 figure artifact or canonical package exists yet; the generator was exercised only with a synthetic unit-test fixture.
- V2-022 defines one immutable 18-metric registry covering applicability, display/estimand, direction, domain, units, aggregation, denominator, probability requirements, selection role and 5,000-draw uncertainty for all five task types. Core tables 1-10 and 13 and supplementary tables 11-13 are exact source-bound, atomic, closed-world stages. Every emitted row carries run/config/scientific-input/source-tree and metric-registry identity plus source path/hash, dataset/model/evaluation/denominator/uncertainty fields. Only synthetic fixtures have exercised these generators; no real table or manuscript value was created.
- V2-012 replaces the legacy counterfactual/actionability stage with a supplementary-only heuristic-search contract. It uses exact OOF cases; outer-training-only models, prototypes, domains and scales; actual serialized fold-model and aggregate model-set SHA-256 identities; four distinct taxonomy-labelled feature scopes; one shared maximum candidate pool; three nested budgets; complete eligible-case coverage; Wilson search-success intervals; and 5,000-draw successful-case bootstrap summaries. Outputs use only `heuristic counterfactual-search success` language and explicitly prohibit causal recourse, prescriptions, intervention evidence and real-world feasibility claims. A bounded real-INX diagnostic passed and was deleted after validation; no supplementary production artifact or canonical claim exists yet.
- V2-013 replaces the mixed legacy external helper in the production builder with a task-bounded supplementary runner. IBM 3/4 PerformanceRating is a restricted binary robustness stratum; IBM attrition and Employee Turnover are separate related-task transfer strata. Each uses exact 10 outer x 5 inner OOF evaluation, primary-policy-only XGBoost selection, same-fold parameter reuse for audit policies, persisted/replay-verified outer models, task-specific metrics with literal `N/A` for ordinal metrics, and a predeclared 5,000-draw paired OOF bootstrap. No stratum is directly comparable with the primary three-class task, direct external validation, or locked-model transport. A bounded real-IBM diagnostic passed with 1,470 rows, 4,410 OOF rows, 30 models and zero replay error, then its temporary output was removed; no production supplementary artifact or numerical claim exists yet.
- V2-015 executable provenance validation now has a compact canonical-eligible [`receipt`](reports/research_log/finalization_v2/11_inx_workbook_equivalence_receipt.json) from clean commit `8f02e55`. A read-only, macro-disabled, no-link-update Excel 16 run found exact 1201 x 28 workbook/CSV equality after the declared normalization contract, zero mismatches and a common normalized SHA-256 of `b5caa2ee...`. The workbook `Data Definitions` sheet is separately hashed and bound through an explicit `RelationshipSatisfaction -> EmpRelationshipSatisfaction` alias, but it documents only 7 of 28 columns and is not a complete or authoritative data dictionary. Source authenticity, licence and citation remain manual-review blockers.
- V2-014 forward sanitation is complete at current tip. All 14 declared source/interim/row-level/fitted files remain locally byte-identical and ignored, while commit `9342b0c` no longer tracks them. The compact canonical-eligible [`technical-export receipt`](reports/research_log/finalization_v2/12_publication_export_receipt.json) independently rebuilds the exact-commit allowlisted archive: 305 closed-world members, SHA-256 `1917059f...`, zero forbidden paths, secrets, machine-path findings, symlinks or Git metadata, and no retained ZIP. This does not erase prior Git history, grant redistribution rights, or publish a scientific/release package.
- Every declared core and supplementary runner/validator now exists and both scopes are technically execution-ready in [`configs/manuscript_final.yaml`](configs/manuscript_final.yaml). This flag authorizes the clean build; it does not assert that a package exists or is canonical. No full current-commit canonical build or verified release manifest exists yet.
- The first clean core attempt, `canonical_v2_20260714T175804Z_4d08ca2`, failed before any model fit because the offline boundary's function replacement for `subprocess.Popen` was not class-compatible with a late Python 3.14 `asyncio` import. The failed 141 KB receipt is preserved locally and inadmissible. The guard is now class-compatible; a fresh-process sklearn import regression and the complete suite pass. A new exact-commit run is still required.
- The second clean core attempt, `canonical_v2_20260714T180533Z_770c8d0`, completed shared folds and the full four-model benchmark, then failed safely before policy-ablation fitting because the reporting stage treated the complete 18-metric applicability registry as its narrower display set. Its failed 92.3 MB package is preserved locally and inadmissible. Policy ablation and calibration now validate their predeclared report subsets against the complete registry; 43 focused tests and the complete regression pass. A new clean exact-commit run is still required.
- The repaired two-scope candidate `canonical_v2_20260714T182539Z_35b121b` completed core in 1,614.358 seconds and supplementary in 1,862.575 seconds; strict read-only release validation returned `status=valid`. It is preserved as valid pre-migration evidence, but strict promotion correctly requires the exact current commit, so it is not canonical after the historical-`latest` migration checkpoint. The scientific source-tree and config hashes are unchanged; one new exact-commit build remains required for pointer promotion.
- V2-020 now wraps the complete core/supplementary scientific build in a process-wide no-network/no-paid-API boundary. DNS, TCP, UDP, listener and socket-send operations fail; caught attempts poison completion; API credentials are cleared/restored; shell, non-Git and remote-Git subprocesses are rejected; package status must carry the exact zero-attempt policy. Hosted enforcement passed in V2-019/V2-021 CI, including run `29353799148`; the current complete local suite passes 747 tests, 2 skips and 11 subtests.
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

### Frozen core figure plan

[`configs/manuscript_final.yaml`](configs/manuscript_final.yaml) now binds the future core figures to exact portable, current-run upstream sources and claim boundaries:

1. Study design and leakage-aware XAI audit pipeline.
2. Feature-policy and leakage-risk ablation trade-off.
3. Primary XGBoost versus the three predeclared baselines.
4. Predeclared cross-fitted sigmoid calibration.
5. Global grouped out-of-fold SHAP attribution.
6. Descriptive grouped out-of-fold SHAP stability.
7. HRDataset_v14 independent mapped-target replication.

### Frozen source-table plan

The production builder now generates only current-run, hash-bound source tables. Core contains tables 1-10 plus table 13 (reproducibility and claim boundaries); supplementary contains table 11 (heuristic search success), table 12 (restricted/binary tasks), and its scope-specific table 13. These CSVs are manuscript-support evidence, not formatted manuscript tables. The exact plan and claim boundaries live in [`table_contract.py`](src/governance/table_contract.py); no real table is canonical until the final clean two-scope package passes every manifest and promotion gate.

This is a tested **plan, generator, and validation contract**, not a real generated figure package. The production runner publishes exactly 29 runner-owned files atomically: seven 300-DPI PNGs, seven SVGs without unsafe declarations, seven source CSVs, seven technical captions, and one identity/source-hash manifest. The orchestrator then adds the closed-world stage contract and revalidates every upstream receipt. Canonical Figure 1-7 artifacts remain pending the clean real-data build, and historical/manual v1 figures cannot satisfy this contract.

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

The conservative HRDataset_v14 primary policy excludes department/position/status/marriage/diversity aliases, identifiers, sensitive fields, raw dates, Salary, State, Zip, and RecruitmentSource. Its exact seven retained feature families are `EmpJobRole`, `EngagementSurvey`, `EmpJobSatisfaction`, `SpecialProjectsCount`, `DaysLateLast30`, `Absences`, and derived `ExperienceYearsAtThisCompany`. Engagement/attendance timing is unverified and has a separate temporality-restricted audit. Two negative tenure durations are explicitly set missing; source/reference date support is schema-bound with no current-date or dataset-maximum fallback.

The HRDataset department proxy diagnostic is expected to be `not_estimated_insufficient_outer_training_class_support` because a singleton class is absent from at least one outer-training split. This is insufficient support, not evidence of fairness or low proxy risk; classes are not silently merged or dropped.

The complete fixed scope and prohibited claims are recorded in [`00_scope_and_fixed_decisions.md`](reports/research_log/finalization_v2/00_scope_and_fixed_decisions.md).

## Verified HRDataset_v14 stage (not a canonical release)

The local run `stage_validation_hrdataset_20260713T175045Z_5af0262e83a3` completed offline on the verified 311-row HRDataset input under config `5af0262e83a3720f8dca0b4d6795bdffc6bb2cefedc901ae0a47f9262d07f305`. Its closed-world stage contract contains 124 hash-verified outputs; 50 persisted fold models, all ten sigmoid calibrators, the 5,000-draw plan, and exact-fold grouped SHAP replay without numerical drift. Raw dataset SHA-256 is `cb199967...`; the parsed-cell content SHA-256 is `e6d5bb36...`; scientific-input and source-tree hashes are `71f1fc46...` and `706690fc...`. The package is local/ignored under D5 and is not expected to resolve from a GitHub checkout.

Stage-validation results establish reporting constraints, not frozen manuscript numbers:

- Raw conservative macro-F1 is `0.666355` (paired-bootstrap 95% CI `0.628073` to `0.704690`); QWK is `0.541220` (`0.485143` to `0.598791`).
- Predeclared sigmoid improves log loss (`0.536253` to `0.422393`) and multiclass Brier (`0.301308` to `0.235226`), but lowers macro-F1 by `-0.041104` (`-0.074614` to `-0.008355`) and produces no class-4 argmax predictions. Probability quality and class-decision performance must therefore be reported separately.
- Removing timing-unverified engagement/attendance fields lowers macro-F1 by `-0.366008` (`-0.412759` to `-0.319906`), so temporality is a major limitation rather than a robustness success.
- Exact-fold grouped SHAP contains only the seven permitted primary features. The leading attributions are `DaysLateLast30`, `EngagementSurvey`, derived tenure and `Absences`; these raw-margin attributions are noncausal.
- Department proxy reconstructability is not estimated because one outer-training split lacks the singleton department class. Locked INX transport is infeasible because only three safe common features pass the overlap contract.

The run has no completed canonical run manifest, final evidence manifest, claim matrix or `latest` promotion. Its provisional input manifest correctly remains nonterminal. The reusable validator is [`unit2g_stage_validator.py`](src/governance/unit2g_stage_validator.py); the compact noncanonical result is [`10_unit2g_checkpoint_summary.json`](reports/research_log/finalization_v2/10_unit2g_checkpoint_summary.json). Full commands, identities, validation results and limitations are recorded in [`COMMAND_LOG.md`](reports/research_log/finalization/COMMAND_LOG.md), [`TEST_LOG.md`](reports/research_log/finalization/TEST_LOG.md), and [`07_artifact_and_claim_map.md`](reports/research_log/finalization_v2/07_artifact_and_claim_map.md).

The stage remains scientifically intact but predates the explicit V2-029 reporting labels. Those labels will be generated only in the future clean canonical rebuild; no historical CSV or JSON was edited in place.

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

The current branch tip no longer tracks 14 declared source, interim, row-level processed, and fitted-working-data paths while preserving and ignoring all 2,335,429 local bytes. Dataset cards, schema mappings, documentation, and placeholders remain tracked. The clean history-free technical-export receipt passes independently. Earlier Git history still contains raw files and is not rewritten; source/licence verification and any public-history strategy remain manual blockers.

## Ethics and publication status

Ethics/IRB confirmation is pending. The previously supplied institution, unit, reference, and date strings were placeholders and are not verified facts. Engineering may continue, but submission readiness remains blocked until authentic institutional metadata is supplied.

The approved publication plan is:

- code plus small source tables/figures/manifests in Git;
- the full evidence package prepared for GitHub Release or Zenodo;
- `reports/manuscript_final/latest` reduced to a small pointer rather than a physical duplicate;
- checkpoint pushes are permitted only on `finalization/leakage-aware-v2` after tests, staged-file review, secret/raw-data checks, and README synchronization; no force-push, merge, release, Zenodo upload, history rewrite, or publication is permitted.

Dataset licence/source verification and ethics confirmation remain manual submission blockers. See [`08_manual_submission_blockers.md`](reports/research_log/finalization_v2/08_manual_submission_blockers.md).

## Build entry points

Use CPython 3.14 and the exact constraints lock. The compatibility entry point installs only the canonical scientific core:

```powershell
py -3.14 -m venv myenv
.\myenv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements-supplementary.txt` adds no network/API package beyond core. `requirements-dev.txt` composes the complete test environment. `requirements-legacy-optional.txt` is explicitly excluded from core and approved supplementary evidence; it contains retained CatBoost/UI/OpenAI modules only. A fresh isolated core install resolved 31 locked non-bootstrap distributions, imported all 13 direct packages, and passed `pip check` plus the production dependency validator. Its canonical-eligible [`dependency receipt`](reports/research_log/finalization_v2/13_dependency_lock_receipt.json) binds clean commit `498e8ad`, source tree `a2b361b8...`, exact lock SHA-256 `482cbf32...`, and zero missing/unlocked/mismatched/core-forbidden packages. The temporary environment was removed after independent validation.

[`ci.yml`](.github/workflows/ci.yml) now runs the exact CPython 3.14 development lock on pushes and pull requests, clears all declared API credentials, checks dependency consistency and manuscript-file immutability, runs the offline-runtime focus plus pytest/unittest/compileall, and enforces tracked-data, secret, large-file, path, issue-register and README-link gates. [`validate-release-candidate.yml`](.github/workflows/validate-release-candidate.yml) is a manual validation-only workflow for a labelled trusted Windows self-hosted runner: it requires an exact run ID, 40-character commit and every ignored local real-data input before it validates both immutable scopes under the offline boundary; it contains no upload, promotion, push or publication step. GitHub Actions run `29351557672` passed all gates at commit `700adb2` (35 focused; 709 passed/23 explicit data-free skips/11 subtests; unittest 177 with 10 skips). The local real-data suite passed all 730 tests with only the two pre-existing skips. No release candidate exists.

The intended clean core command is:

```powershell
.\myenv\Scripts\python.exe -m src.experiments.build_manuscript_evidence --config configs/manuscript_final.yaml --scope core --no-reuse-compatible
```

The supplementary scope uses `--scope supplementary`. Both commands are now technically enabled because every declared stage, contract test, and claim boundary is implemented. Release readiness still requires one clean current-commit, cache-disabled two-scope rebuild, strict manifest validation, table/figure validation, and explicit pointer promotion; the execution flags alone are not a canonical-release claim.

An interrupted build may be resumed only with its explicit existing `--run-id` and compatible reuse enabled. The original manifest must show a clean start and match current commit/source/config/dataset/side-input/scientific identities. A completed sibling scope under the same run ID is excluded from the clean-start check only after strict package validation; all other untracked paths remain disallowed. Per-scope locks prevent concurrent writers, stale locks are never taken over automatically, and promotion rejects locked or non-release-ready packages. New runs still require a completely clean worktree.

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
