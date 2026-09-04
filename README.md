# Employee Performance with XAI

This repository implements a publication-grade **leakage-aware XAI audit protocol** for employee-performance modelling. It is a research and reproducibility package, not an autonomous or deployment-ready HR system.

> **Research use only.** Nothing here authorizes hiring, firing, promotion, compensation, discipline, ranking, screening, or another individual employment decision. The evidence does not establish causality, legal compliance, fairness, human usefulness, or deployment readiness.

## Canonical v2 evidence

The canonical technical evidence package is complete and validated on `finalization/leakage-aware-v2`.

| Identity | Value |
| --- | --- |
| Canonical run | `canonical_v2_20260714T221501Z_483f96f` |
| Clean generation commit | `483f96fdbaab16cb0f32d03d9dbe676a759af44a` |
| Config SHA-256 | `51415c2ce68c89d9ce2b042b0a7a811fe3e98180b72460006f9d0465f6bf49b7` |
| Source-tree SHA-256 | `f1e358e99914563305428cece1b1595bc76a58643184407ec5b222162d650332` |
| Core scientific-input SHA-256 | `06c507bee525ea1daca43b61249764007d4d8baaa05c9333f23446ea723ce160` |
| Supplementary scientific-input SHA-256 | `caffb945d15f990e3a789e9707f7a8a9115be31fecbbd705822994a10cfaf151` |
| Canonical acceptance | 752 pytest passes, 2 skips, 11 subtests; 179 unittest passes, 1 skip |
| Scientific network/API use | zero attempted network operations; zero paid API calls |

The authoritative compact receipt is [`15_canonical_evidence_receipt.json`](reports/research_log/finalization_v2/15_canonical_evidence_receipt.json). Current status, decisions, commands, tests, issues, and blockers are maintained under [`reports/research_log/finalization/`](reports/research_log/finalization/) and [`reports/research_log/finalization_v2/`](reports/research_log/finalization_v2/).

### Major-revision v3 extension

The canonical v2 package remains immutable, but a broader reviewer-remediation brief requires additional ordinal, repeated-CV, policy-retuning, SHAP-faithfulness, replication-sensitivity, and data-quality evidence. The additive v3 work is defined in the [`major revision v3 plan`](reports/research_log/major_revision_v3/PLAN.md) and the [`requirement coverage audit`](reports/research_log/major_revision_v3/REQUIREMENT_COVERAGE_AUDIT.md). Implemented controls include the validated [`feature-availability and governance contract`](reports/research_log/major_revision_v3/FEATURE_AVAILABILITY_GOVERNANCE_CONTRACT.md) and the [`ordinal benchmark protocol`](reports/research_log/major_revision_v3/ORDINAL_BENCHMARK_PROTOCOL.md).

The completed Phase 1B comparison is published as a governed [`compact ordinal-benchmark evidence package`](reports/research_log/major_revision_v3/phase1b_ordinal_benchmark/README.md). Cumulative-threshold XGBoost leads macro-F1 and balanced accuracy, Random Forest leads QWK and ordinal MAE, LightGBM leads RPS and multiclass Brier, and nominal XGBoost leads log loss. The proportional-odds model is weaker than the nominal models, and cumulative-threshold XGBoost's log loss is materially worse than its classification ranking.

The [`repeated nested-CV protocol`](reports/research_log/major_revision_v3/REPEATED_NESTED_CV_PROTOCOL.md) and governed [`compact Phase 1C evidence package`](reports/research_log/major_revision_v3/phase1c_repeated_nested_cv/README.md) report training/fold variability across five prespecified 5×5 repetitions with all nine systems refitted. Macro-F1 winners alternate between LightGBM and XGBoost, cumulative-threshold XGBoost wins balanced accuracy in four repetitions, and Random Forest wins QWK and ordinal MAE in all five. These metric-specific results do not establish a universally best model and do not alter or supersede any canonical v2 result.

The Phase 1D [`fixed-schedule and independently retuned policy protocol`](reports/research_log/major_revision_v3/POLICY_RETUNING_PROTOCOL.md) and governed [`compact policy evidence package`](reports/research_log/major_revision_v3/phase1d_policy_retuning/README.md) separate feature-access sensitivity from within-policy retuning across P0–P5. Retuning improves Macro-F1 point estimates for five policies but is not uniformly favorable across criteria: P0 loses balanced accuracy, while P5 loses QWK and balanced accuracy despite improved Macro-F1 and ordinal MAE. P3 is an exact replay control. These are descriptive, noncausal point differences rather than significance or universal-policy evidence.

## Tracked manuscript-support assets

GitHub contains a deterministic, source-mapped export at [`manuscript/mdpi_information/assets/`](manuscript/mdpi_information/assets/). It was generated from the validated canonical run without refitting models or recomputing scientific evidence. The compact package contains 109 files and 10,338,351 bytes; its closed-world manifest SHA-256 is `fbe7355b956df01ad9817f27b42dc13c0f3e0e33e7f0e5c42a2477beb9d001e1`.

The package includes:

- all seven main figures and three supplementary figures in 300-DPI PNG and safe, editable SVG;
- captions and alt text for every figure;
- eight manuscript-ready main tables plus three supplementary tables in CSV and Markdown;
- exact canonical source-table copies and figure-source CSV files;
- full-precision values, four-decimal display values, denominators, intervals, uncertainty methods, and row-level source references for table records;
- figure, table, result, and claim-boundary source maps;
- a complete evidence ledger, exact-results JSON, insertion guide, and closed-world checksum manifest.

Start with the [asset guide](manuscript/mdpi_information/assets/README.md), [figure/table insertion guide](manuscript/mdpi_information/assets/handoff/figure_table_insertion_guide.md), [claim-boundary handoff](manuscript/mdpi_information/assets/handoff/claim_boundary_handoff.md), or [package manifest](manuscript/mdpi_information/assets/manifests/manuscript_asset_manifest.json).

### Main figures

| Figure | Subject | PNG | SVG |
| --- | --- | --- | --- |
| 1 | Leakage-aware audit rationale and technical protocol | [PNG](manuscript/mdpi_information/assets/figures/main/figure_01_audit_protocol.png) | [SVG](manuscript/mdpi_information/assets/figures/main/figure_01_audit_protocol.svg) |
| 2 | Four-model nested benchmark | [PNG](manuscript/mdpi_information/assets/figures/main/figure_02_model_benchmark.png) | [SVG](manuscript/mdpi_information/assets/figures/main/figure_02_model_benchmark.svg) |
| 3 | Feature-policy sensitivity | [PNG](manuscript/mdpi_information/assets/figures/main/figure_03_feature_policy_sensitivity.png) | [SVG](manuscript/mdpi_information/assets/figures/main/figure_03_feature_policy_sensitivity.svg) |
| 4 | Cross-fitted sigmoid calibration | [PNG](manuscript/mdpi_information/assets/figures/main/figure_04_calibration.png) | [SVG](manuscript/mdpi_information/assets/figures/main/figure_04_calibration.svg) |
| 5 | Exact-fold OOF grouped SHAP | [PNG](manuscript/mdpi_information/assets/figures/main/figure_05_global_grouped_shap.png) | [SVG](manuscript/mdpi_information/assets/figures/main/figure_05_global_grouped_shap.svg) |
| 6 | OOF SHAP stability | [PNG](manuscript/mdpi_information/assets/figures/main/figure_06_shap_stability.png) | [SVG](manuscript/mdpi_information/assets/figures/main/figure_06_shap_stability.svg) |
| 7 | HRDataset_v14 mapped-target replication | [PNG](manuscript/mdpi_information/assets/figures/main/figure_07_hrdataset_replication.png) | [SVG](manuscript/mdpi_information/assets/figures/main/figure_07_hrdataset_replication.svg) |

The manuscript sequence intentionally maps canonical benchmark Figure 3 to manuscript Figure 2 and canonical feature-policy Figure 2 to manuscript Figure 3. [`figure_number_mapping.csv`](manuscript/mdpi_information/assets/manifests/figure_number_mapping.csv) records the exact mapping and hashes. Figure 1 is a post-canonical presentation rendering of the frozen audit protocol; the remaining main figures are byte-for-byte canonical copies.

### Main tables

| Table | Subject | Manuscript-ready CSV |
| --- | --- | --- |
| 1 | Datasets, targets, support, and analytical roles | [CSV](manuscript/mdpi_information/assets/tables/main/table_01_datasets.csv) |
| 2 | Prespecified feature-governance policies | [CSV](manuscript/mdpi_information/assets/tables/main/table_02_feature_governance.csv) |
| 3 | Ten-by-five nested OOF four-model benchmark | [CSV](manuscript/mdpi_information/assets/tables/main/table_03_nested_benchmark.csv) |
| 4 | Matched-fold feature-policy sensitivity | [CSV](manuscript/mdpi_information/assets/tables/main/table_04_feature_policy_sensitivity.csv) |
| 5 | Raw and cross-fitted sigmoid probability metrics | [CSV](manuscript/mdpi_information/assets/tables/main/table_05_calibration.csv) |
| 6 | Grouped SHAP attribution and stability | [CSV](manuscript/mdpi_information/assets/tables/main/table_06_shap_attribution_stability.csv) |
| 7 | Support-aware subgroup and proxy diagnostics | [CSV](manuscript/mdpi_information/assets/tables/main/table_07_subgroup_proxy_diagnostics.csv) |
| 8 | HRDataset_v14 mapped-target replication | [CSV](manuscript/mdpi_information/assets/tables/main/table_08_hrdataset_replication.csv) |

The exact numerical handoff is [`manuscript_exact_results.json`](manuscript/mdpi_information/assets/handoff/manuscript_exact_results.json); the 221-row traceability ledger is [`manuscript_evidence_ledger.csv`](manuscript/mdpi_information/assets/handoff/manuscript_evidence_ledger.csv). These are aggregate result records, not employee-level observations.

## What remains local

The complete canonical evidence package contains 545 files and approximately 446.6 MB. It remains Git-ignored at `reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/`. The tracked compact export deliberately excludes:

- raw or interim datasets and employee identifiers;
- employee-level OOF predictions and local SHAP rows;
- fitted models, calibrator internals, bootstrap indices, and training partitions;
- caches, temporary directories, environments, and secrets.

Those local materials remain bound by the canonical receipt and closed-world manifests. Publication through a separate GitHub Release or repository archive requires explicit licence, redistribution, ethics, and historical-publication decisions. The compact Git package is sufficient for manuscript figures, tables, numerical cells, and scientific source tracing.

## Frozen scientific protocol

- Main identity: leakage-aware XAI audit protocol.
- Reference model: XGBoost.
- Baselines: multinomial Logistic Regression, Random Forest, and LightGBM.
- Nested evaluation: 10 outer folds × 5 inner folds.
- Primary selection metric: macro-F1.
- Secondary/tie-break metric: QWK inside the inclusive `0.001` macro-F1 tie pool.
- Calibration: predeclared cross-fitted sigmoid calibration.
- Uncertainty: paired sample-level OOF bootstrap with 5,000 draws where applicable.
- XAI: exact-fold OOF grouped SHAP from the persisted prediction-producing outer-fold model.
- Fairness: support-aware descriptive subgroup/proxy diagnostics only.
- Counterfactual evidence: supplementary heuristic-search success only.
- LLM, chatbot, and agent evaluations: excluded from the core scientific paper.
- Human evaluation: none.

The primary INX policy excludes `Age`, `Gender`, `MaritalStatus`, `EmpDepartment`, `EmpLastSalaryHikePercent`, `Attrition`, `EmpNumber`, and the target `PerformanceRating`. The conservative HRDataset_v14 primary policy excludes department, position, employment-status, marriage, diversity-job-fair and identifier fields; raw hire dates; `Salary`; `State`; `Zip`; and `RecruitmentSource`.

SHAP values are noncausal raw-margin attributions. Sensitive-field removal does not prove fairness or eliminate proxy risk. HRDataset_v14 is an independently trained mapped-target replication, not locked INX-model transport. IBM and Employee Turnover results are supplementary task-transfer/robustness evidence and are not directly comparable with the primary task.

## Data, ethics, and publication limits

[`configs/data_acquisition.yaml`](configs/data_acquisition.yaml) pins local dataset hashes, schema, dimensions, and target support. Missing local inputs fail closed; recorded mirrors are not automatically approved download sources.

Manual blockers remain:

1. ethics/IRB institution, unit, reference number, date, and determination;
2. dataset authenticity, licence, citation, and redistribution review;
3. a separately authorized strategy for historical raw/noncanonical Git blobs;
4. author approval of supported/prohibited claim wording;
5. manuscript writing, journal metadata, and public Release/Zenodo/DOI publication.

No ethics approval, licence, DOI, release URL, or citation authority is invented here. See [`08_manual_submission_blockers.md`](reports/research_log/finalization_v2/08_manual_submission_blockers.md).

## Reproduction and validation

Use CPython 3.14 with the locked dependency contract:

```powershell
py -3.14 -m venv myenv
.\myenv\Scripts\python.exe -m pip install -r requirements.txt -c constraints/py314-lock.txt
```

Canonical build entry points:

```powershell
.\myenv\Scripts\python.exe -m src.experiments.build_manuscript_evidence --config configs/manuscript_final.yaml --scope core --run-id <run-id> --no-reuse-compatible
.\myenv\Scripts\python.exe -m src.experiments.build_manuscript_evidence --config configs/manuscript_final.yaml --scope supplementary --run-id <same-run-id> --no-reuse-compatible
.\myenv\Scripts\python.exe -m src.experiments.build_manuscript_evidence --config configs/manuscript_final.yaml --validate-run-id <same-run-id>
```

The compact export requires the verified local canonical root and publishes atomically:

```powershell
.\myenv\Scripts\python.exe -m tools.canonical_manuscript_asset_export
.\myenv\Scripts\python.exe -m tools.canonical_manuscript_asset_export --validate-only
```

Routine repository validation:

```powershell
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests tools
.\myenv\Scripts\python.exe -m src.governance.ci_repository_gate --project-root .
```

Scientific execution is deny-all-network and deny-paid-API. [`.github/workflows/ci.yml`](.github/workflows/ci.yml) checks dependencies, tests, compilation, manuscript immutability, secrets, raw-data exposure, path portability, large files, issue-register integrity, and README links on pushes and pull requests.

## Reviewer and agent resumption order

Read [`AGENTS.md`](AGENTS.md), then:

1. [`CURRENT_STATUS.md`](reports/research_log/finalization/CURRENT_STATUS.md)
2. [`DECISION_LOG.md`](reports/research_log/finalization/DECISION_LOG.md)
3. [`NEXT_ACTIONS.md`](reports/research_log/finalization/NEXT_ACTIONS.md)
4. [`COMMAND_LOG.md`](reports/research_log/finalization/COMMAND_LOG.md)
5. [`TEST_LOG.md`](reports/research_log/finalization/TEST_LOG.md)
6. [`02_issue_register.csv`](reports/research_log/finalization_v2/02_issue_register.csv)
7. current branch, HEAD, `git status`, and `git diff`

Historical v1 runs, failed candidates, stage-validation packages, and early trials are audit chronology only. They must not replace the canonical identity or compact asset package documented above.

## Repository map

```text
configs/                              Frozen scope, acquisition, model, metric, and policy contracts
data/                                 Local inputs and tracked schema/provenance metadata
src/experiments/                      Core and supplementary scientific stages
src/explainability/                   Grouped-SHAP lineage and explainability code
src/governance/                       Run, provenance, claim, manifest, and publication contracts
tools/                                Deterministic compact manuscript-asset export
manuscript/mdpi_information/assets/  Tracked figures, tables, source maps, and handoff evidence
reports/manuscript_final/             Ignored complete canonical and historical evidence packages
reports/research_log/finalization/    Current interruption-safe status and logs
reports/research_log/finalization_v2/ Detailed audit, receipts, issues, and readiness records
tests/                                Unit, contract, integration, and publication-gate tests
manuscript/                           Author-owned manuscript files; technical finalization does not edit them
```

Every reported claim must trace to verified real input, the frozen protocol, correct uncertainty and claim boundaries, and artifacts from one clean run/config/commit identity.
