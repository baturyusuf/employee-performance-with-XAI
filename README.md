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
| Validation | 752 pytest passes, 2 skips, 11 subtests; 179 unittest passes, 1 skip |
| Scientific network/API use | zero attempted network operations; zero paid API calls |

The compact machine-readable receipt is [`15_canonical_evidence_receipt.json`](reports/research_log/finalization_v2/15_canonical_evidence_receipt.json). The current status, decisions, commands, tests, issues, and manual blockers are maintained under [`reports/research_log/finalization/`](reports/research_log/finalization/) and [`reports/research_log/finalization_v2/`](reports/research_log/finalization_v2/).

## GitHub-accessible publication assets

The remote branch now includes the complete aggregate publication-support subset from the frozen canonical run:

- all seven final figures in PNG and SVG;
- all seven figure-source CSV files;
- all seven technical captions;
- the figure manifest and stage contract;
- all eleven core source tables and all three supplementary source tables;
- both table manifests and stage contracts;
- core and supplementary supported/prohibited claim-boundary files.

This is a 50-file, 6,337,343-byte immutable subset. Its reproducible canonical-JSON inventory SHA-256 is `645f5295c36f2d1e0a3b7809e67d805f1bb16b906121ca91e032bc1d36e2d228`. It contains aggregate manuscript-support evidence rather than raw datasets, employee identifiers, row-level OOF/SHAP records, fitted models, caches, or secrets. The exact inventory algorithm and component hashes are recorded in [`16_github_publication_assets_receipt.json`](reports/research_log/finalization_v2/16_github_publication_assets_receipt.json).

### Final figures

| Figure | Subject | PNG | SVG | Source data | Caption |
| --- | --- | --- | --- | --- | --- |
| 1 | Study design and leakage-aware pipeline | [PNG](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/figure_1_study_design_leakage_aware_pipeline.png) | [SVG](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/figure_1_study_design_leakage_aware_pipeline.svg) | [CSV](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/source_data/figure_1_study_design_leakage_aware_pipeline_source.csv) | [caption](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/captions/figure_1_study_design_leakage_aware_pipeline_caption.md) |
| 2 | Feature-policy trade-off | [PNG](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/figure_2_feature_policy_tradeoff.png) | [SVG](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/figure_2_feature_policy_tradeoff.svg) | [CSV](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/source_data/figure_2_feature_policy_tradeoff_source.csv) | [caption](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/captions/figure_2_feature_policy_tradeoff_caption.md) |
| 3 | XGBoost versus baselines | [PNG](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/figure_3_xgboost_vs_baselines.png) | [SVG](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/figure_3_xgboost_vs_baselines.svg) | [CSV](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/source_data/figure_3_xgboost_vs_baselines_source.csv) | [caption](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/captions/figure_3_xgboost_vs_baselines_caption.md) |
| 4 | Cross-fitted sigmoid calibration | [PNG](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/figure_4_cross_fitted_sigmoid_calibration.png) | [SVG](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/figure_4_cross_fitted_sigmoid_calibration.svg) | [CSV](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/source_data/figure_4_cross_fitted_sigmoid_calibration_source.csv) | [caption](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/captions/figure_4_cross_fitted_sigmoid_calibration_caption.md) |
| 5 | Global grouped OOF SHAP | [PNG](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/figure_5_global_grouped_oof_shap.png) | [SVG](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/figure_5_global_grouped_oof_shap.svg) | [CSV](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/source_data/figure_5_global_grouped_oof_shap_source.csv) | [caption](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/captions/figure_5_global_grouped_oof_shap_caption.md) |
| 6 | OOF SHAP stability | [PNG](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/figure_6_oof_shap_stability.png) | [SVG](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/figure_6_oof_shap_stability.svg) | [CSV](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/source_data/figure_6_oof_shap_stability_source.csv) | [caption](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/captions/figure_6_oof_shap_stability_caption.md) |
| 7 | HRDataset_v14 mapped-target replication | [PNG](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/figure_7_hrdataset_v14_mapped_target_replication.png) | [SVG](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/figure_7_hrdataset_v14_mapped_target_replication.svg) | [CSV](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/source_data/figure_7_hrdataset_v14_mapped_target_replication_source.csv) | [caption](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/captions/figure_7_hrdataset_v14_mapped_target_replication_caption.md) |

Figure integrity: [`figure_manifest.json`](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/figure_manifest.json) and [`stage_contract.json`](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_figures/stage_contract.json).

### Final source tables

The CSV files contain exact canonical numerical cells and row-level provenance fields for the **table records**, not employee-level observations. Tables 3–8—the values specifically needed for model comparison, policy sensitivity, calibration, SHAP, stability, and subgroup diagnostics—are directly accessible below.

| Table | Scope and subject | Source CSV |
| --- | --- | --- |
| 1 | Dataset roles, mappings, and support | [table 1](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_tables/table_01_dataset_roles_target_mappings_support.csv) |
| 2 | Exact primary feature policy | [table 2](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_tables/table_02_exact_primary_feature_policy.csv) |
| 3 | Four-model nested benchmark | [table 3](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_tables/table_03_four_model_nested_benchmark.csv) |
| 4 | Leakage-policy sensitivity | [table 4](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_tables/table_04_leakage_policy_sensitivity.csv) |
| 5 | Cross-fitted sigmoid calibration | [table 5](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_tables/table_05_cross_fitted_sigmoid_calibration.csv) |
| 6 | Global grouped OOF SHAP | [table 6](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_tables/table_06_global_grouped_oof_shap.csv) |
| 7 | OOF SHAP stability | [table 7](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_tables/table_07_oof_shap_stability.csv) |
| 8 | Support-aware subgroup diagnostics | [table 8](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_tables/table_08_support_aware_subgroup_diagnostics.csv) |
| 9 | Department proxy reconstructability | [table 9](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_tables/table_09_department_proxy_reconstructability.csv) |
| 10 | HRDataset_v14 mapped-target replication | [table 10](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_tables/table_10_hrdataset_v14_mapped_target_replication.csv) |
| 11 | Supplementary heuristic-search success | [table 11](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/supplementary/supplementary_tables/table_11_heuristic_counterfactual_search_success.csv) |
| 12 | Supplementary restricted/binary task evidence | [table 12](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/supplementary/supplementary_tables/table_12_restricted_and_binary_task_evidence.csv) |
| 13 (core) | Reproducibility and claim boundaries | [core table 13](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_tables/table_13_reproducibility_and_claim_boundaries.csv) |
| 13 (supplementary) | Supplementary reproducibility and claim boundaries | [supplementary table 13](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/supplementary/supplementary_tables/table_13_supplementary_reproducibility_and_claim_boundaries.csv) |

Table integrity: [core table manifest](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_tables/table_manifest.json), [core stage contract](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/core_tables/stage_contract.json), [supplementary table manifest](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/supplementary/supplementary_tables/table_manifest.json), and [supplementary stage contract](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/supplementary/supplementary_tables/stage_contract.json).

Claim boundaries: [core](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core/canonical_claim_boundaries.md) and [supplementary](reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/supplementary/canonical_claim_boundaries.md).

## What remains local

The complete canonical evidence package contains 545 files and approximately 446.6 MB. The remote Git subset deliberately excludes:

- raw or interim datasets;
- employee-level OOF predictions and local SHAP rows;
- fitted model binaries and calibrator internals;
- bootstrap index arrays;
- candidate-fit, inner-fold, and training-partition records;
- caches, temporary directories, and environment files.

Those materials remain hash-bound in the compact receipt and local closed-world manifests. They belong in a separately authorized GitHub Release or Zenodo package after licence, redistribution, ethics, and historical-publication decisions. Git is used here for the aggregate manuscript-support assets needed by reviewers and authoring agents.

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

The primary INX policy excludes `Age`, `Gender`, `MaritalStatus`, `EmpDepartment`, `EmpLastSalaryHikePercent`, `Attrition`, `EmpNumber`, and the target `PerformanceRating`. The conservative HRDataset_v14 primary policy additionally excludes department/position/status/marriage/diversity aliases, identifiers, sensitive fields, raw dates, Salary, State, Zip, and RecruitmentSource.

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

Routine repository validation:

```powershell
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
.\myenv\Scripts\python.exe -m src.governance.ci_repository_gate --project-root .
```

Scientific execution is deny-all-network and deny-paid-API. The GitHub Actions workflow in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) checks dependencies, tests, compilation, manuscript immutability, secrets, raw-data exposure, path portability, large files, issue-register integrity, and README links on pushes and pull requests.

## Reviewer and agent resumption order

Read [`AGENTS.md`](AGENTS.md), then:

1. [`CURRENT_STATUS.md`](reports/research_log/finalization/CURRENT_STATUS.md)
2. [`DECISION_LOG.md`](reports/research_log/finalization/DECISION_LOG.md)
3. [`NEXT_ACTIONS.md`](reports/research_log/finalization/NEXT_ACTIONS.md)
4. [`COMMAND_LOG.md`](reports/research_log/finalization/COMMAND_LOG.md)
5. [`TEST_LOG.md`](reports/research_log/finalization/TEST_LOG.md)
6. [`02_issue_register.csv`](reports/research_log/finalization_v2/02_issue_register.csv)
7. current branch, HEAD, `git status`, and `git diff`

Historical v1 runs, failed candidates, stage-validation packages, and early trials are retained only as audit chronology. They are not canonical evidence and must not replace the paths listed in this README.

## Repository map

```text
configs/                           Frozen scope, acquisition, model, metric, and policy contracts
data/                              Local inputs and tracked schema/provenance metadata
src/experiments/                   Core and supplementary scientific stages
src/explainability/                Grouped-SHAP lineage and explainability code
src/governance/                    Run, provenance, claim, manifest, and publication contracts
reports/manuscript_final/          Canonical aggregate publication assets plus ignored full evidence
reports/research_log/finalization/ Current interruption-safe status and logs
reports/research_log/finalization_v2/ Detailed audit, receipts, issues, and readiness records
tests/                             Unit, contract, integration, and publication-gate tests
manuscript/                        Author-owned manuscript; not modified by technical finalization
```

The governing rule is simple: every reported claim must trace to verified real input, the frozen protocol, correct uncertainty and claim boundaries, and artifacts from one clean run/config/commit identity.
