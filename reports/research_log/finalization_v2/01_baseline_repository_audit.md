# Baseline Repository Audit

Audit date: 2026-07-13
Audit mode: read-only scientific/code/artifact audit; no scientific code changed

## Git State

- Branch: `main`
- HEAD: `1c7c343bda401629a3619f92267384916f0708d0`
- Upstream: `origin/main`, ahead 0 / behind 0
- Worktree before audit: clean
- Latest commit: `update`, 809 changed files and approximately 940,655 insertions
- Tracked files: 1,850

## Repository and Publication Size

- Tracked working content: 629,617,692 bytes (about 601 MiB)
- `reports/`: about 586 MiB
- `reports/manuscript_final/latest`: about 158.56 MiB and is a physical mirror
- Successful versioned v1 run: about 158.56 MiB
- The mirror and versioned run contain 215 byte-identical files; `latest` adds only `run_pointer.json`
- No single tracked file exceeds 100 MiB; 55 files exceed 1 MiB and total about 530.84 MiB
- The largest repeated artifacts are LLM audit CSV/JSONL files and local SHAP evidence

## Runtime Baseline

- Python: 3.14.0 (CPython, Windows 11)
- numpy 2.4.4; pandas 3.0.2; scipy 1.17.1; scikit-learn 1.8.0
- xgboost 3.2.0; shap 0.51.0; matplotlib 3.10.8
- lightgbm 4.6.0; catboost 1.2.10
- openai 2.43.0; openai-agents 0.17.6; pytest 9.1.1
- `xlrd` and `openpyxl` are not installed in the current environment despite being declared in broad dependency ranges.
- No exact dependency lock or constraints file exists.

## Dataset Files Observed

| Logical input | Path | SHA-256 | Rows x columns |
| --- | --- | --- | --- |
| INX configured raw CSV | `data/raw/inx_employee_performance.csv` | `b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a` | 1200 x 28 |
| INX implicit interim CSV | `data/interim/inx_employee_performance_validated.csv` | `afc97e8b8aba4bae1e1ca62e17c8c0c18b73f2c6e53c846ffbee2d0d4c495a7e` | 1200 x 28 |
| HRDataset_v14 | `data/external/hrdataset_v14/raw.csv` | `cb19996755c93c0a8d6527f59da4701c80aef65eff854906546dce286249813c` | 311 x 36 |
| IBM HR Analytics | `data/external/ibm_hr_analytics/raw.csv` | `a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659` | 1470 x 35 |
| Employee Turnover | `data/external/employee_turnover/raw.csv` | `2510e274a90547f34c7b0db5a4ab70282c2710eb54252f14921bf980b81a928c` | 14999 x 10 |

The raw INX CSV contains a BOM-prefixed identifier column while the interim file contains normalized `EmpNumber`. Their byte hashes and direct parser frames differ, but a normalized cell-level comparison found the same 1,200 x 28 scientific table, target distribution and values. This reduces concern about current numeric drift, but does not repair the false actual-input provenance/cache binding. All raw, interim, processed, workbook, and mapping files are currently tracked by Git.

INX workbook: `data/raw/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls`, 410,624 bytes, SHA-256 `d7d224e7c8f50693e0aba7621e56a55a84b0c2c156ff95afd2b156daa5d0f003`. A read-only Excel COM audit found the first sheet (`INX_Future_Inc_Employee_Perform`) equivalent to the CSV after normalization: 1,200 data rows, 28 columns, identical cells and zero duplicate employee IDs. This is an audit observation, not yet reproducible evidence: no repository script/test exists, `xlrd`/`openpyxl` are absent, and the `Data Definitions` sheet is not provenance-bound.

External mapped support was verified: HRDataset_v14 `{2: 31, 3: 243, 4: 37}` with no unmapped target; IBM performance `{3: 1244, 4: 226}`; IBM attrition `{0: 1233, 1: 237}`; Employee Turnover `{0: 11428, 1: 3571}`. Current schema-mapping SHA-256 prefixes are HR `4988bde1`, IBM `49f97b70`, and turnover `ef49bf69`; none is bound to the run-level cache contract.

## Existing v1 Canonical Package Rejection

The package at `reports/manuscript_final/latest` is historical v1 evidence, not acceptable v2 evidence:

- Manifest run ID: `manuscript_final_20260712T181754Z_c664ef152ff3`
- Manifest Git commit: `18347488bdb4eed60f115ceeff70c420071ceef0`, not current HEAD
- Manifest records `git_worktree_dirty=true`
- Entrypoint command remains `status=started`, `ended_at=null`, `return_code=null`
- The manifest is overall `complete`, so command completion is not validated
- Manifest hashes the raw INX CSV while core stages silently read the interim file when it exists
- Absolute `C:\Users\Yusuf\...` paths occur in the manifest and many metadata/CSV artifacts
- Core manifest contains LLM, agent, chatbot, and counterfactual artifacts that are excluded from the new core scope

## Protocol Findings

- `src/data/preprocess.py:20-29` implements an implicit interim-first loader. Policy, calibration, SHAP, fairness, and counterfactual core stages call it.
- External `schema_mapping.json` inputs are consumed but not included in the explicit input snapshot/hash chain.
- Cache validation does not compare Git commit and does not bind actual loader-returned input identity.
- Calibration ranks raw/sigmoid/isotonic using outer-test log loss, Brier, and ECE; sigmoid is not currently predeclared independently of final test results.
- Policy/calibration/proxy/external summaries use fold-mean t intervals. Policy differences use ten-fold Wilcoxon tests. Required paired sample-level OOF bootstrap is absent.
- SHAP stability treats 45 dependent fold pairs as independent for t intervals.
- No canonical three-baseline stage exists. The legacy logistic baseline is an 80/20 holdout.
- Stages regenerate nominally identical folds independently instead of consuming one hashed fold assignment.
- Current OOF SHAP implementation does use each test sample's fold-specific model and has deterministic grouped-axis/forbidden-feature checks; these parts are reusable.
- Support-aware subgroup bootstrap and proxy-target removal exist, but proxy performance uncertainty still uses fold t intervals.
- Counterfactual is OOF, but remains core, uses actionability/validity terminology, contains redundant `no_salary`, and has no search-budget sensitivity analysis.
- All external tasks are combined in one core stage rather than split into HRDataset core and IBM/turnover supplementary evidence.
- Figures 1-4 and README still foreground LLM/agent/chatbot governance, contradicting the fixed v2 scope.

## Tests and CI

- `pytest`: 188 passed plus 4 subtests in 14.75 s with paid-service environment variables removed.
- `unittest`: 161 passed in 4.764 s with paid-service environment variables removed.
- `compileall`: passed.
- Worktree stayed clean after tests.
- These tests do not detect the confirmed actual-input, calibration-selection, path-portability, command-finalization, uncertainty, or core-isolation defects.
- `.github/workflows` does not exist.
- GitHub connector returned zero workflow runs for current HEAD; there is no green CI evidence.
- No global socket/network blocker exists for the test suite.

## Data and Legal Status

Every dataset card currently marks source authenticity, licence, and citation verification as `manual_review_required`. Raw files are already present in public Git history, so `git rm --cached` alone can stop future tip tracking but cannot remove historical distribution.
