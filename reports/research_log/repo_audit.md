# Repository Audit - Phase 0

Date: 2026-06-05

Git commit inspected: `6f9df0a85d26bf1519634001357c8f91340a740c`

Brief read before edits: `CODEX_PROJECT_BRIEF_HR_XAI_V2.md`

## Executive Decision

Preserve the current repository structure. Do not rebuild from scratch.

The repository already contains useful data, modeling, leakage-safe CV, SHAP, fairness, counterfactual, calibration, and robustness work. The correct strategy is controlled refactor plus research hardening: centralize configuration, logging, metrics, feature-set construction, taxonomy, tests, and manuscript outputs around the existing modules.

## Audit Method

- Inspected top-level structure, `src/`, `reports/`, `tests/`, `data/`, `models_artifacts/`, `app/`, and `notebooks/`.
- Read key modules in `src/data`, `src/models`, `src/experiments`, `src/explainability`, and `src/utils`.
- Checked generated report coverage under `reports/advanced_experiments`, `reports/leakage_safe`, `reports/leakage_safe_cv`, `reports/robustness`, `reports/fairness`, and `reports/xai`.
- Checked data shape and basic integrity through the Python environment in `myenv`.
- Attempted test execution.
- Checked dependency and config files.

## Current Repository State

### Data

- Raw dataset exists at `data/raw/inx_employee_performance.csv`.
- Dataset parsing requires delimiter handling; the raw CSV is semicolon-delimited and includes a BOM on the first column. `src/data/load_data.py` and `src/data/validate_schema.py` handle this better than ad hoc CSV reads.
- Parsed data shape: 1200 rows, 28 columns.
- Target distribution:
  - Class 2: 194
  - Class 3: 874
  - Class 4: 132
- Missing values: 0 observed in the raw parsed data.
- Duplicate `EmpNumber`: 0 observed.
- Existing processed artifacts are present under `data/processed/`, including raw and preprocessed train/test splits and `sklearn_preprocessor.joblib`.

### Code Layout

- `src/data/` contains usable loading, schema validation, and preprocessing code.
- `src/models/` contains implemented baseline and CatBoost training modules, but `evaluate.py`, `calibrate.py`, `train_xgboost.py`, and `train_ebm.py` are empty placeholders.
- `src/experiments/` contains the strongest research code: leakage-safe CV, leakage-safe holdout pipeline, advanced experiments, and robustness checks.
- `src/explainability/` contains SHAP, grouped SHAP, local explanations, reason codes, fairness reporting, and counterfactual code.
- `src/features/` is currently mostly placeholder files.
- `src/api/` is empty placeholder code.
- `app/streamlit_app.py` is implemented, but currently loads `models_artifacts/catboost/catboost_model.cbm`, not the leakage-safe selected artifacts.
- `notebooks/` files are placeholder notebooks with empty cell lists.

### Tests and Environment

- `tests/test_api.py`, `tests/test_data.py`, `tests/test_features.py`, and `tests/test_models.py` are empty.
- Running `python -m pytest -q` with `python` on PATH failed because `python.exe` is not available on PATH in this shell.
- Running `.\\myenv\\Scripts\\python.exe -m pytest -q` failed because `pytest` is not installed.
- `requirements.txt` and `environment.yml` are empty.
- The local `myenv` contains packages such as pandas, scikit-learn, CatBoost, XGBoost, LightGBM, SHAP, InterpretML, and Streamlit, but this is not a reproducible dependency specification.

## Existing Modules to Reuse

### Reuse with minor hardening

- `src/data/load_data.py`
  - Handles CSV delimiter variation and Excel files.
  - Should be retained as the canonical data loader.
- `src/data/validate_schema.py`
  - Handles column normalization, aliases, target validation, duplicate IDs, and schema checks.
  - Should be extended to produce required data audit files.
- `src/data/preprocess.py`
  - Provides train/test splitting and preprocessing utilities.
  - Should be retained, but feature selection must move to config-backed feature sets and taxonomy.
- `src/experiments/leakage_safe_cv.py`
  - Best current candidate for reproducible model comparison.
  - Uses stratified folds and fold-local fitting for model/preprocessing paths inspected.
  - Already outputs fold metrics, summaries, bootstrap CIs, and paired Wilcoxon tests with Holm correction.
- `src/experiments/leakage_safe_pipeline.py`
  - Useful for holdout diagnostics, model artifacts, calibration files, and SHAP outputs.
  - Should be converted to config/registry-based execution before being used as paper-critical evidence.
- `src/explainability/xgboost_grouped_shap.py`
  - Useful grouped SHAP implementation for one-hot encoded XGBoost.
  - Needs an explicit grouped-SHAP sum-preservation test.
- `src/experiments/robustness_checks.py`
  - Contains useful leakage ablation, calibration, SHAP stability, fairness support filtering, and counterfactual actionability code.
  - Should be split or wrapped by smaller scripts/configs because it is currently a large monolith.

### Reuse as reference, not final pipeline

- `src/models/train_baseline.py`
  - Good baseline artifact generator.
  - Metrics should be moved to a central evaluator.
- `src/models/train_catboost.py`
  - Useful full-feature CatBoost training implementation.
  - Current full-feature artifacts must remain upper-bound/leakage-warning baselines, not final deployable models.
  - Contains duplicated metric/report logic and an unreachable duplicate block in class-weight calculation.
- `src/explainability/shap_global.py`, `shap_local.py`, `shap_extended.py`, `lightgbm_shap.py`, `reason_codes.py`, `counterfactuals.py`, `fairness_report.py`
  - Useful legacy and interpretability components.
  - Need taxonomy-driven warnings and leakage-safe integration before final use.
- `app/streamlit_app.py`
  - Useful prototype.
  - Secondary priority. It must eventually expose only leakage-safe selected models with warnings.

### Placeholder or incomplete

- `src/models/evaluate.py`
- `src/models/calibrate.py`
- `src/models/train_xgboost.py`
- `src/models/train_ebm.py`
- `src/features/build_features.py`
- `src/features/encode_features.py`
- `src/features/fairness_feature_sets.py`
- `src/api/main.py`
- `src/api/routes.py`
- `src/api/schemas.py`
- `tests/*`
- `notebooks/*`

## Duplicated or Inconsistent Logic

- Feature-set definitions are repeated in `leakage_safe_cv.py`, `leakage_safe_pipeline.py`, `xgboost_grouped_shap.py`, and `robustness_checks.py`.
- Metric implementations are repeated across baseline training, CatBoost training, leakage-safe CV, leakage-safe pipeline, and robustness checks.
- Output path construction and JSON serialization helpers are repeated across modules.
- Random seeds, CV folds, model hyperparameters, feature exclusions, and output paths are hard-coded.
- `drop_sensitive` currently means only `Gender` and `MaritalStatus` in `src/utils/config.py`; the research brief requires a broader taxonomy for sensitive, organizational, proxy, leakage, and actionability attributes.
- Counterfactual actionability sets are hard-coded and not driven by a feature taxonomy.
- Existing calibration outputs under `reports/robustness/calibration/` appear tied to high-performing leakage-risk/full-feature settings and should not be treated as final leakage-safe calibration evidence until regenerated.
- Existing generated metadata often includes absolute local Windows paths, which weakens portability.

## Existing Outputs and Reproducibility Status

### Outputs already present

- `reports/advanced_experiments/cv_benchmark/`
  - Fold metrics, summary metrics, metadata.
- `reports/advanced_experiments/ablation/`
  - Ablation fold metrics, summary metrics, metadata.
- `reports/advanced_experiments/statistical_tests/`
  - Bootstrap CIs and pairwise tests for macro-F1 and QWK.
- `reports/robustness/leakage_ablation/`
  - Leakage ablation fold and summary metrics.
- `reports/leakage_safe_cv/`
  - Leakage-safe 10-fold CV fold metrics, summary metrics, bootstrap CIs, and paired tests.
- `reports/leakage_safe/`
  - Holdout diagnostics, calibration, and SHAP outputs for CatBoost, LightGBM, and XGBoost across leakage-safe feature sets.
- `reports/robustness/shap_stability/`
  - Fold feature rankings, top-k Jaccard, Spearman rank correlations, and summary JSON.
- `reports/robustness/counterfactual_actionability/`
  - Actionability summary, sample-level summary, and candidate counterfactuals.
- `reports/robustness/fairness_support/`
  - Group metrics, support summaries, and small-group warnings.
- `reports/fairness/` and `reports/fairness/formal/`
  - Legacy fairness report outputs.
- `reports/xai/`
  - Legacy global/local SHAP and counterfactual outputs.

### Evidence observed in existing reports

- Leakage ablation shows high full-feature performance:
  - `all_features` XGBoost macro-F1 about 0.9038, QWK about 0.8676.
  - `all_features` LightGBM macro-F1 about 0.9032, QWK about 0.8633.
- Leakage-safe CV shows the main performance drop:
  - `no_salary_hike_no_attrition` XGBoost macro-F1 about 0.6020, QWK about 0.6403.
  - `no_salary_hike_no_attrition_no_department` XGBoost macro-F1 about 0.5965, QWK about 0.6330.
- Existing macro-F1 paired tests show XGBoost significantly above LightGBM for `no_salary_hike_no_attrition`, but not significantly above LightGBM for `no_salary_hike_no_attrition_no_department`.
- SHAP stability summary reports:
  - mean top-5 Jaccard: 1.0
  - mean top-10 Jaccard: about 0.827
  - mean top-15 Jaccard: about 0.729
  - mean Spearman: about 0.855
- Counterfactual actionability summary reports:
  - `employee_only` validity rate: 0.0
  - `employee_manager` validity rate: 0.14
  - `full_default` validity rate: 0.97

### Regeneration status

Many outputs appear regenerable from module entry points such as:

- `python -m src.data.load_data`
- `python -m src.data.preprocess`
- `python -m src.models.train_baseline`
- `python -m src.models.train_catboost`
- `python -m src.experiments.leakage_safe_pipeline`
- `python -m src.experiments.leakage_safe_cv`
- `python -m src.experiments.robustness_checks`
- `python -m src.explainability.xgboost_grouped_shap`

However, the repo is not yet fully reproducible because dependencies are not specified, tests are empty, no central config snapshots are stored, no experiment registry existed before this audit, and outputs are not versioned by run ID.

## Missing for the New Research Framing

- `configs/` directory with project, feature-set, model, evaluation, XAI, fairness, counterfactual, and G-XAIR configs.
- `configs/feature_taxonomy.yaml` or equivalent taxonomy table.
- Central config loader and validation tests.
- Central experiment registry writer.
- Versioned output directories with metadata/config/package snapshots.
- Central metrics module in `src/models/evaluate.py`.
- Tests for feature sets, taxonomy, metrics, leakage sensitivity, ordinal metrics, grouped SHAP integrity, fairness support filtering, counterfactual constraints, and registry writing.
- Data audit outputs under `reports/data_audit/`.
- Feature temporality audit.
- Leakage Sensitivity Index output.
- Synthetic leakage stress test.
- Ordinal model comparison.
- Leakage-safe calibration decision outputs.
- Proxy analysis for `EmpDepartment` and `EmpJobRole`.
- G-XAIR dashboard/component score.
- Manuscript asset generation.
- `EXPERIMENT_RUNBOOK.md`.
- `MODEL_GOVERNANCE_CARD.md`.
- `reports/model_cards/`, `reports/manuscript_assets/`, and `reports/gxair/`.

## Preserve or Rebuild

Recommendation: preserve and refactor.

Rationale:

- The existing code already implements much of the intended paper direction.
- Existing reports provide evidence for leakage sensitivity, leakage-safe CV, statistical tests, SHAP stability, fairness support filtering, and counterfactual actionability.
- The main weakness is reproducibility and governance, not absence of useful implementation.
- A full rewrite would create avoidable regression risk and delay scientific validation.

## First Implementation Plan

Use small, reviewable commits or steps.

1. Governance skeleton
   - Add `configs/` with feature sets, evaluation defaults, model defaults, fairness defaults, counterfactual defaults, and project paths.
   - Add `reports/research_log/open_questions.md`.
   - Expand `RESEARCH_DECISION_LOG.md` and `reports/research_log/experiment_registry.csv` as work progresses.

2. Reproducibility infrastructure
   - Add a config loader and output/run metadata utilities.
   - Add experiment registry writer.
   - Add versioned output directory conventions.
   - Add package/environment snapshot writing.

3. Feature-set and taxonomy layer
   - Implement one canonical feature-set builder.
   - Add `feature_taxonomy.yaml` with temporal status, leakage risk, control type, sensitive/proxy status, and final-model eligibility.
   - Replace duplicated feature-set definitions gradually.

4. Metrics and tests
   - Move metrics from experiment scripts into `src/models/evaluate.py`.
   - Add tests for metric correctness, feature exclusion, target leakage prevention, and output directory creation.
   - Pin test dependencies and make `pytest` runnable.

5. Data audit and temporality audit
   - Generate schema, target distribution, missingness, duplicate, taxonomy review, and temporality audit outputs.
   - Explicitly document `EmpLastSalaryHikePercent`, `Attrition`, `EmpDepartment`, and `EmpJobRole`.

6. Re-run leakage-safe CV through configs
   - Reproduce current `reports/leakage_safe_cv/` results using the new registry/config system.
   - Compare regenerated results with existing outputs before trusting downstream conclusions.

7. Add leakage hardening
   - Implement Leakage Sensitivity Index.
   - Add synthetic leakage stress test.
   - Keep full-feature models labeled as upper-bound/leakage-risk only.

8. Add final-model evidence chain
   - Run leakage-safe model selection, ordinal comparison, calibration, fairness/proxy analysis, SHAP stability, and counterfactual actionability before selecting any final model.

9. Manuscript and app follow-up
   - Generate manuscript-ready tables/figures from scripts.
   - Update Streamlit/API only after final leakage-safe model selection and governance card are ready.

## Immediate Researcher Checkpoint to Schedule

Before dependency work, decide the environment target.

Recommendation: use a clean Python 3.11 or 3.12 environment with pinned dependencies rather than relying on the checked-in Python 3.14 `myenv`. This avoids compatibility risk for CatBoost, SHAP, LightGBM, XGBoost, InterpretML, and scientific test tooling.
