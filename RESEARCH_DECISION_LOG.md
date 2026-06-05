# Research Decision Log

This log records major scientific and engineering decisions for the leakage-aware, fairness-audited, actionability-constrained HR XAI research pipeline.

## 2026-06-05 - Preserve existing structure and harden research pipeline

### Context
The project brief requires a phase-0 audit before code changes. The repository already contains data loaders, preprocessing, model training, leakage-safe CV, leakage-safe holdout pipelines, SHAP analysis, fairness outputs, counterfactual actionability outputs, and robustness reports. The decision required was whether to rebuild the repository from scratch or preserve and refactor the current structure.

### Alternatives considered
- Full rewrite into a new research pipeline.
- Preserve the current repository with minimal patching only.
- Controlled refactor: preserve useful modules and reports, centralize repeated logic, add configuration, logging, tests, and governance documentation.

### Evidence examined
- Project brief: `CODEX_PROJECT_BRIEF_HR_XAI_V2.md`.
- Repository inventory: `src/`, `reports/`, `tests/`, `data/`, `models_artifacts/`, `app/`, `notebooks/`.
- Data check: `data/raw/inx_employee_performance.csv` has 1200 rows, 28 columns, no missing values, no duplicate `EmpNumber`, and target distribution 194/874/132 for classes 2/3/4 when parsed through the intended delimiter handling.
- Leakage-safe CV outputs: `reports/leakage_safe_cv/summary_metrics.csv`, `reports/leakage_safe_cv/statistical_tests/*`.
- Leakage ablation outputs: `reports/robustness/leakage_ablation/leakage_ablation_summary.csv`.
- SHAP stability outputs: `reports/robustness/shap_stability/shap_stability_summary.json`.
- Counterfactual actionability outputs: `reports/robustness/counterfactual_actionability/actionability_summary.csv`.
- Fairness support outputs: `reports/robustness/fairness_support/*`.
- Test status: checked-in test files are empty and `pytest` is not installed in the checked-in virtual environment.
- Dependency status: `requirements.txt` and `environment.yml` are empty; the checked-in `myenv` has packages but is not a reproducible dependency specification.

### Decision
Do not rebuild the repository from scratch. Preserve the current package layout and existing generated research outputs as the foundation, then perform a controlled research-hardening refactor.

### Rationale
The current repository already supports the core thesis: full-feature performance is much higher than leakage-safe performance, leakage-safe CV exists, grouped SHAP exists, actionability analysis exists, and fairness/support diagnostics exist. A full rewrite would discard useful evidence and increase regression risk. The scientifically strongest path is to centralize configuration, metrics, feature-set construction, metadata logging, and tests around the existing working modules.

### Risks and limitations
- Existing feature sets and metric implementations are duplicated across several modules.
- Existing reports were generated without a central experiment registry or config snapshots.
- Some existing outputs appear full-feature or leakage-risk and must not be used as final deployable evidence.
- Dependency files are empty, so another machine cannot reliably reproduce the current results.
- Tests are placeholders, so current behavior is not protected against regressions.
- The Streamlit app currently loads the older CatBoost artifact rather than leakage-safe selected artifacts.

### Follow-up actions
- Create central config files and a feature taxonomy.
- Add an experiment registry writer and run metadata utilities.
- Move metrics into `src/models/evaluate.py` and add tests.
- Replace duplicated feature-set definitions with one config-backed implementation.
- Regenerate data audit, leakage-safe CV, calibration, fairness, SHAP stability, and counterfactual actionability through scripted, logged runs before selecting any final model.

## 2026-06-05 - Add config-first infrastructure without new runtime dependency

### Context
The project needs a central config layer before refactoring duplicated feature sets, metrics, output paths, and experiment metadata. The local environment does not currently include `PyYAML` or `pytest`, and `requirements.txt` / `environment.yml` are empty.

### Alternatives considered
- Add YAML configs and immediately require `PyYAML`.
- Use JSON configs only.
- Use `.yaml` files written as JSON-compatible YAML, with a loader that uses `PyYAML` if available and falls back to Python's standard `json` parser.

### Evidence examined
- Local environment check: `yaml=False`, `pytest=False`.
- Existing brief recommends `configs/*.yaml`.
- Unit test command: `.\\myenv\\Scripts\\python.exe -m unittest discover -s tests -v`.
- Test result: 17 tests passed.

### Decision
Use `.yaml` config files with JSON-compatible YAML content for the first infrastructure layer. Add an optional `PyYAML` path in the loader but do not require it yet. Use `unittest`-compatible tests so they run in the current environment and can still be collected by `pytest` later.

### Rationale
This keeps the repository moving toward the requested config-first design without making an unreviewed environment/dependency decision. It also gives immediate test coverage for feature-set exclusion, taxonomy assumptions, metric correctness, leakage sensitivity, and registry writing.

### Risks and limitations
- JSON-compatible YAML is less readable than idiomatic block-style YAML.
- A proper dependency file still must be created.
- The taxonomy is a first draft; `Age`, `EmpDepartment`, `EducationBackground`, and `EmpJobRole` need researcher review before final-model decisions.
- Existing experiment scripts still use duplicated hard-coded feature sets until the next refactor step.

### Follow-up actions
- Decide the project Python version and dependency specification.
- Wire existing experiment scripts to the new config-backed feature-set builder.
- Generate data audit and temporality audit from the taxonomy.
- Replace duplicated metric code with `src/models/evaluate.py`.

## 2026-06-05 - Generate data audit and temporality audit before new model experiments

### Context
The project needs evidence that the raw dataset, target distribution, missingness, duplicate status, and feature temporality assumptions are documented before additional modeling decisions. This is required before leakage-safe model selection or fairness/actionability claims.

### Alternatives considered
- Continue directly to model refactoring and CV regeneration.
- Generate data audit outputs manually from notebooks.
- Add a scriptable data audit module that reuses the canonical loader and writes the required report files.

### Evidence examined
- Existing raw dataset: `data/raw/inx_employee_performance.csv`.
- Canonical loader and schema validation: `src/data/load_data.py`, `src/data/validate_schema.py`.
- New data audit script: `src/data/audit.py`.
- Generated outputs:
  - `reports/data_audit/schema_report.csv`
  - `reports/data_audit/target_distribution.csv`
  - `reports/data_audit/missingness_report.csv`
  - `reports/data_audit/duplicate_report.json`
  - `reports/data_audit/feature_taxonomy_review.csv`
  - `reports/data_audit/feature_temporality_audit.md`
- Test command: `.\\myenv\\Scripts\\python.exe -m unittest discover -s tests -v`.
- Test result: 18 tests passed.

### Decision
Use `src/data/audit.py` as the reproducible data and temporality audit entry point.

### Rationale
The script avoids ad hoc CSV parsing, reuses the repository's delimiter/BOM-safe loader, validates the schema, joins data columns to the feature taxonomy, and produces the exact report family required by the project brief. This provides a defensible evidence layer before any final model or feature decision.

### Risks and limitations
- The temporality audit is based on dataset fields and HR-process assumptions; it must be reviewed by the researcher.
- The current taxonomy is a first draft and may need changes after domain review.
- This audit does not prove leakage; it flags leakage/proxy risk for follow-up ablation and sensitivity experiments.

### Follow-up actions
- Review open taxonomy questions in `reports/research_log/open_questions.md`.
- Implement Leakage Sensitivity Index outputs from existing ablation results.
- Refactor leakage-safe CV to use central configs and registry metadata.
- Add proxy analysis for `EmpDepartment` reconstruction.

## 2026-06-05 - Derive Leakage Sensitivity Index from existing ablation outputs

### Context
The project brief requires a transparent Leakage Sensitivity Index (LSI) and leakage ablation interpretation. Existing robustness reports already contain cross-validation leakage ablation outputs, so the next question was whether to rerun models immediately or first produce a reproducible LSI wrapper from the existing evidence.

### Alternatives considered
- Rerun the full leakage ablation immediately.
- Manually calculate LSI values in the audit report.
- Add a script that derives LSI from existing ablation outputs and writes the required `reports/leakage/` artifacts.

### Evidence examined
- Source fold metrics: `reports/robustness/leakage_ablation/leakage_ablation_fold_metrics.csv`.
- Source summary: `reports/robustness/leakage_ablation/leakage_ablation_summary.csv`.
- New LSI script: `src/experiments/leakage_sensitivity.py`.
- Generated outputs:
  - `reports/leakage/leakage_ablation_fold_metrics.csv`
  - `reports/leakage/leakage_ablation_summary.csv`
  - `reports/leakage/leakage_sensitivity_index.csv`
  - `reports/leakage/leakage_ablation_interpretation.md`
  - `reports/leakage/figures/leakage_performance_drop.png`
- Test command: `.\\myenv\\Scripts\\python.exe -m unittest discover -s tests -v`.
- Test result: 19 tests passed.

### Decision
Use `src/experiments/leakage_sensitivity.py` as the current reproducible LSI report generator, with the explicit caveat that it derives from existing ablation outputs and does not rerun model training.

### Rationale
This creates the required leakage reporting artifacts without mixing infrastructure work with expensive model reruns. It makes the leakage sensitivity calculation explicit, testable, and auditable while preserving the current evidence chain.

### Risks and limitations
- The source ablation report was generated before the new config/registry layer and should be regenerated later.
- `all_features` in the existing source report reflects the source script's feature handling and should be labelled as an upper-bound/leakage-warning baseline.
- LSI is evidence of sensitivity to leakage-risk variables, not proof of causal leakage.

### Follow-up actions
- Refactor and rerun leakage ablation from central configs.
- Add synthetic leakage stress tests.
- Compare full-feature vs leakage-safe explanation shift.
- Keep full-feature models out of final deployable model consideration.

## 2026-06-05 - Apply researcher-approved Age exclusion and rerun leakage-safe CV

### Context
The researcher approved the recommended policy: exclude `Age` from final model candidates, keep age only for audit/sensitivity, and keep both department-including and department-free candidates under review until proxy/fairness evidence is available.

### Alternatives considered
- Keep existing leakage-safe feature-set definitions unchanged and only document the policy.
- Rename all final feature sets to new age-excluded names.
- Update the existing final-candidate feature-set definitions to drop `Age`, while adding explicit `*_with_age_audit` feature sets for audit-only comparisons.

### Evidence examined
- Updated config: `configs/feature_sets.yaml`.
- Updated tests: `tests/test_feature_sets.py`.
- Config-backed CV command: `.\\myenv\\Scripts\\python.exe -m src.experiments.leakage_safe_cv --feature-sets all --models all --n-splits 10 --drop-sensitive --output-dir reports/leakage_safe_cv_config`.
- Generated outputs:
  - `reports/leakage_safe_cv_config/fold_metrics.csv`
  - `reports/leakage_safe_cv_config/summary_metrics.csv`
  - `reports/leakage_safe_cv_config/statistical_tests/`
- Test command: `.\\myenv\\Scripts\\python.exe -m unittest discover -s tests -v`.
- Test result after proxy-analysis additions: 22 tests passed.

### Decision
Use age-excluded `no_salary_hike_no_attrition` and age-excluded `no_salary_hike_no_attrition_no_department` as the current final-model candidate feature sets. Keep age-included versions only as audit-only feature sets.

### Rationale
This aligns the code with the approved HR governance policy while preserving an audit path to quantify the cost of excluding age. The first config-backed CV run shows that removing age did not materially degrade the leading candidate performance relative to previous age-included leakage-safe outputs.

### Risks and limitations
- These CV results cover performance, ordinal metrics, and calibration proxies, but they do not yet include final calibration, fairness gaps, SHAP stability, counterfactual actionability, or proxy-risk penalties.
- Existing older reports under `reports/leakage_safe_cv/` used previous feature definitions and should be treated as historical reference, not the current final-candidate evidence.
- CatBoost produced training-log side effects under `catboost_info/`; future CatBoost runs should disable file writing or redirect training logs to run-specific output directories.

### Follow-up actions
- Prevent CatBoost from writing to the top-level `catboost_info/` directory in CV.
- Generate preliminary model-selection reports from `reports/leakage_safe_cv_config/`.
- Run calibration, fairness, SHAP stability, and counterfactual actionability on the age-excluded candidates.

## 2026-06-05 - Department-free model still has high department proxy risk

### Context
The approved policy keeps department-including and department-free candidates under review. To test whether removing `EmpDepartment` meaningfully reduces organisational group dependence, a proxy analysis was needed.

### Alternatives considered
- Assume department removal is sufficient for fairness sensitivity.
- Use only pairwise feature associations.
- Train a simple proxy classifier to reconstruct `EmpDepartment` from remaining final-candidate features and report feature associations.

### Evidence examined
- Proxy script: `src/experiments/proxy_analysis.py`.
- Command: `.\\myenv\\Scripts\\python.exe -m src.experiments.proxy_analysis --feature-set no_salary_hike_no_attrition_no_department --n-splits 5`.
- Outputs:
  - `reports/fairness/proxy_analysis_department.csv`
  - `reports/fairness/proxy_analysis_department_cv_fold_metrics.csv`
  - `reports/fairness/proxy_analysis_department_cv_summary.csv`
  - `reports/fairness/proxy_analysis_department_interpretation.md`
- Proxy CV summary:
  - Accuracy: 0.9825
  - Balanced accuracy: 0.9820
  - Macro-F1: 0.9724
- Top proxy association:
  - `EmpJobRole`: mutual information 1.3841, Cramer's V 0.9811
  - `EducationBackground`: mutual information 0.2119, Cramer's V 0.3664

### Decision
Do not treat department removal as sufficient evidence of fairness or low proxy risk. Keep department-free modeling as a serious candidate, but require explicit proxy warnings and downstream fairness comparisons before final selection.

### Rationale
`EmpDepartment` is almost perfectly reconstructable from the remaining features, primarily through `EmpJobRole`. Removing the direct department column reduces direct dependence but does not eliminate organisational proxy information. This is a governance finding that should strengthen the paper rather than be hidden.

### Risks and limitations
- The proxy classifier measures reconstructability, not discrimination or causal unfairness.
- High proxy risk may be structurally unavoidable if job roles are nested within departments.
- Removing `EmpJobRole` may reduce proxy risk but could also remove important job-context information; this needs an explicit sensitivity experiment.

### Follow-up actions
- Add an `EmpJobRole`-removed sensitivity feature set.
- Compare performance/fairness for department-free and job-role-free variants.
- Add reason-code proxy warnings for `EmpJobRole`.
- Ask the researcher whether job role should remain allowed with warnings or be excluded in a stricter fairness sensitivity model.

## 2026-06-05 - Evaluate job-role-free strict proxy sensitivity

### Context
The researcher approved a stricter sensitivity experiment excluding `EmpJobRole` after proxy analysis showed that `EmpDepartment` was highly reconstructable from department-free features.

### Alternatives considered
- Remove `EmpJobRole` from final candidates immediately.
- Keep `EmpJobRole` with warnings and do not quantify the cost.
- Add and evaluate a job-role-free sensitivity feature set before any final model decision.

### Evidence examined
- Updated feature set: `no_salary_hike_no_attrition_no_department_no_job_role` in `configs/feature_sets.yaml`.
- CV outputs:
  - `reports/leakage_safe_cv_job_role_sensitivity/fold_metrics.csv`
  - `reports/leakage_safe_cv_job_role_sensitivity/summary_metrics.csv`
  - `reports/leakage_safe_cv_job_role_sensitivity/statistical_tests/`
- Proxy outputs:
  - `reports/fairness/proxy_with_job_role/proxy_analysis_department_cv_summary.csv`
  - `reports/fairness/proxy_no_job_role/proxy_analysis_department_cv_summary.csv`
- Comparison report:
  - `reports/fairness/job_role_sensitivity_comparison.csv`
  - `reports/fairness/job_role_sensitivity_interpretation.md`
- Test command: `.\\myenv\\Scripts\\python.exe -m unittest discover -s tests -v`.
- Test result: 23 tests passed.

### Decision
Keep `no_salary_hike_no_attrition_no_department_no_job_role` as a strict fairness/proxy sensitivity baseline, not as the default final model candidate at this stage.

### Rationale
Removing `EmpJobRole` reduced department proxy macro-F1 from 0.9724 to 0.2412, a relative reduction of about 75.2%. However, the best job-role-free employee-performance model dropped from macro-F1 0.5987 / QWK 0.6380 to macro-F1 0.5612 / QWK 0.5430. This is a meaningful performance and ordinal-agreement cost. The scientifically strongest position is to retain this as a strict sensitivity baseline while adding proxy warnings if job role remains in candidate models.

### Risks and limitations
- The proxy classifier measures reconstructability, not discrimination.
- The job-role-free performance comparison still needs fairness, calibration, SHAP stability, and counterfactual actionability checks before any final policy conclusion.
- Removing job role may remove legitimate job-context information, so a blanket ban is not yet justified by this evidence alone.

### Follow-up actions
- Add proxy warnings for `EmpJobRole` in reason-code/model-card outputs.
- Include job-role-free sensitivity in G-XAIR proxy-risk scoring.
- Run fairness gap comparison across the three relevant feature sets:
  - `no_salary_hike_no_attrition`
  - `no_salary_hike_no_attrition_no_department`
  - `no_salary_hike_no_attrition_no_department_no_job_role`
- Continue to calibration and actionability before final model selection.

## 2026-06-05 - Feature-set fairness sensitivity shows exclusion alone is not mitigation

### Context
After the researcher approved the recommended next step, the project needed an OOF fairness comparison across the three relevant leakage-safe feature sets while holding the model fixed. The goal was to isolate feature-set effects from model-choice effects before any final model selection.

### Alternatives considered
- Continue model selection using performance and proxy results only.
- Compare subgroup metrics only for the currently best-performing feature set.
- Run a controlled 10-fold OOF fairness sensitivity using the same XGBoost protocol across all three feature sets.

### Evidence examined
- Script: `src/experiments/fairness_sensitivity.py`.
- Test: `tests/test_fairness_support_filter.py`.
- Command: `.\myenv\Scripts\python.exe -m src.experiments.fairness_sensitivity --feature-sets default --model xgboost --n-splits 10 --min-support 30`.
- Outputs:
  - `reports/fairness/feature_set_sensitivity/fairness_oof_predictions.csv`
  - `reports/fairness/feature_set_sensitivity/fairness_group_metrics.csv`
  - `reports/fairness/feature_set_sensitivity/fairness_disparity_summary.csv`
  - `reports/fairness/feature_set_sensitivity/small_group_warnings.csv`
  - `reports/fairness/feature_set_sensitivity/fairness_feature_set_sensitivity_interpretation.md`
- OOF performance with fixed XGBoost:
  - `no_salary_hike_no_attrition`: accuracy 0.8542, macro-F1 0.6036, QWK 0.6442, ordinal MAE 0.1475.
  - `no_salary_hike_no_attrition_no_department`: accuracy 0.8492, macro-F1 0.5988, QWK 0.6376, ordinal MAE 0.1525.
  - `no_salary_hike_no_attrition_no_department_no_job_role`: accuracy 0.7975, macro-F1 0.5488, QWK 0.5351, ordinal MAE 0.2050.
- Support-filtered department gaps:
  - Department-including set: EmpDepartment macro-F1 gap 0.2733, accuracy gap 0.0873.
  - Department-free set: EmpDepartment macro-F1 gap 0.2689, accuracy gap 0.0984.
  - Department/job-role-free set: EmpDepartment macro-F1 gap 0.2148, accuracy gap 0.1286.
- Class-specific department gaps remained high in multiple cases, including class-2 TPR/precision gaps and class-3 false-positive-rate gaps.
- Small-group warnings were isolated to `EmpDepartment=Data Science` (n=20) and `EducationBackground=Human Resources` (n=21); these groups were excluded from support-filtered disparity summaries.

### Decision
Do not treat removal of `EmpDepartment` or `EmpJobRole` as a fairness mitigation by itself. Keep the department-free feature set with `EmpJobRole` as a main candidate under explicit proxy and subgroup-risk warnings. Keep the job-role-free feature set as a strict fairness/proxy sensitivity baseline, not as the default final model candidate.

### Rationale
Removing `EmpDepartment` preserved utility but did not materially reduce department subgroup disparity because organisational proxy information remains, especially through `EmpJobRole`. Removing both `EmpDepartment` and `EmpJobRole` reduced some department macro-F1 disparity but caused a substantial utility loss and did not uniformly improve subgroup gaps. The scientifically defensible interpretation is that feature exclusion is diagnostic evidence, not proof of fair treatment.

### Risks and limitations
- Fairness metrics are audit diagnostics, not legal or causal discrimination claims.
- Class-4 precision gaps can be unstable because class-4 predictions are sparse even when total group support is above 30.
- This comparison used XGBoost only to isolate feature-set effects; final model selection still requires calibration, SHAP stability, actionability, and bootstrap uncertainty.
- The fairness result is sufficiently concerning that final deployment-oriented claims should wait for a researcher review and a model governance card.

### Follow-up actions
- Add bootstrap confidence intervals for subgroup gaps before final fairness claims.
- Continue with calibration diagnostics for main candidates before selecting a final model.
- Add proxy/fairness warnings to reason-code and model-card outputs.
- Include these gaps in the future G-XAIR fairness robustness and proxy-risk components.

## 2026-06-05 - Start final model selection evidence package from current config-backed state

### Context
The project entered the final model selection evidence package phase. The decision was whether to reuse current config-backed evidence or restart/regenerate the repository from scratch.

### Alternatives considered
- Restart the project and rebuild the repository structure.
- Use all existing reports equally, regardless of whether they predate the config-backed pipeline.
- Preserve the current structure, label older reports as historical, and regenerate only the final-candidate evidence needed for model selection.

### Evidence examined
- `RESEARCH_DECISION_LOG.md`.
- `configs/feature_sets.yaml`.
- Current modules under `src/experiments`, `src/models`, `src/explainability`, `src/features`, `src/data`, and `src/utils`.
- Current config-backed reports:
  - `reports/leakage_safe_cv_config/`
  - `reports/leakage_safe_cv_job_role_sensitivity/`
  - `reports/fairness/feature_set_sensitivity/`
  - `reports/fairness/proxy_with_job_role/`
  - `reports/fairness/proxy_no_job_role/`
  - `reports/fairness/job_role_sensitivity_comparison.csv`
- Historical reports under `reports/robustness/*` and older `reports/xai/*`.
- Phase-start audit: `reports/model_selection/final_evidence_phase_audit.md`.

### Decision
Preserve the existing repository structure and generate new final-candidate evidence under clearly named final output directories. Treat older robustness/XAI outputs as historical unless regenerated in this phase.

### Rationale
The current repository already contains reusable config-backed feature-set logic, CV-safe XGBoost fitting, central metrics, OOF fairness predictions, proxy analysis, and registry infrastructure. Restarting would discard useful evidence and increase regression risk. The strongest research-engineering approach is to add narrow final-evidence scripts and preserve provenance.

### Risks and limitations
- Some older outputs remain in the repository and must be labelled historical in manuscript use.
- The final evidence package still uses a public cross-sectional dataset and cannot support causal or deployment-ready claims.
- The dependency files remain incomplete, so environment reproducibility still needs a dedicated pass.

### Follow-up actions
- Generate final-candidate fairness uncertainty, calibration, SHAP stability, counterfactual actionability, dashboard, reason codes, and model card.
- Add tests for new utilities and schema checks.
- Ask the researcher for a checkpoint before manuscript table/figure generation.

## 2026-06-05 - Final model selection evidence package recommends department-free XGBoost with governance warnings

### Context
The project needed a scientifically justified final model recommendation using more than macro-F1. The active final evidence candidates were three XGBoost feature sets: department-including leakage-safe baseline, department-free primary candidate, and strict department/job-role-free proxy sensitivity baseline.

### Alternatives considered
- Select the highest macro-F1 model: `no_salary_hike_no_attrition` + XGBoost.
- Select the strictest proxy-minimizing model: `no_salary_hike_no_attrition_no_department_no_job_role` + XGBoost.
- Select the governance-aware department-free candidate: `no_salary_hike_no_attrition_no_department` + XGBoost, while retaining the other two as comparison/sensitivity baselines.

### Evidence examined
- Fairness bootstrap outputs:
  - `reports/fairness/feature_set_sensitivity/bootstrap_disparity_ci.csv`
  - `reports/fairness/feature_set_sensitivity/bootstrap_disparity_interpretation.md`
- Calibration outputs:
  - `reports/calibration/final_candidates/calibration_summary.csv`
  - `reports/calibration/final_candidates/calibration_bins.csv`
  - `reports/calibration/final_candidates/calibration_interpretation.md`
  - `reports/calibration/final_candidates/figures/`
- SHAP stability outputs:
  - `reports/xai/final_candidates/shap_stability_summary.csv`
  - `reports/xai/final_candidates/fold_feature_rankings.csv`
  - `reports/xai/final_candidates/shap_stability_interpretation.md`
- Counterfactual actionability outputs:
  - `reports/counterfactuals/final_candidates/actionability_summary.csv`
  - `reports/counterfactuals/final_candidates/actionability_by_sample.csv`
  - `reports/counterfactuals/final_candidates/actionability_interpretation.md`
- Dashboard and recommendation:
  - `reports/model_selection/final_candidate_dashboard.csv`
  - `reports/model_selection/final_candidate_dashboard.md`
  - `reports/model_selection/final_recommendation.md`
  - `reports/model_selection/model_selection_rationale.json`
- Reason-code/governance outputs:
  - `reports/xai/final_candidates/reason_code_examples/`
  - `reports/xai/final_candidates/reason_code_governance_notes.md`
- Model card:
  - `reports/model_card/hr_xai_model_card.md`
- Test command: `.\myenv\Scripts\python.exe -m unittest discover -s tests -v`.
- Test result: 29 tests passed.

### Decision
Recommend `no_salary_hike_no_attrition_no_department` + XGBoost as the primary research model. Keep `no_salary_hike_no_attrition` + XGBoost as the main leakage-safe comparison baseline. Keep `no_salary_hike_no_attrition_no_department_no_job_role` + XGBoost as the strict fairness/proxy sensitivity baseline. Full-feature models remain historical leakage-warning upper-bound baselines only.

### Rationale
The department-free candidate preserves most utility relative to the department-including baseline while excluding direct department membership: macro-F1 0.5988 vs 0.6036 and QWK 0.6376 vs 0.6442. The strict job-role-free model reduces department proxy reconstruction but has a meaningful utility and ordinal-agreement cost: macro-F1 0.5488 and QWK 0.5351. Fairness bootstrap CIs show that removing EmpDepartment alone does not materially reduce department-related disparity. SHAP stability is adequate for global discussion but EmpJobRole appears as a recurring top feature in the department-free model, requiring proxy warnings. Counterfactual actionability shows employee-only validity is 0.0000 for the primary candidate, so counterfactuals must not be framed as employee prescriptions. Calibration evidence favors sigmoid by aggregate probability-quality rank for the primary candidate, but probability bands with calibration warnings are safer than precise probability claims.

### Risks and limitations
- Department proxy risk remains high when EmpJobRole is retained: department reconstruction macro-F1 0.9724.
- Removing EmpDepartment does not prove fairness, and subgroup gaps remain concerning.
- The actionability analysis uses a prototype-search heuristic over representative samples, not a causal recourse method.
- SHAP is attribution, not causality; local explanations require governance warnings.
- Calibration was CV-safe but uses inner calibration splits; external validation is still required before deployment.
- The dataset is public and cross-sectional, so the model card prohibits autonomous HR decisions and causal claims.

### Follow-up actions
- Ask the researcher to approve whether to proceed to manuscript tables/figures, run additional robustness experiments, revise feature policy, or revise the model recommendation.
- If proceeding, generate manuscript-ready tables and figures from the final-candidate outputs.
- Add environment/dependency reproducibility documentation before paper release.
