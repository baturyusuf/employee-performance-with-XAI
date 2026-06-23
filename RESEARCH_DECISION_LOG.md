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
- Test command: `.\\myenv\\Scripts\\python.exe -m unittest discover -s tests -v`.
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

## 2026-06-22 - LLM-assisted multi-agent XAI governance layer added as optional research prototype

### Context
The project was extended from a leakage-safe XAI model-selection repository into an LLM-assisted multi-agent XAI governance prototype. The predictive model remains XGBoost; the LLM layer is restricted to interpreting structured evidence from the existing ML/XAI pipeline.

### Alternatives considered
- Use an LLM as an additional predictor.
- Let a chatbot answer from general HR knowledge.
- Add a deterministic, evidence-bound LLM/agent layer that consumes only structured XAI and governance evidence.

### Evidence examined
- Final model-selection evidence package under `reports/model_selection/`, `reports/fairness/`, `reports/calibration/`, `reports/xai/`, and `reports/counterfactuals/`.
- Generated governed explanation examples: `reports/llm_explanations/governed_explanation_examples.md`.
- Multi-agent audit: `reports/agent_audits/multi_agent_governance_audit.md`.
- Chatbot guardrail evaluation: `reports/chatbot_eval/guardrail_evaluation.md`.
- LLM governance evaluation summary: `reports/llm_explanations/llm_governance_eval_summary.csv`.
- G-XAIR LLM-agent dashboard: `reports/governance_reports/gxair_llm_agent_dashboard.csv`.
- Final LLM-agent research summary: `reports/governance_reports/final_llm_agent_research_summary.md`.
- Test commands: `.\\myenv\\Scripts\\python.exe -m py_compile app\\streamlit_app.py` and `.\\myenv\\Scripts\\python.exe -m unittest discover -s tests -v`.
- Test result: 45 tests passed.

### Decision
Implement the LLM/agent/chatbot layer as an optional governed interpretation layer, not as a predictive model. Use the deterministic offline LLM stub as the default reproducible backend. Keep external LLM use out of the default pipeline until it passes the same faithfulness, compliance, and guardrail checks.

### Rationale
The deterministic layer preserves the project's Explainable AI and Responsible AI positioning. It provides governed explanations, multi-agent audit summaries, chatbot refusals for unsafe HR requests, and a transparent G-XAIR extension without changing the leakage-safe XGBoost predictor or inventing empirical evidence.

### Results
The automatic offline evaluation reported faithfulness pass rate 1.0, unsupported claim rate 0.0, forbidden claim rate 0.0, missing warning rate 0.0, unsafe prompt refusal rate 1.0, deterministic consistency rate 1.0, and rule-based agent agreement 1.0. These are software-governance checks, not human-subject evaluation results.

### Risks and limitations
- The offline LLM stub is deterministic and conservative; external LLM behavior remains unvalidated.
- The chatbot retrieves local project evidence and must not answer from general HR intuition. The Streamlit tab is report-backed and requires generated governance artifacts.
- Agent outputs are governance diagnostics, not legal or causal determinations.
- The system remains research-only and is not suitable for autonomous HR decisions.
- External validation and human-centered evaluation are required before any operational use.

### Follow-up actions
- Use the generated manuscript support file for Q3-level extension framing.
- If an external LLM backend is added, rerun faithfulness, missing-evidence, consistency, and unsafe-prompt evaluations.
- Refine the Streamlit governance tab after any future dashboard redesign or external LLM integration.



## 2026-06-22 - Real OpenAI-backed LLM and LLM-assisted agent path initiated

### Context
The previous LLM layer used a deterministic offline stub as the default execution mode. The project objective was clarified: the final product path must use a real LLM and a professional multi-agent governance system, while still preserving XGBoost as the predictive model and XAI/audit evidence as the only evidence source for LLM interpretation.

### Alternatives considered
- Keep the deterministic offline stub as the main layer. Rejected as insufficient for the requested final product direction.
- Use OpenAI API structured outputs with a custom evidence-bound orchestrator. Selected as the first production path because it is testable, constrained by JSON Schema, and minimally disruptive to the existing XAI pipeline.
- Move immediately to OpenAI Agents SDK. Deferred as an explicit architecture decision because it adds a larger runtime dependency and may require refactoring current audit agents into SDK tools/handoffs.
- Use LangGraph or local LLM runtimes. Deferred until the OpenAI path is stable.

### Evidence examined
- Official OpenAI structured-output documentation for JSON Schema-constrained outputs.
- Official OpenAI Agents SDK documentation for agent loops, tools, handoffs, guardrails, and tracing.
- Local environment check showed `openai` and `openai-agents` were initially missing and `OPENAI_API_KEY` was not set.

### Changes implemented
- Added OpenAI structured-output client: `src/llm/openai_client.py`.
- Added LLM runtime configuration and `.env` loading: `src/llm/runtime_config.py`.
- Added setup checker: `src/llm/check_llm_setup.py`.
- Added JSON Schema contract for governed explanations: `src/llm/output_schema.py`.
- Added client factory with `auto`, `offline`, and `openai` providers: `src/llm/client_factory.py`.
- Added LLM-assisted multi-agent orchestrator: `src/agents/llm_agent_orchestrator.py`.
- Added LLM-assisted audit runner: `src/agents/run_llm_governance_audit.py`.
- Added production roadmap and architecture decision notes: `LLM_AGENT_PRODUCTION_ROADMAP.md`.
- Added `.env.example` and updated `requirements.txt`.

### Current setup status
The Python dependencies were installed successfully into `myenv`. `openai` and `openai-agents` are now installed. `OPENAI_API_KEY` is still missing, so real LLM calls cannot run yet. The command `--require-real-llm` correctly fails when the API key is absent, preventing silent fallback to the offline stub.

### Validation
- Setup command: `.\myenv\Scripts\python.exe -m src.llm.check_llm_setup`.
- Real LLM fail-fast command: `.\myenv\Scripts\python.exe -m src.llm.generate_governed_explanations --provider openai --require-real-llm --limit 1`.
- Test command: `.\myenv\Scripts\python.exe -m unittest discover -s tests -v`.
- Test result: 50 tests passed.

### Decision
Proceed with OpenAI API structured outputs as the immediate real-LLM integration path. Keep offline stub only as a reproducibility/test fallback. Before deeper agent-runtime work, ask the researcher to choose whether to keep the custom orchestrator or migrate to OpenAI Agents SDK.

### Required user action
Set `OPENAI_API_KEY` before real LLM runs. Recommended shell setup:

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:HR_XAI_LLM_PROVIDER = "openai"
$env:HR_XAI_REQUIRE_REAL_LLM = "1"
$env:HR_XAI_OPENAI_MODEL = "gpt-5.4-mini"
```

### Next decision checkpoint
Choose the agent runtime:
- Option A: Keep custom orchestrator over deterministic audit tools plus OpenAI structured-output synthesis.
- Option B: Migrate to OpenAI Agents SDK with tools, handoffs, guardrails, sessions, and tracing.
- Option C: Use LangGraph for explicit state-machine orchestration.

## 2026-06-23 - OpenAI Agents SDK runtime, secret hygiene, and cost controls added

### Context
The researcher approved migrating from the custom LLM-assisted orchestrator toward OpenAI Agents SDK. The repository will be public, so API keys and other secrets must not be committed to source control.

### Security decision
Secrets must be supplied only through environment variables or a local `.env` file. `.env` and `.env.*` are ignored by git, while `.env.example` is kept as a safe placeholder template. No API key is embedded in code, reports, tests, or documentation.

### Changes implemented
- Added OpenAI Agents SDK runtime: `src/agents/openai_agents_runtime.py`.
- Exposed CLI selection via `--agent-runtime openai-agents` in `src/agents/run_llm_governance_audit.py`.
- Wrapped deterministic governance checks as Agents SDK function tools.
- Added Pydantic structured outputs for specialist agents and supervisor synthesis.
- Added OpenAI tracing context for real Agents SDK runs.
- Added cost estimator: `src/llm/cost_estimator.py`.
- Generated cost report: `reports/llm_explanations/llm_cost_estimate.md`.
- Strengthened `.gitignore` for `.env.*` while preserving `.env.example`.
- Changed default development model to `gpt-5.4-mini` to reduce accidental cost; `gpt-5.5` remains selectable for final high-quality artifacts.

### Cost assumptions
The cost estimator assumes eight LLM calls per case: one governed explanation, six specialist agent calls, and one supervisor call. Default estimate assumes 8,000 input tokens and 600 output tokens per call. These are planning estimates, not billing guarantees.

### Validation
- Compile command: `.\myenv\Scripts\python.exe -m py_compile src\agents\openai_agents_runtime.py src\agents\run_llm_governance_audit.py src\llm\cost_estimator.py src\llm\runtime_config.py`.
- Offline compatibility command: `.\myenv\Scripts\python.exe -m src.agents.run_llm_governance_audit --agent-runtime custom --provider offline`.
- Setup command: `.\myenv\Scripts\python.exe -m src.llm.check_llm_setup`.
- Test command: `.\myenv\Scripts\python.exe -m unittest discover -s tests -v`.
- Test result: 53 tests passed.

### Current blocker
`OPENAI_API_KEY` is still not set in the environment, so real OpenAI Agents SDK execution has not been run yet. The runtime correctly fails fast when `--require-real-llm` is requested without the key.

### Required user action
Set `OPENAI_API_KEY` locally before real LLM execution. Do not commit `.env`.

### Next action after API key
Run:

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:HR_XAI_LLM_PROVIDER = "openai"
$env:HR_XAI_REQUIRE_REAL_LLM = "1"
$env:HR_XAI_OPENAI_MODEL = "gpt-5.4-mini"
```

Then:

```bash
.\myenv\Scripts\python.exe -m src.llm.check_llm_setup
.\myenv\Scripts\python.exe -m src.agents.run_llm_governance_audit --agent-runtime openai-agents --provider openai --require-real-llm
```

## 2026-06-23 - OpenAI API key visible but real LLM run blocked by quota/billing

### Context
The API key was moved from user environment variables to system environment variables. The current Codex process does not automatically inherit newly created environment variables, so the key was read from the Windows Machine environment for the command session without printing or logging the key value.

### Evidence
- Setup checker result: OpenAI SDK installed, OpenAI Agents SDK installed, `OPENAI_API_KEY` present, provider ready.
- Real LLM smoke command: `.\myenv\Scripts\python.exe -m src.llm.generate_governed_explanations --provider openai --require-real-llm --limit 1`.
- OpenAI response: `insufficient_quota` / quota-billing unavailable.

### Changes implemented
- Added clean OpenAI API runtime error handling in `src/llm/openai_client.py`.
- Updated CLI error handling in `src/llm/generate_governed_explanations.py` and `src/agents/run_llm_governance_audit.py` so API quota/auth failures show concise messages instead of stack traces.

### Decision
Do not proceed to full LLM-agent evaluation until billing/quota is enabled or a project with available API credits is selected. The code path is ready, but real model execution is blocked by OpenAI account/project quota.

### Validation
- Test command: `.\myenv\Scripts\python.exe -m unittest discover -s tests -v`.
- Test result: 53 tests passed.

### Required user action
Check the OpenAI dashboard for billing, project budget, usage limits, and model access. After quota is available, rerun the one-case smoke test before any multi-case evaluation.

## 2026-06-23 - Real OpenAI governed explanation and Agents SDK audit succeeded

### Context
After quota was added to the OpenAI account/project, the real OpenAI smoke tests were retried using the system `OPENAI_API_KEY`. The key value was never printed or written to files.

### Evidence
- Setup checker showed OpenAI SDK installed, OpenAI Agents SDK installed, API key present, and provider ready.
- Real governed explanation command succeeded: `.\myenv\Scripts\python.exe -m src.llm.generate_governed_explanations --provider openai --require-real-llm --limit 1`.
- Governed explanation faithfulness result: pass, score 100, unsupported claims 0, forbidden claims 0, missing warnings 0.
- Real Agents SDK audit command succeeded: `.\myenv\Scripts\python.exe -m src.agents.run_llm_governance_audit --agent-runtime openai-agents --provider openai --require-real-llm`.
- Agents SDK supervisor status: research_only.
- Specialist agents completed with statuses: leakage pass/high, fairness pass_with_warnings/high, calibration pass_with_warnings/medium, SHAP stability pass_with_warnings/low, counterfactual actionability pass_with_warnings/high, explanation compliance pass/low.

### Fixes implemented after first real run
- Updated faithfulness numeric checking so rounded negative SHAP values do not become false unsupported numeric claims.
- Added mandatory warning completion in governed explanations.
- Changed OpenAI Agents SDK tools to bound no-argument tools so the LLM does not need to reconstruct large evidence JSON as a tool argument.
- Added a regression test for negative SHAP numeric tolerance.

### Validation
- Test command: `.\myenv\Scripts\python.exe -m unittest discover -s tests -v`.
- Test result: 54 tests passed.
- Secret scan over text files found no `sk-` style API key pattern.

### Decision
The real OpenAI/Agents SDK path is now functional for one-case smoke testing. Next work should add persistent token/cost logging for Agents SDK runs, then run a small 5-case evaluation before any larger batch.

## 2026-06-23 - Five-case real OpenAI LLM-agent evaluation completed

### Context
After the real one-case smoke test succeeded, the approved next phase was to add token/cost logging, run a 5-case real OpenAI evaluation, and update faithfulness, guardrail, agent consistency, and cost reports.

### Changes implemented
- Added persistent usage/cost logger: `src/llm/usage_logger.py`.
- Connected OpenAI Chat Completions governed explanation calls to the usage log.
- Connected OpenAI Agents SDK specialist/supervisor runs to the usage log.
- Added real OpenAI small-batch evaluation runner: `src/llm/run_real_llm_evaluation.py`.
- Added usage logger tests: `tests/test_llm_usage_logger.py`.
- Added mandatory leakage-warning completion to governed explanations.
- Updated warning consistency calculation to average warning-set overlap per agent across cases.

### Evidence generated
- Real OpenAI eval summary: `reports/llm_explanations/real_llm_eval_summary.md` and `.csv`.
- Usage log: `reports/llm_explanations/llm_usage_log.csv`.
- Governed explanation examples/eval: `reports/llm_explanations/governed_explanation_examples.md` and `governed_explanation_eval.csv`.
- Real OpenAI Agents SDK audit reports for cases 528, 376, 568, 18, and 392 under `reports/agent_audits/`.

### Results
- Cases: 5 (`528`, `376`, `568`, `18`, `392`).
- Faithfulness pass rate: 1.0.
- Unsupported claim rate: 0.0.
- Forbidden claim rate: 0.0.
- Missing warning rate: 0.0.
- Agent success rate: 1.0 after remediation.
- Warning consistency rate: 0.679861, averaged per agent across cases.
- Unsafe prompt refusal rate: 1.0.
- Logged usage rows: 53.
- Input tokens: 312817.
- Output tokens: 25920.
- Total tokens: 338737.
- Estimated cost: 0.267099 USD. Billing dashboard remains the source of truth.

### Remediation note
The first 5-case run identified one case-level compliance failure for case 568 because the generated explanation did not explicitly include the required full-feature leakage-warning wording. The governed explanation layer was patched to add that mandatory warning, case 568 was rerun, and all agent statuses now pass or pass_with_warnings.

### Validation
- Test command: `.\myenv\Scripts\python.exe -m unittest discover -s tests -v`.
- Test result: 56 tests passed.
- Text secret scan found no `sk-` style API key pattern.

### Decision
The real OpenAI + OpenAI Agents SDK layer is functional for small-batch governance evaluation. Do not scale beyond small batches until persistent budget limits and more granular token/cost reporting are reviewed.

## 2026-06-23 - Real OpenAI metrics interpreted for 5-case LLM-agent evaluation

### Context
The real OpenAI governed explanation and OpenAI Agents SDK audit batch had completed for five representative cases. The raw metrics were technically correct but needed research interpretation, especially the warning-consistency result.

### Evidence interpreted
- Source summary: `reports/llm_explanations/real_llm_eval_summary.csv`.
- Interpretation report: `reports/llm_explanations/real_llm_eval_interpretation.md`.
- Cases: 528, 376, 568, 18, and 392.

### Interpretation
- Faithfulness pass rate of 1.0 supports that the governed explanation layer followed structured XAI/governance evidence for this small batch.
- Unsupported claim rate, forbidden claim rate, and missing warning rate of 0.0 indicate no detected hallucinated metrics/features, forbidden HR claims, or missing mandatory warnings in the final evaluated outputs.
- Agent success rate of 1.0 shows that the OpenAI Agents SDK audit path completed with pass or pass_with_warnings for all specialist agents after case 568 remediation.
- Unsafe prompt refusal rate of 1.0 supports the tested chatbot guardrails on the current unsafe prompt set.
- Warning consistency of 0.679861 is mixed but not a failure: leakage and fairness/proxy warnings are stable model-level warnings, while counterfactual, calibration, compliance, and SHAP warnings vary by case and need taxonomy normalization before larger claims.

### Decision
Treat the five-case real OpenAI run as a successful small-batch engineering validation of the LLM interpretation and agent audit layer. Do not treat it as proof of deployment safety, fairness, causality, or human trust. The next technical improvement should be warning taxonomy normalization for case-specific agents before a larger 10-case or 20-case run.

### Validation
- New focused test command: `.\myenv\Scripts\python.exe -m unittest tests.test_real_llm_eval_interpretation -v`.
- New focused test result: 4 tests passed.
- The interpretation script made no OpenAI API calls.

### Follow-up validation
- Full test command after interpretation report: `.\myenv\Scripts\python.exe -m unittest discover -s tests -v`.
- Full test result: 60 tests passed.

## 2026-06-23 - Five follow-up LLM-agent governance stages completed

### Context
The approved follow-up work covered five stages: warning taxonomy normalization, larger real OpenAI LLM-agent evaluation, Streamlit governance interface improvement, manuscript asset generation, and stronger chatbot adversarial testing. The work preserved the core architecture: XGBoost predicts, XAI/audit modules create structured evidence, and the LLM/agents interpret and govern that evidence.

### Changes implemented
- Added canonical governance warning taxonomy in `src/governance/warning_taxonomy.py` and report writer `src/governance/warning_taxonomy_report.py`.
- Normalized warning IDs/messages in governed explanations and OpenAI/custom agent audit outputs.
- Updated real LLM warning-consistency calculation to use canonical `normalized_warning_ids` instead of raw free text.
- Expanded chatbot guardrails with prompt-injection, sensitive-attribute, autonomous-decision, Turkish HR-decision, and decision-bypass patterns.
- Expanded chatbot evaluation with unsafe, adversarial, and safe-control prompt groups.
- Improved `app/streamlit_app.py` so the LLM Governance & Audit tab remains available even when the legacy CatBoost dashboard model is missing.
- Added manuscript asset generator `src/governance/manuscript_assets.py`.
- Added supplemental reason-code builder `src/llm/build_supplemental_reason_codes.py`, using existing XGBoost local grouped SHAP tables and filtering final-policy forbidden features before LLM exposure.
- Added no-API refresh script `src/llm/refresh_real_llm_reports.py` to recompute faithfulness and summary reports from saved OpenAI outputs.
- Added project continuation handoff: `PROJECT_CONTINUATION_HANDOFF.md`.

### Real OpenAI evidence
- Requested larger batch after only five original representative cases were available.
- Built five supplemental reason-code cases from existing XGBoost local grouped SHAP evidence.
- Final real OpenAI batch cases: 528, 376, 568, 18, 392, 405, 125, 176, 662, 906.
- Runtime: OpenAI API governed explanations + OpenAI Agents SDK specialist/supervisor audits.
- Model: `gpt-5.4-mini`.
- Final refreshed metrics: faithfulness pass rate 1.0, unsupported claim rate 0.0, forbidden claim rate 0.0, missing warning rate 0.0, agent success rate 1.0, warning consistency rate 0.829497, unsafe/adversarial prompt refusal rate 1.0.

### Remediation note
The initial 10-case run produced one false compliance failure for case 176 because the faithfulness checker did not recognize the wording variant `not be employee-actionable`. The governed explanation already contained the required actionability warning. The checker was patched, a regression test was added, and saved OpenAI outputs were refreshed without additional API calls.

### Generated reports
- `reports/governance_reports/warning_taxonomy.md` and `.csv`.
- `reports/chatbot_eval/guardrail_evaluation.md` and `.csv`.
- `reports/llm_explanations/real_llm_eval_summary.md` and `.csv`.
- `reports/llm_explanations/real_llm_eval_interpretation.md`.
- `reports/manuscript_assets/llm_agent_extension_tables.md`.
- `reports/manuscript_assets/llm_agent_extension_summary.md`.
- `reports/manuscript_assets/*_table.csv`.
- `PROJECT_CONTINUATION_HANDOFF.md`.

### Validation
- Full test command: `.\myenv\Scripts\python.exe -m unittest discover -s tests -v`.
- Full test result: 67 tests passed.
- Text secret scan result: `text_secret_pattern_hits=0`.

### Decision
The LLM-agent layer is now a stronger Q3-level research prototype with real OpenAI small-batch evidence, canonical warning normalization, adversarial chatbot guardrail evaluation, manuscript-support artifacts, and a continuation handoff. It remains a research prototype and must not be represented as a deployable autonomous HR system.
