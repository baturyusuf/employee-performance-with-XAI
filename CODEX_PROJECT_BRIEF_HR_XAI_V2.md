# CODEX PROJECT BRIEF — Employee Performance with XAI, v2

## 0. Role and operating mode

You are the engineering lead for this repository. Treat this document as the product, research, and engineering specification. Your job is not only to “make the code run”; your job is to help transform the repository into a publishable, Q1–Q2-level research codebase.

Work like a scientific engineering lead:
- Inspect the current repository before changing anything.
- Preserve useful existing work.
- Refactor only where it materially improves reproducibility, scientific validity, or maintainability.
- Do not blindly chase the highest accuracy.
- Every major modelling decision must be supported by experiment results, statistical checks, diagnostics, and written reasoning.
- When evidence is insufficient, stop and ask the human researcher instead of guessing.

The target research framing is:

> **Beyond Accuracy in Employee Performance Prediction: A Leakage-Aware, Actionability-Constrained, Fairness-Audited XAI Framework for HR Decision Support**

This is not primarily a “build the highest-scoring employee performance classifier” project. It is a responsible HR analytics / XAI governance project. Predictive performance is one component. The stronger contribution is showing how employee performance models should be evaluated under leakage, explanation stability, calibration, subgroup robustness, and constrained actionability.

---

## 1. Current repository assessment

The repository already has a useful structure. Do **not** recreate the project from scratch unless the audit proves the current code is irreparably inconsistent.

The current repo appears to contain:
- `src/data/` for loading, preprocessing, schema validation.
- `src/features/` for feature building, encoding, and fairness feature sets.
- `src/models/` for baseline, CatBoost, XGBoost, EBM, calibration, and evaluation.
- `src/experiments/` for advanced experiments, leakage-safe CV, leakage-safe pipeline, and robustness checks.
- `src/explainability/` for SHAP, grouped SHAP, local explanations, counterfactuals, reason codes, and fairness reporting.
- `reports/` with many existing outputs: CV benchmark, leakage ablation, leakage-safe outputs, calibration, fairness, counterfactual actionability, SHAP stability, statistical tests.
- `notebooks/` for EDA, preprocessing, baseline ordinal logit, tree models, SHAP, fairness, and counterfactuals.
- `app/` and `src/api/` for application/API surfaces.

This means the repo is already aligned with the intended paper direction. The right strategy is **controlled refactor + research hardening**, not a full rewrite.

### Main engineering decision

Use the existing structure as the foundation.

Add a small number of missing governance/research components:
- `configs/`
- `reports/research_log/`
- `reports/model_cards/`
- `reports/manuscript_assets/`
- `reports/gxair/`
- `src/governance/` or `src/audit/`
- `src/ordinal/` if ordinal modelling is added as a separate module
- `scripts/` for reproducible command-line experiment entry points, if not already available

Avoid scattering new one-off notebooks. Prefer reusable Python modules and scripts. Notebooks can remain for exploration and paper figures, but paper-critical results must be reproducible from scripts/configs.

---

## 2. Scientific objective

The final repository should support the following claim:

> Employee performance models may appear highly accurate when they exploit outcome-proximal or post-evaluation variables. A responsible HR decision-support model should therefore be evaluated not only by accuracy, but also by leakage sensitivity, explanation stability, calibration reliability, subgroup robustness, and constrained counterfactual actionability.

The codebase must make it possible to reproduce the full evidence chain behind this claim.

---

## 3. Core research questions

Implement the code and reporting workflow around these research questions:

### RQ1 — Leakage sensitivity
How much of the apparent performance is driven by potentially outcome-proximal variables such as `EmpLastSalaryHikePercent` and `Attrition`?

### RQ2 — Leakage-safe decision support
After removing leakage-risk variables, which model offers the best balance of macro-F1, ordinal agreement, calibration, and robustness?

### RQ3 — Explanation shift and stability
How do SHAP explanations change after leakage-risk variables are removed? Are the explanations stable across folds/seeds?

### RQ4 — Fairness and proxy risk
Do model errors and predictions vary across demographic or organisational subgroups? Does removing `EmpDepartment` reduce direct department dependence, and do proxy effects remain through variables such as `EmpJobRole`?

### RQ5 — Counterfactual actionability
Are counterfactual changes actually actionable by the employee, or do they require manager/organisation-level intervention?

### RQ6 — XAI governance readiness
Can we provide a transparent dashboard or score that summarizes whether a model is suitable for HR decision-support use?

---

## 4. Non-negotiable research principles

1. **Full-feature models are upper-bound baselines, not final models.**  
   If a full-feature model achieves very high accuracy because it uses `EmpLastSalaryHikePercent`, this is not a victory. It is a leakage warning.

2. **Leakage-safe results are the main results.**  
   The primary model candidates should exclude `EmpLastSalaryHikePercent` and `Attrition`. A stricter sensitivity model should also exclude `EmpDepartment`.

3. **SHAP is attribution, not causality.**  
   Do not write or generate outputs implying “this feature causes performance.” Use language like “the model attributes the prediction to...” or “the model relies on...”.

4. **Counterfactuals are intervention hypotheses, not employee prescriptions.**  
   Never generate simplistic recommendations like “the employee should change X.” Distinguish employee-controllable, manager-controllable, organisation-controllable, immutable, and forbidden variables.

5. **Model selection must be evidence-based.**  
   Do not select a model because its mean macro-F1 is slightly higher. Use cross-validation, confidence intervals, paired tests, calibration, fairness, and explanation stability.

6. **Every major decision must be logged.**  
   Maintain a decision record explaining what was tried, what results were obtained, why a choice was made, and what remains uncertain.

---

## 5. Required new documentation files

Create or update these files:

### 5.1 `RESEARCH_DECISION_LOG.md`
This is the project’s scientific decision diary. It must be updated whenever a major modelling, feature, metric, or interpretation decision is made.

Each entry should use this format:

```markdown
## YYYY-MM-DD — Decision title

### Context
What question or uncertainty required a decision?

### Alternatives considered
- Option A
- Option B
- Option C

### Evidence examined
List exact result files, metrics, tables, charts, confidence intervals, tests, or diagnostics.

### Decision
What was selected?

### Rationale
Why is this the scientifically strongest choice?

### Risks and limitations
What could still be wrong?

### Follow-up actions
What must be checked next?
```

### 5.2 `EXPERIMENT_RUNBOOK.md`
A reproducible command guide showing how to regenerate:
- data validation outputs,
- feature temporality audit,
- full-feature benchmark,
- leakage ablation,
- leakage-safe CV,
- statistical tests,
- SHAP stability,
- fairness/proxy audit,
- counterfactual actionability,
- G-XAIR dashboard,
- manuscript tables and figures.

### 5.3 `MODEL_GOVERNANCE_CARD.md`
A human-readable model card for the final decision-support model:
- intended use,
- not intended use,
- data,
- target,
- feature exclusions,
- performance,
- calibration,
- fairness warnings,
- actionability warnings,
- explanation warnings,
- deployment risks.

### 5.4 `reports/research_log/experiment_registry.csv`
A structured registry with at least:
- run_id
- date_time
- git_commit_if_available
- script
- config
- feature_set
- model
- seed
- cv_strategy
- primary_metrics
- output_dir
- notes
- decision_status: `candidate`, `rejected`, `selected`, `needs_review`

### 5.5 `reports/research_log/open_questions.md`
Questions Codex must ask the researcher when uncertain. Do not silently guess.

---

## 6. Configuration-first design

Create a central config layer. Recommended files:

```text
configs/
  project.yaml
  feature_sets.yaml
  model_grid.yaml
  evaluation.yaml
  xai.yaml
  fairness.yaml
  counterfactuals.yaml
  gxair.yaml
```

The code should avoid hard-coded feature lists, thresholds, seeds, and output paths where possible.

### 6.1 Feature set definitions

At minimum support:

```yaml
feature_sets:
  full_feature_upper_bound:
    description: "All non-ID predictors except target. Used only as an upper-bound and leakage warning."
    drop: ["EmpNumber", "PerformanceRating"]

  no_salary_hike:
    description: "Removes primary outcome-proximal compensation variable."
    drop: ["EmpNumber", "PerformanceRating", "EmpLastSalaryHikePercent"]

  no_salary_hike_no_attrition:
    description: "Primary leakage-safe decision-support feature set."
    drop: ["EmpNumber", "PerformanceRating", "EmpLastSalaryHikePercent", "Attrition"]

  no_salary_hike_no_attrition_no_department:
    description: "Fairness-aware sensitivity feature set."
    drop: ["EmpNumber", "PerformanceRating", "EmpLastSalaryHikePercent", "Attrition", "EmpDepartment"]

  no_sensitive_demographic:
    description: "Sensitivity set excluding direct demographic attributes."
    drop: ["EmpNumber", "PerformanceRating", "Gender", "MaritalStatus"]

  actionable_employee_only:
    description: "Only employee-controllable variables, if any exist after taxonomy review."
    include_by_control_type: ["employee_controllable"]

  employee_manager:
    description: "Variables controllable by employee or manager."
    include_by_control_type: ["employee_controllable", "manager_controllable"]
```

### 6.2 Feature taxonomy

Create `configs/feature_taxonomy.yaml` or equivalent with columns:

- `feature_name`
- `data_type`: numeric, categorical, ordinal, binary
- `temporal_status`: pre_evaluation, concurrent, post_evaluation_risk, unknown
- `leakage_risk`: none, low, medium, high
- `control_type`: employee_controllable, manager_controllable, organisation_controllable, immutable, forbidden, unknown
- `sensitive_or_proxy`: direct_sensitive, organisational_group, possible_proxy, not_sensitive
- `allowed_for_final_model`: true/false
- `notes`

This taxonomy is central. Use it for leakage filtering, fairness, counterfactual actionability, and reason-code warnings.

---

## 7. Required pipeline phases

Implement or harden the project through these phases.

---

### Phase 0 — Repository audit

Before coding new features, inspect:
- package layout,
- existing scripts,
- existing outputs,
- current dependency files,
- duplicated logic,
- hard-coded paths,
- whether functions are reusable,
- whether tests pass,
- whether current outputs are reproducible.

Create:

```text
reports/research_log/repo_audit.md
```

This audit must answer:
1. Which existing modules are reliable and should be reused?
2. Which modules are duplicated or inconsistent?
3. Which outputs already exist and can be regenerated?
4. What is missing for the new research framing?
5. Is a full rewrite necessary?

Expected answer unless strong evidence says otherwise: **no full rewrite; refactor and extend existing pipeline.**

---

### Phase 1 — Reproducibility and logging infrastructure

Implement:
- deterministic seeds,
- consistent output directories,
- structured logging,
- experiment registry,
- config loader,
- run metadata writer,
- versioned result folders.

Every script should write:
- config copy,
- metrics,
- metadata,
- warnings,
- timestamp,
- seed,
- feature set,
- model name,
- package versions if feasible.

Add tests for:
- config loading,
- output directory creation,
- metric calculation,
- feature set construction.

---

### Phase 2 — Data validation and feature taxonomy audit

Use existing `src/data/validate_schema.py` if reliable. Extend it if needed.

Produce:
```text
reports/data_audit/
  schema_report.csv
  target_distribution.csv
  missingness_report.csv
  duplicate_report.json
  feature_taxonomy_review.csv
  feature_temporality_audit.md
```

The temporality audit must explicitly discuss:
- why `EmpLastSalaryHikePercent` is high leakage risk,
- why `Attrition` is risky depending on prediction time,
- why `EmpDepartment` is not necessarily protected but can encode organisational group effects,
- why `EmpJobRole` may remain a proxy after removing `EmpDepartment`.

Do not delete features only because they are suspicious. The framework should compare feature sets and show the effect.

---

### Phase 3 — Robust evaluation framework

Standardise metric calculation in `src/models/evaluate.py`.

Required metrics:
- accuracy,
- balanced accuracy,
- macro-F1,
- weighted-F1,
- per-class precision/recall/F1,
- confusion matrix,
- quadratic weighted kappa,
- ordinal MAE,
- adjacent accuracy,
- severe error rate,
- log loss,
- multiclass Brier score,
- expected calibration error,
- class-wise ECE if feasible.

Evaluation protocol:
- Stratified 10-fold CV as the default.
- Repeat CV with multiple seeds if runtime allows.
- Use identical folds across models for paired comparisons.
- Fit preprocessing within each fold only. Never fit encoders/scalers on the full dataset before CV.
- For models requiring one-hot encoding, use a scikit-learn pipeline.
- For CatBoost, prefer native categorical handling if implemented cleanly.

Statistical comparison:
- bootstrap confidence intervals for key metrics,
- paired fold-level tests,
- Holm correction for multiple comparisons,
- nonparametric Wilcoxon signed-rank if distribution assumptions are questionable.

Decision rule:
- Select the final model using a multi-criteria rule, not mean macro-F1 alone.
- If two models are statistically tied on macro-F1/QWK, prefer the model with better calibration, simpler pipeline, more stable explanations, and fewer fairness/actionability risks.

---

### Phase 4 — Full-feature benchmark and leakage ablation

Implement or harden the experiment that compares:

1. `full_feature_upper_bound`
2. `no_salary_hike`
3. `no_salary_hike_no_attrition`
4. `no_salary_hike_no_attrition_no_department`

Models:
- Logistic Regression / class-balanced Logistic Regression
- Random Forest / class-balanced Random Forest
- XGBoost
- LightGBM
- CatBoost
- EBM
- optional SVM/KNN/MLP/stacking only if runtime is acceptable

The full-feature benchmark must be labelled clearly as:
> **Upper-bound / leakage-risk benchmark, not deployable final model**

Produce:
```text
reports/leakage/
  leakage_ablation_fold_metrics.csv
  leakage_ablation_summary.csv
  leakage_sensitivity_index.csv
  leakage_ablation_interpretation.md
  figures/leakage_performance_drop.png
```

### Leakage Sensitivity Index

Implement a transparent metric such as:

```text
LSI(metric) = (score_full_feature - score_leakage_safe) / abs(score_full_feature)
```

Also report absolute drop:

```text
absolute_drop = score_full_feature - score_leakage_safe
```

Use macro-F1 and QWK as primary LSI metrics.

Interpretation guide:
- high LSI = model is strongly dependent on leakage-risk variables.
- low LSI = performance is less sensitive to leakage-risk variables.

Do not overclaim causality. This is evidence of leakage/proxy dependence, not definitive proof of temporal leakage.

---

### Phase 5 — Synthetic leakage stress test

This is a key addition that can strengthen the paper.

Create a controlled experiment where artificial leakage proxies are injected into the dataset. For example:
- `synthetic_target_proxy_weak`
- `synthetic_target_proxy_medium`
- `synthetic_target_proxy_strong`

Possible construction:
- noisy copy of target,
- target-correlated ordinal variable,
- target-correlated categorical variable,
- probability-controlled proxy with adjustable noise.

Purpose:
1. Show how model performance inflates when leakage proxies exist.
2. Show whether the leakage audit detects the inflation.
3. Validate the usefulness of LSI or the proposed governance dashboard.

Produce:
```text
reports/leakage_stress_test/
  synthetic_leakage_design.md
  synthetic_leakage_results.csv
  synthetic_leakage_lsi.csv
  synthetic_leakage_interpretation.md
  figures/synthetic_leakage_curve.png
```

Decision requirement:
- If synthetic leakage does not produce a clear monotonic pattern, inspect implementation and explain why.
- Do not force the result.

---

### Phase 6 — Leakage-safe model selection

Primary candidate feature sets:
- `no_salary_hike_no_attrition`
- `no_salary_hike_no_attrition_no_department`

Primary candidate models:
- XGBoost
- LightGBM
- CatBoost
- EBM
- Logistic/ordinal baseline

Optional:
- cost-sensitive XGBoost,
- ordinal logistic regression,
- cumulative binary ordinal decomposition,
- CORAL/CORN-style ordinal models if feasible.

Model selection should produce:

```text
reports/model_selection/
  candidate_model_cv_metrics.csv
  candidate_model_summary.csv
  candidate_model_statistical_tests.csv
  model_selection_decision.md
  selected_model_config.yaml
```

The final model is not necessarily the highest macro-F1 model. It should be selected using:
- macro-F1,
- QWK,
- ordinal MAE,
- severe error rate,
- calibration,
- explanation stability,
- fairness gaps,
- counterfactual actionability,
- complexity and reproducibility.

---

### Phase 7 — Ordinal modelling enhancement

The target values `2`, `3`, and `4` are ordinal. Current multiclass modelling is acceptable as a baseline, but the repo should test whether explicitly ordinal methods improve scientific consistency.

Implement at least one of:

1. **Ordinal logistic regression**  
   Use `mord` if available, or implement a clean fallback.

2. **Cumulative binary decomposition**  
   Train two binary classifiers:
   - `P(y > 2)`
   - `P(y > 3)`  
   Then reconstruct class probabilities carefully.

3. **Cost-sensitive multiclass model**  
   Penalise severe ordinal errors more strongly than adjacent errors.

Compare with standard multiclass models using:
- QWK,
- ordinal MAE,
- adjacent accuracy,
- severe error rate,
- macro-F1.

Produce:
```text
reports/ordinal/
  ordinal_model_comparison.csv
  ordinal_model_interpretation.md
```

Decision rule:
- If ordinal models improve QWK/severe-error without unacceptable macro-F1/calibration loss, consider them strong candidates.
- If not, keep multiclass XGBoost/LightGBM/CatBoost but explain why ordinal metrics are still used.

---

### Phase 8 — Calibration diagnostics

For candidate models:
- log loss,
- multiclass Brier,
- ECE,
- class-wise calibration,
- reliability curves.

Try:
- raw probabilities,
- isotonic calibration,
- Platt/sigmoid calibration,
- temperature scaling if implemented reliably.

Produce:
```text
reports/calibration/
  calibration_metrics.csv
  calibration_bins_by_model.csv
  calibration_decision.md
  figures/reliability_curve_<model>.png
```

Decision rule:
- Do not choose calibration that improves ECE while destroying macro-F1/QWK or producing unstable probabilities.
- If calibration model requires a holdout set, ensure it is nested or properly cross-validated.

---

### Phase 9 — Explanation analysis

Use SHAP for tree models and appropriate alternatives for EBM/logistic baselines.

Required outputs:
```text
reports/xai/
  grouped_shap_global_importance.csv
  grouped_shap_by_class.csv
  shap_full_vs_safe_shift.csv
  shap_stability_summary.csv
  local_reason_codes/
  figures/
```

For XGBoost with one-hot encoding:
- group one-hot SHAP contributions back to raw feature families.
- validate that the sum of grouped SHAP values approximately equals the original total contribution.

Explanation shift:
- compare full-feature explanations vs leakage-safe explanations.
- report rank overlap, Spearman/Kendall correlation, and top-k changes.
- highlight if salary-hike dominates full-feature explanations.

Stability:
- compute top-k Jaccard for k=5, 10, 15 across CV folds/seeds.
- compute Spearman rank correlation across folds.
- optionally compute bootstrap stability.

Local explanations:
- high-confidence class 2, 3, 4 examples,
- misclassified example,
- uncertain example,
- fairness-warning example if available.

Reason codes must include:
- model confidence,
- top positive/negative factors,
- whether factors are employee/manager/organisation controllable,
- warning that SHAP is not causal,
- warning if subgroup support is low,
- warning if feature is a possible proxy.

---

### Phase 10 — Fairness and proxy analysis

Fairness is an audit, not proof of absence of discrimination.

Attributes to audit:
- `Gender`
- `MaritalStatus`
- `EmpDepartment`
- `EducationBackground`
- `BusinessTravelFrequency`
- optionally Age bins if ethically and statistically justified

Required metrics:
- group sample size,
- per-group accuracy,
- per-group macro-F1 if feasible,
- true positive rate by class,
- false positive rate by class,
- precision by class,
- positive prediction rate by class,
- calibration by group if feasible,
- max gap,
- support warnings.

Support filtering:
- default minimum group support = 30.
- report small-group warnings separately.
- do not overinterpret low-support gaps.

Proxy analysis:
- After removing `EmpDepartment`, test whether remaining features predict `EmpDepartment`.
- Use a simple classifier or mutual information/Cramér’s V to identify proxy variables.
- Pay special attention to `EmpJobRole`.

Outputs:
```text
reports/fairness/
  fairness_group_metrics.csv
  fairness_disparity_summary.csv
  small_group_warnings.csv
  fairness_bootstrap_ci.csv
  proxy_analysis_department.csv
  fairness_interpretation.md
```

Decision rule:
- Removing `EmpDepartment` is a sensitivity analysis, not proof of fairness.
- If department-free model has similar performance, keep it as a serious candidate.
- If proxy risk remains high, document it and include reason-code warnings.

---

### Phase 11 — Counterfactual actionability

Counterfactuals must be constrained by the feature taxonomy.

Intervention sets:
1. `full_default`
2. `no_salary`
3. `employee_only`
4. `employee_manager`
5. `organisation_required`
6. optional `fairness_safe_no_sensitive`

Evaluate:
- validity,
- proximity,
- sparsity,
- plausibility,
- probability gain,
- number of changes,
- control type of changed features,
- whether suggested changes are allowed,
- whether counterfactual crosses unrealistic values.

Outputs:
```text
reports/counterfactual_actionability/
  actionability_summary.csv
  actionability_summary_by_sample.csv
  counterfactual_candidates.csv
  invalid_counterfactual_reasons.csv
  actionability_interpretation.md
```

Interpretation rule:
- If employee-only validity is near zero, this is a meaningful governance finding, not a failure.
- Explain that performance shifts in this dataset may require managerial or organisational interventions.
- Do not present counterfactuals as individual employee advice.

---

### Phase 12 — G-XAIR dashboard / score

Implement a transparent governance dashboard called something like:

> **G-XAIR: Governance-aware XAI Readiness Dashboard**

Do not hide everything behind a single arbitrary number. Provide components and optionally a composite score.

Suggested components:

1. **Performance Adequacy Score, PAS**
   - based on macro-F1, QWK, ordinal MAE, severe error.

2. **Leakage Robustness Score, LRS**
   - inverse of leakage sensitivity.
   - high score means low dependence on leakage-risk features.

3. **Explanation Stability Score, ESS**
   - based on top-k Jaccard and Spearman stability.

4. **Calibration Reliability Score, CRS**
   - based on ECE/Brier/log-loss.

5. **Fairness Robustness Score, FRS**
   - based on support-filtered max gaps and uncertainty.

6. **Actionability Realism Score, ARS**
   - based on constrained counterfactual validity/plausibility under employee and manager intervention sets.

7. **Proxy Risk Penalty, PRP**
   - penalty if sensitive/organisational attributes can be reconstructed from remaining features.

Outputs:
```text
reports/gxair/
  gxair_component_scores.csv
  gxair_composite_score.csv
  gxair_weight_sensitivity.csv
  gxair_interpretation.md
  figures/gxair_radar.png
```

Important:
- Provide equal-weight and sensitivity-weighted variants.
- Do not claim the composite score is universal.
- Present it as an audit/dashboard framework.

---

## 8. Manuscript asset generation

Create reproducible tables/figures for the paper:

```text
reports/manuscript_assets/
  tables/
    table_dataset_overview.csv
    table_feature_taxonomy.csv
    table_full_feature_benchmark.csv
    table_leakage_ablation.csv
    table_leakage_safe_model_selection.csv
    table_calibration.csv
    table_shap_stability.csv
    table_fairness_gaps.csv
    table_counterfactual_actionability.csv
    table_gxair_scores.csv
  figures/
    fig_pipeline.png
    fig_leakage_drop.png
    fig_model_selection.png
    fig_calibration.png
    fig_grouped_shap_class_2.png
    fig_grouped_shap_class_3.png
    fig_grouped_shap_class_4.png
    fig_shap_stability.png
    fig_counterfactual_actionability.png
    fig_fairness_gaps.png
    fig_gxair_dashboard.png
  manuscript_results_summary.md
```

Every table should include:
- source script,
- date generated,
- feature set,
- model,
- metric definitions,
- interpretation notes.

---

## 9. Application/API guidance

The Streamlit app and API are secondary. They should not drive the research design.

If maintained:
- expose only leakage-safe selected models,
- clearly label predictions as decision support,
- show uncertainty,
- show reason codes with warnings,
- show actionability category of each explanation,
- do not show full-feature/leakage-risk predictions as deployable,
- include fairness/support warnings.

If time is limited, prioritize research pipeline and manuscript assets over app polish.

---

## 10. Suggested technologies

Use existing dependencies when possible. Try new tools only if they serve the research question.

### Core
- Python
- pandas
- numpy
- scikit-learn
- scipy
- statsmodels if useful
- matplotlib
- joblib
- pydantic or dataclasses for config validation

### Models
- XGBoost
- LightGBM
- CatBoost
- scikit-learn LogisticRegression
- RandomForestClassifier
- HistGradientBoostingClassifier
- InterpretML EBM
- optional `mord` for ordinal models

### XAI
- SHAP
- InterpretML for EBM
- permutation importance
- optional Alibi or DiCE for counterfactuals

### Fairness
- Fairlearn if it integrates cleanly
- custom group metrics if Fairlearn becomes cumbersome

### Experiment management
- MLflow if already configured and stable
- otherwise use structured CSV/JSON metadata first

### Testing
- pytest
- pandera or Great Expectations if not too heavy
- ruff/black optional

Avoid adding large new frameworks unless they directly improve reproducibility or scientific validity.

---

## 11. Required tests

Add or update tests for:

```text
tests/
  test_config_loading.py
  test_feature_sets.py
  test_feature_taxonomy.py
  test_metrics.py
  test_leakage_sensitivity.py
  test_ordinal_metrics.py
  test_grouped_shap_integrity.py
  test_fairness_support_filter.py
  test_counterfactual_constraints.py
  test_experiment_registry.py
```

Minimum guarantees:
- no target column in features,
- excluded leakage variables are actually excluded from leakage-safe models,
- preprocessing is fit inside CV folds,
- grouped SHAP preserves total contribution approximately,
- ECE and Brier calculations are correct on small synthetic examples,
- actionability constraints prevent forbidden feature changes.

---

## 12. Decision-making protocol

When you need to choose between alternatives, follow this process:

1. State the decision to be made.
2. List options.
3. Define evaluation criteria before running the experiment.
4. Run the smallest sufficient experiment.
5. Inspect:
   - mean metrics,
   - variance,
   - confidence intervals,
   - statistical tests,
   - calibration,
   - fairness,
   - explanation stability,
   - runtime/complexity,
   - failure modes.
6. Make a decision.
7. Write it to `RESEARCH_DECISION_LOG.md`.
8. If uncertainty remains high, ask the human researcher.

Never silently choose a method because it “seems better.”

---

## 13. Human-question triggers

Ask the researcher before proceeding if:

- the raw dataset is missing or multiple versions exist,
- column meanings are ambiguous,
- a feature’s temporal status is unclear,
- package installation requires major changes,
- results contradict the current paper narrative,
- runtime cost becomes excessive,
- external dataset use/licensing is unclear,
- the final model choice is statistically ambiguous,
- fairness results are concerning or difficult to interpret,
- an interpretation might imply a sensitive HR decision.

Use this question format:

```markdown
## Researcher checkpoint needed

### Issue
...

### Why it matters
...

### Options
1. ...
2. ...
3. ...

### My recommendation
...

### What I need from you
...
```

---

## 14. Immediate execution plan

### Step 1
Read this brief and inspect the repo.

### Step 2
Create `reports/research_log/repo_audit.md`.

### Step 3
Create the missing config and logging infrastructure.

### Step 4
Run or repair tests.

### Step 5
Regenerate the current baseline/leakage-safe outputs from scripts, not notebooks.

### Step 6
Compare regenerated outputs with existing `reports/` artifacts.

### Step 7
Implement missing scientific hardening:
- feature taxonomy,
- leakage sensitivity index,
- synthetic leakage stress test,
- ordinal model comparison,
- proxy analysis,
- G-XAIR dashboard,
- manuscript asset generator.

### Step 8
Write or update:
- `RESEARCH_DECISION_LOG.md`
- `EXPERIMENT_RUNBOOK.md`
- `MODEL_GOVERNANCE_CARD.md`
- `reports/manuscript_assets/manuscript_results_summary.md`

---

## 15. Acceptance criteria

This project is considered ready for paper writing only if:

1. The entire pipeline can be rerun from command-line scripts/configs.
2. Leakage-safe feature sets are verified by tests.
3. Full-feature results are clearly labelled as non-deployable upper-bound.
4. Leakage ablation is reproducible.
5. Synthetic leakage stress test is implemented and interpreted.
6. Final model selection is backed by CV, CI, statistical tests, calibration, and robustness checks.
7. SHAP explanations are grouped, class-specific, and stability-tested.
8. Fairness audit includes support filtering and small-group warnings.
9. Counterfactual analysis uses actionability constraints.
10. G-XAIR dashboard or component score is generated.
11. All major decisions are logged.
12. The final output includes manuscript-ready tables and figures.
13. The repo contains a runbook and model governance card.
14. Tests pass.

---

## 16. Final instruction

Be proactive, but not reckless. You are allowed to improve the architecture, add modules, and challenge existing choices. However, every change must serve the paper’s core thesis:

> High accuracy is not enough for employee performance AI. The model must be leakage-aware, explanation-stable, calibrated, fairness-audited, and actionability-constrained.

If you are unsure, ask. If you decide, justify. If you run experiments, record them. If a result weakens the paper’s narrative, do not hide it; analyze it scientifically and update the research direction.
