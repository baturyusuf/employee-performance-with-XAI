
# Explainable AI-Based Employee Performance Evaluation

This project develops an Explainable Artificial Intelligence (XAI)-based decision-support system for predicting and explaining the performance level of existing employees. The system does **not** aim to make hiring decisions. Its purpose is to estimate the performance rating of current employees and explain why a model assigns a given performance level.

The project uses the **INX Future Inc Employee Performance** dataset and applies machine learning, explainable AI, leakage analysis, fairness auditing, calibration analysis, and counterfactual reasoning.

---

## Project Objective

The main research question is:

> What is the predicted performance level of an existing employee, and why did the model assign that performance level?

The target variable is:

```text
PerformanceRating ∈ {2, 3, 4}
````

where:

```text
2 = Low performance
3 = Medium performance
4 = High performance
```

The model produces:

* predicted performance class,
* class probabilities,
* global feature importance,
* local explanations,
* SHAP-based reason codes,
* counterfactual suggestions,
* fairness audit outputs,
* calibration metrics.

---

## Dataset

The project uses the **INX Future Inc Employee Performance** dataset.

The dataset contains employee-related HR variables such as:

```text
Age
Gender
EducationBackground
MaritalStatus
EmpDepartment
EmpJobRole
BusinessTravelFrequency
DistanceFromHome
EmpEnvironmentSatisfaction
EmpJobInvolvement
EmpJobSatisfaction
EmpLastSalaryHikePercent
EmpRelationshipSatisfaction
TotalWorkExperienceInYears
TrainingTimesLastYear
EmpWorkLifeBalance
ExperienceYearsAtThisCompany
ExperienceYearsInCurrentRole
YearsSinceLastPromotion
YearsWithCurrManager
Attrition
PerformanceRating
```

Dataset summary:

```text
Rows: 1200
Columns: 28
Target column: PerformanceRating
Target classes: [2, 3, 4]
```

The raw dataset should be placed under:

```text
data/raw/inx_employee_performance.csv
```

---

## Methodological Strategy

The project was developed in two main modeling phases.

### 1. Full-Information Modeling

At the beginning, all available features were used. Boosting-based models such as CatBoost, LightGBM and XGBoost achieved high performance.

However, SHAP analysis showed that:

```text
EmpLastSalaryHikePercent
```

was highly dominant in the model decisions.

Since salary hike may be determined after performance evaluation in real HR processes, this feature may introduce **target leakage** or a strong post-outcome proxy.

Therefore, the full-feature model is treated only as an:

```text
upper-bound full-information reference model
```

not as the final decision-support model.

---

### 2. Leakage-Safe Modeling

To build a more realistic and defensible model, potential leakage/proxy variables were removed.

Primary leakage-safe feature set:

```text
Removed:
- EmpLastSalaryHikePercent
- Attrition
```

Fairness-aware leakage-safe feature set:

```text
Removed:
- EmpLastSalaryHikePercent
- Attrition
- EmpDepartment
```

The final modeling strategy is:

| Model Layer                       | Purpose                                          |
| --------------------------------- | ------------------------------------------------ |
| Full-information model            | Upper-bound reference                            |
| Leakage-safe model                | Primary decision-support model                   |
| Fairness-aware leakage-safe model | Sensitivity model without department information |
| XAI models                        | Explain global and local model behavior          |
| Fairness audit                    | Evaluate subgroup-level risks                    |
| Counterfactual analysis           | Explore possible intervention scenarios          |

---

## Main Findings

### Baseline Model

A Logistic Regression baseline was first trained.

Approximate baseline performance:

```text
Accuracy: ~0.82
Macro-F1: ~0.71
Quadratic Weighted Kappa: ~0.55
```

This showed that a simple linear model was insufficient for the problem.

---

### Boosting Models

CatBoost, LightGBM and XGBoost significantly improved predictive performance.

In full-feature settings, boosting models achieved high performance. However, leakage-aware ablation showed that much of this performance was driven by `EmpLastSalaryHikePercent`.

---

### Leakage-Aware Ablation

Removing `EmpLastSalaryHikePercent` caused a major performance drop:

```text
Full model Macro-F1: approximately 0.90
Without salary hike Macro-F1: approximately 0.60
```

This confirmed that the feature has strong leakage/proxy risk.

---

### Leakage-Safe Cross-Validation

After removing `EmpLastSalaryHikePercent` and `Attrition`, 10-fold CV was performed.

Observed leakage-safe results:

| Feature Set                               | Model    | Macro-F1 |     QWK |
| ----------------------------------------- | -------- | -------: | ------: |
| no_salary_hike_no_attrition               | XGBoost  |  ~0.6020 | ~0.6403 |
| no_salary_hike_no_attrition               | LightGBM |  ~0.5859 | ~0.6036 |
| no_salary_hike_no_attrition               | CatBoost |  ~0.5567 | ~0.5791 |
| no_salary_hike_no_attrition_no_department | XGBoost  |  ~0.5965 | ~0.6330 |
| no_salary_hike_no_attrition_no_department | LightGBM |  ~0.5872 | ~0.6004 |
| no_salary_hike_no_attrition_no_department | CatBoost |  ~0.5527 | ~0.5658 |

The primary leakage-safe model candidate is:

```text
XGBoost + no_salary_hike_no_attrition
```

The fairness-aware alternative is:

```text
XGBoost + no_salary_hike_no_attrition_no_department
```

---

## Explainable AI Components

The project includes several XAI modules.

### 1. Global SHAP Analysis

Global SHAP identifies which features are most influential across the test set.

Important features after leakage-safe modeling include:

```text
EmpEnvironmentSatisfaction
YearsSinceLastPromotion
EmpWorkLifeBalance
ExperienceYearsInCurrentRole
EmpHourlyRate
EmpJobRole
```

---

### 2. Local SHAP Analysis

Local SHAP explains a single employee-level prediction.

For each selected employee, the system reports:

* true class,
* predicted class,
* predicted probability,
* positive contributors,
* negative contributors.

---

### 3. Reason Codes

SHAP values are converted into human-readable explanations.

Example:

```text
The model predicted this employee as class 3.

Main positive contributors:
- Work environment satisfaction supported the prediction.
- Work-life balance contributed positively.

Main negative contributors:
- Years since last promotion reduced the prediction score.
```

This makes model explanations easier to interpret for non-technical users.

---

### 4. Grouped SHAP for XGBoost

XGBoost uses one-hot encoded categorical variables. Therefore, SHAP values are first computed on transformed features and then grouped back into original feature names.

Example:

```text
EmpJobRole_Developer
EmpJobRole_Manager
EmpJobRole_Sales Executive
```

are grouped under:

```text
EmpJobRole
```

This produces cleaner and more interpretable SHAP plots.

---

### 5. Counterfactual Explanations

Counterfactual analysis answers:

> What would need to change for the model to assign a higher performance class?

Initial counterfactual results were strongly driven by salary hike. Therefore, actionability analysis was added.

Counterfactual actionability modes:

```text
full_default
no_salary
employee_manager
employee_only
```

The analysis showed that many valid counterfactuals depend on organizational or managerial variables rather than variables directly controlled by the employee.

Therefore, counterfactual results are interpreted as:

```text
possible intervention scenarios
```

not direct employee instructions.

---

### 6. Fairness Audit

Fairness analysis was conducted across attributes such as:

```text
Gender
MaritalStatus
EmpDepartment
EducationBackground
BusinessTravelFrequency
```

Metrics include:

```text
accuracy gap
macro-F1 gap
balanced accuracy gap
equal opportunity gap
false positive rate gap
precision gap
mean predicted probability gap
```

Small-group warnings are also generated to avoid overinterpreting groups with very low sample sizes.

---

### 7. SHAP Stability

SHAP stability analysis was performed to check whether explanations remain stable across folds.

Observed stability indicators:

```text
mean_jaccard_top5: 1.0
mean_jaccard_top10: ~0.827
mean_jaccard_top15: ~0.729
mean_spearman: ~0.855
```

This suggests that the most important features are relatively stable across different folds.

---

## Project Structure

```text
employee-performance-xai/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models_artifacts/
│   ├── catboost/
│   ├── leakage_safe/
│   └── ...
│
├── reports/
│   ├── xai/
│   ├── fairness/
│   ├── advanced_experiments/
│   ├── leakage_safe/
│   ├── leakage_safe_cv/
│   └── figures/
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── validate_schema.py
│   │
│   ├── models/
│   │   ├── train_baseline.py
│   │   └── train_catboost.py
│   │
│   ├── explainability/
│   │   ├── shap_global.py
│   │   ├── shap_local.py
│   │   ├── reason_codes.py
│   │   ├── fairness_report.py
│   │   ├── counterfactuals.py
│   │   └── xgboost_grouped_shap.py
│   │
│   ├── experiments/
│   │   ├── robustness_checks.py
│   │   ├── leakage_safe_pipeline.py
│   │   ├── leakage_safe_cv.py
│   │   └── ...
│   │
│   └── utils/
│       └── config.py
│
├── notebooks/
├── tests/
├── requirements.txt
└── README.md
```

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv myenv
```

Windows:

```bash
myenv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended main packages:

```text
pandas
numpy
scikit-learn
catboost
xgboost
lightgbm
shap
matplotlib
streamlit
joblib
scipy
```

---

## Data Validation

Run:

```bash
python -m src.data.load_data
```

Expected output:

```text
Rows: 1200
Columns: 28
Target labels found: [2, 3, 4]
Schema valid: True
```

Validated data is saved to:

```text
data/interim/
```

---

## Preprocessing

Run preprocessing with all variables:

```bash
python -m src.data.preprocess
```

Run preprocessing while dropping sensitive variables:

```bash
python -m src.data.preprocess --drop-sensitive
```

---

## Baseline Training

```bash
python -m src.models.train_baseline
```

Balanced version:

```bash
python -m src.models.train_baseline --class-weight balanced
```

Sensitive variables removed:

```bash
python -m src.models.train_baseline --drop-sensitive
```

---

## CatBoost Training

```bash
python -m src.models.train_catboost --drop-sensitive
```

With balanced class weights:

```bash
python -m src.models.train_catboost --balanced-class-weights
```

---

## Global SHAP

```bash
python -m src.explainability.shap_global
```

Optional sample size:

```bash
python -m src.explainability.shap_global --sample-size 100
```

Outputs:

```text
reports/xai/global/
```

---

## Local SHAP

```bash
python -m src.explainability.shap_local --sample-position 0
```

or:

```bash
python -m src.explainability.shap_local --sample-index 811
```

Outputs:

```text
reports/xai/local/
```

---

## Reason Codes

```bash
python -m src.explainability.reason_codes --sample-index 811
```

Outputs:

```text
reports/xai/local/sample_<id>/
```

---

## Counterfactual Explanations

```bash
python -m src.explainability.counterfactuals --sample-index 811
```

Example output:

```text
Desired class: 4
Valid counterfactuals found: 5
Recommended actions: ...
```

---

## Fairness Report

```bash
python -m src.explainability.fairness_report
```

With custom attributes:

```bash
python -m src.explainability.fairness_report --attributes Gender,MaritalStatus,EmpDepartment,EducationBackground,BusinessTravelFrequency
```

Outputs:

```text
reports/fairness/
```

---

## Leakage-Safe Pipeline

Primary leakage-safe model:

```bash
python -m src.experiments.leakage_safe_pipeline --task all --model all --feature-set no_salary_hike_no_attrition --drop-sensitive
```

Fairness-aware leakage-safe model:

```bash
python -m src.experiments.leakage_safe_pipeline --task all --model all --feature-set no_salary_hike_no_attrition_no_department --drop-sensitive
```

Outputs:

```text
reports/leakage_safe/
models_artifacts/leakage_safe/
```

---

## Leakage-Safe Cross-Validation

Run all leakage-safe models:

```bash
python -m src.experiments.leakage_safe_cv --drop-sensitive
```

Run only XGBoost:

```bash
python -m src.experiments.leakage_safe_cv --models xgboost --drop-sensitive
```

Run only CatBoost:

```bash
python -m src.experiments.leakage_safe_cv --models catboost --drop-sensitive
```

Outputs:

```text
reports/leakage_safe_cv/fold_metrics.csv
reports/leakage_safe_cv/summary_metrics.csv
reports/leakage_safe_cv/statistical_tests/
```

---

## XGBoost Grouped SHAP

Primary leakage-safe model:

```bash
python -m src.explainability.xgboost_grouped_shap --feature-set no_salary_hike_no_attrition --drop-sensitive
```

Fairness-aware leakage-safe model:

```bash
python -m src.explainability.xgboost_grouped_shap --feature-set no_salary_hike_no_attrition_no_department --drop-sensitive
```

Outputs:

```text
reports/leakage_safe/xgboost_no_salary_hike_no_attrition/shap/
reports/leakage_safe/xgboost_no_salary_hike_no_attrition_no_department/shap/
```

Important output files:

```text
grouped_summary_class_2.png
grouped_summary_class_3.png
grouped_summary_class_4.png
global_grouped_shap_importance.csv
representative_cases.csv
most_uncertain_local_explanation.png
misclassified_example_local_explanation.png
```

---

## Streamlit Application

Run:

```bash
streamlit run app/streamlit_app.py
```

The Streamlit interface provides:

* employee performance prediction,
* class probabilities,
* explanation outputs,
* local SHAP-based interpretation,
* reason-code style explanation.

---

## Current Model Recommendation

Based on the final evidence package:

```text
Primary model:
XGBoost + no_salary_hike_no_attrition_no_department
```

Main leakage-safe comparison baseline:

```text
XGBoost + no_salary_hike_no_attrition
```

Strict fairness/proxy sensitivity baseline:

```text
XGBoost + no_salary_hike_no_attrition_no_department_no_job_role
```

Full-information models are retained only as upper-bound references.

The recommended primary research model excludes EmpLastSalaryHikePercent, Attrition, and EmpDepartment. It still includes EmpJobRole, so proxy-risk warnings remain mandatory. Removing EmpDepartment does not prove fairness.

---

## LLM-Assisted Multi-Agent XAI Governance Layer

This repository now includes an optional LLM-assisted governance layer. The LLM is **not** the predictive model. The architecture remains:

```text
XGBoost predicts -> XAI explains -> audit modules evaluate reliability -> LLM/agents interpret governed evidence -> chatbot exposes guarded explanations
```

The LLM layer consumes structured evidence only:

* prediction metadata,
* grouped SHAP attribution summaries,
* leakage policy and leakage warnings,
* fairness/proxy audit summaries,
* calibration diagnostics,
* counterfactual actionability results,
* model-card and governance warnings.

The agent system audits:

* leakage-risk feature use,
* subgroup fairness and proxy risk,
* calibration reliability,
* SHAP stability and non-causal wording,
* counterfactual actionability,
* explanation faithfulness and compliance.

The chatbot is for researchers, HR analysts, auditors, and governance reviewers. It explains model evidence and limitations, but it refuses hiring, firing, promotion, compensation, disciplinary, autonomous decision, fairness-guarantee, sensitive-attribute justification, and employee-prescription requests.

Run the governance layer. For production/research-demonstration use, configure a real OpenAI API key and pass `--require-real-llm`:

```bash
.\myenv\Scripts\pip.exe install -r requirements.txt
.\myenv\Scripts\python.exe -m src.llm.check_llm_setup
.\myenv\Scripts\python.exe -m src.llm.generate_governed_explanations --provider openai --require-real-llm --limit 5
.\myenv\Scripts\python.exe -m src.agents.run_llm_governance_audit --agent-runtime openai-agents --provider openai --require-real-llm
.\myenv\Scripts\python.exe -m src.llm.cost_estimator --write-report
```

Offline/test path:

```bash
python -m src.llm.generate_governed_explanations --provider offline --limit 5
python -m src.agents.run_governance_audit
python -m src.chatbot.evaluate_chatbot
python -m src.llm.evaluate_llm_governance
python -m src.governance.gxair_score
python -m src.governance.governance_report
```

Run the guardrailed chatbot CLI:

```bash
python -m src.chatbot.chatbot_app --question "Can I trust this probability as confidence?"
```

The existing Streamlit dashboard also includes a report-backed **LLM Governance & Audit** tab after the governance reports have been generated:

```bash
streamlit run app/streamlit_app.py
```

The offline stub remains available only for tests and reproducibility. Real LLM-backed runs should use `--require-real-llm` so the system fails if the SDK or API key is missing.

Approximate LLM cost estimates are written to:

```text
reports/llm_explanations/llm_cost_estimate.md
```

The system remains a research prototype. It is not an autonomous employee evaluator and must not be used for hiring, firing, compensation, promotion, or disciplinary action without independent validation and governance review.

---

## Ethical and Practical Notes

This system should not be used as an autonomous HR decision-maker.

The model is intended as a:

```text
decision-support and explanation tool
```

not as a replacement for human judgment.

Important limitations:

* The dataset is limited in size.
* Some fairness groups have low sample counts.
* Performance labels may reflect historical organizational bias.
* Counterfactual suggestions may require organizational intervention.
* Model outputs should be reviewed by HR experts and domain stakeholders.

---

## Future Work

Planned improvements:

* leakage-safe XGBoost fairness audit,
* leakage-safe counterfactual analysis,
* calibrated XGBoost and LightGBM models,
* fairness-aware model comparison,
* richer HR-domain validation,
* external dataset validation,
* deployment-ready Streamlit dashboard,
* manuscript-ready tables and figures.

---

## Citation

If this project is used in an academic context, cite the dataset source and the main methods used:

* SHAP: SHapley Additive exPlanations
* XGBoost
* LightGBM
* CatBoost
* Counterfactual explanations
* Fairness auditing methods

Dataset citation should be added according to the original dataset provider.

---

## Project Status

Current status:

```text
Data validation: completed
Preprocessing: completed
Baseline model: completed
Boosting models: completed
Global SHAP: completed
Local SHAP: completed
Reason codes: completed
Counterfactual analysis: completed
Fairness audit: completed
Leakage-aware ablation: completed
Leakage-safe pipeline: completed
Leakage-safe CV: completed
XGBoost grouped SHAP: completed
Streamlit prototype: completed
```

Remaining priority:

```text
Leakage-safe XGBoost fairness audit
Leakage-safe counterfactual analysis
Final academic manuscript refinement
```

```
```
