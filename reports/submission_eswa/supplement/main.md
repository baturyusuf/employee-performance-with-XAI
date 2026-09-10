# Supplementary material

**Author-review draft — not for submission. Mandatory author, institutional, rights, and portal decisions remain open.**

This anonymous scientific supplement reports method contracts and aggregate evidence for the accompanying manuscript. It contains no author identity, employee-level record, fitted model, approval history, internal review, or development diary. Exact, unabridged versions of Tables S1--S4 and S6--S9 are supplied as CSV files in the `tables` directory.

# Table S1. Feature-availability and governance matrix

`I` denotes included and `E` excluded. A question mark marks plausible pre-existence without a verified row-level timestamp. Full semantic descriptions, governance classes, justifications, and timestamp caveats appear in `tables/Table_S1.csv`.

## Table S1a. Timing and risk classification

| Feature | Timing | Risk |
| --- | --- | --- |
| EmpNumber | Identifier | Identifier |
| Age | Pre-existing? | Sensitive |
| Gender | Pre-existing? | Sensitive |
| EducationBackground | Pre-existing? | Ordinary |
| MaritalStatus | Pre-existing? | Sensitive |
| EmpDepartment | Pre-existing? | Org. proxy |
| EmpJobRole | Pre-existing? | Org. proxy |
| BusinessTravelFrequency | Pre-existing? | Org. proxy |
| DistanceFromHome | Pre-existing? | Ordinary |
| EmpEducationLevel | Pre-existing? | Ordinary |
| EmpEnvironmentSatisfaction | Uncertain | Timing |
| EmpHourlyRate | Pre-existing? | Org. proxy |
| EmpJobInvolvement | Uncertain | Timing |
| EmpJobLevel | Pre-existing? | Org. proxy |
| EmpJobSatisfaction | Uncertain | Timing |
| NumCompaniesWorked | Pre-existing? | Ordinary |
| OverTime | Uncertain | Timing |
| EmpLastSalaryHikePercent | May postdate | Outcome-near |
| EmpRelationshipSatisfaction | Uncertain | Timing |
| TotalWorkExperienceInYears | Pre-existing? | Ordinary |
| TrainingTimesLastYear | Uncertain | Timing |
| EmpWorkLifeBalance | Uncertain | Timing |
| ExperienceYearsAtThisCompany | Pre-existing? | Ordinary |
| ExperienceYearsInCurrentRole | Pre-existing? | Org. proxy |
| YearsSinceLastPromotion | Pre-existing? | Org. proxy |
| YearsWithCurrManager | Pre-existing? | Org. proxy |
| Attrition | May postdate | temporal leakage |
| PerformanceRating | unavailable by design target | target direct leakage |

## Table S1b. Policy inclusion matrix

| Feature | P0 | P1 | P2 | P3 | P4 | P5 |
| --- | --- | --- | --- | --- | --- | --- |
| EmpNumber | E | E | E | E | E | E |
| Age | I | I | E | E | E | E |
| Gender | I | I | E | E | E | E |
| EducationBackground | I | I | I | I | I | I |
| MaritalStatus | I | I | E | E | E | E |
| EmpDepartment | I | I | I | E | E | E |
| EmpJobRole | I | I | I | I | I | E |
| BusinessTravelFrequency | I | I | I | I | I | E |
| DistanceFromHome | I | I | I | I | I | I |
| EmpEducationLevel | I | I | I | I | I | I |
| EmpEnvironmentSatisfaction | I | I | I | I | E | E |
| EmpHourlyRate | I | I | I | I | I | E |
| EmpJobInvolvement | I | I | I | I | E | E |
| EmpJobLevel | I | I | I | I | I | E |
| EmpJobSatisfaction | I | I | I | I | E | E |
| NumCompaniesWorked | I | I | I | I | I | I |
| OverTime | I | I | I | I | E | E |
| EmpLastSalaryHikePercent | I | E | E | E | E | E |
| EmpRelationshipSatisfaction | I | I | I | I | E | E |
| TotalWorkExperienceInYears | I | I | I | I | I | I |
| TrainingTimesLastYear | I | I | I | I | E | E |
| EmpWorkLifeBalance | I | I | I | I | E | E |
| ExperienceYearsAtThisCompany | I | I | I | I | I | I |
| ExperienceYearsInCurrentRole | I | I | I | I | I | E |
| YearsSinceLastPromotion | I | I | I | I | I | E |
| YearsWithCurrManager | I | I | I | I | I | E |
| Attrition | I | E | E | E | E | E |
| PerformanceRating | E | E | E | E | E | E |

# Table S2. Preprocessing and search contract

All preprocessing and candidate selection are confined to the current training partition. Exact estimator paths, fixed parameters, candidate grids, tolerances, and failure rules appear in `tables/Table_S2.csv`.

| Type | Component | Candidates | Fit scope | Primary | Tie break | Outer test |
| --- | --- | --- | --- | --- | --- | --- |
| preprocessing | Numeric preprocessing | 0 | Training partitions only | not applicable | not applicable | Evaluation only |
| preprocessing | Categorical preprocessing | 0 | Training partitions only | not applicable | not applicable | Evaluation only |
| trained model | Multinomial Logistic Regression | 6 | Training partitions only | macro f1 | quadratic weighted kappa | Evaluation only |
| trained model | Random Forest | 8 | Training partitions only | macro f1 | quadratic weighted kappa | Evaluation only |
| trained model | LightGBM | 8 | Training partitions only | macro f1 | quadratic weighted kappa | Evaluation only |
| trained model | XGBoost | 8 | Training partitions only | macro f1 | quadratic weighted kappa | Evaluation only |
| trained model | Proportional-Odds Ordinal Logistic Regression | 6 | Training partitions only | macro f1 | quadratic weighted kappa | Evaluation only |
| trained model | Cumulative-Threshold XGBoost | 8 | Training partitions only | macro f1 | quadratic weighted kappa | Evaluation only |
| selection regime | macro f1 selection |  | Training partitions only | macro f1 | quadratic weighted kappa; then lowest candidate index | Evaluation only |
| selection regime | qwk selection |  | Training partitions only | quadratic weighted kappa | macro f1; then lowest candidate index | Evaluation only |

# Table S3. Cross-validation overview

Exact seed schedules, fold identities, out-of-fold coverage, calibration isolation, selection-score sources, and reuse/refit boundaries appear in `tables/Table_S3.csv`.

| Analysis | Population | n | Outer design | Inner design | Outer test |
| --- | --- | --- | --- | --- | --- |
| Canonical benchmark | INX/P3 | 1 | 10-fold stratified | 5-fold stratified | Evaluation only |
| Repeated-CV sensitivity | INX/P3 | 5 | 5-fold stratified | 5-fold stratified | Evaluation only |
| Fixed-policy sensitivity | INX/P0-P5 | 1 | 10-fold stratified | 5-fold stratified | Evaluation only |
| Retuned-policy sensitivity | INX/P0-P5 | 1 | 10-fold stratified | 5-fold stratified | Evaluation only |
| Selection-objective sensitivity | INX/P3 | 1 | 10-fold stratified | 5-fold stratified | Evaluation only |
| Repeated selection-objective sensitivity | INX/P3 | 5 | Same persisted 5-fold identities | Same persisted 5-fold identities | Evaluation only |
| Sigmoid calibration | INX/P3/XGBoost | 1 | 10-fold stratified | Persisted folds | Evaluation only |
| Synthetic HR target/CV sensitivity | HRDataset v14/conservative seven features | 5 | 5-fold stratified | 5-fold stratified | Evaluation only |
| Synthetic HR target-alias sensitivity | HRDataset v14/311 historical and matched 309 | 1 | 5-fold stratified | 5-fold stratified | Evaluation only |

# Table S4. Per-class performance by selection objective

| Selection | System | Rating | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- | --- | --- |
| macro f1 | cumulative threshold xgboost | 2 | 0.7974 | 0.9330 | 0.8599 | 194 |
| macro f1 | cumulative threshold xgboost | 3 | 0.8700 | 0.6739 | 0.7595 | 874 |
| macro f1 | cumulative threshold xgboost | 4 | 0.1858 | 0.4167 | 0.2570 | 132 |
| macro f1 | lightgbm | 2 | 0.8233 | 0.9124 | 0.8655 | 194 |
| macro f1 | lightgbm | 3 | 0.8497 | 0.8924 | 0.8705 | 874 |
| macro f1 | lightgbm | 4 | 0.1194 | 0.0606 | 0.0804 | 132 |
| macro f1 | logistic regression | 2 | 0.5485 | 0.6701 | 0.6032 | 194 |
| macro f1 | logistic regression | 3 | 0.8010 | 0.7277 | 0.7626 | 874 |
| macro f1 | logistic regression | 4 | 0.1361 | 0.1742 | 0.1528 | 132 |
| macro f1 | proportional odds logistic | 2 | 0.5126 | 0.7320 | 0.6030 | 194 |
| macro f1 | proportional odds logistic | 3 | 0.7920 | 0.5183 | 0.6266 | 874 |
| macro f1 | proportional odds logistic | 4 | 0.1538 | 0.4091 | 0.2236 | 132 |
| macro f1 | random forest | 2 | 0.8364 | 0.9227 | 0.8775 | 194 |
| macro f1 | random forest | 3 | 0.8517 | 0.9531 | 0.8996 | 874 |
| macro f1 | random forest | 4 | 0.0000 | 0.0000 | 0.0000 | 132 |
| macro f1 | xgboost | 2 | 0.8211 | 0.9227 | 0.8689 | 194 |
| macro f1 | xgboost | 3 | 0.8542 | 0.8112 | 0.8322 | 874 |
| macro f1 | xgboost | 4 | 0.1513 | 0.1742 | 0.1620 | 132 |
| qwk | cumulative threshold xgboost | 2 | 0.8447 | 0.8969 | 0.8700 | 194 |
| qwk | cumulative threshold xgboost | 3 | 0.8491 | 0.9657 | 0.9036 | 874 |
| qwk | cumulative threshold xgboost | 4 | 0.0000 | 0.0000 | 0.0000 | 132 |
| qwk | lightgbm | 2 | 0.8293 | 0.8763 | 0.8521 | 194 |
| qwk | lightgbm | 3 | 0.8460 | 0.9428 | 0.8918 | 874 |
| qwk | lightgbm | 4 | 0.1905 | 0.0303 | 0.0523 | 132 |
| qwk | logistic regression | 2 | 0.5375 | 0.6649 | 0.5945 | 194 |
| qwk | logistic regression | 3 | 0.7985 | 0.7208 | 0.7577 | 874 |
| qwk | logistic regression | 4 | 0.1345 | 0.1742 | 0.1518 | 132 |
| qwk | proportional odds logistic | 2 | 0.4836 | 0.8351 | 0.6125 | 194 |
| qwk | proportional odds logistic | 3 | 0.8121 | 0.4005 | 0.5364 | 874 |
| qwk | proportional odds logistic | 4 | 0.1567 | 0.5152 | 0.2403 | 132 |
| qwk | random forest | 2 | 0.8249 | 0.9227 | 0.8710 | 194 |
| qwk | random forest | 3 | 0.8520 | 0.9554 | 0.9008 | 874 |
| qwk | random forest | 4 | 0.0000 | 0.0000 | 0.0000 | 132 |
| qwk | xgboost | 2 | 0.8524 | 0.9227 | 0.8861 | 194 |
| qwk | xgboost | 3 | 0.8535 | 0.9600 | 0.9036 | 874 |
| qwk | xgboost | 4 | 0.1429 | 0.0076 | 0.0144 | 132 |

Hard predictions use the declared argmax rule with no class-specific threshold optimization. The exact table is `tables/Table_S4.csv`.

# Table S5. Complete audit gates

| Stage | Input | Audit | Failure | Permissible claim |
| --- | --- | --- | --- | --- |
| Information | Features and intended context | Availability/timing contract | Timing unverified | Sensitivity only |
| Selection | Outer-training partitions | Nested candidate selection | Test-informed selection | Evaluation invalid |
| Prediction | Untouched outer test | OOF probabilities; ordinal/class metrics | Extreme-class collapse | Bound aggregate claim |
| Calibration | Cross-fitted training probabilities | Predeclared sigmoid fit | Outer-test fit/selection | Calibration invalid |
| Explanation | Prediction-producing fold model | Identity, additivity, stability, deletion | Model mismatch | No explanation claim |
| Subgroup/proxy | Held-out outputs/categories | Support, interval, distinct proxy questions | Unsupported denominator | Descriptive/not estimated |
| Synthetic protocol sensitivity | Separately fitted HR systems on a fictitious teaching dataset | Target, folds, calibration, support | Construct/transport conflation | Methodological portability only |
| Evidence | Aggregate source rows | Selector, value, hash, qualifier | Unresolved/stale number | No numerical claim |

# Table S6. P3 nominal-XGBoost subgroup gap uncertainty and support

P3 is the feature policy and nominal XGBoost is the fitted-system family. Intervals are 95% simultaneous exploratory intervals based on the studentized maximum absolute bootstrap deviation over all estimable prespecified P3 nominal-XGBoost attribute, support-threshold, and metric gap cells. Eligibility is fixed before resampling; fitted-model variability is excluded. These diagnostics apply to the named system and do not certify fairness or describe all six trained systems.

## Age

| Metric | Gap | Minimum group | Maximum group | Simultaneous low | Simultaneous high |
| --- | --- | --- | --- | --- | --- |
| macro f1 | 0.0301 | 30-39 | 40-49 | 0.0000 | 0.1657 |
| quadratic weighted kappa | 0.0653 | 30-39 | 50-59 | 0.0000 | 0.3335 |
| ordinal mae | 0.0770 | 50-59 | 30-39 | 0.0000 | 0.2411 |

## Gender

| Metric | Gap | Minimum group | Maximum group | Simultaneous low | Simultaneous high |
| --- | --- | --- | --- | --- | --- |
| macro f1 | 0.0363 | Male | Female | 0.0000 | 0.1515 |
| quadratic weighted kappa | 0.0527 | Male | Female | 0.0000 | 0.2420 |
| ordinal mae | 0.0508 | Female | Male | 0.0000 | 0.1719 |

## Marital Status

| Metric | Gap | Minimum group | Maximum group | Simultaneous low | Simultaneous high |
| --- | --- | --- | --- | --- | --- |
| macro f1 | 0.0101 | Married | Divorced | 0.0000 | 0.1135 |
| quadratic weighted kappa | 0.0381 | Single | Married | 0.0000 | 0.2431 |
| ordinal mae | 0.0291 | Divorced | Single | 0.0000 | 0.1485 |

## Business Travel

| Metric | Gap | Minimum group | Maximum group | Simultaneous low | Simultaneous high |
| --- | --- | --- | --- | --- | --- |
| macro f1 | 0.0528 | Travel Frequently | Travel Rarely | 0.0000 | 0.1987 |
| quadratic weighted kappa | 0.0711 | Travel Frequently | Travel Rarely | 0.0000 | 0.3472 |
| ordinal mae | 0.0613 | Travel Rarely | Travel Frequently | 0.0000 | 0.2195 |

## Department

| Metric | Gap | Minimum group | Maximum group | Simultaneous low | Simultaneous high |
| --- | --- | --- | --- | --- | --- |
| macro f1 | 0.2179 | Development | Finance | 0.0000 | 0.5281 |
| quadratic weighted kappa | 0.4388 | Development | Research & Development | 0.1275 | 0.7501 |
| ordinal mae | 0.1193 | Finance | Development | 0.0000 | 0.3687 |

## Education Background

| Metric | Gap | Minimum group | Maximum group | Simultaneous low | Simultaneous high |
| --- | --- | --- | --- | --- | --- |
| macro f1 | 0.0413 | Other | Medical | 0.0000 | 0.3766 |
| quadratic weighted kappa | 0.1190 | Technical Degree | Marketing | 0.0000 | 0.5064 |
| ordinal mae | 0.0891 | Medical | Technical Degree | 0.0000 | 0.3211 |

The exact support counts, pointwise intervals, resample counts, source identities, and claim boundaries appear in `tables/Table_S6.csv`.

# Table S7. Selection-objective confusion matrices

| Selection | System | True | Predicted | Count |
| --- | --- | --- | --- | --- |
| macro f1 | cumulative threshold xgboost | 2 | 2 | 181 |
| macro f1 | cumulative threshold xgboost | 2 | 3 | 13 |
| macro f1 | cumulative threshold xgboost | 2 | 4 | 0 |
| macro f1 | cumulative threshold xgboost | 3 | 2 | 44 |
| macro f1 | cumulative threshold xgboost | 3 | 3 | 589 |
| macro f1 | cumulative threshold xgboost | 3 | 4 | 241 |
| macro f1 | cumulative threshold xgboost | 4 | 2 | 2 |
| macro f1 | cumulative threshold xgboost | 4 | 3 | 75 |
| macro f1 | cumulative threshold xgboost | 4 | 4 | 55 |
| macro f1 | lightgbm | 2 | 2 | 177 |
| macro f1 | lightgbm | 2 | 3 | 16 |
| macro f1 | lightgbm | 2 | 4 | 1 |
| macro f1 | lightgbm | 3 | 2 | 36 |
| macro f1 | lightgbm | 3 | 3 | 780 |
| macro f1 | lightgbm | 3 | 4 | 58 |
| macro f1 | lightgbm | 4 | 2 | 2 |
| macro f1 | lightgbm | 4 | 3 | 122 |
| macro f1 | lightgbm | 4 | 4 | 8 |
| macro f1 | logistic regression | 2 | 2 | 130 |
| macro f1 | logistic regression | 2 | 3 | 59 |
| macro f1 | logistic regression | 2 | 4 | 5 |
| macro f1 | logistic regression | 3 | 2 | 97 |
| macro f1 | logistic regression | 3 | 3 | 636 |
| macro f1 | logistic regression | 3 | 4 | 141 |
| macro f1 | logistic regression | 4 | 2 | 10 |
| macro f1 | logistic regression | 4 | 3 | 99 |
| macro f1 | logistic regression | 4 | 4 | 23 |
| macro f1 | proportional odds logistic | 2 | 2 | 142 |
| macro f1 | proportional odds logistic | 2 | 3 | 52 |
| macro f1 | proportional odds logistic | 2 | 4 | 0 |
| macro f1 | proportional odds logistic | 3 | 2 | 124 |
| macro f1 | proportional odds logistic | 3 | 3 | 453 |
| macro f1 | proportional odds logistic | 3 | 4 | 297 |
| macro f1 | proportional odds logistic | 4 | 2 | 11 |
| macro f1 | proportional odds logistic | 4 | 3 | 67 |
| macro f1 | proportional odds logistic | 4 | 4 | 54 |
| macro f1 | random forest | 2 | 2 | 179 |
| macro f1 | random forest | 2 | 3 | 15 |
| macro f1 | random forest | 2 | 4 | 0 |
| macro f1 | random forest | 3 | 2 | 33 |
| macro f1 | random forest | 3 | 3 | 833 |
| macro f1 | random forest | 3 | 4 | 8 |
| macro f1 | random forest | 4 | 2 | 2 |
| macro f1 | random forest | 4 | 3 | 130 |
| macro f1 | random forest | 4 | 4 | 0 |
| macro f1 | xgboost | 2 | 2 | 179 |
| macro f1 | xgboost | 2 | 3 | 14 |
| macro f1 | xgboost | 2 | 4 | 1 |
| macro f1 | xgboost | 3 | 2 | 37 |
| macro f1 | xgboost | 3 | 3 | 709 |
| macro f1 | xgboost | 3 | 4 | 128 |
| macro f1 | xgboost | 4 | 2 | 2 |
| macro f1 | xgboost | 4 | 3 | 107 |
| macro f1 | xgboost | 4 | 4 | 23 |
| qwk | cumulative threshold xgboost | 2 | 2 | 174 |
| qwk | cumulative threshold xgboost | 2 | 3 | 20 |
| qwk | cumulative threshold xgboost | 2 | 4 | 0 |
| qwk | cumulative threshold xgboost | 3 | 2 | 30 |
| qwk | cumulative threshold xgboost | 3 | 3 | 844 |
| qwk | cumulative threshold xgboost | 3 | 4 | 0 |
| qwk | cumulative threshold xgboost | 4 | 2 | 2 |
| qwk | cumulative threshold xgboost | 4 | 3 | 130 |
| qwk | cumulative threshold xgboost | 4 | 4 | 0 |
| qwk | lightgbm | 2 | 2 | 170 |
| qwk | lightgbm | 2 | 3 | 24 |
| qwk | lightgbm | 2 | 4 | 0 |
| qwk | lightgbm | 3 | 2 | 33 |
| qwk | lightgbm | 3 | 3 | 824 |
| qwk | lightgbm | 3 | 4 | 17 |
| qwk | lightgbm | 4 | 2 | 2 |
| qwk | lightgbm | 4 | 3 | 126 |
| qwk | lightgbm | 4 | 4 | 4 |
| qwk | logistic regression | 2 | 2 | 129 |
| qwk | logistic regression | 2 | 3 | 60 |
| qwk | logistic regression | 2 | 4 | 5 |
| qwk | logistic regression | 3 | 2 | 101 |
| qwk | logistic regression | 3 | 3 | 630 |
| qwk | logistic regression | 3 | 4 | 143 |
| qwk | logistic regression | 4 | 2 | 10 |
| qwk | logistic regression | 4 | 3 | 99 |
| qwk | logistic regression | 4 | 4 | 23 |
| qwk | proportional odds logistic | 2 | 2 | 162 |
| qwk | proportional odds logistic | 2 | 3 | 32 |
| qwk | proportional odds logistic | 2 | 4 | 0 |
| qwk | proportional odds logistic | 3 | 2 | 158 |
| qwk | proportional odds logistic | 3 | 3 | 350 |
| qwk | proportional odds logistic | 3 | 4 | 366 |
| qwk | proportional odds logistic | 4 | 2 | 15 |
| qwk | proportional odds logistic | 4 | 3 | 49 |
| qwk | proportional odds logistic | 4 | 4 | 68 |
| qwk | random forest | 2 | 2 | 179 |
| qwk | random forest | 2 | 3 | 15 |
| qwk | random forest | 2 | 4 | 0 |
| qwk | random forest | 3 | 2 | 36 |
| qwk | random forest | 3 | 3 | 835 |
| qwk | random forest | 3 | 4 | 3 |
| qwk | random forest | 4 | 2 | 2 |
| qwk | random forest | 4 | 3 | 130 |
| qwk | random forest | 4 | 4 | 0 |
| qwk | xgboost | 2 | 2 | 179 |
| qwk | xgboost | 2 | 3 | 15 |
| qwk | xgboost | 2 | 4 | 0 |
| qwk | xgboost | 3 | 2 | 29 |
| qwk | xgboost | 3 | 3 | 839 |
| qwk | xgboost | 3 | 4 | 6 |
| qwk | xgboost | 4 | 2 | 2 |
| qwk | xgboost | 4 | 3 | 129 |
| qwk | xgboost | 4 | 4 | 1 |

The exact table, including the dataset key, is `tables/Table_S7.csv`.

# Table S8. Bounded literature comparison

## Li et al. (2021)

**Problem and evidence:** Employee-performance classification; Three linked internal HR tables from a publishing organization. **Method and XAI:** Logistic regression; decision tree; naive Bayes; Permutation importance reported. **Boundary relative to this study:** Organization-specific comparison; present work joins ordinal failure analysis to an explicit evidence contract. Official proceedings PDF; no claim of an exhaustive absence audit.

## Adeniyi et al. (2022)

**Problem and evidence:** Employee-performance classification; Public HR table with 311 rows and 36 attributes. **Method and XAI:** PCA; ANN; RF; DT; reported 80:20 split; not_reported_in_accessible_source. **Boundary relative to this study:** Present work separately audits target aliases, target mapping and training-only evaluation. Publisher full text. Existing title retained over conflicting Crossref deposit.

## Putri and Sitohang (2026)

**Problem and evidence:** Employee-performance classification; INX employee data. **Method and XAI:** Random Forest; multiple train/test proportions; Feature importance. **Boundary relative to this study:** Present work separates nested candidate selection from final evaluation and audits metric-specific extreme-class failures. Publisher PDF methods/results; non-reporting is not proof of absence.

## Löfström et al. (2024)

**Problem and evidence:** Reliable local explanations with uncertainty; 25 benchmark datasets. **Method and XAI:** Calibrated Explanations based on Venn-Abers; Conditional feature explanations and counterfactuals. **Boundary relative to this study:** A new explanation method; present study audits existing exact-model SHAP and a declared probability-calibration path. Publisher abstract and metadata; do not infer unseen evaluation controls.

## Sepúlveda et al. (2025)

**Problem and evidence:** Stability of local interpretability; Synthetic and real anomaly-detection data. **Method and XAI:** Top-rank-weighted stability measure; SHAP-based feature-ranking consistency under small changes. **Boundary relative to this study:** Local perturbation stability differs from fold/seed/training-resample comparisons and model-level deletion in the present audit. Publisher abstract and section excerpts.

## Manafi Varkiani et al. (2025)

**Problem and evidence:** Employee attrition and contextual explanation; Italian financial-company employee records. **Method and XAI:** Machine-learning case study; Random Forest explanation; SHAP contribution magnitude and direction. **Boundary relative to this study:** Strong HR-XAI precedent; different outcome and organization, without evidence of the present shared audit contract in accessible material. Publisher abstract and available methods excerpts.

## Rass and Dallinger (2026)

**Problem and evidence:** Detecting unwanted patterns in training data; Real-life and synthetic datasets. **Method and XAI:** Prespecified rules; fuzzy reasoning; regression significance tests; Rule-based explainability. **Boundary relative to this study:** Proposes data-testing machinery; present study evaluates how multiple system-audit views constrain claims on fixed HR evidence. Publisher abstract and metadata; final publication year 2026 despite DOI containing 2025.

## Abusitta et al. (2024)

**Problem and evidence:** Taxonomy and limitations of XAI; Literature survey. **Method and XAI:** Review and comparative taxonomy; Central topic. **Boundary relative to this study:** General XAI guidance; present paper must show a concrete operational audit rather than treat a component list as novelty. Publisher abstract and section excerpts; not a benchmark competitor.

## Kapoor et al. (2024), REFORMS

**Problem and evidence:** Reporting and reproducibility of ML-based science; Consensus and methodological literature. **Method and XAI:** Checklist and guidelines; Reporting context. **Boundary relative to this study:** Important prior protocol. Present contribution should be the executable linkage plus empirical interactions among audit findings, not the invention of transparent evaluation. Author-maintained project and registered Science Advances metadata.

## Bar-Gil et al. (2024)

**Problem and evidence:** Embedding AI ethics in HR analytics; Two organizational HR case studies. **Method and XAI:** Comparative case-study analysis; Organizational governance context. **Boundary relative to this study:** Shows why technical diagnostics cannot settle organizational governance or employee consequences. Publisher abstract and full-text excerpts.

## Lones (2024)

**Problem and evidence:** Avoiding common ML practice errors; Tutorial and examples. **Method and XAI:** Practical methodological guidance; Interpretation guidance. **Boundary relative to this study:** Broad practical precedent; present study instantiates connected checks in an ordinal HR setting. Publisher tutorial identity and abstract; individual empirical comparisons are not inferred.

## Present study

**Problem and evidence:** Auditing ordinal employee-performance prediction; INX; secondary protocol sensitivity on synthetic HRDataset_v14. **Method and XAI:** Nested benchmark; repeated objective/policy sensitivities; exact evidence identities; exact-model grouped SHAP; stability; model-level deletion. **Boundary relative to this study:** Integration links conflicting findings to the evaluated system; synthetic teaching data do not supply observed organizational transport evidence. Frozen claim boundary; not an additional predictor or deployment result.

The full comparison dimensions and official-source links appear in `tables/Table_S8.csv`.

# Numerical evidence ledger

The machine-readable ledger and its twenty anonymous aggregate source tables are supplied in the `evidence` directory. `SOURCE_INDEX.csv` records every anonymous source hash. The ledger preserves exact values, selectors, rounding, required qualifiers, and prohibited interpretations for the 99 active numerical claims.
# Table S9. Repeated selection-objective sensitivity

This analysis reuses the exact five Phase 1C repeated 5×5 nested-CV split identities, preprocessing, candidate registries, seeds, inclusive 0.001 tolerance, and deterministic tie logic. Macro-F1-selected OOF predictions were reused. QWK selection added 150 outer fits and no inner or baseline fits; outer-test outcomes remained evaluation-only. Values below are descriptive means over the five fixed repetitions.

| System | Selection | Macro-F1 | Balanced acc. | QWK | MAE | RPS | Log loss | Brier | Rating-4 recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nominal XGBoost | macro-f1 | 0.6288 | 0.6446 | 0.5833 | 0.2335 | 0.0863 | 0.5677 | 0.3453 | 0.1894 |
| Nominal XGBoost | qwk | 0.6021 | 0.6279 | 0.6377 | 0.1550 | 0.0682 | 0.4863 | 0.2700 | 0.0167 |
| Cumulative-threshold XGBoost | macro-f1 | 0.6155 | 0.6524 | 0.5451 | 0.3062 | 0.1007 | 1.2592 | 0.4069 | 0.3288 |
| Cumulative-threshold XGBoost | qwk | 0.5933 | 0.6251 | 0.6374 | 0.1522 | 0.0673 | 0.4795 | 0.2653 | 0.0000 |
| Random Forest | macro-f1 | 0.5955 | 0.6283 | 0.6311 | 0.1597 | 0.0854 | 0.6228 | 0.3561 | 0.0106 |
| Random Forest | qwk | 0.5952 | 0.6278 | 0.6351 | 0.1565 | 0.0837 | 0.6107 | 0.3475 | 0.0061 |

Candidate selections changed in 15, 20, 17, 20, and 20 of 30 model-fold cells (92/150 overall). Full six-model ordering changed for all seven compared aggregate metrics in every repetition. Nominal XGBoost gained QWK and improved MAE while rating-4 recall fell in 5/5 repetitions. Cumulative-threshold XGBoost gained QWK in 5/5 and had zero QWK-selected rating-4 recall in 5/5. Random Forest gained QWK in 4/5 with one zero change; its rating-4 recall fell in 2/5 and was unchanged in 3/5.

The exact 60 repetition × model × selection rows, including all required aggregate and per-class metrics plus the five selected candidate IDs, are supplied in `tables/Table_S9.csv`. Direction counts are not confidence intervals. Repetitions reuse the same observations, and the effect is not claimed to be universal.
