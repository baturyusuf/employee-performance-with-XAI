# Beyond Predictive Accuracy: A Traceable Leakage- and Governance-Aware XAI Audit Protocol for Ordinal Employee Performance Prediction

**Article type:** Original Research Article

**Authors:** Muhammed Yusuf Batur^1,*^ and Mehmet Göktürk^2^

**Affiliations:** ^1^ Rumeli University, [AUTHOR SIGN-OFF REQUIRED: complete postal address]; ^2^ Gebze Technical University, [AUTHOR SIGN-OFF REQUIRED: complete postal address]

**Correspondence:** myusuf.batur@rumeli.edu.tr; [AUTHOR SIGN-OFF REQUIRED]

## Abstract

Employee-performance prediction is often summarized by a single score even though model selection, ordinal error severity, information availability, and extreme-class behavior can change the conclusion. We present a traceable audit protocol for ordinal prediction using two public cross-sectional HR tables. In the INX benchmark, cumulative-threshold XGBoost led macro-F1 (0.6255), whereas Random Forest led quadratic weighted kappa (QWK; 0.6317) and ordinal mean absolute error (MAE; 0.1583) while failing to identify any rating-4 cases. Changing inner selection from macro-F1 to QWK altered 37 of 60 model-fold candidate choices and 6 of 9 metric leaders; QWK-selected nominal XGBoost reached QWK 0.6418 but rating-4 recall fell to 0.0076. In the nominal-XGBoost policy-sensitivity experiment, moving from the Primary Leakage-Aware policy (P3) to the timestamp-unverified Prospective-Plausibility policy (P4) reduced macro-F1/QWK by 0.2094/0.3581 under the fixed schedule and by 0.1937/0.3308 after independent retuning. Subgroup, calibration, explanation-stability, proxy, target-mapping, and partial cross-dataset replication analyses further bound interpretation. Aggregate ordinal improvement therefore does not guarantee extreme-class recognition, and the contribution is the operational integration of these checks into a traceable evaluation contract rather than a new prediction algorithm.

**Keywords:** human-resource analytics; ordinal classification; explainable artificial intelligence; data leakage; nested cross-validation; SHAP stability; probability calibration; proxy diagnostics; traceability

## 1. Introduction

Algorithmic analysis of employee records can affect people even when a model is framed as decision support. HR data are typically small, organizationally produced, and entangled with prior managerial decisions. Consequently, an apparently strong classifier may exploit variables recorded during or after an evaluation, reproduce structural context, or report probabilities and explanations that have not been audited for reliability. HR scholarship therefore emphasizes accountability, employee reactions, personal integrity, and the limits of purely technical optimization [@tambe2019artificial; @leichtdeobald2019challenges; @kochling2020discriminated; @giermindl2021dark].

The employee-performance literature has compared conventional classifiers on the 1,200-row INX table and related organizational data [@archana2019application; @lather2019prediction; @li2021employee; @patel2022ranker; @adeniyi2022comparison; @nayem2024unbiased; @putri2026klasifikasi]. Such studies establish practical interest, but predictive ranking alone does not answer whether information was available before the rating, whether tuning remained inside training data, whether an ordinal error crossed two rating levels, or whether post-hoc explanations are stable and faithful.

This study contributes:

- a prespecified P0-P5 information contract distinguishing outcome-proximal, sensitive, timing-uncertain, and organizational-proxy fields;
- a nine-system ordinal benchmark plus repeated nested cross-validation and a nominal-XGBoost fixed/retuned policy-sensitivity experiment, all based on exactly-once held-out predictions;
- exact-fold TreeSHAP, stability, deletion, calibration, subgroup, and proxy-use diagnostics with explicit noncausal and support-aware boundaries; and
- a partial cross-dataset protocol replication on HRDataset_v14, target-mapping/CV-design sensitivities, and a hash-bound evidence ledger linking reported numerical claims to versioned source records.

The intended use is methodological research and audit. The study does not establish a production HR system, causal determinants of performance, certified fairness, or prospective validity. Its targets are recorded organizational ratings, not validated measures of objective capability or productivity.

## 2. Related Work

### 2.1 Employee-performance prediction and HR decision support

Four verified studies in the bounded literature set use the exact INX data, while others use psychometric, organization-specific, HRDataset_v14, or newly collected employee-performance data [@archana2019application; @lather2019prediction; @li2021employee; @patel2022ranker; @adeniyi2022comparison; @nayem2024unbiased; @putri2026klasifikasi]. Related churn studies illustrate broader machine-learning and explanation work in people analytics but address different outcomes [@abufaty2025integrating; @chaudhary2025integrated]. Success on churn cannot be reinterpreted as validation of an ordinal performance-rating model.

### 2.2 Ordinal modelling and evaluation

Ordinal labels require evaluation that respects order without hiding class-specific behavior. Proportional-odds models impose a shared-slope cumulative-link structure [@mccullagh1980regression], while threshold decompositions adapt binary classifiers to ordered outcomes [@frank2001simple]. Ordinal assessment can combine nominal discrimination with distance-aware measures [@cardoso2011measuring], including weighted kappa [@cohen1968weighted] and ranked probability scoring [@epstein1969scoring; @gneiting2007strictly]. These aggregate criteria must still be accompanied by per-class results because a favorable distance-weighted score does not guarantee detection of a rare extreme category.

The benchmark includes Random Forest [@breiman2001random], XGBoost [@chen2016xgboost], and LightGBM [@ke2017lightgbm] alongside linear and ordinal systems. Algorithm pedigree does not determine which metric a system will lead, so selection objectives and class-specific behavior are treated as empirical parts of the evaluation contract.

### 2.3 Leakage, selection, and reproducible evaluation

Leakage arises when information encodes the target or would not be available at the claimed decision point [@kaufman2012leakage; @kapoor2023leakage]. Evaluating a configuration on the data used to select it also produces optimistic estimates, motivating nested separation of selection and evaluation [@cawley2010overfitting]. Reproducibility guidance calls for transparent processing, sample allocation, hyperparameters, uncertainty, and artifact reporting [@mitchell2019model; @pineau2021improving]. Our protocol combines these principles through persisted folds, training-only selection, exact output identities, and content hashes.

### 2.4 Explanation stability and faithfulness

SHAP provides additive feature attributions, including efficient algorithms for tree models [@lundberg2017unified; @lundberg2020local]. Yet an attribution plot is not evidence of causality or reliability. Nearby inputs can produce unstable explanations, sensitivity and infidelity can be quantified, and post-hoc explanations can be manipulated [@alvarezmelis2018robustness; @yeh2019infidelity; @slack2020fooling]. We bind each held-out explanation to the exact prediction model, aggregate encoded columns to declared raw-feature families, and examine ranking stability separately from deletion behavior.

### 2.5 Calibration and subgroup/proxy boundaries

Classification and ordinal scores do not show whether probabilities are reliable. Post-hoc calibration can improve some metrics, but calibration is multidimensional and scalar summaries depend on the event and binning scheme [@guo2017calibration; @vaicenavicius2019evaluating]. In HR settings, removing direct attributes also does not remove every proxy channel or establish fairness. Our subgroup results are support-aware descriptive diagnostics, while department reconstructability is kept separate from performance-model output dependence.

### 2.6 Bounded positioning

The contribution is not the novelty of any single component, but their operational integration into a traceable evaluation contract. This integration joins information policy, nested selection, exact held-out prediction-explanation identity, training-only calibration, explanation stability and deletion, subgroup/proxy boundaries, partial cross-dataset replication, and manuscript-number provenance. Supplementary Table S5 operationalizes the contribution as reusable audit gates with explicit inputs, outputs, failure conditions, and claim boundaries. The positioning is bounded and is neither a world-first nor an exhaustive-review claim.

## 3. Materials and Methods

### 3.1 Study design, estimand, and intended use

The INX analysis is a cross-sectional sensitivity study under explicit feature-availability assumptions, not an observed prospective prediction exercise. The estimand treats `PerformanceRating` as an ordinal organizational rating with labels 2, 3, and 4. Feature-observation times and rating-decision time are absent, so retained variables are not verified as prospectively available.

The protocol is intended for research, model audit, and reproducibility-oriented assessment. It is not intended for autonomous hiring, dismissal, promotion, compensation, discipline, or employee ranking. Predictions and SHAP values describe fitted models under observed cross-sectional data, not individual prescriptions.

![Figure 1. Audit protocol and evidence-identity flow. Training-only operations remain separated from untouched outer-test prediction, explanation, and evaluation paths.](../../manuscript/mdpi_information/assets/figures/main/figure_01_audit_protocol.png)

### 3.2 Datasets, targets, and data quality

INX is the primary development and internal out-of-fold (OOF) evaluation table. It contains 1,200 rows and 28 columns; target support is 194/874/132 for ratings 2/3/4. HRDataset_v14 contains 311 rows and 36 raw columns. The retained three-class mapping combines `PIP` and `Needs Improvement` as class 2, preserves `Fully Meets` as 3, and maps `Exceeds` as 4, producing support 31/243/37. A prespecified sensitivity retains the two lower source categories as distinct ordered classes, producing four-class support 13/18/243/37. These are different estimands; neither validates equivalence between organizations' rating constructs.

The aggregate audit applies whitespace-aware missingness, duplicate checks, identifier rules, schema hashing, numeric-domain rules, and temporal/consistency rules fixed before analysis. No source value was silently repaired. Provenance and redistribution rights require manual resolution, so raw employee-level tables are excluded.

**Table 1. Dataset roles, target mappings, and aggregate audit status**

| Dataset/estimand | Analytical role | Rows | Target support | Key boundary |
| --- | --- | ---: | --- | --- |
| INX, three ratings | Primary development and internal OOF evaluation | 1200 | 2=194; 3=874; 4=132 | Cross-sectional; feature and decision timestamps unavailable |
| HRDataset_v14, retained three-class mapping | Partial cross-dataset protocol replication | 311 | 2=31; 3=243; 4=37 | Different features, semantics, parameters, and population |
| HRDataset_v14, four-class sensitivity | Target-formulation sensitivity | 311 | 13/18/243/37 | Distinct estimand; not an accuracy-improvement comparison |

### 3.3 Prespecified information policies

Six policies progressively restrict information. P0 excludes only identifier and target and is an information-rich diagnostic comparator. P1 removes declared outcome-proximal and temporal-risk variables. P2 also removes direct sensitive demographics. P3, the primary policy, removes department. P4 removes timing-uncertain surveys and related measures but remains only a prospective-plausibility sensitivity because timestamps are absent. P5 further removes declared role, compensation, assignment, promotion, and manager-context proxy channels; residual proxies may remain.

**Table 2. P0-P5 information policies**

| Policy | Role | Retained | Interpretation |
| --- | --- | ---: | --- |
| P0 Information-Rich Diagnostic | Diagnostic only | 26 | Outcome-proximal/timing-risk information retained |
| P1 Leakage-Risk-Controlled | Outcome/temporal-risk ablation | 24 | High-risk fields removed; other risks remain |
| P2 Direct-Sensitive-Attribute-Excluded | Sensitive-feature ablation | 21 | Direct sensitive fields removed; not a fairness result |
| P3 Primary Leakage-Aware | Canonical primary | 20 | Department removed; timing uncertainty and proxies remain |
| P4 Prospective-Plausibility | Timestamp-unverified sensitivity | 13 | Semantically plausible prior fields only |
| P5 Strict Proxy-Reduced Prospective-Plausibility | Organizational-proxy sensitivity | 6 | Declared strong proxies removed; residual proxies possible |

### 3.4 Preprocessing, models, ordinal construction, and hard decisions

Preprocessing is fitted anew within the current inner-development or outer-training partition. Numeric variables receive median imputation and standard scaling. Categorical variables receive most-frequent imputation and dense one-hot encoding with unknown categories ignored. Outer-test rows are evaluation-only and never fit preprocessing, choose a candidate, calibrator, seed, or information policy.

The P3 benchmark contains six trained systems—multinomial logistic regression, Random Forest [@breiman2001random], LightGBM [@ke2017lightgbm], nominal XGBoost [@chen2016xgboost], proportional-odds logistic regression, and cumulative-threshold XGBoost—plus stratified, majority, and ordinal-median baselines. Logistic and proportional-odds models search six combinations of regularization strength (0.1, 1, 10) and optional balancing. Random Forest searches eight combinations of maximum depth (unrestricted or 8), minimum leaf size (1 or 3), and optional balancing. LightGBM searches eight combinations of leaves (15 or 31), minimum child size (20 or 40), and optional balancing. Nominal XGBoost searches eight combinations of depth (3 or 5), minimum child weight (1 or 5), and optional balancing; cumulative-threshold XGBoost uses the analogous eight combinations with depth 2 or 3. Fixed estimator settings and complete registries are given in Supplementary Table S2.

The proportional-odds model is a cumulative-link model with shared feature coefficients across ordered cut points and increasing fitted thresholds [@mccullagh1980regression]. Its shared-slope assumption is a model restriction, not an empirical claim of identical effects. Cumulative-threshold XGBoost independently estimates `P(Y>2)` and `P(Y>3)` [@frank2001simple]. A rowwise nonincreasing pool-adjacent-violators projection corrects crossing cumulative probabilities before class probabilities are obtained by differencing and normalized. This construction supplies ordered probabilities but does not guarantee recognition of the extreme class.

Hard labels were generated by each fitted estimator's unmodified multiclass prediction rule. The two custom ordinal estimators explicitly select the class by argmax over their class-probability vector. No class-specific threshold optimization was performed, and alternative class-specific decision thresholds were not evaluated. Per-class results and confusion matrices therefore characterize the declared hard-decision rules rather than an optimized thresholding policy.

### 3.5 Nested benchmark and selection-objective sensitivity

The canonical benchmark uses persisted stratified ten-fold outer splits and five-fold inner splits shared across systems. Every row receives exactly one outer-test prediction per system. The canonical selection regime maximizes inner macro-F1, admits candidates within an inclusive 0.001 tolerance, then uses QWK and the lowest candidate index as deterministic tie-breaks. The sensitivity regime reverses macro-F1 and QWK while retaining identical folds, candidate registries, tolerance, and outer evaluation; canonical OOF predictions are reused, whereas QWK-selected models are refitted in every outer fold. No MAE- or RPS-selected regime was added after inspecting the results.

The revision-stage analysis definitions used for these sensitivities were frozen before their execution. This is a within-revision prespecification statement, not a claim that the overall research program was preregistered before any inspection of the datasets.

We report selected-candidate changes, metric-leader changes, complete-ordering changes, and metric-effect magnitudes separately. A lower-rank swap alone is not interpreted as material dependence. Metrics are macro-F1, balanced accuracy, QWK [@cohen1968weighted], ordinal MAE, two-level reversal rate, normalized ranked probability score (RPS) [@epstein1969scoring; @gneiting2007strictly], log loss, multiclass Brier score, and top-label expected calibration error (ECE). An empirical-prior predictor, estimated from outer-training labels, is retained as a probability-quality reference; zero top-label ECE for this constant predictor is not evidence of perfect calibration.

![Figure 2. Nine-system ordinal benchmark. Metric-specific leaders are shown under the common P3 protocol; the plot is not a universal leaderboard.](../../manuscript/mdpi_information/phase5b_figures/figure_02_nine_system_benchmark.png)

### 3.6 Repeated nested cross-validation

Training and split variability use five fixed repetitions of 5-fold outer by 5-fold inner nested cross-validation. Each trained system is selected and refitted in every repetition. Means, sample SDs, ranges, winner counts, and rank correlations are descriptive across the five repetition identities. Ranges are not confidence intervals, and repetition pairs share samples.

### 3.7 Nominal-XGBoost fixed-schedule and retuned information-policy contrasts

The P0-P5 fixed/retuned policy analysis is a nominal-XGBoost policy-sensitivity experiment, not a six-model policy benchmark. All six policies use the same nominal-XGBoost model family and the same eight-candidate registry. The fixed analysis applies the P3-selected fold-specific candidate schedule while feature access changes; the retuned analysis performs independent inner selection for each policy within outer-training data. P3 is the common anchor. The P3-to-P4 and P4-to-P5 contrasts therefore expose timing/information sensitivity for nominal XGBoost under both schedules, but neither schedule estimates a causal feature-policy effect. No confidence intervals or significance tests are attached to retuned-minus-fixed differences.

![Figure 3. Nominal-XGBoost fixed-schedule feature-policy sensitivity. P0 is diagnostic only; P4/P5 do not establish prospective validity or absence of proxies.](../../manuscript/mdpi_information/assets/figures/main/figure_03_feature_policy_sensitivity.png)

### 3.8 Exact-fold SHAP stability and deletion diagnostics

For nominal XGBoost under P3, every held-out observation is explained by the persisted outer-fold model that generated its prediction. TreeSHAP values use raw-margin space; encoded columns are signed-summed to feature families before absolute values are averaged across classes and OOF observations. Stability is evaluated across outer-fold, model-seed, and stratified 80% outer-training-resample pairs using top-k Jaccard and all-feature Spearman agreement.

Deletion compares SHAP-ranked masking against 20 random repetitions at one, three, and five deleted features. Median/mode masking may generate out-of-distribution records. Stability and deletion behavior therefore characterize the fitted model and declared interventions, not causal effects, actionability, human usefulness, or advice.

![Figure 4. Global grouped exact-fold TreeSHAP attribution. Magnitudes are raw-margin model attributions and must not be read as causal effects.](../../manuscript/mdpi_information/assets/figures/main/figure_05_global_grouped_shap.png)

![Figure 5. SHAP ranking stability across folds, model seeds, and outer-training resamples. Pairwise comparisons are dependent and descriptive.](../../manuscript/mdpi_information/assets/figures/main/figure_06_shap_stability.png)

### 3.9 Cross-fitted calibration

Within each nominal-XGBoost outer fold, three one-vs-rest sigmoid calibrators are fitted only to five-fold cross-fitted outer-training probabilities. Outputs are renormalized; untouched outer-test outcomes are evaluation-only. Raw and sigmoid outputs are compared with log loss, Brier score, top-label ECE, macro classwise ECE, cumulative ECE, and normalized RPS. ECE uses ten fixed equal-width bins and retains empty bins. Conclusions are metric-specific.

![Figure 6. Raw and cross-fitted sigmoid calibration diagnostics. Reliability curves retain bin support and do not provide prospective probability validation.](../../manuscript/mdpi_information/assets/figures/main/figure_04_calibration.png)

### 3.10 Support-aware subgroup and proxy diagnostics

The subgroup audit covers three systems, all six prespecified attributes (Age, Gender, Marital Status, Business Travel, Department, and Education), nine metrics, and support thresholds 20/30/50. Unsupported groups or class denominators remain explicit. The P3 manuscript summary uses the prespecified threshold of 30 and 5,000 bootstrap repetitions stratified by fold and class, with eligibility fixed before resampling. The exploratory simultaneous intervals condition on the fitted models and are descriptive diagnostics, not confirmatory fairness inference. Publication-facing point estimates, eligibility counts, and simultaneous intervals are given in Supplementary Table S6.

Proxy diagnostics separate prediction changes after refitting P3 without JobRole, output sensitivity when JobRole is shuffled within fold, and independent department reconstruction. Reconstructability shows information in a feature space, not use by the performance model. None establishes discrimination, fairness, causality, or legal compliance.

### 3.11 HR target mapping, CV design, and target-alias sensitivity

HRDataset_v14 is used for a partial cross-dataset protocol replication with models trained anew on that dataset. Its conservative policy contains exactly seven features: `EmpJobRole`, `EngagementSurvey`, `EmpJobSatisfaction`, `SpecialProjectsCount`, `DaysLateLast30`, `Absences`, and `ExperienceYearsAtThisCompany`. These features are not claimed to be prospectively verified; HRDataset_v14 does not provide observation timestamps sufficient to establish that each feature preceded the performance decision.

For target-formulation sensitivity, the retained three-class mapping and distinct four-class mapping each use five fixed repetitions of 5-fold outer by 5-fold inner nested validation, with selection and cross-fitted sigmoid calibration confined to outer-training data. The two mappings define different estimands and are not treated as an improvement comparison.

CV-design sensitivity compares the canonical 10-fold-outer/5-fold-inner estimates with the ranges from repeated 5-fold-outer/5-fold-inner analysis. Range inclusion is a descriptive stability check, not an equivalence test or confidence statement. Target-alias sensitivity preserves the historical 311-row result, removes the two target-text/identifier disagreement rows from the existing canonical predictions to obtain fit-free metrics on 309 rows, and separately selects/refits on those same 309 rows using consistently restricted fold identities. The primary training/data-rule comparison is therefore matched-population 309-row restricted canonical predictions versus 309-row exclusion/refit predictions; the 311-to-309 contrast is reported separately as a sample-removal effect.

![Figure 7. HRDataset_v14 partial cross-dataset protocol replication and mapped-target sensitivity. Results do not imply target equivalence, prospective validity, or locked-model transport.](../../manuscript/mdpi_information/assets/figures/main/figure_07_hrdataset_replication.png)

### 3.12 Evidence identity and claim control

All reported numerical claims were programmatically checked against a versioned evidence ledger containing source paths, selectors, values, qualifiers, and content hashes. Evidence packages record the relevant inputs, exclusions, source identities, and manifests so that manuscript numbers can be traced to frozen aggregate evidence without exposing employee-level rows. Complete claim identifiers and source locators are provided in the Supplementary Evidence Ledger rather than displayed in ordinary prose.

## 4. Results

### 4.1 Metric-specific benchmark leaders and extreme-class behavior

The P3 comparison did not yield one winner. Cumulative-threshold XGBoost had the highest macro-F1 (0.6255) and balanced accuracy (0.6745). Random Forest had the highest QWK (0.6317) and lowest ordinal MAE (0.1583). LightGBM had the lowest normalized RPS (0.0804), whereas nominal XGBoost had the lowest raw log loss (0.5515). Lower is preferable for MAE, RPS, and log loss; these ranks apply only to evaluated P3 systems and metrics.

Aggregate ordinal performance did not guarantee extreme-class success. Despite leading QWK and MAE, Random Forest had rating-4 recall 0.0000. Under the macro-F1-selected regime, cumulative-threshold XGBoost had rating-4 recall 0.4167 but substantially worse MAE (0.3142) than Random Forest. Thus, neither an aggregate improvement nor a metric-leader label should be read as adequate recognition of rating 4. Full classwise precision, recall, F1, and support for the four leading trained systems are reported in Supplementary Table S4, with corresponding confusion matrices supplied as supplementary data.

**Table 3. Nine-system exactly-once OOF benchmark under P3**

| System | Macro-F1 | Balanced acc. | QWK | Ordinal MAE | RPS | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cumulative-threshold XGBoost | 0.6255 | 0.6745 | 0.5550 | 0.3142 | 0.1029 | 1.3053 |
| Nominal XGBoost | 0.6210 | 0.6360 | 0.5676 | 0.2433 | 0.0860 | 0.5515 |
| LightGBM | 0.6055 | 0.6218 | 0.5883 | 0.1983 | 0.0804 | 0.5883 |
| Random Forest | 0.5923 | 0.6253 | 0.6317 | 0.1583 | 0.0822 | 0.5982 |
| Multinomial logistic | 0.5062 | 0.5240 | 0.3710 | 0.3550 | 0.1134 | 0.7142 |
| Proportional-odds logistic | 0.4844 | 0.5531 | 0.3927 | 0.4683 | 0.1405 | 0.8982 |
| Stratified baseline | 0.3304 | 0.3310 | 0.0243 | 0.4467 | — | — |
| Ordinal-median baseline | 0.2809 | 0.3333 | 0.0000 | 0.2717 | — | — |
| Majority baseline | 0.2809 | 0.3333 | 0.0000 | 0.2717 | — | — |

The three hard-label baselines are retained only as hard-prediction comparators and are not treated as probabilistic comparators. Probability-quality comparisons use the training-only empirical class-prior probability reference and the trained probabilistic systems.

### 4.2 Selection-objective sensitivity

Changing the inner selection objective from macro-F1 to QWK changed 37 of 60 outer-fold selected candidates. It changed the metric leader for 6 of 9 reported metrics and the complete ordering for all 9, but those three facts are not collapsed into a binary claim that the ranking "depends." The effect magnitudes and class-specific consequences varied by system. QWK selection raised nominal-XGBoost QWK from 0.5676 to 0.6418 and reduced MAE from 0.2433 to 0.1525, while macro-F1 fell from 0.6210 to 0.6014 and rating-4 recall fell from 0.1742 to 0.0076. Cumulative-threshold XGBoost showed a similar trade-off: QWK rose by 0.0769, from 0.5550 to 0.6319, and MAE fell from 0.3142 to 0.1533, while rating-4 recall fell from 0.4167 to 0.0000. Smaller lower-order changes are reported without being interpreted automatically as material.

**Table 4. Selection-objective sensitivity for the four leading trained systems**

| System | Regime | Macro-F1 | QWK | MAE | RPS | Rating-4 recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cumulative-threshold XGBoost | Macro-F1 selected | 0.6255 | 0.5550 | 0.3142 | 0.1029 | 0.4167 |
| Cumulative-threshold XGBoost | QWK selected | 0.5912 | 0.6319 | 0.1533 | 0.0666 | 0.0000 |
| LightGBM | Macro-F1 selected | 0.6055 | 0.5883 | 0.1983 | 0.0804 | 0.0606 |
| LightGBM | QWK selected | 0.5987 | 0.6097 | 0.1700 | 0.0756 | 0.0303 |
| Random Forest | Macro-F1 selected | 0.5923 | 0.6317 | 0.1583 | 0.0822 | 0.0000 |
| Random Forest | QWK selected | 0.5906 | 0.6335 | 0.1567 | 0.0832 | 0.0000 |
| Nominal XGBoost | Macro-F1 selected | 0.6210 | 0.5676 | 0.2433 | 0.0860 | 0.1742 |
| Nominal XGBoost | QWK selected | 0.6014 | 0.6418 | 0.1525 | 0.0675 | 0.0076 |

These class-4 failures were observed under the declared unmodified hard-decision rules. Alternative class-specific thresholding was not tested, so the results do not establish that every possible threshold policy would produce the same extreme-class behavior.

The empirical-prior probability reference had log loss 0.7683, Brier score 0.4313, RPS 0.1167, and top-label ECE 0.0000. Its zero top-label ECE results from the constant confidence construction and does not imply perfect classwise or ordinal calibration.

### 4.3 Repetition variability and ranking

Across five repeated designs, nominal XGBoost mean macro-F1 was 0.6288, versus 0.6249 for LightGBM. LightGBM won macro-F1 in three repetitions and XGBoost in two. Random Forest ranked first for QWK in 5/5 repetitions, and cumulative-threshold XGBoost ranked first for balanced accuracy in 4/5 repetitions. Mean pairwise macro-F1 rank Spearman correlation was 0.926 across ten dependent repetition pairs. These describe the fixed designs rather than population uncertainty.

**Table 5. Five-repetition nested-CV summaries**

| System | Macro-F1 mean ± SD | Balanced acc. mean ± SD | QWK mean ± SD | Ordinal MAE mean ± SD |
| --- | ---: | ---: | ---: | ---: |
| Nominal XGBoost | 0.6288 ± 0.0099 | 0.6446 ± 0.0094 | 0.5833 ± 0.0156 | 0.2335 ± 0.0150 |
| LightGBM | 0.6249 ± 0.0137 | 0.6364 ± 0.0127 | 0.6070 ± 0.0136 | 0.1902 ± 0.0051 |
| Cumulative-threshold XGBoost | 0.6155 ± 0.0059 | 0.6524 ± 0.0074 | 0.5451 ± 0.0083 | 0.3062 ± 0.0092 |
| Random Forest | 0.5955 ± 0.0027 | 0.6283 ± 0.0012 | 0.6311 ± 0.0022 | 0.1597 ± 0.0015 |

### 4.4 Nominal-XGBoost P3-to-P4 timing/information sensitivity

Within the nominal-XGBoost policy-sensitivity experiment, the restriction from P3 Primary Leakage-Aware to P4 Prospective-Plausibility produced the largest policy step after the diagnostic P0 contrast. Under the fixed schedule, P3-to-P4 changed macro-F1 by -0.2094, QWK by -0.3581, and MAE by +0.2400. With independent policy retuning, the corresponding changes were -0.1937, -0.3308, and +0.2400. From P4 to P5, the fixed changes were -0.0657 macro-F1, -0.1150 QWK, and +0.1125 MAE; the retuned changes were -0.0727, -0.1767, and +0.0683. Retuned macro-F1 was 0.8943 at P0, 0.6210 at P3, 0.4273 at P4, and 0.3547 at P5. In cross-schedule comparisons, P2 retuning changed macro-F1 by +0.0185, whereas P5 retuning changed QWK by -0.0344 even though its MAE improved. Because source timestamps are unavailable, these are nominal-XGBoost information-policy sensitivities rather than prospective validation.

**Table 6. Nominal-XGBoost fixed-schedule and independently retuned policy results**

| Policy | Fixed macro-F1 | Retuned macro-F1 | Fixed QWK | Retuned QWK | Fixed MAE | Retuned MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | 0.8914 | 0.8943 | 0.8496 | 0.8529 | 0.0775 | 0.0733 |
| P1 | 0.6279 | 0.6396 | 0.5814 | 0.5892 | 0.2308 | 0.2300 |
| P2 | 0.6150 | 0.6335 | 0.5569 | 0.5892 | 0.2483 | 0.2217 |
| P3 | 0.6210 | 0.6210 | 0.5676 | 0.5676 | 0.2433 | 0.2433 |
| P4 | 0.4117 | 0.4273 | 0.2095 | 0.2368 | 0.4833 | 0.4833 |
| P5 | 0.3460 | 0.3547 | 0.0945 | 0.0601 | 0.5958 | 0.5517 |

### 4.5 Explanation stability and calibration

The top five feature families were identical across 15 model-seed pairs: mean top-5 Jaccard 1.0000. Across ten outer-training-resample pairs, mean all-feature Spearman correlation was 0.9847; canonical outer-fold pairs had mean all-feature Spearman 0.9066. Pairwise comparisons are dependent and have no confidence interval. Deleting the top-ranked feature produced mean probability-drop contrast +0.2676 versus random deletion; contrasts remained +0.2481 at three features and +0.2063 at five. These support fitted-model sensitivity under the masking rule, not real-world intervention effects.

Sigmoid calibration reduced log loss from 0.5515 to 0.4556 and Brier score from 0.3426 to 0.2634. Macro classwise ECE declined from 0.1070 to 0.0249, mean cumulative ECE from 0.0779 to 0.0184, and normalized RPS from 0.0860 to 0.0669. Top-label ECE instead changed from 0.0375 to 0.0419, a worse point estimate by 0.0044 whose paired interval spans zero. Calibration performance therefore depends on event and metric.

**Table 7. SHAP and probability-quality diagnostics**

| Diagnostic | Result | Boundary |
| --- | ---: | --- |
| Seed-pair top-5 Jaccard | 1.0000 | 15 dependent descriptive pairs |
| Resample all-feature Spearman | 0.9847 | 10 dependent descriptive pairs |
| Top-1 guided-minus-random drop | +0.2676 | Masking diagnostic, not causal effect |
| Raw -> sigmoid log loss | 0.5515 -> 0.4556 | Training-only cross-fitted calibrator |
| Raw -> sigmoid Brier | 0.3426 -> 0.2634 | Metric-specific improvement |
| Raw -> sigmoid top-label ECE | 0.0375 -> 0.0419 | Point estimate worsened by 0.0044 |

### 4.6 Subgroup results across all prespecified attributes

At the prespecified support threshold of 30, the widest descriptive macro-F1/QWK/MAE gaps were: Age, 0.0301/0.0653/0.0770; Gender, 0.0363/0.0527/0.0508; Marital Status, 0.0101/0.0381/0.0291; Business Travel, 0.0528/0.0711/0.0613; Department, 0.2179/0.4388/0.1193; and Education, 0.0413/0.1190/0.0891. All six prespecified attributes are retained rather than reporting only the largest result. Full exploratory simultaneous intervals and eligibility details are reported in Supplementary Table S6. These diagnostics do not establish fairness or discrimination.

**Table 8. P3 descriptive subgroup gaps at support threshold 30**

| Attribute | Macro-F1 gap | QWK gap | MAE gap | Eligible groups, macro-F1/QWK/MAE |
| --- | ---: | ---: | ---: | --- |
| Age | 0.0301 | 0.0653 | 0.0770 | 4/4/4 of 5 |
| Gender | 0.0363 | 0.0527 | 0.0508 | 2/2/2 of 2 |
| Marital Status | 0.0101 | 0.0381 | 0.0291 | 3/3/3 of 3 |
| Business Travel | 0.0528 | 0.0711 | 0.0613 | 3/3/3 of 3 |
| Department | 0.2179 | 0.4388 | 0.1193 | 5/3/5 of 6 |
| Education | 0.0413 | 0.1190 | 0.0891 | 5/4/5 of 6 |

### 4.7 Proxy diagnostics

Refitting P3 without JobRole changed the argmax prediction for 10.75% of cases, with proxy-reduced-minus-primary macro-F1 -0.0330. Marginal within-fold JobRole permutation produced mean total variation 0.1024; department-conditional permutation produced 0.0529. Marginal perturbation can create implausible combinations, and conditioning only on department is not a fully conditional test.

Separate department reconstruction accuracy was 0.9792 with JobRole and 0.2908 without it. This documents an information channel but not how the performance model used department, absence of residual information, or discrimination.

**Table 9. Proxy-use and reconstructability diagnostics**

| Diagnostic | Value | Scope |
| --- | ---: | --- |
| P3 vs no-JobRole prediction-change rate | 10.75% | Paired OOF predictions after refitting |
| Macro-F1 difference, reduced minus P3 | -0.0330 | Descriptive refit contrast |
| Marginal permutation total variation | 0.1024 | 20 outcome-blind shuffles |
| Department-conditional total variation | 0.0529 | 20 outcome-blind shuffles |
| Reconstruction accuracy with JobRole | 0.9792 | Separate proxy task |
| Reconstruction accuracy without JobRole | 0.2908 | Separate proxy task |

### 4.8 HR target-mapping and CV-design sensitivity

Across five repeated designs under the retained three-class mapping, raw XGBoost had mean macro-F1 0.6531, QWK 0.5339, MAE 0.1916, log loss 0.5593, Brier score 0.3033, top-label ECE 0.0721, and RPS 0.0759. Its cross-fitted sigmoid counterpart had 0.6274, 0.6044, 0.1280, 0.4216, 0.2324, 0.0522, and 0.0589. Under the distinct four-class estimand, raw results were 0.5847, 0.6288, 0.2360, 0.6646, 0.3618, 0.0995, and 0.0604; sigmoid results were 0.5481, 0.6621, 0.1640, 0.5201, 0.2771, 0.0386, and 0.0488. These values are reported side by side without subtracting one target formulation from the other.

Of the 14 raw and sigmoid canonical 10x5 estimates assessed against repeated-5x5 ranges, 11 fell inside. The three outside-range estimates were raw macro-F1 and sigmoid Brier score and RPS. Range inclusion is descriptive and does not establish equivalence between validation designs.

**Table 10. HR target-mapping and CV-design sensitivity**

| Mapping/system | Macro-F1 | QWK | MAE | Log loss | Brier | ECE | RPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Three-class raw | 0.6531 | 0.5339 | 0.1916 | 0.5593 | 0.3033 | 0.0721 | 0.0759 |
| Three-class sigmoid | 0.6274 | 0.6044 | 0.1280 | 0.4216 | 0.2324 | 0.0522 | 0.0589 |
| Four-class raw | 0.5847 | 0.6288 | 0.2360 | 0.6646 | 0.3618 | 0.0995 | 0.0604 |
| Four-class sigmoid | 0.5481 | 0.6621 | 0.1640 | 0.5201 | 0.2771 | 0.0386 | 0.0488 |
| Canonical estimates inside repeated range | — | — | — | — | — | — | 11 of 14 overall |

### 4.9 HR target-alias and data-quality sensitivity

The two target-text/identifier disagreement rows were isolated without silently changing the historical result. Removing only those rows from the existing canonical OOF predictions changed the 311-row metrics to fit-free 309-row values: macro-F1 0.6570 -> 0.6624, QWK 0.5495 -> 0.5465, MAE 0.1801 -> 0.1780, and RPS 0.0725 -> 0.0719. These are sample-removal effects, not refitting effects.

On the matched 309-row population, exclusion and refitting relative to restricted canonical predictions changed macro-F1 by +0.004210, balanced accuracy by -0.002102, QWK by -0.000339, MAE by +0.000000, RPS by +0.000024, log loss by +0.010803, Brier score by +0.000668, and ECE by -0.011197. Selected candidates changed in 2 of 5 outer folds. The primary comparison thus separates training/data-rule sensitivity from the change in evaluation population. The separate fit-free 311-to-309 sample-removal changes were +0.005357 macro-F1, -0.003035 QWK, -0.002071 MAE, and -0.000636 RPS.

**Table 11. HR target-alias sensitivity with matched evaluation population**

| Arm | Rows | Macro-F1 | Balanced acc. | QWK | MAE | RPS | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical canonical OOF | 311 | 0.6570 | 0.6618 | 0.5495 | 0.1801 | 0.0725 | 0.5350 | 0.2903 | 0.0754 |
| Canonical OOF restricted, fit-free | 309 | 0.6624 | 0.6627 | 0.5465 | 0.1780 | 0.0719 | 0.5317 | 0.2876 | 0.0734 |
| Exclusion and refit | 309 | 0.6666 | 0.6606 | 0.5461 | 0.1780 | 0.0719 | 0.5425 | 0.2882 | 0.0622 |

The HRDataset_v14 audit also found 215 effective missing cells: 207 termination dates aligned with non-terminated records and eight manager identifiers. Three of 29 declared rules had findings, totaling six rule occurrences: two review-before-hire cases, two target-text/identifier disagreements, and two department-text minority cases. Occurrences are not asserted as six unique erroneous employees; source values were retained.

## 5. Discussion

### 5.1 Selection objective and extreme-class failure

Model choice is not captured by a single leaderboard. The shift from macro-F1 to QWK selection changed 37/60 selected candidates, 6/9 metric leaders, and 9/9 complete orderings, but these describe distinct levels of sensitivity. The scientifically important effect is metric- and model-specific: QWK selection greatly improved nominal-XGBoost QWK and MAE while almost eliminating rating-4 recall. Random Forest already led aggregate QWK and MAE with zero rating-4 recall. These failures were observed under the declared unmodified hard-decision rules; alternative threshold optimization was outside the prespecified scope. Aggregate QWK or MAE improvement therefore does not guarantee class-4 success, and class-specific metrics must remain visible beside ordinal summaries.

### 5.2 Timing and information access define the estimand

The nominal-XGBoost policy-sensitivity experiment shows that the information contract matters. P0's high performance is an information-rich diagnostic comparison, not evidence that a high-performing prospective system exists. The large P3-to-P4 deterioration under both fixed and retuned schedules shows sensitivity to removing timing-uncertain information. P4/P5 are semantically stricter but timestamps do not verify that retained fields precede the decision. Moreover, P3, P4, and P5 retain 20, 13, and 6 features, respectively; the contrasts therefore change feature identity and dimensionality together and cannot isolate timing-risk status from the general predictive information removed. The justified interpretation is nominal-XGBoost timing/information sensitivity under declared assumptions, not leakage freedom, a pure timing effect, or prospective validity.

### 5.3 Subgroup and proxy evidence are descriptive boundaries

Reporting all six attributes prevents the largest Department gap from becoming the whole subgroup narrative. The Department result is nevertheless salient, especially the QWK gap of 0.4388, and its restricted group eligibility limits interpretation. Exploratory simultaneous intervals are reported with eligibility in Supplementary Table S6 rather than being used as confirmatory fairness tests. JobRole refitting, permutation, and Department reconstruction answer different questions about performance-model output and available organizational information. None is a causal, legal, or confirmatory fairness test.

### 5.4 Target mapping and validation design change the question

The three- and four-class HR formulations change class support and target meaning, so their metrics are separate estimands rather than an improvement sequence. The HRDataset component is a partial cross-dataset protocol replication with dataset-specific fitting, not locked-model external validation. Likewise, 11/14 canonical estimates falling within repeated-design ranges is descriptive stability evidence, not equivalence. The matched 309-row alias analysis further shows why sample removal and model refitting must be separated: the primary refit contrasts are small for several aggregate metrics, while log loss and ECE move in opposite directions.

### 5.5 Explanations and calibration require metric-specific evidence

Exact-fold explanation prevents presenting a full-data-model explanation beside cross-validated performance. High top-k agreement and stronger deletion drops support internal consistency under tested procedures, but do not validate causal or employee-level meaning. Sigmoid calibration improved several probability metrics while worsening the top-label ECE point estimate, reinforcing that probability quality is multidimensional. The empirical-prior example also shows why one scalar ECE value can be misleading.

### 5.6 Relation to prior work

Prior employee studies supply classifier comparisons, HR scholarship explains sociotechnical stakes, and ordinal modelling, explanation, leakage, calibration, and reproducibility research supplies individual audit methods. Within the bounded set, this study's contribution is the shared evidence contract. It guards against model/explanation mismatch, held-out calibration fitting, conflation of information-policy and retuning effects, unsupported fairness inference, target-formulation conflation, and stale numbers. Supplementary Table S5 makes these requirements operational as reusable audit gates. Relevant unobserved work may overlap individual components.

## 6. Limitations

First, both datasets are public cross-sectional tables with unresolved source-to-byte provenance and redistribution rights. Public availability does not establish authenticity, ownership, representativeness, reuse permission, or redistribution permission. No raw dataset is approved for publication from the evidence currently available.

Second, feature and decision timestamps are absent. P0-P5 encode assumptions, not observed temporal order. P4 is a prospective-plausibility sensitivity; P5 cannot establish absence of residual proxies. The P3-to-P4-to-P5 policy contrasts also reduce retained dimensionality from 20 to 13 to 6 features, so they do not isolate timing-risk status from the predictive information removed with those features.

Third, targets are recorded organizational ratings. Documentation does not establish objective capability, productivity, or future potential, and the audit does not establish construct validity. The HRDataset target mappings do not prove category equivalence.

Fourth, samples are modest and imbalanced. Five-repetition ranges are not confidence intervals. Some subgroup/class cells are unsupported; exploratory intervals condition on observed sample, models, folds, and eligibility. Rating-4 failures remain especially important despite favorable aggregate metrics.

Fifth, SHAP values are noncausal raw-margin attributions. Stability pairs are dependent, masking can create out-of-distribution records, and no human study establishes explanation usefulness.

Sixth, calibration is retrospective on OOF predictions. ECE depends on event, binning, and support; future reliability, decision thresholds, and organizational utility remain untested.

Seventh, proxy analyses are diagnostic. Refitting changes model and feature set; shuffles are artificial; reconstructability does not establish department use, discrimination, fairness, or legal compliance.

Eighth, HRDataset_v14 is a partial cross-dataset protocol replication with models trained anew, not locked-model transport. Feature space, semantics, parameters, target formulation, and population differ. CV-range inclusion is not equivalence, and alias exclusion does not validate either source field as ground truth. Observation timestamps are also unavailable for the seven-feature HRDataset policy, so prospective availability is not verified there either.

Ninth, the revision-stage analyses were frozen before execution, but the broader research program evolved after prior inspection of the same dataset. Nested cross-validation protects against within-run model-selection bias but does not eliminate higher-level researcher adaptation to the dataset. The study should therefore be read as a traceable retrospective audit, not as a study preregistered before initial dataset inspection.

## 7. Conclusions

An ordinal employee-performance study changes meaning when selection objective, class-specific behavior, information policy, evaluation nesting, explanation identity, calibration path, subgroup support, target mapping, and proxy boundaries are explicit. Under P3, different systems led classification, ordinal, and probability metrics; QWK-oriented selection improved aggregate ordinal scores while extreme-class recall collapsed for important systems. In the separate nominal-XGBoost policy-sensitivity experiment, the P3-to-P4 step showed substantial timestamp-unverified information sensitivity. Subgroup, partial cross-dataset replication, target-mapping, target-alias, and CV-design results further bounded generalization.

The durable output is a traceable audit contract, not a production decision system. Future work requires timestamped data, validated constructs, authorized provenance, preregistered prospective evaluation, stronger conditional proxy tests, human-centered explanation studies, and institutionally approved governance before real employment use.

## Supplementary Materials

The journal-facing supplementary package contains: Supplementary Table S1, feature availability and information-policy contract; Supplementary Table S2, preprocessing and hyperparameter-search contract; Supplementary Table S3, CV and seed contract; Supplementary Table S4, full per-class metrics for the four leading trained systems; supplementary confusion matrices; Supplementary Table S5, protocol/audit gates; Supplementary Table S6, detailed subgroup gaps, eligibility, and exploratory simultaneous intervals; and the versioned Supplementary Evidence Ledger. Internal reviewer simulations, approval records, revision diffs, and development logs are not part of the journal supplementary materials.

## Author Contributions

[AUTHOR SIGN-OFF REQUIRED. Proposed CRediT statement is provided in `reports/submission_ready/AUTHOR_DECLARATIONS_DRAFT.md`; do not submit this placeholder.]

## Funding

[AUTHOR SIGN-OFF REQUIRED: provide funder/grant details or explicitly confirm no external funding.]

## Institutional Review Board Statement

[AUTHOR/INSTITUTION SIGN-OFF REQUIRED: provide the responsible institution/review unit's determination, reference/application number if applicable, date, and approved wording. No approval, exemption, waiver, or not-applicable determination is inferred here.]

## Informed Consent Statement

[AUTHOR/INSTITUTION SIGN-OFF REQUIRED: provide wording consistent with the institutional ethics determination and verified dataset provenance.]

## Data Availability Statement

Aggregate evidence, schemas, hashes, code, and qualified upstream locators are available in the project repository. Raw employee-level datasets are excluded from the submission package because the current evidence does not establish a complete authoritative source-to-byte chain together with redistribution permission. A final immutable release identifier and archive DOI may be added only after the authors resolve rights and explicitly authorize a release.

## Acknowledgments

[AUTHOR SIGN-OFF REQUIRED: approve final acknowledgment text or remove this section.]

## Conflicts of Interest

[AUTHOR SIGN-OFF REQUIRED: provide disclosures approved by all authors or explicitly confirm none.]

## Use of AI-Assisted Technologies

[AUTHOR SIGN-OFF REQUIRED. Proposed disclosure: During preparation of this work, the authors used OpenAI ChatGPT and Codex for code assistance and manuscript drafting/revision support. AI-assisted outputs were not treated as scientific evidence; numerical claims and manuscript-facing results were checked against versioned source evidence and deterministic validation procedures. The authors remain responsible for the content, interpretation, citations, and final submitted text. No generative-AI system is listed as an author.]

## References

The authoritative bibliography is `references.bib`; the submission audit retains the 34 source-verified references and checks citation-to-bibliography consistency before submission.