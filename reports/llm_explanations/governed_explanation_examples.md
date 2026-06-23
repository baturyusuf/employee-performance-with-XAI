# Governed Explanation Examples

## Case 528

### Structured Evidence Summary
- Predicted class: 2
- Confidence: 0.9836298763239094
- Feature policy: no_salary_hike_no_attrition_no_department

### Governed Explanation
For case 528, the model predicts class 2 with probability 0.9836298763239094 under the leakage-safe feature policy `no_salary_hike_no_attrition_no_department`. The provided evidence says this is a research-only decision-support setting and not an autonomous HR evaluator. The strongest positive SHAP attributions are EmpEnvironmentSatisfaction (1.6565), EmpJobRole (0.7740), YearsSinceLastPromotion (0.7484), EmpHourlyRate (0.2580), and EmpJobLevel (0.2341). The strongest negative SHAP attributions are EmpWorkLifeBalance (-0.1076), NumCompaniesWorked (-0.0450), EmpEducationLevel (-0.0285), TrainingTimesLastYear (-0.0013), and BusinessTravelFrequency (-0.0006). SHAP here is attribution, not causality, so these values should not be read as causes of the prediction. The SHAP stability summary reports Spearman 0.8717 and top-10 Jaccard 0.7606, which indicates some stability in the explanation ranking. Calibration evidence is limited: Brier score is 0.2608, expected calibration error is 0.0638, and the evidence explicitly warns that probabilities should be interpreted as approximate confidence bands, not calibrated truth. Counterfactual evidence is not employee-actionable: actionability_label is `not_employee_actionable`, validity is 0.0, no features changed, and the counterfactual failed because employee-only validity is zero or unavailable. Fairness evidence includes an EmpDepartment macro F1 gap of 0.2689 with bootstrap CI [0.1849, 0.3492], plus low-support warnings for EmpDepartment/Data Science (n=20) and EducationBackground/Human Resources (n=21). Proxy-risk warnings state that EmpJobRole may proxy EmpDepartment and that removing EmpDepartment does not prove fairness. Leakage evidence shows the full-feature score is 0.9038 versus leakage-safe score 0.5965, with leakage sensitivity index 0.3400; full-feature models are explicitly leakage-warning upper-bound baselines only.

### Compliance Check
- Pass: True
- Score: 100
- Missing warnings: []

### Warnings
- causality / high: SHAP is attribution, not causality.
- calibration / medium: Probability estimates should be interpreted as approximate confidence, not objective certainty.
- actionability / high: Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- fairness / high: Removing EmpDepartment does not prove fairness.
- proxy / high: EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- leakage / high: Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- deployment / high: Prediction requires human review.
- other / medium: No subgroup_metrics were provided, so subgroup-level evidence is unavailable for this case.
- deployment / high: This model is decision support only and is not for autonomous HR decisions.
- validation / high: External validation is required before deployment.

## Case 376

### Structured Evidence Summary
- Predicted class: 3
- Confidence: 0.9948591941166247
- Feature policy: no_salary_hike_no_attrition_no_department

### Governed Explanation
Using the provided research-only evidence, the model predicted class 3 for case 376 with class probability 0.9948591941166247 under the feature policy no_salary_hike_no_attrition_no_department. The prediction matches the provided true class. The calibration summary says probabilities should be interpreted as approximate confidence bands, not calibrated truth; reported calibration metrics are Brier score 0.2608022278073826, expected calibration error 0.0638464082070905, and log loss 0.45509175349537. SHAP attribution for class 3 shows the largest positive contributions from DistanceFromHome (0.6831), TotalWorkExperienceInYears (0.4654), EmpEnvironmentSatisfaction (0.3063), YearsSinceLastPromotion (0.2346), and OverTime (0.1791), while the largest negative contributions include TrainingTimesLastYear (-0.1147), BusinessTravelFrequency (-0.0862), and EmpJobLevel (-0.0761). The SHAP stability summary reports Spearman 0.8717 and top-10 Jaccard 0.7606, but SHAP remains attribution, not causality. Counterfactual evidence is not employee-actionable: actionability_label is not_employee_actionable, validity is 0.0, changed_features is empty, and the failed reason is employee_only validity is zero or unavailable. Fairness evidence includes an EmpDepartment macro F1 gap of 0.2688590531332467 with bootstrap CI [0.1848783532680399, 0.3492426064838457], plus low-support warnings for EmpDepartment/Data Science (n=20) and EducationBackground/Human Resources (n=21). Proxy-risk warnings state that EmpJobRole may proxy EmpDepartment and that removing EmpDepartment does not prove fairness. Leakage evidence shows the full-feature score is 0.9037820216268916 versus leakage-safe score 0.5965327889347274, with leakage_sensitivity_index 0.3399594430292904; full-feature models are leakage-warning upper-bound baselines only. This is research-only decision support with strong warnings, not an autonomous HR evaluator.

### Compliance Check
- Pass: True
- Score: 100
- Missing warnings: []

### Warnings
- calibration / medium: Probability estimates should be interpreted as approximate confidence, not objective certainty.
- causality / high: SHAP is attribution, not causality.
- actionability / high: Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- fairness / high: Removing EmpDepartment does not prove fairness.
- proxy / high: EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- leakage / high: Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- deployment / high: This model is decision support only and is not for autonomous HR decisions.
- other / medium: No evidence was provided for subgroup_metrics beyond the listed warnings and gaps.
- deployment / high: Prediction requires human review.
- validation / high: External validation is required before deployment.

## Case 568

### Structured Evidence Summary
- Predicted class: 4
- Confidence: 0.9053656427743827
- Feature policy: no_salary_hike_no_attrition_no_department

### Governed Explanation
For case 568, the leakage-safe XGBoost model predicts class 4 with probability 0.9053656427743827, and the recorded true class is 4. The model is marked as final_candidate_leakage_safe under the feature policy no_salary_hike_no_attrition_no_department. The strongest positive SHAP attributions are DistanceFromHome (0.6562257409095764), EmpEnvironmentSatisfaction (0.5128120183944702), EmpJobSatisfaction (0.3224063813686371), ExperienceYearsInCurrentRole (0.22711484134197235), and BusinessTravelFrequency (0.14459040760993958). The strongest negative attribution is EmpJobRole (-0.3753000497817993), followed by smaller negative contributions from EmpEducationLevel, EmpWorkLifeBalance, EmpHourlyRate, and YearsSinceLastPromotion. These are model attributions only; they do not establish causal effects. Calibration evidence says probabilities should be interpreted as approximate confidence bands, not calibrated truth. Counterfactual evidence is not employee-actionable: actionability_label is not_employee_actionable, validity is 0.0, and no changed features were found. Fairness evidence includes an EmpDepartment macro F1 gap of 0.2688590531332467 with bootstrap CI [0.1848783532680399, 0.3492426064838457], plus low-support warnings for EmpDepartment/Data Science (n=20) and EducationBackground/Human Resources (n=21). Proxy-risk warnings note that EmpJobRole may proxy EmpDepartment and that removing EmpDepartment does not prove fairness. Deployment status is research_only_decision_support_with_strong_warnings, and human review is required.

### Compliance Check
- Pass: True
- Score: 100
- Missing warnings: []

### Warnings
- calibration / medium: Probability estimates should be interpreted as approximate confidence, not objective certainty.
- causality / high: SHAP is attribution, not causality.
- actionability / high: Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- fairness / high: Removing EmpDepartment does not prove fairness.
- proxy / high: EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- deployment / high: This model is decision support only and is not for autonomous HR decisions.
- deployment / high: Prediction requires human review.
- leakage / high: Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- validation / high: External validation is required before deployment.

## Case 18

### Structured Evidence Summary
- Predicted class: 3
- Confidence: 0.8735988525633045
- Feature policy: no_salary_hike_no_attrition_no_department

### Governed Explanation
For case 18, the leakage-safe XGBoost model predicts class 3 with probability 0.8735988525633045 under the feature policy `no_salary_hike_no_attrition_no_department`. The true class is 4. The calibration evidence says probabilities should be treated as approximate confidence bands, not calibrated truth. SHAP attribution for this case is moderately stable overall (Spearman 0.8717293233082707; top-10 Jaccard 0.7606060606060606), but SHAP remains attribution, not causality.

The largest positive SHAP attributions are EmpEnvironmentSatisfaction (0.26594287157058716), EmpJobRole (0.217536062002182), YearsWithCurrManager (0.1778436154127121), EmpHourlyRate (0.1300954818725586), and OverTime (0.09803373366594315). The largest negative attributions are YearsSinceLastPromotion (-0.14117829501628876), EducationBackground (-0.13574816286563873), EmpJobInvolvement (-0.06191493570804596), TotalWorkExperienceInYears (-0.047165095806121826), and EmpJobSatisfaction (-0.044420480728149414). These are model attributions only and should not be read as causes or prescriptions.

Counterfactual evidence is not employee-actionable: actionability_label is `not_employee_actionable`, validity is 0.0, no features changed, and the counterfactual failed because `employee_only validity is zero or unavailable`. Fairness evidence includes an EmpDepartment macro F1 gap of 0.2688590531332467 with bootstrap CI [0.1848783532680399, 0.3492426064838457], plus low-support warnings for EmpDepartment/Data Science (n=20) and EducationBackground/Human Resources (n=21). Proxy-risk warnings note that EmpJobRole may proxy EmpDepartment and that removing EmpDepartment does not prove fairness. Leakage evidence shows the full-feature score is 0.9037820216268916 versus leakage-safe score 0.5965327889347274, with leakage_sensitivity_index 0.3399594430292904; full-feature models are leakage-warning upper-bound baselines only.

### Compliance Check
- Pass: True
- Score: 100
- Missing warnings: []

### Warnings
- calibration / medium: Probability estimates should be interpreted as approximate confidence, not objective certainty.
- causality / high: SHAP is attribution, not causality.
- actionability / high: Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- fairness / high: Removing EmpDepartment does not prove fairness.
- proxy / high: EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- leakage / high: Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- deployment / high: Prediction requires human review.
- deployment / high: This model is decision support only and is not for autonomous HR decisions.
- validation / high: External validation is required before deployment.

## Case 392

### Structured Evidence Summary
- Predicted class: 3
- Confidence: 0.4269939724653061
- Feature policy: no_salary_hike_no_attrition_no_department

### Governed Explanation
The leakage-safe XGBoost model predicted class 3 for case 392, and the true class is also 3. The reported class probability for class 3 is 0.4269939724653061, with uncertainty flagged. Calibration evidence says probabilities should be treated as approximate confidence bands, not calibrated truth. The calibration metrics provided are Brier score 0.2608022278073826, expected calibration error 0.0638464082070905, and log loss 0.45509175349537.

The strongest positive SHAP attributions were EmpHourlyRate (0.18021847307682037), YearsWithCurrManager (0.1623276025056839), EmpJobSatisfaction (0.10358555614948273), DistanceFromHome (0.08926171064376831), and BusinessTravelFrequency (0.06151972711086273). The strongest negative SHAP attributions were EmpEnvironmentSatisfaction (-0.6402941346168518), YearsSinceLastPromotion (-0.33148953318595886), EmpJobRole (-0.12770596146583557), EmpWorkLifeBalance (-0.0724664255976677), and ExperienceYearsAtThisCompany (-0.05383184552192688). These are model attributions only, not causal effects.

The SHAP stability summary reports Spearman 0.8717293233082707 and top-10 Jaccard 0.7606060606060606, which supports some stability in the ranked explanations. However, several features have proxy or temporality warnings: EmpJobRole may proxy EmpDepartment, YearsSinceLastPromotion may encode prior decisions, and some features may be measured near evaluation time. Fairness evidence includes an EmpDepartment macro F1 gap of 0.2688590531332467 with bootstrap CI low 0.1848783532680399 and high 0.3492426064838457, plus low-support warnings for Data Science (n=20) and Human Resources in EducationBackground (n=21). The evidence also states that removing EmpDepartment does not prove fairness.

Counterfactual evidence is not employee-actionable: actionability_label is not_employee_actionable, changed_features is empty, validity is 0.0, and the failed reason is that employee_only validity is zero or unavailable. The model card and governance evidence state this is research-only decision support with strong warnings, not for hiring, firing, promotion, compensation, disciplinary action, or individual employment decisions. External validation is required.

### Compliance Check
- Pass: True
- Score: 100
- Missing warnings: []

### Warnings
- calibration / medium: Probability estimates should be interpreted as approximate confidence, not objective certainty.
- causality / high: SHAP is attribution, not causality.
- deployment / high: This model is decision support only and is not for autonomous HR decisions.
- actionability / high: Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- fairness / high: Removing EmpDepartment does not prove fairness.
- proxy / high: EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- leakage / high: Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- deployment / high: Prediction requires human review.
- validation / high: External validation is required before deployment.

## Case 405

### Structured Evidence Summary
- Predicted class: 2
- Confidence: 0.9832764863967896
- Feature policy: no_salary_hike_no_attrition_no_department

### Governed Explanation
For case 405, the leakage-safe XGBoost model predicts class 2 and the provided true class is also 2. The reported class probability for class 2 is 0.9832764863967896. Calibration evidence says probabilities should be treated as approximate confidence bands, not calibrated truth. SHAP attributions show the strongest positive contribution from EmpEnvironmentSatisfaction (1.71826171875), followed by YearsSinceLastPromotion (0.6144664287567139), EmpJobRole (0.4599297344684601), EmpJobLevel (0.273281991481781), and YearsWithCurrManager (0.2513692677021026). Negative attributions include EmpWorkLifeBalance (-0.1105070188641548), EmpEducationLevel (-0.0702903196215629), TotalWorkExperienceInYears (-0.0552969574928283), EducationBackground (-0.0294284634292125), and DistanceFromHome (-0.0258172713220119). The SHAP stability summary reports spearman 0.8717293233082707 and top10_jaccard 0.7606060606060606, but SHAP remains attribution only, not causal explanation. Fairness evidence includes an EmpDepartment macro F1 gap of 0.2688590531332467 with bootstrap CI low 0.1848783532680399 and high 0.3492426064838457, plus low-support warnings for EmpDepartment/Data Science (n=20) and EducationBackground/Human Resources (n=21). Proxy-risk warnings state that EmpJobRole may proxy EmpDepartment and that removing EmpDepartment does not prove fairness. Counterfactual evidence is not employee-actionable: actionability_label is not_employee_actionable, validity is 0.0, changed_features is empty, and the failed reason is that employee_only validity is zero or unavailable. Governance evidence states deployment is research_only_decision_support_with_strong_warnings, prohibited for autonomous HR decisions, and human review is required.

### Compliance Check
- Pass: True
- Score: 100
- Missing warnings: []

### Warnings
- calibration / medium: Probability estimates should be interpreted as approximate confidence, not objective certainty.
- causality / high: SHAP is attribution, not causality.
- fairness / high: Removing EmpDepartment does not prove fairness.
- proxy / high: EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- actionability / high: Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- deployment / high: This model is decision support only and is not for autonomous HR decisions.
- deployment / high: Prediction requires human review.
- leakage / high: Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- validation / high: External validation is required before deployment.

## Case 125

### Structured Evidence Summary
- Predicted class: 3
- Confidence: 0.9922711849212646
- Feature policy: no_salary_hike_no_attrition_no_department

### Governed Explanation
For case 125, the leakage-safe XGBoost model predicted class 3 with probability 0.9922711849212646 and the true class is also 3. The feature policy is no_salary_hike_no_attrition_no_department, and the case is marked final_candidate_leakage_safe. The calibration evidence says probabilities should be interpreted as approximate confidence bands, not calibrated truth; reported calibration metrics are Brier score 0.2608022278073826, expected calibration error 0.0638464082070905, and log loss 0.45509175349537. SHAP values are provided as grouped attributions only, and SHAP is attribution, not causality. The largest positive attributions were EmpJobRole (0.4692350029945373), EducationBackground (0.419499784708023), YearsSinceLastPromotion (0.3233237564563751), ExperienceYearsInCurrentRole (0.2717743813991546), and EmpEnvironmentSatisfaction (0.1255460083484649). The largest negative attributions were TotalWorkExperienceInYears (-0.1689964234828949), EmpRelationshipSatisfaction (-0.1095330566167831), and ExperienceYearsAtThisCompany (-0.0455411076545715). Several features are flagged as possible proxies or historical-decision signals, including EmpJobRole, EducationBackground, YearsSinceLastPromotion, TotalWorkExperienceInYears, and ExperienceYearsAtThisCompany. Fairness evidence includes an EmpDepartment macro F1 gap of 0.2688590531332467 with bootstrap CI [0.1848783532680399, 0.3492426064838457], plus low-support warnings for EmpDepartment=Data Science (n=20) and EducationBackground=Human Resources (n=21). Proxy-risk warnings state that EmpJobRole may proxy EmpDepartment and that removing EmpDepartment does not prove fairness. Counterfactual evidence is not employee-actionable: actionability_label is not_employee_actionable, validity is 0.0, changed_features is empty, and the failed reason is employee_only validity is zero or unavailable. Deployment status is research_only_decision_support_with_strong_warnings, and human review is required. This output must not be used for autonomous HR decisions.

### Compliance Check
- Pass: True
- Score: 100
- Missing warnings: []

### Warnings
- calibration / medium: Probability estimates should be interpreted as approximate confidence, not objective certainty.
- causality / high: SHAP is attribution, not causality.
- deployment / high: This model is decision support only and is not for autonomous HR decisions.
- actionability / high: Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- fairness / high: Removing EmpDepartment does not prove fairness.
- proxy / high: EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- other / medium: Some subgroup metrics are unavailable; subgroup_metrics is empty and evidence is limited to the provided JSON.
- leakage / high: Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- deployment / high: Prediction requires human review.
- validation / high: External validation is required before deployment.

## Case 176

### Structured Evidence Summary
- Predicted class: 4
- Confidence: 0.5255113840103149
- Feature policy: no_salary_hike_no_attrition_no_department

### Governed Explanation
For case 176, the leakage-safe XGBoost model predicts class 4 with probability 0.5255113840103149 and flags uncertainty. The strongest positive SHAP attributions are ExperienceYearsInCurrentRole (0.3240), YearsWithCurrManager (0.2188), EmpJobSatisfaction (0.1951), DistanceFromHome (0.1950), and TotalWorkExperienceInYears (0.1950). The strongest negative SHAP attributions are EmpEnvironmentSatisfaction (-0.2897), BusinessTravelFrequency (-0.1187), OverTime (-0.0979), EmpJobInvolvement (-0.0219), and EmpJobRole (-0.0080). SHAP here is attribution, not causality, so these values should not be treated as reasons that changes would necessarily alter the outcome. Calibration evidence says probabilities should be interpreted as approximate confidence bands, not calibrated truth. Counterfactual evidence is unavailable for employee-actionable guidance: actionability_label is not_employee_actionable, validity is 0.0, and no features changed. Fairness evidence includes an EmpDepartment macro F1 gap of 0.2689 with bootstrap CI [0.1849, 0.3492], plus low-support warnings for EmpDepartment/Data Science (n=20) and EducationBackground/Human Resources (n=21). Proxy-risk warnings note that EmpJobRole may proxy EmpDepartment and that removing EmpDepartment does not prove fairness. Leakage evidence shows the full-feature score is 0.9038 versus leakage-safe score 0.5965, with leakage sensitivity index 0.3400; full-feature models are leakage-warning upper-bound baselines only. Deployment status is research-only decision support with strong warnings, and human review is required.

### Compliance Check
- Pass: True
- Score: 100
- Missing warnings: []

### Warnings
- calibration / medium: Probability estimates should be interpreted as approximate confidence, not objective certainty.
- causality / high: SHAP is attribution, not causality.
- actionability / high: Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- fairness / high: Removing EmpDepartment does not prove fairness.
- proxy / high: EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- leakage / high: Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- deployment / high: This model is decision support only and is not for autonomous HR decisions.
- deployment / high: Prediction requires human review.
- validation / high: External validation is required before deployment.

## Case 662

### Structured Evidence Summary
- Predicted class: 3
- Confidence: 0.984155535697937
- Feature policy: no_salary_hike_no_attrition_no_department

### Governed Explanation
The model output for case 662 is predicted class 3 with class probability 0.984155535697937 under the feature policy `no_salary_hike_no_attrition_no_department`. The true class is 2. Calibration evidence says probabilities should be treated as approximate confidence bands, not calibrated truth. SHAP attributions indicate positive contributions from EmpEnvironmentSatisfaction (0.5424), EducationBackground (0.5040), ExperienceYearsInCurrentRole (0.1962), YearsWithCurrManager (0.1700), and EmpJobSatisfaction (0.1422). Negative contributions include TrainingTimesLastYear (-0.1549), YearsSinceLastPromotion (-0.1364), EmpHourlyRate (-0.0917), EmpJobRole (-0.0882), and OverTime (-0.0310). SHAP is attribution, not causality, so these values should not be read as causes of the prediction. Counterfactual evidence is not employee-actionable: validity is 0.0, no features changed, and the counterfactual failed because employee_only validity is zero or unavailable. Fairness evidence shows an EmpDepartment macro F1 gap of 0.2689 with bootstrap CI [0.1849, 0.3492], plus low-support warnings for EmpDepartment=Data Science (n=20) and EducationBackground=Human Resources (n=21). Proxy-risk warnings note that EmpJobRole may proxy EmpDepartment and that removing EmpDepartment does not prove fairness. Leakage evidence shows the full-feature score is 0.9038 versus leakage-safe score 0.5965, with leakage sensitivity index 0.3400; full-feature models are leakage-warning upper-bound baselines only. Deployment status is research-only decision support with strong warnings, and predictions require human review.

### Compliance Check
- Pass: True
- Score: 100
- Missing warnings: []

### Warnings
- calibration / medium: Probability estimates should be interpreted as approximate confidence, not objective certainty.
- causality / high: SHAP is attribution, not causality.
- actionability / high: Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- fairness / high: Removing EmpDepartment does not prove fairness.
- proxy / high: EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- leakage / high: Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- deployment / high: This model is decision support only and is not for autonomous HR decisions.
- deployment / high: Prediction requires human review.
- validation / high: External validation is required before deployment.

## Case 906

### Structured Evidence Summary
- Predicted class: 3
- Confidence: 0.5060936212539673
- Feature policy: no_salary_hike_no_attrition_no_department

### Governed Explanation
This is research-only decision support, not an autonomous HR decision tool. For case 906, the model predicted class 3 and the provided true class is also 3. The reported confidence is 0.5060936212539673, and the calibration note says probabilities should be interpreted as approximate confidence bands, not calibrated truth. The uncertainty flag is true, so human review is required.

The SHAP explanation shows the strongest positive attributions for EmpJobRole (0.2232), EmpEnvironmentSatisfaction (0.1159), EmpJobLevel (0.0292), TrainingTimesLastYear (0.0259), and EmpJobSatisfaction (0.0252). The strongest negative attributions were YearsSinceLastPromotion (-0.2100), YearsWithCurrManager (-0.1763), BusinessTravelFrequency (-0.1380), ExperienceYearsInCurrentRole (-0.1221), and ExperienceYearsAtThisCompany (-0.0947). These are model attributions only; they do not establish causal effects.

Counterfactual evidence is unavailable for employee-actionable guidance: actionability_label is not_employee_actionable, changed_features is empty, validity is 0.0, and the failed reason states employee_only validity is zero or unavailable. The counterfactual warning also states that counterfactuals are model scenarios, not employee prescriptions.

Fairness evidence shows an audited disparity for EmpDepartment with a macro F1 gap of 0.2689 and bootstrap CI from 0.1849 to 0.3492. Low-support warnings are present for EmpDepartment/Data Science (n=20) and EducationBackground/Human Resources (n=21), both below 30 samples. Proxy-risk warnings state that EmpJobRole may proxy EmpDepartment and that removing EmpDepartment does not prove fairness.

Leakage evidence indicates the feature policy excludes EmpLastSalaryHikePercent, Attrition, Age, and EmpDepartment. The leakage-safe score is 0.5965, while the full-feature score is 0.9038; the leakage sensitivity index is 0.3400. The full-feature model is explicitly described as a leakage-warning upper-bound baseline only.

Overall, the evidence supports a cautious, human-reviewed interpretation only. No hiring, firing, promotion, compensation, disciplinary, or individual employment decision should be made from this output alone.

### Compliance Check
- Pass: True
- Score: 100
- Missing warnings: []

### Warnings
- calibration / medium: Probability estimates should be interpreted as approximate confidence, not objective certainty.
- actionability / high: Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- causality / high: SHAP is attribution, not causality.
- fairness / high: Removing EmpDepartment does not prove fairness.
- proxy / high: EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- leakage / high: Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- deployment / high: This model is decision support only and is not for autonomous HR decisions.
- deployment / high: Prediction requires human review.
- validation / high: External validation is required before deployment.

