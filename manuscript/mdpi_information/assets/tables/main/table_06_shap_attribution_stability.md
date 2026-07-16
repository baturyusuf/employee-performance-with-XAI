# Grouped SHAP attribution and stability

Panel A is raw-margin model attribution. Panel B summarizes dependent fold pairs descriptively; no population confidence interval is constructed.

| panel | rank | feature_family | mean_absolute_grouped_shap_display | fold_top10_frequency | governance_category | top_k | fold_pair_count | jaccard_median_display | jaccard_iqr_display | jaccard_range_display | spearman_median_display | spearman_iqr_display | spearman_range_display | inference_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_attribution | 1 | EmpEnvironmentSatisfaction | 1.0789 | 10 | manager_controllable; sensitive_or_proxy=not_sensitive; leakage_risk=medium |  |  |  |  |  |  |  |  |  |
| A_attribution | 2 | YearsSinceLastPromotion | 0.5252 | 10 | organisation_controllable; sensitive_or_proxy=possible_proxy; leakage_risk=medium |  |  |  |  |  |  |  |  |  |
| A_attribution | 3 | EmpJobRole | 0.2653 | 10 | organisation_controllable; sensitive_or_proxy=possible_proxy; leakage_risk=low |  |  |  |  |  |  |  |  |  |
| A_attribution | 4 | EmpWorkLifeBalance | 0.2518 | 10 | manager_controllable; sensitive_or_proxy=possible_proxy; leakage_risk=medium |  |  |  |  |  |  |  |  |  |
| A_attribution | 5 | ExperienceYearsInCurrentRole | 0.2243 | 10 | organisation_controllable; sensitive_or_proxy=possible_proxy; leakage_risk=low |  |  |  |  |  |  |  |  |  |
| A_attribution | 6 | EmpHourlyRate | 0.1322 | 10 | organisation_controllable; sensitive_or_proxy=possible_proxy; leakage_risk=low |  |  |  |  |  |  |  |  |  |
| A_attribution | 7 | YearsWithCurrManager | 0.1025 | 10 | organisation_controllable; sensitive_or_proxy=possible_proxy; leakage_risk=low |  |  |  |  |  |  |  |  |  |
| A_attribution | 8 | DistanceFromHome | 0.0993 | 9 | immutable; sensitive_or_proxy=possible_proxy; leakage_risk=none |  |  |  |  |  |  |  |  |  |
| A_attribution | 9 | TotalWorkExperienceInYears | 0.0910 | 9 | immutable; sensitive_or_proxy=possible_proxy; leakage_risk=none |  |  |  |  |  |  |  |  |  |
| A_attribution | 10 | EmpJobSatisfaction | 0.0770 | 4 | manager_controllable; sensitive_or_proxy=not_sensitive; leakage_risk=medium |  |  |  |  |  |  |  |  |  |
| A_attribution | 11 | ExperienceYearsAtThisCompany | 0.0729 | 3 | immutable; sensitive_or_proxy=possible_proxy; leakage_risk=none |  |  |  |  |  |  |  |  |  |
| A_attribution | 12 | TrainingTimesLastYear | 0.0643 | 1 | employee_controllable; sensitive_or_proxy=not_sensitive; leakage_risk=low |  |  |  |  |  |  |  |  |  |
| A_attribution | 13 | BusinessTravelFrequency | 0.0580 | 1 | organisation_controllable; sensitive_or_proxy=possible_proxy; leakage_risk=low |  |  |  |  |  |  |  |  |  |
| A_attribution | 14 | NumCompaniesWorked | 0.0558 | 0 | immutable; sensitive_or_proxy=possible_proxy; leakage_risk=none |  |  |  |  |  |  |  |  |  |
| A_attribution | 15 | EducationBackground | 0.0541 | 1 | immutable; sensitive_or_proxy=possible_proxy; leakage_risk=low |  |  |  |  |  |  |  |  |  |
| A_attribution | 16 | EmpEducationLevel | 0.0530 | 2 | employee_controllable; sensitive_or_proxy=possible_proxy; leakage_risk=none |  |  |  |  |  |  |  |  |  |
| A_attribution | 17 | EmpRelationshipSatisfaction | 0.0454 | 0 | manager_controllable; sensitive_or_proxy=not_sensitive; leakage_risk=medium |  |  |  |  |  |  |  |  |  |
| A_attribution | 18 | EmpJobLevel | 0.0434 | 0 | organisation_controllable; sensitive_or_proxy=possible_proxy; leakage_risk=low |  |  |  |  |  |  |  |  |  |
| A_attribution | 19 | EmpJobInvolvement | 0.0329 | 0 | employee_controllable; sensitive_or_proxy=not_sensitive; leakage_risk=medium |  |  |  |  |  |  |  |  |  |
| A_attribution | 20 | OverTime | 0.0324 | 0 | manager_controllable; sensitive_or_proxy=possible_proxy; leakage_risk=medium |  |  |  |  |  |  |  |  |  |
| B_stability |  |  |  |  |  | 5 | 45.0 | 1.0000 | [1.0000, 1.0000] | [1.0000, 1.0000] | 0.9173 | [0.8812, 0.9368] | [0.7940, 0.9774] | descriptive_dependent_fold_pairs_no_confidence_interval |
| B_stability |  |  |  |  |  | 10 | 45.0 | 0.8182 | [0.8182, 0.8182] | [0.5385, 1.0000] | 0.9173 | [0.8812, 0.9368] | [0.7940, 0.9774] | descriptive_dependent_fold_pairs_no_confidence_interval |
| B_stability |  |  |  |  |  | 15 | 45.0 | 0.8750 | [0.7647, 0.8750] | [0.6667, 1.0000] | 0.9173 | [0.8812, 0.9368] | [0.7940, 0.9774] | descriptive_dependent_fold_pairs_no_confidence_interval |
