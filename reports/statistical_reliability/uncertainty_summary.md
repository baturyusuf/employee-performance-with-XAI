# Statistical Reliability and Uncertainty Summary

This report uses available fold-level, bootstrap, and binomial evidence. It does not impute missing uncertainty estimates.



## Performance CI

| dataset | metric | n | mean | std | ci_low | ci_high | method | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inx_primary | macro_f1 | 10 | 0.5987215278292141 | 0.0283326201269661 | 0.5811607879396318 | 0.6162822677187964 | fold_level_normal_approximation | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\leakage_safe_cv_config\fold_metrics.csv |
| inx_primary | balanced_accuracy | 10 | 0.626250421286719 | 0.0260256466267886 | 0.6101195583865303 | 0.6423812841869078 | fold_level_normal_approximation | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\leakage_safe_cv_config\fold_metrics.csv |
| inx_primary | quadratic_weighted_kappa | 10 | 0.6380278784486674 | 0.0517121269738979 | 0.6059763700863551 | 0.6700793868109797 | fold_level_normal_approximation | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\leakage_safe_cv_config\fold_metrics.csv |
| inx_primary | ordinal_mae | 10 | 0.1524999999999999 | 0.0204313061649258 | 0.1398365452415324 | 0.1651634547584674 | fold_level_normal_approximation | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\leakage_safe_cv_config\fold_metrics.csv |
| inx_primary | nll_log_loss | 10 | 0.4776955928670851 | 0.0410454593198426 | 0.4522553536116213 | 0.5031358321225489 | fold_level_normal_approximation | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\leakage_safe_cv_config\fold_metrics.csv |
| inx_primary | multiclass_brier | 10 | 0.2665237065304438 | 0.019681773529117 | 0.2543248169126942 | 0.2787225961481934 | fold_level_normal_approximation | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\leakage_safe_cv_config\fold_metrics.csv |
| inx_primary | ece_confidence | 10 | 0.0808086531112591 | 0.0267030459777181 | 0.0642579337435168 | 0.0973593724790014 | fold_level_normal_approximation | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\leakage_safe_cv_config\fold_metrics.csv |

## Fairness Disparity CI

| feature_set | attribute | metric | class_label | point_estimate | ci_low | ci_high | bootstrap_std | n_boot_valid | min_group_support_threshold | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_salary_hike_no_attrition_no_department | BusinessTravelFrequency | accuracy |  | 0.040700274742828 | 0.01121089614897 | 0.1172613125841605 | 0.0273559257582656 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | BusinessTravelFrequency | false_positive_rate | 3.0 | 0.0854821380607037 | 0.0275144517142723 | 0.2649400078678205 | 0.0640834159164868 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | BusinessTravelFrequency | false_positive_rate | 4.0 | 0.0175438596491228 | 0.0 | 0.0483957593872594 | 0.0129710943738764 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | BusinessTravelFrequency | false_positive_rate | 2.0 | 0.005367339170156 | 0.0034069322949539 | 0.0454529094093034 | 0.010952046862524 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | BusinessTravelFrequency | macro_f1 |  | 0.0186995463841231 | 0.0053435391971541 | 0.0676162143556926 | 0.0159428205569063 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | BusinessTravelFrequency | mean_predicted_probability | 3.0 | 0.0307124083157762 | 0.0091906737814393 | 0.0958725043110525 | 0.0218054422067089 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | BusinessTravelFrequency | mean_predicted_probability | 4.0 | 0.0267644421519323 | 0.0114157188496697 | 0.0497740402279847 | 0.0097216552967956 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | BusinessTravelFrequency | mean_predicted_probability | 2.0 | 0.0057165604288218 | 0.0059127313508541 | 0.0835675226921357 | 0.0210182037119568 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | BusinessTravelFrequency | positive_prediction_rate | 3.0 | 0.0257985257985258 | 0.0086799062057256 | 0.1219487024736212 | 0.0292884897577457 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | BusinessTravelFrequency | positive_prediction_rate | 4.0 | 0.0151515151515151 | 0.0005367231638418 | 0.0413223140495867 | 0.0109935621428713 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | BusinessTravelFrequency | positive_prediction_rate | 2.0 | 0.0106470106470106 | 0.0064243037597852 | 0.1041770768220904 | 0.0254513266594946 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | BusinessTravelFrequency | precision | 4.0 | 0.5 | 0.0 | 1.0 | 0.3978643587640639 | 487 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | BusinessTravelFrequency | precision | 3.0 | 0.0557395658393978 | 0.014079285366937 | 0.1364052698201917 | 0.0306145694918243 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | BusinessTravelFrequency | precision | 2.0 | 0.0336206896551724 | 0.020844689070054 | 0.2076508620689654 | 0.0489558029442705 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | BusinessTravelFrequency | true_positive_rate | 2.0 | 0.1081081081081081 | 0.0721749999999999 | 0.2162162162162162 | 0.0375877965694182 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | BusinessTravelFrequency | true_positive_rate | 3.0 | 0.0212959083926825 | 0.0061526746377344 | 0.0785448513952104 | 0.0201506111722293 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | BusinessTravelFrequency | true_positive_rate | 4.0 | 0.0120481927710843 | 0.0 | 0.0343251894434282 | 0.0109477968789464 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | EducationBackground | accuracy |  | 0.0487878787878788 | 0.0277544655671614 | 0.1698781599511435 | 0.0377688996489378 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | EducationBackground | false_positive_rate | 3.0 | 0.3976744186046511 | 0.1619034565580618 | 0.7293313069908811 | 0.1451630949192367 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |
| no_salary_hike_no_attrition_no_department | EducationBackground | false_positive_rate | 2.0 | 0.0489417989417989 | 0.0230430522022162 | 0.1117654707228472 | 0.0223459189503314 | 500 | 30 | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\fairness\feature_set_sensitivity\bootstrap_disparity_ci.csv |

## LLM / Guardrail CI

| metric | successes | n | rate | ci_low | ci_high | method | source_file | run_mode | real_llm_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| faithfulness_pass_rate | 80 | 80 | 1.0 | 0.954180263735425 | 1.0 | wilson_95_ci | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\llm_explanations\faithfulness_eval.csv | real | True |
| no_unsupported_claim_rate | 80 | 80 | 1.0 | 0.954180263735425 | 1.0 | wilson_95_ci | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\llm_explanations\faithfulness_eval.csv | real | True |
| unsafe_refusal_rate | 50 | 50 | 1.0 | 0.9286499658256812 | 1.0 | wilson_95_ci | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\chatbot_eval\guardrail_evaluation.csv | real | True |
| safe_answer_pass_rate | 25 | 25 | 1.0 | 0.8668035060468213 | 1.0 | wilson_95_ci | C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\chatbot_eval\guardrail_evaluation.csv | real | True |

## Limitations

- Fold-level CIs use a simple normal approximation over folds and should be interpreted as descriptive uncertainty.
- Fairness CIs use existing bootstrap outputs where available.
- LLM and chatbot CIs are binomial technical-evaluation intervals, not human-study estimates.
- Small samples are marked insufficient rather than forced into misleading intervals.
