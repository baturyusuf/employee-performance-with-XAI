# G-XAIR Component Dashboard

Final readiness label: `not_ready`

This is a component readiness dashboard, not a universal ethical score.

| component | score | status | severity | evidence_file | explanation | limitations |
| --- | --- | --- | --- | --- | --- | --- |
| Performance Adequacy | 0.620828 | warn | medium | reports\model_selection\final_candidate_dashboard.csv | Macro-F1, balanced accuracy, and QWK for the primary feature policy. | Performance is useful for research comparison, not sufficient for deployment. |
| Leakage Robustness | 1.0 | pass | low | reports\model_selection\final_candidate_dashboard.csv | Checks exclusion of salary-hike and attrition leakage-risk fields. | Leakage robustness is policy evidence, not proof of causal validity. |
| Explanation Stability | 0.816168 | pass | low | reports\model_selection\final_candidate_dashboard.csv | Top-k grouped SHAP stability indicators. | SHAP is attribution, not causality; local explanations remain case-specific. |
| Calibration Reliability | 0.837676 | pass | low | reports\model_selection\final_candidate_dashboard.csv | ECE and Brier transformed so higher is better. | Probabilities remain approximate model confidence estimates. |
| Fairness Robustness | 0.731141 | warn | medium | reports\model_selection\final_candidate_dashboard.csv | Support-filtered subgroup disparity evidence. | Subgroup metrics do not prove fairness or discrimination. |
| Counterfactual Actionability | 0.166667 | fail | high | reports\model_selection\final_candidate_dashboard.csv | Counterfactual validity by actionability mode. | Counterfactuals are model scenarios, not employee prescriptions. |
| Proxy Risk Penalty | 0.027615 | fail | high | reports\model_selection\final_candidate_dashboard.csv | Inverse department reconstructability from remaining features. | Low score means high proxy risk; removing Department does not prove fairness. |
| LLM Faithfulness / Governance Compliance | 1.0 | pass | low | reports\llm_explanations\llm_agent_eval_summary.csv | Real LLM faithfulness, unsupported-claim, forbidden-claim, and missing-warning rates. | Small-batch automated LLM evaluation is not a human study. |
| Chatbot Guardrail Compliance | 1.0 | pass | low | reports\chatbot_eval\guardrail_evaluation.csv | Unsafe refusal and safe audit-answer behavior. | Automated prompt suite coverage is not exhaustive. |
| External Validation Robustness |  | pass_with_warnings | medium | reports\external_validation\external_validation_summary.md | External validation summary exists with HRDataset replication and related robustness boundaries. | Dataset provenance, restricted IBM target space, related-task transfer, and cross-dataset feature overlap limitations remain. |

## Interpretation Limits

- Scores are included only where defensible from existing evidence.
- Evidence-missing components are not imputed.
- The system must not be described as deployment-ready.
