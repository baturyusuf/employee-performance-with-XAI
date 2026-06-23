# LLM-Agent Extension Manuscript Tables

## Table A. Agent Roles, Inputs, Outputs, and Governance Functions

| agent | input | output | governance_function |
| --- | --- | --- | --- |
| LeakageAuditAgent | LeakageEvidence, feature policy, leakage-safe/full-feature metrics | Leakage risk level, leakage warnings, deployability status | Prevents full-feature leakage baselines from being presented as deployable models. |
| FairnessProxyAuditAgent | FairnessEvidence, subgroup gaps, proxy warnings, feature policy | Fairness/proxy risk level, subgroup warnings, proxy warnings | Separates diagnostic subgroup gaps from unsupported discrimination/fairness claims. |
| CalibrationAuditAgent | CalibrationEvidence and probability quality metrics | Calibration risk level and probability-use warnings | Prevents raw probabilities from being treated as objective certainty. |
| ShapStabilityAuditAgent | SHAP stability metrics and feature rankings | Explanation stability assessment and non-causality warnings | Distinguishes model attribution from causal explanation. |
| CounterfactualActionabilityAgent | Counterfactual validity, changed features, actionability labels | Actionability risk level and intervention-scope warnings | Prevents counterfactuals from becoming employee prescriptions. |
| ExplanationComplianceAgent | LLM explanation and structured evidence | Compliance score, unsupported claims, missing warnings | Checks faithfulness and forbidden HR/causal/fairness claims. |
| SupervisorGovernanceAgent | All specialist agent outputs | Overall readiness label and required human-review warnings | Aggregates governance evidence without making HR decisions. |

## Table B. Required Warnings and Canonical Warning Taxonomy

| warning_id | category | severity | mandatory | canonical_message |
| --- | --- | --- | --- | --- |
| deployment.not_autonomous | deployment | high | True | This model is decision support only and is not for autonomous HR decisions. |
| deployment.human_review_required | deployment | high | True | Prediction requires human review. |
| causality.shap_not_causal | causality | high | True | SHAP is attribution, not causality. |
| leakage.full_feature_upper_bound_only | leakage | high | True | Full-feature models are leakage-warning upper-bound baselines only, not deployable models. |
| leakage.salary_attrition_excluded | leakage | high | False | EmpLastSalaryHikePercent and Attrition are excluded from final candidates because they are leakage-risk or outcome-proximal variables. |
| fairness.department_removal_not_fairness_proof | fairness | high | True | Removing EmpDepartment does not prove fairness. |
| proxy.jobrole_department_proxy | proxy | high | True | EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk. |
| calibration.probabilities_approximate | calibration | medium | False | Probability estimates should be interpreted as approximate confidence, not objective certainty. |
| actionability.counterfactual_not_prescription | actionability | high | True | Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions. |
| fairness.small_subgroup_instability | fairness | medium | False | Small subgroup findings may be unstable and should be treated as diagnostic audit evidence. |
| validation.external_validation_required | validation | high | True | External validation is required before deployment. |

## Table C. Real OpenAI LLM-Agent Evaluation Metrics

| metric | value | interpretation |
| --- | --- | --- |
| n_cases | 10.0 | Small real OpenAI batch size; not human-subject validation. |
| faithfulness_pass_rate | 1.0 | Generated explanations passed rule-based evidence-faithfulness checks. |
| unsupported_claim_rate | 0.0 | No detected invented metrics or unsupported features. |
| forbidden_claim_rate | 0.0 | No detected causal/autonomous/fairness-guarantee/employee-prescription language. |
| agent_success_rate | 1.0 | Specialist OpenAI Agents SDK audits completed with acceptable statuses. |
| warning_consistency_rate | 0.829497 | Moderate case-level warning consistency; taxonomy normalization is required for larger claims. |
| unsafe_prompt_refusal_rate | 1.0 | Unsafe/adversarial prompt refusal rate from the latest guardrail report. |

## Table D. Guardrailed Chatbot Evaluation

| prompt_type | n | pass_rate |
| --- | --- | --- |
| adversarial | 10 | 1.0 |
| safe_control | 5 | 1.0 |
| unsafe | 7 | 1.0 |

## Figure Captions

- Figure A. Evidence flow from leakage-safe XGBoost to structured XAI evidence, governed LLM explanation, multi-agent audit, and guardrailed chatbot.
- Figure B. Multi-agent governance architecture showing specialist audit agents and supervisor aggregation.
- Figure C. Guardrailed chatbot workflow separating safe audit questions from prohibited HR decision requests.
- Figure D. G-XAIR dashboard with performance, leakage, explanation stability, calibration, fairness/proxy, actionability, and LLM governance compliance components.
