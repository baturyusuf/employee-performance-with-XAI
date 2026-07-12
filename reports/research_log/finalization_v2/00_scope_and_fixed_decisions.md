# Scope and Fixed Decisions

Date: 2026-07-13  
Status: active contract

## Study Identity

The target study is a **leakage-aware XAI audit protocol** for three-class employee `PerformanceRating` prediction. It is research evidence, not an autonomous or deployable HR decision system.

## User-Fixed Decisions

- Use the term **leakage-aware**, not leakage-safe.
- XGBoost is the primary model; compare it with three standard predictive baselines. Detailed XAI is required only for XGBoost.
- Remove LLM/chatbot stages, artifacts, figures, tables, results, and claims from the core paper pipeline. Legacy code may remain clearly isolated as experimental/supplementary.
- Counterfactual analysis is supplementary-only and may claim only heuristic search success, never causal recourse, actionability, or employee advice.
- HRDataset_v14 is the primary independent mapped-target external replication.
- IBM performance, IBM attrition, and Employee Turnover are supplementary/secondary robustness evidence only.
- Sigmoid is the predeclared primary calibration method. Outer-test evidence must not select it.
- Raw datasets with unverified redistribution rights must not be distributed in the publication repository.
- No paid API call is permitted in core or supplementary builds.
- No ethics approval, exemption, licence, source authenticity, citation, or legal status may be invented.
- The manuscript must not be edited before technical freeze and user approval of the claim matrix.

## Prohibited Claims

No human-usefulness, trust, usability, adoption, decision-quality, human-evaluation, deployment-readiness, legal-compliance, fairness-guarantee, causal-SHAP, causal/prescriptive-counterfactual, real-LLM, chatbot-safety, locked-transport, or autonomous-HR claim is supported.
