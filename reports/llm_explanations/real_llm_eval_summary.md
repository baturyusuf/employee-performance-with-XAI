# Real OpenAI LLM-Agent Evaluation Summary

## Scope
- Cases: 10 (528;376;568;18;392;405;125;176;662;906)
- Model/provider: OpenAI API with OpenAI Agents SDK for governance audit.
- Cost policy: small-batch evaluation with gpt-5.4-mini and temperature 0.

## Results
- Faithfulness pass rate: 1.0
- Unsupported claim rate: 0.0
- Forbidden claim rate: 0.0
- Missing warning rate: 0.0
- Agent success rate: 1.0
- Warning consistency rate: 0.829497
- Warning consistency by agent: `{"CalibrationAuditAgent": 0.7777777777777778, "CounterfactualActionabilityAgent": 0.8888888888888888, "ExplanationComplianceAgent": 0.7777777777777778, "FairnessProxyAuditAgent": 0.9888888888888889, "LeakageAuditAgent": 0.9047619047619047, "ShapStabilityAuditAgent": 0.6388888888888888}`
- Unsafe prompt refusal rate: 1.0

## Usage
- Usage log rows: 90
- Input tokens: 531633
- Output tokens: 44783
- Total tokens: 576416
- Estimated cost USD: 0.440754

## Artifacts
- Governed explanations: `reports/llm_explanations/governed_explanation_examples.md`
- Governed explanation eval: `reports/llm_explanations/governed_explanation_eval.csv`
- Usage log: `reports/llm_explanations/llm_usage_log.csv`
- Guardrail eval: `C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\chatbot_eval\guardrail_evaluation.md`
- Agent audit: `C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\agent_audits\openai_agents_sdk_case_528_governance_audit.md`
- Agent audit: `C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\agent_audits\openai_agents_sdk_case_376_governance_audit.md`
- Agent audit: `C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\agent_audits\openai_agents_sdk_case_568_governance_audit.md`
- Agent audit: `C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\agent_audits\openai_agents_sdk_case_18_governance_audit.md`
- Agent audit: `C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\agent_audits\openai_agents_sdk_case_392_governance_audit.md`
- Agent audit: `C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\agent_audits\openai_agents_sdk_case_405_governance_audit.md`
- Agent audit: `C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\agent_audits\openai_agents_sdk_case_125_governance_audit.md`
- Agent audit: `C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\agent_audits\openai_agents_sdk_case_176_governance_audit.md`
- Agent audit: `C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\agent_audits\openai_agents_sdk_case_662_governance_audit.md`
- Agent audit: `C:\Users\Yusuf\Documents\GitHub\employee performance with XAI\reports\agent_audits\openai_agents_sdk_case_906_governance_audit.md`

## Limitations
- This is a small-batch technical validation, not a human evaluation.
- Warning consistency is computed across same-batch warning sets, not repeated LLM reruns.
- Cost is estimated from logged token usage and configured model prices; billing dashboard remains the source of truth.
