# Canonical Deterministic Multi-Agent Governance Audit

Run ID: `manuscript_final_20260712T181754Z_c664ef152ff3`  
Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`  
Cases audited: `80`

This is an offline, deterministic evidence-flow and governance-composition audit. It made zero API calls and is not evidence of real-LLM faithfulness or safety.

## Supervisor readiness distribution

| status | n_cases |
| --- | --- |
| research_only | 40 |
| not_ready | 40 |

## Agent finding distribution

| agent_name | status | risk_level | n_findings |
| --- | --- | --- | --- |
| CalibrationAuditAgent | pass_with_warnings | medium | 80 |
| CounterfactualActionabilityAgent | pass_with_warnings | high | 36 |
| CounterfactualActionabilityAgent | pass_with_warnings | medium | 44 |
| ExplanationComplianceAgent | pass | low | 80 |
| FairnessProxyAuditAgent | pass_with_warnings | medium | 80 |
| LeakageAuditAgent | fail | high | 40 |
| LeakageAuditAgent | pass | high | 40 |
| ShapStabilityAuditAgent | pass_with_warnings | high | 80 |
| SupervisorGovernanceAgent | not_ready | high | 40 |
| SupervisorGovernanceAgent | research_only | high | 40 |

## Claim boundaries

- Agents issue warnings and research-readiness diagnostics only; they make no HR decision.
- The offline renderer is deterministic pipeline-validation infrastructure, not a substitute for an approved real-LLM evaluation.
- Audit findings do not establish legal compliance, fairness, causality, deployment readiness, or zero failure probability.
