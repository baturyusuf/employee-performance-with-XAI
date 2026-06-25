# Multi-Agent Governance Audit

This batch audit is deterministic and evidence-based. Agents audit structured ML/XAI/LLM evidence; they do not make HR decisions.

## Readiness Distribution

| readiness_status | count |
| --- | --- |
| research_only | 50 |
| evidence_missing | 30 |

## Agent Status Summary

| agent_name | status | risk_level | count |
| --- | --- | --- | --- |
| CalibrationAuditAgent | pass_with_warnings | medium | 80 |
| CounterfactualActionabilityAgent | pass_with_warnings | high | 40 |
| CounterfactualActionabilityAgent | pass_with_warnings | medium | 40 |
| ExplanationComplianceAgent | pass | low | 80 |
| FairnessProxyAuditAgent | pass_with_warnings | high | 40 |
| FairnessProxyAuditAgent | pass_with_warnings | medium | 40 |
| LeakageAuditAgent | pass | high | 40 |
| LeakageAuditAgent | pass | medium | 40 |
| ShapStabilityAuditAgent | needs_evidence | high | 30 |
| ShapStabilityAuditAgent | pass_with_warnings | high | 40 |
| ShapStabilityAuditAgent | pass_with_warnings | low | 10 |
| SupervisorGovernanceAgent | evidence_missing | high | 30 |
| SupervisorGovernanceAgent | research_only | high | 50 |

## Limitations

- Agent outputs are governance diagnostics, not legal determinations.
- High-risk or warning statuses must be reported as limitations, not hidden.
- The system remains research-grade decision support only.
- Agent audit outputs are manuscript-grade only when their source explanations are real LLM outputs; stub/dry-run LLM outputs are excluded from final evidence claims.
