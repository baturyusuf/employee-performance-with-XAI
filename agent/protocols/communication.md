# Agent Communication Protocol

## Rule 1 — Repository artifacts are messages

Do not assume another Zed thread can see your chat history. Communicate by creating or updating a durable repository artifact.

## Message types

### Work Package (WP)
Scientific layer -> implementation layer.

Defines the scientific question, rationale, exact requirements, constraints, outputs and acceptance criteria.

### Evidence Package (EP)
Implementation layer -> verification/scientific layer.

Records what was actually implemented and executed, with commit, commands, inputs, outputs, metrics, deviations and limitations.

### Review Record (RR)
Verifier/advisor -> task owner.

Records PASS / REVISE / ESCALATE / REJECT with precise reasons and required remediation.

### Decision Record (DR)
Scientific Director / Human PI.

Records consequential decisions that affect scientific interpretation, scope, protocol or manuscript claims.

### Handoff
Any agent -> another agent when context must transfer without changing scientific authority.

## IDs

Use:
- `WP-<DOMAIN>-NNN`
- `EP-<DOMAIN>-NNN`
- `RR-<DOMAIN>-NNN`
- `DR-NNN`
- `HO-<FROM>-<TO>-NNN`

Domains: DATA, ML, STAT, XAI, NOV, PUB, REPRO, GOV.

## Communication rules

1. Every implementation change must reference a WP ID.
2. Every experiment result returned for scientific interpretation must have an EP.
3. Every verifier finding must cite concrete files, commands or evidence.
4. Do not overwrite negative results; preserve them.
5. Do not silently reinterpret another agent's result.
6. If a WP is ambiguous, mark it BLOCKED and write a handoff/question rather than guessing.
7. If implementation must deviate from the WP, record the deviation before drawing conclusions.
8. Scientific conclusions belong to SCI-* agents, not IMP-* agents.
9. Consequential scientific disagreements are escalated to SCI-DIRECTOR.
10. Update management summaries only from durable artifacts, never from uncommitted chat claims.
