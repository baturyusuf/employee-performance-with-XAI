# Zed Thread Setup

Zed is the execution surface. GitHub is the durable memory and coordination layer.

## Thread naming

Use:

`[AGENT-ID] Short Role Name`

Examples:
- `[SCI-DIRECTOR] Scientific Director`
- `[SCI-DATA] Data & Validity`
- `[SCI-LEAKAGE] Leakage Audit`
- `[IMP-LEAD] Implementation Lead`
- `[VER-TECH] Technical Verifier`

## Bootstrap

Do not paste long prompts into Zed. Use the one-line bootstraps in:

`agent/zed/BOOTSTRAP_COMMANDS.md`

Each thread must read its repository prompt and treat it as the persistent role contract.

## Persistent threads

Keep these long-lived:
- SCI-DIRECTOR
- SCI-NOVELTY
- SCI-DATA
- SCI-ML
- SCI-STATS
- SCI-XAI
- SCI-PUB
- IMP-LEAD
- VER-TECH
- MGT-SECRETARY

These are the stable organizational backbone.

## Task-scoped scientific specialists

Open only when a concrete problem exists:
- SCI-LEAKAGE
- SCI-ORDINAL
- SCI-CAL
- SCI-SHAP
- SCI-CF
- SCI-FAIR
- SCI-XROBUST
- SCI-REPRO
- SCI-GOV
- SCI-LLMFAITH only if LLM explanations are explicitly returned to manuscript scope

This avoids keeping 15–20 expensive Sol contexts alive while preserving one-prompt-per-problem specialization.

## Task-scoped local workers

Open only against a READY Work Package:
- IMP-DATA
- IMP-ML
- IMP-STATS
- IMP-XAI
- IMP-RUN
- IMP-REPRO

The worker bootstrap must include the exact WP path.

## Context discipline

When a thread becomes large, do not trust conversation history as project memory. Create or refresh a Handoff and resume from:
- AGENTS.md;
- role prompt;
- management current state;
- active Work Package;
- latest Evidence/Review/Decision records.

## Parallel code work

Never let two write-capable agents modify the same checkout concurrently.

Use one branch/worktree per active implementation package:

`agent/<agent-id-lower>/<wp-id-lower>`

Example:

`agent/imp-ml/wp-ml-004`

Scientific advisors normally do not require isolated code worktrees because they do not edit production source.

## Recommended first launch order

1. SCI-DIRECTOR
2. SCI-NOVELTY, SCI-DATA, SCI-ML, SCI-STATS, SCI-XAI, SCI-PUB
3. MGT-SECRETARY
4. SCI-DIRECTOR creates a gap inventory
5. specialist Sol threads are opened only for substantiated gaps
6. approved WPs go to IMP-LEAD
7. local workers execute in isolated worktrees
8. VER-TECH verifies
9. relevant SCI advisor interprets the EP
10. SCI-DIRECTOR resolves cross-domain conflicts and updates decisions

## Human control

The Human PI may stop, reprioritize, reject or supersede any agent action. No agent can self-authorize a change to the frozen scientific protocol merely because it is technically convenient.
