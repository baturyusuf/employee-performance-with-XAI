# Zed Thread Setup

Create one long-lived Zed thread for each scientific advisor and the implementation lead. Worker and verifier threads may be persistent or task-scoped.

## Thread naming

Use:

`[AGENT-ID] Short Role Name`

Examples:

- `[SCI-DIRECTOR] Scientific Director`
- `[SCI-DATA] Data & Validity`
- `[IMP-LEAD] Implementation Lead`
- `[VER-TECH] Technical Verifier`

## Initial instruction

For each thread, the only manual bootstrap instruction should be:

```text
Read AGENTS.md and <PROMPT_PATH>. Adopt that file as your persistent operating contract for this thread. Execute its startup sequence before doing any task.
```

The prompt path is listed in `agent/registry.yaml`.

Do not paste the full role prompt into chat. The repository file is the canonical prompt.

## Persistent vs temporary threads

Persistent:
- SCI-DIRECTOR
- all SCI-* advisors
- IMP-LEAD
- VER-TECH
- MGT-SECRETARY

Task-scoped:
- implementation workers when a work package is short
- search/literature subagents
- one-off audit agents

## Context discipline

When context becomes large, do not rely on thread history. Write or refresh a handoff file, then resume from:
- role prompt
- current management state
- active Work Package
- latest Evidence/Review artifact

## Parallel code work

Never let two code-writing agents modify the same checkout concurrently.

Use one branch/worktree per active implementation package:

`agent/<agent-id>/<wp-id>`

Example:

`agent/imp-ml/wp-ml-004`

Scientific advisors normally do not require isolated worktrees because they should not edit source code.

## Human control

The Human PI may stop, reprioritize, reject, or supersede any agent action. No agent can self-authorize a change to the frozen scientific protocol merely because it is technically convenient.
