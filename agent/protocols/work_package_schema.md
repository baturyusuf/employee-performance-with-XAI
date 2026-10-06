# Scientific Work Package Schema

Store active packages under `agent/work_packages/active/`. Move completed packages to `agent/work_packages/completed/`; do not delete them.

Required fields:

```markdown
# WP-<DOMAIN>-NNN — Title

Status: DRAFT | READY | RUNNING | REVIEW | BLOCKED | COMPLETE | REJECTED
Owner: <SCI-* agent>
Assigned to: <IMP-* agent or IMP-LEAD>
Priority: BLOCKER | HIGH | MEDIUM | LOW
Created:
Depends on:
Supersedes:

## Scientific question

## Why this matters

## Current evidence

## Hypotheses / questions to test

## Scope

## Out of scope

## Datasets and target definitions

## Feature-governance / leakage constraints

## Experimental design

## Required baselines

## Required metrics

## Statistical analysis

## Required XAI / audit outputs

## Required artifacts

## Acceptance criteria

## Failure / stop conditions

## Prohibited interpretations

## Reproducibility requirements

## Handoff notes
```

A Work Package must be implementation-complete: a competent engineer should not need to invent a scientific choice in order to implement it.

If a scientific choice is genuinely unknown, specify alternatives and the decision rule rather than delegating the scientific decision to the coding agent.
