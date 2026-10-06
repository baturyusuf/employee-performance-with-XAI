# Multi-Agent Research Organization

This directory is the persistent operating layer for the Zed + GitHub multi-agent research workflow.

## Principle

GitHub is the system of record. Zed threads are execution surfaces, not durable memory.

Scientific agents decide **what should be investigated, why it matters, what evidence is required, and what claims are justified**. They do not implement production research code.

Local implementation agents decide **how to faithfully implement an approved scientific work package**. They may write and run code, but they may not silently change hypotheses, datasets, target definitions, feature-governance policy, evaluation metrics, or scientific acceptance criteria.

Independent verification agents audit implementation and evidence before results return to the scientific layer.

## Hierarchy

```text
Human PI / Author
      |
Scientific Director — GPT-5.6 Sol
      |
+-----+--------------------+-------------------+
|                          |                   |
Scientific Advisors     Scientific Advisors  Scientific Advisors
      |
Scientific Work Packages
      |
Implementation Lead — local LLM
      |
Local implementation workers
      |
Technical / reproducibility verification
      |
Evidence Packages
      |
Relevant Scientific Advisor
      |
Scientific Director
      |
Accepted decision / revised work package / manuscript claim boundary
```

## Durable communication

Agents communicate through repository artifacts:

1. `agent/work_packages/` — approved scientific instructions.
2. `agent/handoffs/` — bounded task transfers or context handoffs.
3. `agent/reviews/` — independent verification and scientific reviews.
4. `management/` — human-readable summaries, priorities, risks and decisions.
5. Existing canonical evidence under `reports/research_log/` and `manuscript/mdpi_information/assets/` remains authoritative unless an explicitly approved new scientific work package supersedes it.

## Startup

Every Zed thread must first read:

1. `AGENTS.md`
2. its role prompt under `agent/prompts/`
3. `agent/protocols/communication.md`
4. `management/PROJECT_STATUS.md`
5. `management/CURRENT_PRIORITIES.md`
6. any referenced Work Package / Evidence Package / Decision Record
7. the existing finalization records listed in `AGENTS.md` when the task touches canonical evidence

Do not use chat history as the only source of project state.

## Lifecycle

```text
Problem detected
    -> scientific analysis
    -> Work Package (WP)
    -> implementation
    -> Evidence Package (EP)
    -> independent verification
    -> scientific review
    -> PASS / REVISE / ESCALATE / REJECT
    -> Decision Record (when consequential)
    -> management summary update
```

## Zed

Use one persistent Zed thread per long-lived advisor or lead. Use short-lived subagents only for bounded research or implementation subtasks. Parallel code-writing agents should use separate Git branches/worktrees.

See `agent/zed/THREAD_SETUP.md`.
