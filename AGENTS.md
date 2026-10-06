# Agent Handoff Contract

Before resuming finalization work, read in this order:

1. `reports/research_log/finalization/CURRENT_STATUS.md`
2. `reports/research_log/finalization/DECISION_LOG.md`
3. `reports/research_log/finalization/NEXT_ACTIONS.md`
4. `reports/research_log/finalization/COMMAND_LOG.md`
5. `reports/research_log/finalization/TEST_LOG.md`
6. `reports/research_log/finalization_v2/02_issue_register.csv`
7. Current `git status`, `git diff`, branch, and HEAD.

Do not infer completion from old chat or the earlier `manuscript_remediation` logs. The existing `reports/manuscript_final/latest` package is historical v1 evidence and is not accepted as the v2 canonical package. Do not edit `manuscript/mdpi_information/main.md` before the claim matrix is frozen and approved by the user. Never make paid API calls.

# Zed + GitHub Multi-Agent Research Organization

The persistent multi-agent operating system is defined under `agent/`. This section supplements the finalization handoff contract above; it does not supersede any frozen-evidence or publication restriction.

Before performing a role-specific task:

1. Read `agent/README.md`.
2. Read `agent/registry.yaml`.
3. Read the exact role prompt assigned to the current Zed thread under `agent/prompts/`.
4. Read `agent/protocols/communication.md`.
5. Read `management/PROJECT_STATUS.md` and `management/CURRENT_PRIORITIES.md`.
6. If implementation is requested, require a READY Work Package and follow the Git/worktree protocol.
7. If the task touches canonical evidence, also follow the original finalization resumption order at the top of this file.

## Authority boundaries

- `SCI-*` GPT-5.6 Sol agents are scientific advisors. They may analyze, research, design methodology, create Work Packages, review evidence and make scientific recommendations. They must not implement production research code.
- `IMP-*` local agents may implement approved Work Packages and run experiments. They must not silently change scientific hypotheses, dataset roles, target definitions, feature policies, primary metrics, acceptance criteria or claim boundaries.
- `VER-*` agents independently verify implementation or claim-evidence alignment. Reviewers should not silently repair the work they are auditing.
- `MGT-SECRETARY` maintains management summaries from durable repository artifacts and has no scientific decision authority.
- The Human PI / Author remains final authority.

## Durable communication

Agent-to-agent communication must be repository-mediated using Work Packages, Evidence Packages, Review Records, Decision Records and Handoffs. Zed chat history is not the system of record.

See:
- `agent/protocols/work_package_schema.md`
- `agent/protocols/evidence_package_schema.md`
- `agent/protocols/decision_and_conflict.md`
- `agent/protocols/git_worktree.md`
- `agent/zed/THREAD_SETUP.md`

Do not create new experiments merely because the agent architecture exists. First establish a substantiated scientific gap and an approved Work Package.
