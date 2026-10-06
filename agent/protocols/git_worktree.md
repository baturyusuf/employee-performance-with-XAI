# Git / Worktree Protocol

## Branch isolation

Every code-writing Work Package should use a dedicated branch/worktree.

Naming:
`agent/<agent-id-lower>/<wp-id-lower>`

Example:
`agent/imp-ml/wp-ml-004`

## Rules

1. Never run two write-capable agents against the same working tree concurrently.
2. Never force-push unless explicitly authorized by the Human PI.
3. Never rewrite repository history as an agent convenience.
4. Do not commit raw employee-level data, secrets, fitted models or ignored canonical internals.
5. Preserve the existing publication and repository gates.
6. Each implementation commit message should include the WP ID.
7. Before handoff, record:
   - branch,
   - HEAD,
   - git status,
   - changed files,
   - test command/results.
8. Verification should inspect the exact candidate commit, not a mutable working directory.
9. Merge only after the required review record is PASS.
10. Scientific advisor threads should not edit `src/`, `tests/`, experiment code or scientific configs.

Existing finalization restrictions in `AGENTS.md` take precedence.
