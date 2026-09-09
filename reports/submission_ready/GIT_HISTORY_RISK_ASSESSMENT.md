# Git History / Raw-Data Risk Assessment

**Assessment date:** 2026-09-09

## 1. Current tree

**Status: PASS for current-tip publication hygiene.**

The pre-finalization repository gate recorded zero current-tree raw-data, environment, large-file, secret, or machine-path findings. The raw employee-level files are not present in the current publication-facing tree/package.

## 2. Historical Git objects

**Status: RISK CONFIRMED.**

GitHub commit history confirms that both raw-data paths existed in earlier commits before the later sanitization commit:

- `data/raw/inx_employee_performance.csv` appears in historical commits, including pre-sanitization history; commit `9342b0c9a02788ff9e9867b13f2f824662fd1cf3` is the later `feat(publication): sanitize current data tip` change.
- `data/external/hrdataset_v14/raw.csv` also appears in historical commits, including commit `248530e8ce4d6fc8fd74ec613713d5499366f729` (`2 more dataset added`), before the same later sanitization commit.

A file removed from the current tree can remain recoverable from reachable historical commits. Path history therefore establishes that the public repository has historical exposure risk even though the current tree is clean.

## 3. Submission/release consequence

**Status: BLOCKED FOR A CLEAN PUBLIC RELEASE DECISION.**

This is material because INX redistribution permission is unresolved and HRDataset_v14 sharing is subject to upstream licence terms. A clean current tree alone does not erase historically committed raw content.

## Remediation options

### Option A — Preserve history

Leave repository history unchanged and explicitly avoid representing the repository as free of historical raw-data exposure. This avoids destructive Git operations but does not remove already reachable historical copies.

### Option B — Authorized history rewrite

Use a coordinated history-rewrite procedure (for example, `git filter-repo` or an equivalent vetted tool) to purge the affected raw paths, then force-update the relevant public refs and require collaborators to re-clone or carefully reset. This is destructive and changes commit identities.

**Not authorized by this finalization pass. No force push or history rewrite was performed.**

### Option C — Separate clean archival release

Create a new, clean release/archive containing only publication-safe code, aggregate evidence, manuscript materials, and permitted assets. This can provide a clean immutable research artifact without pretending the existing public Git history was purged. Whether this satisfies the authors' release/governance goals is an author decision.

## Required author decision

Before creating a final public release/tag/DOI, explicitly decide whether to:

1. accept and disclose the existing historical exposure,
2. authorize a destructive history purge, or
3. publish a separate clean archival release.

No destructive remediation, tag, release, or DOI is authorized by this document.