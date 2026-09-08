# Phase 4B Provenance and Release-Readiness Package

Generation commit: `1b08313633ba0c3bd343dc5b7175f87029b9425b`
Verification snapshot: `2026-09-07`

This deterministic package records dataset provenance/license evidence, unresolved ethics and author declarations, compact artifact inventory, and an immutable-release procedure. It is preparation evidence only: the release state is `blocked_preparation_only`.

No tag, GitHub release, archive record, DOI, repository software license, ethics determination, or raw-data publication is created or claimed here. All four raw datasets remain excluded from release payloads.

## Contents

- `DATA_PROVENANCE_LICENSE_REGISTER.csv` and `PROVENANCE_LICENSE_REPORT.md`: source, byte, license, citation, and redistribution evidence kept as distinct fields.
- `ETHICS_DECLARATIONS_REGISTER.csv` and `ETHICS_DECLARATIONS_HANDOFF.md`: author/institution inputs and safe interim boundaries.
- `ARTIFACT_INVENTORY.csv`: ten tracked compact evidence components and their manifest hashes.
- `BLOCKER_REGISTER.csv`: eleven open release/submission blockers.
- `RELEASE_READINESS.md` and `RELEASE_NOTES_DRAFT.md`: proposed tag and ordered release/DOI workflow without publication.
- `release_candidate_manifest.json`: machine-readable candidate state with a null final commit and null publication identifiers.
- `provenance_receipt.json` and `manifest.json`: contract/generation identity and closed-world hashes.
