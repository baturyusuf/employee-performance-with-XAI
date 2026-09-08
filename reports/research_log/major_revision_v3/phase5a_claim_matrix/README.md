# Phase 5A Claim-Matrix Package

Generation commit: `fcfc4692466d7bc9f414db4b2fe693073d0d7f4e`
Claim-set SHA-256: `1664b188df14135d3b3de8642de3d080c25a3fbadc440615c2bd04b2f14ddabe`
Status: `pending_user_approval`

This deterministic package binds every proposed numerical sentence to one exact CSV row/value and every proposed narrative sentence to a hash-bound text anchor. Each row carries its evidence scope, required qualifier, and prohibited overclaim.

The package does not edit or authorize edits to the manuscript or bibliography. It also does not resolve the separate provenance/licence, ethics/declaration, Git-history, release, tag, or DOI blockers.

## Contents

- `CLAIM_MATRIX.csv` and `CLAIM_MATRIX.md`: machine-readable and review-readable claim sets.
- `CLAIM_BOUNDARIES.md`: global non-negotiable language boundaries.
- `SOURCE_REGISTER.csv`: source hashes, sizes, claim coverage, and parent manifest hashes.
- `APPROVAL_REQUEST.md` and `approval_record.json`: the exact digest awaiting an explicit decision.
- `provenance_receipt.json` and `manifest.json`: offline generation identity and closed-world hashes.
