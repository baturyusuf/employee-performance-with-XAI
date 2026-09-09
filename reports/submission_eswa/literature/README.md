# Literature handoff

Fresh public-source check: 2026-09-09. Existing inventory: 34 entries; new verified candidate additions: eight, including five ESWA papers. This is a bounded editorial literature update, not a systematic review.

- `VERIFIED_ADDITIONS.bib`: eight complete author lists and verified DOI/venue/year records; cite only the additions actually used.
- `EDITOR_INSERTIONS.md`: related-work bridge, bounded gap paragraph, governance paragraph, and keyword recommendation.
- `COMPARISON_TABLE.csv` / `.md`: 11 prior-work comparisons plus the present study, with access limitations and primary-source links.
- `KEYWORD_COMPARISON.csv`: exact keyword lists from five 2024–2026 ESWA publisher records.
- `SEARCH_PROTOCOL.md`: search scope, query families, inclusion decisions and limits.
- `../qc/REFERENCE_AUDIT.csv` / `.md`: per-reference fresh metadata checks, support roles, corrections, unresolved details and integrity-check limits.
- `existing_reference_metadata.json` / `addition_metadata.json`: fresh public metadata receipts, retrieval times and response hashes.

The four small Python scripts are literature utilities, not scientific experiment code. `audit_metadata.py` and `audit_additions.py` make read-only public metadata requests; `build_literature_outputs.py` and `build_comparison.py` regenerate the local reports. Do not rerun successful requests unnecessarily. Full publisher pages/PDFs are not redistributed in this package.

The current audit verifies identities with explicit discrepancies. It does not certify every citation as unretracted, establish source data rights, or authorize submission. Apply corrections only to the ESWA copy. No MDPI file, scientific code/configuration, or frozen numerical evidence was changed.
