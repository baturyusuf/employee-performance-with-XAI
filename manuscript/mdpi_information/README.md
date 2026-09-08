# MDPI Information Manuscript Package — Phase 5B

This directory contains the revised manuscript for the leakage- and governance-aware ordinal XAI audit study.

## Sources

- `main.md`: authoritative manuscript text.
- `main.tex`: deterministically generated MDPI-style peer source.
- `references.bib`: 25 entries from the frozen source-verified literature set.
- `assets/`: canonical tables, figures, source maps, captions, alt text, and manifests.
- `reports/research_log/major_revision_v3/phase5b_manuscript/`: Phase 5B reports, final asset copies, result/claim comparison, validation, and diff.

The sole claim boundary is SHA-256 `1664b188df14135d3b3de8642de3d080c25a3fbadc440615c2bd04b2f14ddabe`.

## Rebuild

```powershell
& .\myenv\Scripts\python.exe -m src.governance.manuscript_revision_v3 --replace-output
```

No scientific computation or paid service call occurs during this build.

## Submission status

The scientific text is revised and structurally validated, but the package is not submission-ready. Author affiliations/ORCIDs, contributions, funding, institutional review, consent, conflicts, AI-use wording, data rights, software licensing, immutable release/archive identifiers, and official-template PDF compilation remain unresolved.
