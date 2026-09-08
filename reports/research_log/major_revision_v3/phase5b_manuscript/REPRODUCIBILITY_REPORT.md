# Reproducibility Report

## Evidence identity

- Approved claim set: `1664b188df14135d3b3de8642de3d080c25a3fbadc440615c2bd04b2f14ddabe`
- Phase 5B comparison base: `af4668aff4f52abcb4b6294f54434f00b59123d5`
- Manuscript source of truth: `manuscript/mdpi_information/main.md`
- Generated peer source: `manuscript/mdpi_information/main.tex`
- Verified bibliography input: `configs/literature_positioning_v3.json`
- Scientific inputs: compact Phase 1B–4B packages and canonical manuscript assets

## What was and was not run

Phase 5B performed document generation, deterministic validation, asset copying, hashing, and tests. It did not download data, modify raw source values, fit or select a model, fit a calibrator, recompute SHAP, rerun a bootstrap, call a paid API, or publish employee-level rows.

## Rebuild

From the repository root and its pinned Python environment:

```powershell
& .\myenv\Scripts\python.exe -m src.governance.manuscript_revision_v3 --replace-output
```

After authoring the narrative deliverables and completing checks, regenerate the package manifest:

```powershell
& .\myenv\Scripts\python.exe -c "from src.governance.manuscript_revision_v3 import finalize_manifest; finalize_manifest()"
```

## Deterministic checks

The builder fails if the approval record does not match the exact digest, any claim is not approved, a cited key lies outside the 25-work verified set, any of 32 numerical claim IDs is missing, obsolete v1 concepts reappear, prohibited overclaim phrases appear, required sections are absent, or Markdown/LaTeX table and figure counts differ.

The package preserves CSV source hashes and copies only aggregate tables, publication figures, captions, and alt text. Raw data, row-level predictions, folds, per-row SHAP, fitted models, and calibration objects remain excluded.

## Build limitation

The workstation has no `latexmk`, `pdflatex`, `pandoc`, or `tectonic` executable. Phase 5B therefore validates generated LaTeX structurally and by Markdown/LaTeX parity, but cannot claim a compiled-PDF check. The official MDPI `Definitions/` class bundle is also intentionally absent. Final journal-template compilation remains a submission action.

