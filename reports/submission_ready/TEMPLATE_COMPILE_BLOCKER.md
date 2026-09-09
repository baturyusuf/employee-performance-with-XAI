# Official MDPI Template / Compile Gate

## Current status

**BLOCKED — no PDF verification claim is made.**

The official MDPI LaTeX page was checked during finalization and the current indexed template packages are reported as updated 23 June 2026. The repository does not contain that exact current official template bundle, and an exact current bundle was not safely materialized from an authoritative downloadable artifact during this pass.

## Why no substitute was used

An older repository class file, an unofficial mirror, or a hand-built generic article class would not satisfy the requested requirement to apply and verify the **current official MDPI template**. Compiling such a substitute could produce a PDF but would not prove submission-template compliance.

## Production steps once the exact official bundle is available

1. Place the official `template.tex` and complete `Definitions` directory in the controlled submission workspace.
2. Merge the finalized `reports/submission_ready/main.md` content into the journal template without altering validated numerical claims.
3. Use the authoritative `reports/submission_ready/references.bib` and publication-safe figure assets.
4. Compile with the engine supported by the official package and resolve every warning/error affecting citations, references, figures, tables, and layout.
5. Render the compiled PDF to page images and visually inspect every page for clipping, overflows, float placement, readability, table width, figure quality, references, special characters, and author metadata.
6. Only after those checks pass may `PDF verified` and `official-template applied` be marked PASS.

No `main.tex` or PDF in `reports/submission_ready/` is represented as final until this gate is completed.