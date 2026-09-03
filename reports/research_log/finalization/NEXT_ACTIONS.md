# Next Actions

## Active additive v3 unit — 2026-09-03

The broader reviewer brief is now the active extension. Preserve the complete v2 package, but do not infer that its narrower protocol closes the new brief.

1. Completed and pushed: validated v3 planning/coverage checkpoint (`618a3936e040db119eee3d43f6887d3ed4f447ea`).
2. Implemented for INX and awaiting its own tested Git checkpoint: prediction-time scenario, complete 28-field availability contract, and exact P0–P5 semantic policy map. HR-specific semantics remain in Phase 3A.
3. Implemented and awaiting clean checkpoint: two ordinal-aware models, three naive baselines, RPS, per-class metrics, confusion outputs, and a hash-bound runner using the exact canonical-v2 folds and nominal OOF evidence.
4. Commit/push Phase 1B implementation, then run the complete local nine-system benchmark from that exact clean commit. Validate and export only compact non-row-level evidence.
5. Add repeated nested-CV and independently retuned policy estimands after the primary ordinal benchmark evidence is accepted by its automated gates.
6. Continue with the SHAP, calibration, subgroup/proxy, HR sensitivity, data-quality, literature, provenance, and deliverable phases in `../major_revision_v3/PLAN.md`.
7. Freeze and request approval for the final v3 claim matrix before editing `manuscript/mdpi_information/main.md` or `main.tex`.

Current planning evidence: `../major_revision_v3/REQUIREMENT_COVERAGE_AUDIT.md`.

## Historical v2 handoff state — 2026-07-16

The technical evidence build is complete. Do not start another scientific unit or rerun the canonical package without a concrete integrity failure.

Next independent unit for a new chat: **manual claim-matrix review and manuscript-authoring handoff**.

Required manual/external actions before journal submission:

1. Obtain and record ethics/IRB institution, unit, application/reference number, and date.
2. Verify dataset source authenticity, licence, citation, and redistribution permissions.
3. Decide the separately authorized strategy for historical raw/noncanonical blobs in public Git history.
4. Review and approve supported/prohibited claims from the canonical source tables and claim-boundary files.
5. Only after that approval, begin manuscript writing/formatting in a separate task.
6. Create public release/DOI URLs only when they actually exist; do not invent them.

Canonical handoff inputs:

- `../finalization_v2/15_canonical_evidence_receipt.json`
- `../../manuscript_final/latest/pointer.json`
- `../../../manuscript/mdpi_information/assets/README.md`
- `../../../manuscript/mdpi_information/assets/handoff/figure_table_insertion_guide.md`
- `../../../manuscript/mdpi_information/assets/handoff/manuscript_exact_results.json`
- `../../../manuscript/mdpi_information/assets/handoff/manuscript_evidence_ledger.csv`
- `../../../manuscript/mdpi_information/assets/handoff/claim_boundary_handoff.md`
- `../../../manuscript/mdpi_information/assets/manifests/manuscript_asset_manifest.json`

The compact publication-support export is tracked in Git and linked from README. The complete 545-file canonical package is intentionally local/ignored and is not replaced by this export. Use a separately approved Release/Zenodo-sized workflow only if publication of those internals is later authorized.
