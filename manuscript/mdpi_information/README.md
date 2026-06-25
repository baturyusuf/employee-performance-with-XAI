# MDPI Information Manuscript Package

This directory contains a draft manuscript package for *Information* (MDPI), article type: Original Research Article.

## Scope

The manuscript is based on the repository snapshot tagged:

```text
v0.3-real-llm-governance-evidence
```

It uses existing repository evidence only. No new OpenAI calls, model training runs, or LLM-output regeneration were performed for this manuscript package.

## Main Files

```text
main.tex
main.md
references.bib
template_notes.md
submission_checklist.md
claim_audit.md
missing_items.md
reference_verification_needed.md
figure_manifest.csv
table_manifest.csv
```

Generated tables are under:

```text
tables/
```

Generated figures are under:

```text
figures/
```

Each figure is exported as SVG and PNG. The manifest files record source data paths.

## Evidence Boundaries

- The predictive model remains XGBoost.
- The LLM is not the predictive model.
- The final real OpenAI LLM-agent evidence covers 80 cases: 40 INX and 40 HRDataset_v14.
- The final LLM model recorded by the evidence manifest is `gpt-5.4-mini`.
- Stub/dry-run outputs are retained only for reproducibility and pipeline testing and are not manuscript-grade real LLM evidence.
- Final readiness remains `not_ready` due to proxy-risk and counterfactual-actionability blockers.
- No deployment-ready or autonomous HR decision claim is made.

## Template Status

The official MDPI template files are not bundled here. See `template_notes.md`.

## Author Review Required

The manuscript is ready for scientific and formatting review, not direct submission. All `AUTHOR_TO_COMPLETE` fields must be completed before submission.
