# Manual Submission Blockers

## Ethics — Blocking

The user reports that an ethics/IRB application is pending. Institution/university, ethics unit, application number and application date were supplied only as bracketed placeholders, so they remain unknown. No approval, exemption or reference number will be inferred. Scope supplied by the user: pre-existing secondary HR datasets; no recruitment, intervention, participant contact, survey, interview or human evaluation.

## Dataset Source, Licence and Citation — Blocking

INX, HRDataset_v14, IBM HR Analytics and Employee Turnover all remain `manual_review_required` for source authenticity, licence and citation verification. D3 acquisition behavior is accepted and the current files are hash/schema verified, but that does not verify legal/source authenticity. Raw redistribution and final data-availability wording remain blocked.

HRDataset_v14 is presently Git-tracked even though its acquisition record says user-provided local file and licence/source authenticity are unverified. The fixed non-distribution decision requires removal from the current publication tip without deleting the user's local copy; public history cleanup remains a separate authorization-bound issue.

## Public Git History — Blocking for Strict Non-Distribution

Raw datasets already exist in public history. Removing them from the current tip does not remove historical copies. A sanitized publication repository/history strategy requires explicit user authorization because force-push/history rewriting is prohibited by default.

## INX Workbook Equivalence — Open

The workbook exists and is hashed. A one-time Excel COM comparison found its first sheet equivalent to the normalized CSV, but executable repository-level equivalence is not yet established because no script/test exists and the locked Python environment lacks `xlrd`/`openpyxl`. The data-definition sheet also remains outside the provenance chain.

## Claim Matrix — Pending

Macro-F1 is already the predeclared primary metric and QWK the fixed tie-breaker; D1-D5 are answered. The claim matrix remains pending because no clean complete v2 package exists and no source-of-truth result/table/figure map can yet be frozen. Author approval is required only after that package exists. Ethics and dataset source/licence facts remain separate explicit blockers; the bracketed D4 identity/reference fields remain intentionally unverified placeholders.

## HRDataset Department Proxy Estimate — Unsupported by Available Support

The configured nominal department proxy target contains a singleton class. Under the exact approved outer folds, at least one outer-training partition cannot contain all target classes, so the fail-closed external diagnostic is `not_estimated_insufficient_outer_training_class_support`. This does not block the mapped performance-target replication itself, but it blocks any numerical HRDataset department-reconstructability claim. Merging or dropping classes would change the estimand and has not been authorized.

## Noncanonical Evidence in Git History - Publication Hygiene Blocker

Commit `e25f403` pushed 126 Unit 2G stage-validation files totaling 65,412,766 bytes, including persisted models and row-level derivative evidence, contrary to D5. Local cleanup `b7b2ad3` removes the directory through a normal index-only forward change and preserves the validated package locally; tested work now extends through `1639e18`. The latest normal non-force push failed because noninteractive HTTPS credentials were unavailable, and public verification shows remote still at `e25f403`. Because force-push and history rewriting are prohibited, the blobs also remain retrievable from repository history after a future successful forward push. A later sanitized-publication/history strategy requires explicit authorization; until then this remains a publication-hygiene and repository-size blocker, especially while dataset source/licence authenticity is unverified.
