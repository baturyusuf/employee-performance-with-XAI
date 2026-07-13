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

The claim matrix cannot be approved until the Unit 2B primary metric is predeclared, all D1-D5 implementations and manual provenance/ethics gates are resolved as applicable, and a clean real-data v2 build completes. D1-D5 themselves have been answered; the bracketed D4 identity/reference fields remain intentionally unverified placeholders.
