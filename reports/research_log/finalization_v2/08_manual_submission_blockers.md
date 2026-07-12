# Manual Submission Blockers

## Ethics — Blocking

The user reports that an ethics/IRB application is pending. Institution/university, ethics unit, application number and application date were supplied only as bracketed placeholders, so they remain unknown. No approval, exemption or reference number will be inferred. Scope supplied by the user: pre-existing secondary HR datasets; no recruitment, intervention, participant contact, survey, interview or human evaluation.

## Dataset Source, Licence and Citation — Blocking

INX, HRDataset_v14, IBM HR Analytics and Employee Turnover all remain `manual_review_required` for source authenticity, licence and citation verification. Until D3 and manual verification are resolved, raw redistribution and final data-availability wording are blocked.

## Public Git History — Blocking for Strict Non-Distribution

Raw datasets already exist in public history. Removing them from the current tip does not remove historical copies. A sanitized publication repository/history strategy requires explicit user authorization because force-push/history rewriting is prohibited by default.

## INX Workbook Equivalence — Open

The workbook exists and is hashed. A one-time Excel COM comparison found its first sheet equivalent to the normalized CSV, but executable repository-level equivalence is not yet established because no script/test exists and the locked Python environment lacks `xlrd`/`openpyxl`. The data-definition sheet also remains outside the provenance chain.

## Claim Matrix — Pending

The claim matrix cannot be approved until D1-D5 are decided and a clean, real-data v2 build completes.
