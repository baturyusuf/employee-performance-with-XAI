# Dataset Card: INX Future Inc Employee Performance

Run ID: `manuscript_final_20260712T181754Z_c664ef152ff3`  
Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`
Provenance config hash: `f8349018704f719e9a2455322f983e3896e1f52bc4a93f7da5ee52d0a3a6cb64`

## File identity

- Raw file: `data/raw/inx_employee_performance.csv`
- SHA-256: `b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a`
- Shape: 1200 rows x 28 columns

## Role and claim boundary

- Role: `primary_development_and_internal_oof_evaluation`
- Task type: `ordinal_multiclass_performance`
- Allowed claim: Internal out-of-fold employee-performance evidence; not independent external validation.

## Target mapping and observed support

Raw target: `PerformanceRating`. Identity mapping for observed ordinal performance labels 2, 3, and 4.

- `2`: n=194 -> `2`
- `3`: n=874 -> `3`
- `4`: n=132 -> `4`

## Source, mirror, and licence status

- Retrieval URL: `manual_review_required`
- Retrieval date: `manual_review_required`
- Known source/mirror status: `repository_local_copy_upstream_source_unverified`
- Source-authenticity status: `manual_review_required`
- Licence: `manual_review_required`
- Licence verification: `manual_review_required`
- Citation/source: INX Future Inc Employee Performance dataset title and repository-local raw file; upstream bibliographic source is not independently verified.
- Citation verification: `manual_review_required`

## Unresolved manual verification

- Confirm the upstream retrieval URL and actual retrieval date.
- Confirm the authoritative source and mirror chain.
- Confirm the dataset licence and reuse terms.
- Confirm the authoritative bibliographic citation.

Automated hashing and schema checks do not authenticate the upstream source or determine legal reuse rights. `manual_review_required` fields must be resolved by the author or another authorised reviewer before making stronger provenance or licence claims.
