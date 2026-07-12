# Dataset Card: HRDataset_v14 / Human Resources Data Set

Run ID: `manuscript_final_20260712T181754Z_c664ef152ff3`  
Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`
Provenance config hash: `f8349018704f719e9a2455322f983e3896e1f52bc4a93f7da5ee52d0a3a6cb64`

## File identity

- Raw file: `data/external/hrdataset_v14/raw.csv`
- SHA-256: `cb19996755c93c0a8d6527f59da4701c80aef65eff854906546dce286249813c`
- Shape: 311 rows x 36 columns

## Role and claim boundary

- Role: `independent_external_performance_target_replication`
- Task type: `ordinal_multiclass_performance`
- Allowed claim: Independent external performance-target replication; not locked-model transport unless a separate transported-model result is generated.

## Target mapping and observed support

Raw target: `PerformanceScore`. Predeclared performance-score mapping to canonical ordinal labels 2, 3, and 4.

- `Exceeds`: n=37 -> `4`
- `Fully Meets`: n=243 -> `3`
- `Needs Improvement`: n=18 -> `2`
- `PIP`: n=13 -> `2`

## Source, mirror, and licence status

- Retrieval URL: `https://raw.githubusercontent.com/pouyasattari/HR-Dataset-Analysis/main/HRDataset_v14.csv`
- Retrieval date: `manual_review_required`
- Known source/mirror status: `public_mirror_recorded_upstream_authenticity_unverified`
- Source-authenticity status: `manual_review_required`
- Licence: `manual_review_required`
- Licence verification: `manual_review_required`
- Citation/source: HRDataset_v14 / Human Resources Data Set; repository records the listed GitHub raw-file mirror, not an independently authenticated original source.
- Citation verification: `manual_review_required`

## Unresolved manual verification

- Confirm the actual retrieval date rather than treating the repository-add date as retrieval evidence.
- Confirm the authoritative original source and mirror chain.
- Confirm the dataset licence and reuse terms.
- Confirm the authoritative bibliographic citation.

Automated hashing and schema checks do not authenticate the upstream source or determine legal reuse rights. `manual_review_required` fields must be resolved by the author or another authorised reviewer before making stronger provenance or licence claims.
