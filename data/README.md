# Data acquisition and verification

The scientific pipelines treat the raw datasets as user-provided inputs. The
pinned, machine-readable contract is
[`configs/data_acquisition.yaml`](../configs/data_acquisition.yaml). It records
the expected repository-relative path, SHA-256 digest, CSV decoding rules,
ordered schema, dimensions, and raw target distribution for each physical
dataset. The IBM performance and attrition tasks intentionally bind to the same
physical IBM file with different target profiles.

## User-provided workflow

1. Obtain each dataset through a source and under reuse terms that you have
   independently verified.
2. Place the file at the exact `local_path` declared in the acquisition
   contract. Do not rename columns, reorder columns, rewrite delimiters, or
   resave the file before verification; those changes alter the pinned input.
3. Run the repository acquisition preflight before any scientific build. The
   preflight must verify the SHA-256 digest, encoding, delimiter, ordered schema,
   row and column counts, target column, and raw target distribution.
4. If any check differs, stop. Do not use the file, silently normalize it, or
   substitute an older interim/cache copy. Produce a mismatch report and obtain
   a scientific decision before changing the pinned contract or dataset.

The local datasets currently present in the working checkout match the pinned
contract. They are ignored and absent from the current publication branch tip;
their presence is not evidence that source authenticity, licence, citation, or
redistribution rights have been verified. Dataset cards, schema mappings, and
directory placeholders remain tracked because they contain the reproducibility
contract rather than employee-level records.

## Missing files and downloads

Automatic download is allowed only when a physical dataset record has both an
explicitly approved `approved_download_url` and
`automatic_download_allowed: true`. A `recorded_unverified_url` is provenance
context only: it is not an approved acquisition source and must never be used
as an automatic fallback. The current contract permits no automatic downloads.

If a local file is missing and no approved download is configured, the
preflight and scientific build must fail closed with a clear acquisition error.
They must not search for mirrors, use a similarly named file, consume an
interim dataset, generate synthetic data, or reuse cached scientific results.

## Local preservation and publication

Acquisition and publication tooling must not delete a user's local raw files.
Removing a file from version control, when separately authorized, must preserve
the working-tree copy (for example, by using an index-only untracking step).

The current branch tip does not track the declared raw, interim, or processed
working-data paths. Existing Git history is development history and still
contains earlier copies. The approved publication strategy is therefore a
separate allowlisted `git archive` export from a verified commit tree; it
contains no `.git` history and no unverified data path. Creating that export
does not rewrite this development repository's history. Code, small
manuscript-facing tables and figures, and integrity manifests may remain in
Git; larger evidence packages are prepared for a separately approved GitHub
Release or Zenodo deposit. No release or deposit is authorized merely by this
data policy.

## Manual submission blockers

All four physical datasets currently have unresolved source/licence review
status. Before submission or redistribution, the author must independently
confirm the authoritative source chain, licence/reuse terms, and citation. The
software must not convert `manual_review_required` into an approval.
