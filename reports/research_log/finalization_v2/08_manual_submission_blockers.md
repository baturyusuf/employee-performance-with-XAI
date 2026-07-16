# Manual Submission Blockers

The canonical technical evidence package is complete. The following items remain external or manual and are not resolved by code, tests, or scientific execution.

## 1. Ethics/IRB record

Status: **pending submission blocker**.

Required before submission:

- institution;
- ethics/IRB unit;
- application or reference number;
- application date;
- final determination and any required wording.

No approval, exemption, identifier, or date may be invented.

## 2. Dataset authenticity, licence, citation, and redistribution

Status: **manual review required** for INX, HRDataset_v14, IBM HR Analytics, and Employee Turnover inputs.

The repository binds the exact local bytes and recorded source metadata. It does not independently authenticate the upstream sources, determine licence validity, authorize redistribution, or supply a verified citation/DOI.

## 3. Historical Git publication strategy

Status: **open manual/publication blocker**.

The current branch tip excludes raw datasets and the large noncanonical Unit 2G package, but public history still contains previously committed blobs. History rewriting and force-push were explicitly prohibited in this task. Any cleanup, repository replacement, archive strategy, or disclosure requires separate authorization and coordination.

## 4. Claim approval and manuscript work

Status: **not started by design**.

Authors must review the canonical source tables, figures, and supported/prohibited claim boundaries before writing. Manuscript drafting, formatting, DOCX/PDF/LaTeX export, journal metadata, authorship declarations, acknowledgements, conflicts, funding, and submission forms are outside this technical task.

## 5. Public release and DOI

Status: **not created**.

The aggregate canonical publication-support subset is committed so reviewers can access every final table and figure. The remaining full-package internals are local/ignored. A separately approved Release/Zenodo-sized workflow must verify redistribution rights, checksums, access controls, and final URLs before those internals are published. No release URL or DOI is claimed.
