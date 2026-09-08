# Immutable Release Readiness

Status: `blocked_preparation_only`

Proposed future tag: `v1.0.0-paper-revision`
Final release commit: `not recorded`
GitHub release URL: `not created`
Archive record/DOI: `not created`

## Verified repository snapshot

- Audit commit: `45fe0016f629f7ba9a9874e049d3a14dcc2e4360` on `finalization/leakage-aware-v2`.
- Public repository/default branch: https://github.com/baturyusuf/employee-performance-with-XAI / `main`.
- Detected repository software license: none.
- GitHub releases: 0; remote tags: 0.
- Historical local-only tag: `v0.3-real-llm-governance-evidence` at `a47dcdfcf9caacd59629156c37ce60547d6f9f5e`; it is not an origin tag and must not be cited as the final immutable release.
- Current tracked raw datasets: 0; five raw dataset paths remain in Git history and require a separately authorized strategy.

## Open blockers

- `B01` (authors): Author contributions/CRediT roles are not approved.
- `B02` (authors): Funding statement is not supplied or explicitly confirmed as none.
- `B03` (ethics): IRB/ethics institution, unit, reference, date, and determination are absent.
- `B04` (authors): Conflict-of-interest declarations are not approved by all authors.
- `B05` (authors): AI/tool-use disclosure requires author and target-journal policy review.
- `B06` (data_rights): No raw dataset has a complete authoritative source-to-local-byte and redistribution-rights chain.
- `B07` (software_license): The public repository has no detected software LICENSE; authors must select or explicitly decline one.
- `B08` (git_history): Five raw dataset paths remain in public Git history; remediation/disclosure needs separate authorization and coordination.
- `B09` (manuscript): The final v3 claim matrix is not frozen/approved and the manuscript/reviewer response/final simulation are incomplete.
- `B10` (immutable_identity): A final release commit and remote immutable tag do not yet exist.
- `B11` (publication): No GitHub release, archival deposit, release URL, or DOI exists.

## Authorized payload boundary

Tracked source, configuration, tests, manuscript, and validated compact aggregate evidence at the final approved commit.

Excluded: Raw/interim employee records, row-level predictions or SHAP values, fold/training partitions, fitted models/calibrators, secrets, caches, environments, and unlicensed local evidence.

## Ordered release and DOI procedure

1. Resolve author, ethics, software-license, dataset-rights, and Git-history decisions.
2. Freeze and obtain user approval for the final v3 claim matrix.
3. Rewrite and validate the manuscript, response package, declarations, figures, tables, and final review simulation.
4. Run full tests, independent package validators, repository hygiene, link, manifest, and manuscript render checks on a clean candidate commit.
5. Record the exact final commit in the release candidate manifest; create the proposed tag only on that commit after explicit authorization.
6. Publish a GitHub release only after explicit authorization and verify its immutable URL and attached artifact hashes.
7. Connect or upload the authorized release to Zenodo/OSF; reserve or mint a DOI through the archive UI; then record only the returned identifiers and URLs.
8. Update citation metadata and manuscript data/code-availability statements with the verified tag, commit, release URL, and DOI, followed by a final integrity pass.

## Official guidance

- https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
- https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content
- https://help.zenodo.org/docs/github/archive-software/github-upload/
- https://help.zenodo.org/docs/get-started/quickstart/
- https://support.zenodo.org/help/en-gb/18-general/216-what-is-a-doi
