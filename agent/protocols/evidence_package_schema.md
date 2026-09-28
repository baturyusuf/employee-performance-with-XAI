# Evidence Package Schema

Store Evidence Packages under `agent/evidence/`.

Required format:

```markdown
# EP-<DOMAIN>-NNN — Evidence for WP-...

Work Package:
Status: COMPLETE | PARTIAL | FAILED | BLOCKED
Implemented by:
Verified by:
Branch:
Commit:
Environment:

## Implementation summary

## Files changed

## Exact commands executed

## Inputs and provenance

## Data split / fold identity

## Tests

## Outputs generated

## Results

## Statistical outputs

## Integrity checks

## Deviations from Work Package

## Failures / negative results

## Known limitations

## Artifact paths

## Reproduction command

## Implementation conclusion

No scientific conclusion is asserted here. Scientific interpretation is returned to the owning scientific advisor.
```

Rules:
- Do not report a metric without its evaluation population/split.
- Do not hide failed runs.
- Do not round away material differences in the raw evidence file.
- Persist seeds/configs where applicable.
- If results cannot be reproduced from the declared commit/config, status is not COMPLETE.
