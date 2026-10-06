# Work Packages

Directories should use `active/`, `completed/`, and `rejected/` as work accumulates.

Use the schema in `agent/protocols/work_package_schema.md`.

A code-writing agent must not begin substantive implementation without a READY Work Package unless the Human PI explicitly requests an emergency diagnostic. Even then, the diagnostic must not silently alter scientific protocol.

Completed, rejected, failed, and superseded packages are preserved for audit rather than deleted.
