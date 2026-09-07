"""Create and validate the publication-safe Phase 3B data-quality package."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.data.canonical_loader import sha256_file
from src.governance.data_quality_run_validator_v3 import (
    DEFAULT_DATA_QUALITY_RUN,
    validate_data_quality_run_v3,
)


DEFAULT_OUTPUT = Path("reports/research_log/major_revision_v3/phase3b_data_quality")
MANIFEST_NAME = "manifest.json"
REPORT_NAME = "DATA_QUALITY_REPORT.md"
DIRECT_EXPORTS = {
    "categorical_cardinality.csv": "categorical_cardinality.csv",
    "cleaned_schema.csv": "cleaned_schema.csv",
    "column_profiles.csv": "column_profiles.csv",
    "dataset_summary.csv": "dataset_summary.csv",
    "duplicate_audit.csv": "duplicate_audit.csv",
    "identifier_audit.csv": "identifier_audit.csv",
    "manuscript_ready_data_quality.csv": "manuscript_ready_data_quality.csv",
    "raw_schema.csv": "raw_schema.csv",
    "rule_anomaly_audit.csv": "rule_anomaly_audit.csv",
    "target_distribution.csv": "target_distribution.csv",
}
CSV_ROW_COUNTS = {
    "categorical_cardinality.csv": 27,
    "cleaned_schema.csv": 67,
    "column_profiles.csv": 64,
    "dataset_summary.csv": 2,
    "duplicate_audit.csv": 2,
    "identifier_audit.csv": 6,
    "manuscript_ready_data_quality.csv": 2,
    "raw_schema.csv": 64,
    "rule_anomaly_audit.csv": 52,
    "target_distribution.csv": 10,
}
EXPECTED_EXPORT_FILES = frozenset(
    {*DIRECT_EXPORTS, REPORT_NAME, "provenance_receipt.json", MANIFEST_NAME}
)
FORBIDDEN_ROW_LEVEL_COLUMNS = {
    "sample_index",
    "sample_key",
    "row_index",
    "row_id",
    "raw_value",
    "example_value",
    "y_true",
    "y_pred",
}


class DataQualityCompactExportV3Error(RuntimeError):
    """Raised when the compact Phase 3B package is unsafe or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataQualityCompactExportV3Error(message)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _report(run_id: str) -> str:
    return "\n".join(
        [
            "# Core Dataset Data Quality Report",
            "",
            f"Source run: `{run_id}`",
            "",
            "## Scope and method",
            "",
            "This aggregate-only audit covers the two datasets in the canonical core evidence scope: INX and HRDataset_v14. The IBM and turnover datasets are supplementary and are not represented here as core datasets. Exact pinned CSV bytes were loaded offline; no automatic download, model fitting, paid API, source-row repair, or row-level publication occurred.",
            "",
            "Missingness treats whitespace-only strings as missing for audit purposes. Duplicate checks report exact rows, predictor-equal rows after excluding the raw target, and conflicting-target predictor groups. A nonmissing mode share of at least 99% defines near-constant. Identifier candidates require a declaration, an exact row-order sequence, or both a name signal and at least 98% uniqueness. Numeric, temporal, and text/code rules were fixed in `configs/data_quality_v3.json` before the exact run.",
            "",
            "Raw and cleaned schema hashes are SHA-256 digests of canonical JSON records containing ordered position, column name, and pandas dtype. The cleaned schema is the canonical pre-model table: INX remains the verified 28-column frame; HRDataset_v14 has 39 columns after governed renaming/string trimming, a generated excluded row key, review-minus-hire tenure derivation, and mapped 2/3/4 target creation. Per-fold preprocessing remains outside this audit and is training-only in the modeling protocols.",
            "",
            "## Manuscript-ready summary",
            "",
            "| Dataset | Rows | Raw columns | Cleaned columns | Missing cells | Exact duplicate extra rows | Near-constant columns | Identifier candidates in primary features | Rules with findings |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| INX (primary) | 1,200 | 28 | 28 | 0 (0.00%) | 0 | 0 | 0 | 0 |",
            "| HRDataset_v14 (independent replication) | 311 | 36 | 39 | 215 (1.92%) | 0 | 0 | 0 | 3 |",
            "",
            "The machine-readable version of this table is `manuscript_ready_data_quality.csv`.",
            "",
            "## Findings",
            "",
            "### INX",
            "",
            "The 1,200 × 28 raw table has no effective missing cells, blank strings, exact duplicate rows, predictor-equal duplicate rows, constant columns, or near-constant columns. The declared numeric domains and four tenure relations have no violations. PerformanceRating support is 194/874/132 for labels 2/3/4. EmpNumber is the single detected identifier candidate and is absent from the P3 primary feature set.",
            "",
            "These are results under the declared rules, not proof that the data are error-free, representative, authentic, prospectively valid, or licensed for every use.",
            "",
            "### HRDataset_v14",
            "",
            "The 311 × 36 raw table has no exact or predictor-equal duplicate rows, no constant/near-constant columns, and 215 effective missing cells: 207 DateofTermination values and 8 ManagerID values. The termination-date missingness is structurally aligned with the 207 records whose Termd flag is zero; all 104 terminated records have a termination date. Missing ManagerID remains missing and is not repaired.",
            "",
            "Three declared rules have findings, each affecting two rows (0.64% of the checked rows): the last performance-review date precedes the hire date; PerformanceScore text disagrees with PerfScoreID; and two rows are minority Department texts within their DeptID group. The governed tenure derivation preserves the first finding by setting the two negative durations to missing. Performance modeling uses the PerformanceScore text mapping, not PerfScoreID, and excludes both target aliases. The conservative primary policy also excludes Department/DeptID and every declared or generated identifier candidate.",
            "",
            "Raw target support is PIP 13, Needs Improvement 18, Fully Meets 243, and Exceeds 37. The retained mapped target support is 31/243/37 for labels 2/3/4. This dataset-specific mapping does not establish construct or prevalence equivalence with INX.",
            "",
            "## Identifier and leakage interpretation",
            "",
            "The audit detects/examines INX EmpNumber and HRDataset_v14 Employee_Name, EmpID, ManagerName, ManagerID, plus the generated ExternalSampleId. All are excluded from their declared primary model feature sets. High cardinality alone is not treated as identity, so Salary and DOB are not mislabeled merely because their observed values are mostly unique. Identifier exclusion reduces a direct memorization channel; it does not prove that indirect proxy leakage is absent.",
            "",
            "## Construct and generalization boundaries",
            "",
            "PerformanceRating/PerformanceScore are recorded organizational performance ratings. The pinned source documentation does not establish either target as true employee capability, objective productivity, or future potential. The data-quality audit does not create external generalization evidence, validate a locked INX model on HRDataset_v14, establish causal validity or fairness, certify deployment readiness, or resolve source authenticity and licence review.",
            "",
            "No source values were corrected. Any future correction would require a separately justified, versioned sensitivity analysis rather than silent cleaning.",
            "",
            "## Package contents",
            "",
            "- `dataset_summary.csv`: dataset-level dimensions, schema hashes, missingness, duplicates, identifiers, and rule findings.",
            "- `column_profiles.csv` and `categorical_cardinality.csv`: complete aggregate raw-column profiles without example values.",
            "- `duplicate_audit.csv` and `identifier_audit.csv`: duplicate/target-conflict counts and primary-feature exclusion checks.",
            "- `rule_anomaly_audit.csv`: all 52 declared domain, temporal, and consistency rules, including zero-retaining rows.",
            "- `target_distribution.csv`: complete raw and governed mapped target support.",
            "- `raw_schema.csv` and `cleaned_schema.csv`: ordered schemas, transformations, missing counts, and schema hashes.",
            "- `manuscript_ready_data_quality.csv`: compact two-row table for later claim-matrix/manuscript work.",
            "- `provenance_receipt.json` and `manifest.json`: exact source-run validation, publication controls, byte sizes, and hashes.",
        ]
    ) + "\n"


def export_data_quality_compact_v3(
    source_run: Path | str = DEFAULT_DATA_QUALITY_RUN,
    output_dir: Path | str = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Independently validate the source run and atomically export aggregates."""

    source = Path(source_run)
    destination = Path(output_dir)
    _require(not destination.exists(), f"Compact destination already exists: {destination.as_posix()}.")
    run_receipt = validate_data_quality_run_v3(source)
    source_metadata = json.loads((source / "stage_metadata.json").read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for output_name, source_name in DIRECT_EXPORTS.items():
            shutil.copyfile(source / source_name, staging / output_name)
        _write_bytes(staging / REPORT_NAME, _report(run_receipt["run_id"]).encode("utf-8"))
        source_files = {
            path.name: {"sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}
            for path in sorted(source.iterdir())
            if path.is_file()
        }
        provenance = {
            "schema_version": 1,
            "package_kind": "phase3b_core_data_quality_compact_evidence",
            "source_run": source.as_posix(),
            "source_run_id": run_receipt["run_id"],
            "source_generation_commit": run_receipt["generation_commit"],
            "contract_sha256": run_receipt["contract_sha256"],
            "scientific_input_sha256": run_receipt["scientific_input_sha256"],
            "source_created_at_utc": source_metadata["created_at_utc"],
            "independent_run_validation": run_receipt,
            "source_files": source_files,
            "directly_included_source_files": sorted(DIRECT_EXPORTS.values()),
            "excluded_source_files": sorted(set(source_files) - set(DIRECT_EXPORTS.values())),
            "publication_controls": {
                "raw_data_included": False,
                "row_level_values_or_identifiers_included": False,
                "source_rows_repaired": False,
                "supplementary_datasets_represented_as_core": False,
                "model_outputs_or_fitted_objects_included": False,
                "source_authenticity_or_licence_established": False,
                "construct_validity_established": False,
                "external_generalization_established": False,
                "locked_model_transport_established": False,
                "causal_or_fairness_claim_allowed": False,
                "deployment_claim_allowed": False,
            },
            "row_counts": CSV_ROW_COUNTS,
        }
        _write_bytes(staging / "provenance_receipt.json", _json_bytes(provenance))
        records = [
            {"path": path.name, "sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}
            for path in sorted(staging.iterdir())
            if path.is_file()
        ]
        manifest = {
            "schema_version": 1,
            "package_kind": "phase3b_core_data_quality_compact_evidence",
            "source_run_id": run_receipt["run_id"],
            "file_count_excluding_manifest": len(records),
            "files": records,
        }
        _write_bytes(staging / MANIFEST_NAME, _json_bytes(manifest))
        _require(
            {path.name for path in staging.iterdir() if path.is_file()} == EXPECTED_EXPORT_FILES,
            "Compact Phase 3B staging inventory drifted.",
        )
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            for child in sorted(staging.iterdir(), reverse=True):
                if child.is_file():
                    child.unlink()
            staging.rmdir()
        raise
    return validate_data_quality_compact_v3(destination, source_run=source)


def validate_data_quality_compact_v3(
    package_dir: Path | str = DEFAULT_OUTPUT,
    *,
    source_run: Path | str = DEFAULT_DATA_QUALITY_RUN,
) -> dict[str, Any]:
    """Validate compact contents, source equivalence, exclusions, and hashes."""

    package = Path(package_dir)
    source = Path(source_run)
    _require(package.is_dir(), f"Compact Phase 3B package is absent: {package.as_posix()}.")
    inventory = {path.name for path in package.iterdir() if path.is_file()}
    _require(inventory == EXPECTED_EXPORT_FILES, f"Compact Phase 3B closed-world inventory drifted: {sorted(inventory ^ EXPECTED_EXPORT_FILES)}.")
    _require(not any(path.is_dir() for path in package.iterdir()), "Compact Phase 3B package contains a directory.")
    manifest = json.loads((package / MANIFEST_NAME).read_text(encoding="utf-8"))
    records = manifest.get("files")
    _require(isinstance(records, list), "Compact Phase 3B manifest records are absent.")
    _require(manifest.get("file_count_excluding_manifest") == len(EXPECTED_EXPORT_FILES) - 1, "Compact Phase 3B manifest count drifted.")
    _require({record.get("path") for record in records} == EXPECTED_EXPORT_FILES - {MANIFEST_NAME}, "Compact Phase 3B manifest inventory drifted.")
    for record in records:
        path = package / str(record["path"])
        _require(path.stat().st_size == int(record["size_bytes"]), f"Compact Phase 3B size drifted for {path.name}.")
        _require(sha256_file(path) == record["sha256"], f"Compact Phase 3B hash drifted for {path.name}.")
    run_receipt = validate_data_quality_run_v3(source)
    _require(manifest.get("source_run_id") == run_receipt["run_id"], "Compact Phase 3B manifest run id drifted.")
    provenance = json.loads((package / "provenance_receipt.json").read_text(encoding="utf-8"))
    _require(provenance.get("source_run_id") == run_receipt["run_id"], "Compact Phase 3B provenance run id drifted.")
    _require(provenance.get("source_generation_commit") == run_receipt["generation_commit"], "Compact Phase 3B provenance commit drifted.")
    _require(provenance.get("contract_sha256") == run_receipt["contract_sha256"], "Compact Phase 3B provenance contract drifted.")
    _require(provenance.get("scientific_input_sha256") == run_receipt["scientific_input_sha256"], "Compact Phase 3B provenance scientific input drifted.")
    _require(provenance.get("independent_run_validation") == run_receipt, "Compact Phase 3B validation receipt drifted.")
    controls = provenance.get("publication_controls")
    _require(isinstance(controls, Mapping) and controls, "Compact Phase 3B publication controls are absent.")
    for field, value in controls.items():
        _require(value is False, f"Compact Phase 3B publication control {field} drifted.")
    row_counts: dict[str, int] = {}
    for output_name, source_name in DIRECT_EXPORTS.items():
        compact_path = package / output_name
        _require(sha256_file(compact_path) == sha256_file(source / source_name), f"Compact/source bytes drifted for {output_name}.")
        frame = pd.read_csv(compact_path)
        row_counts[output_name] = len(frame)
        _require(FORBIDDEN_ROW_LEVEL_COLUMNS.isdisjoint(str(column).casefold() for column in frame.columns), f"Compact Phase 3B exposes a row-level column: {output_name}.")
    _require(row_counts == CSV_ROW_COUNTS, f"Compact Phase 3B row counts drifted: {row_counts}.")
    _require(provenance.get("row_counts") == CSV_ROW_COUNTS, "Compact Phase 3B provenance row counts drifted.")
    _require(provenance.get("directly_included_source_files") == sorted(DIRECT_EXPORTS.values()), "Compact Phase 3B included-source inventory drifted.")
    _require(provenance.get("excluded_source_files") == ["stage_metadata.json"], "Compact Phase 3B excluded-source inventory drifted.")
    report = (package / REPORT_NAME).read_text(encoding="utf-8")
    for phrase in (
        "two datasets in the canonical core evidence scope",
        "IBM and turnover datasets are supplementary",
        "no automatic download, model fitting, paid API, source-row repair, or row-level publication",
        "215 effective missing cells",
        "Three declared rules have findings",
        "All are excluded from their declared primary model feature sets",
        "not establish either target as true employee capability, objective productivity, or future potential",
        "does not create external generalization evidence",
        "does not prove that indirect proxy leakage is absent",
        "No source values were corrected",
    ):
        _require(phrase.casefold() in report.casefold(), f"Compact Phase 3B report boundary is absent: {phrase}.")
    return {
        "status": "passed",
        "package_dir": package.as_posix(),
        "source_run_id": run_receipt["run_id"],
        "source_generation_commit": run_receipt["generation_commit"],
        "file_count": len(EXPECTED_EXPORT_FILES),
        "total_size_bytes": sum(path.stat().st_size for path in package.iterdir() if path.is_file()),
        "manifest_sha256": sha256_file(package / MANIFEST_NAME),
        "row_counts": row_counts,
        "raw_data_included": False,
        "row_level_values_or_identifiers_included": False,
        "source_rows_repaired": False,
        "supplementary_datasets_represented_as_core": False,
        "construct_validity_established": False,
        "external_generalization_established": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_DATA_QUALITY_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    receipt = (
        validate_data_quality_compact_v3(arguments.output, source_run=arguments.source_run)
        if arguments.validate_only
        else export_data_quality_compact_v3(arguments.source_run, arguments.output)
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CSV_ROW_COUNTS",
    "DEFAULT_OUTPUT",
    "DIRECT_EXPORTS",
    "EXPECTED_EXPORT_FILES",
    "DataQualityCompactExportV3Error",
    "export_data_quality_compact_v3",
    "validate_data_quality_compact_v3",
]
