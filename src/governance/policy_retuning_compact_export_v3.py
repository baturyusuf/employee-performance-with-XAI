"""Create and validate a publication-safe compact Phase 1D evidence package."""

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
from src.governance.policy_retuning_contract_v3 import POLICY_IDS
from src.governance.policy_retuning_run_validator_v3 import (
    DEFAULT_POLICY_RETUNING_RUN,
    validate_policy_retuning_run_v3,
)


DEFAULT_OUTPUT = Path(
    "reports/research_log/major_revision_v3/phase1d_policy_retuning"
)
MANIFEST_NAME = "manifest.json"
DIRECT_EXPORTS = {
    "aggregate_metrics.csv": "aggregate_metrics.csv",
    "metric_comparison.csv": "metric_comparison.csv",
    "headline_policy_comparison.csv": "headline_policy_comparison.csv",
    "selected_candidate_frequency.csv": "selected_candidate_frequency.csv",
}
EXPECTED_EXPORT_FILES = frozenset(
    {"README.md", "provenance_receipt.json", MANIFEST_NAME, *DIRECT_EXPORTS}
)
FORBIDDEN_PUBLIC_COLUMN_TOKENS = (
    "sample_index",
    "employee",
    "empnumber",
    "y_true",
    "y_pred",
    "prob_class_",
    "sample_key",
    "outer_fold",
    "inner_fold",
)


class V3PolicyRetuningCompactExportError(RuntimeError):
    """Raised when a compact Phase 1D export is unsafe or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V3PolicyRetuningCompactExportError(message)


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


def _markdown_summary(headline: pd.DataFrame, *, run_id: str) -> str:
    values = headline.set_index("policy_id")
    _require(set(values.index.astype(str)) == set(POLICY_IDS), "Headline policy set drifted.")
    lines = [
        "# Phase 1D Fixed-Schedule and Independently Retuned Policies — Compact Evidence",
        "",
        f"Source run: `{run_id}`",
        "",
        "This package contains aggregate policy evidence only. Employee-level OOF predictions, fold assignments, fold metrics, candidate-search rows, raw data, and fitted models are deliberately excluded.",
        "",
        "## Headline results",
        "",
        "Raw differences are independently retuned minus fixed primary-schedule values. For ordinal MAE, a negative raw difference is an improvement.",
        "",
        "| Policy | Features | Fixed Macro-F1 | Retuned Macro-F1 | Δ Macro-F1 | Fixed QWK | Retuned QWK | Δ QWK | Fixed balanced accuracy | Retuned balanced accuracy | Δ balanced accuracy | Fixed ordinal MAE | Retuned ordinal MAE | Δ ordinal MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy_id in POLICY_IDS:
        row = values.loc[policy_id]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{policy_id} {row['policy_name']}",
                    str(int(row["n_features"])),
                    f"{row['fixed_macro_f1']:.4f}",
                    f"{row['retuned_macro_f1']:.4f}",
                    f"{row['raw_difference_macro_f1']:+.4f}",
                    f"{row['fixed_quadratic_weighted_kappa']:.4f}",
                    f"{row['retuned_quadratic_weighted_kappa']:.4f}",
                    f"{row['raw_difference_quadratic_weighted_kappa']:+.4f}",
                    f"{row['fixed_balanced_accuracy']:.4f}",
                    f"{row['retuned_balanced_accuracy']:.4f}",
                    f"{row['raw_difference_balanced_accuracy']:+.4f}",
                    f"{row['fixed_ordinal_mae']:.4f}",
                    f"{row['retuned_ordinal_mae']:.4f}",
                    f"{row['raw_difference_ordinal_mae']:+.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Bounded interpretation",
            "",
            "- P3 is an exact replay control: its fixed and retuned values are identical because independent P3 tuning reproduces the canonical primary fold-specific candidate schedule and predictions.",
            f"- Retuning raises macro-F1 for P0, P1, P2, P4, and P5; the largest point difference is P2 ({values.loc['P2', 'raw_difference_macro_f1']:+.4f}). These are descriptive point differences, not confidence intervals or significance tests.",
            f"- The direction is not uniformly favorable across criteria. P0 balanced accuracy changes by {values.loc['P0', 'raw_difference_balanced_accuracy']:+.4f}; P5 QWK changes by {values.loc['P5', 'raw_difference_quadratic_weighted_kappa']:+.4f} and balanced accuracy by {values.loc['P5', 'raw_difference_balanced_accuracy']:+.4f}, even though its macro-F1 and ordinal MAE improve.",
            "- P0 retains outcome-proximal/timing-risk fields and is an information-rich diagnostic upper bound, not a deployable policy. Its high scores cannot be interpreted as prospective performance.",
            "- P4 is a prospective-plausibility sensitivity under timestamp-unverified cross-sectional data, not prospective validation. P5 removes declared organisational proxies but does not prove absence of residual proxies or fairness.",
            "- Fixed-schedule contrasts isolate model/schedule control more tightly; retuned contrasts combine feature access with policy-specific model selection. Neither is a causal feature or retuning effect.",
            "- No universally best policy, leakage-free system, fairness result, or deployment-ready HR decision system is identified.",
            "",
            "## Files",
            "",
            "- `aggregate_metrics.csv`: all 16 metrics for both estimands and six policies.",
            "- `metric_comparison.csv`: fixed, retuned, raw-difference, and direction-aligned values for every metric.",
            "- `headline_policy_comparison.csv`: macro-F1, QWK, balanced accuracy, and ordinal MAE summary.",
            "- `selected_candidate_frequency.csv`: policy-specific retuned candidate frequencies without fold or employee rows.",
            "- `provenance_receipt.json` and `manifest.json`: independent validation, immutable source identities, exclusions, and byte hashes.",
        ]
    )
    return "\n".join(lines) + "\n"


def export_policy_retuning_compact_v3(
    source_run: Path | str = DEFAULT_POLICY_RETUNING_RUN,
    output_dir: Path | str = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Validate the full run and atomically export only compact safe evidence."""

    source = Path(source_run)
    destination = Path(output_dir)
    _require(not destination.exists(), f"Compact destination already exists: {destination.as_posix()}.")
    run_receipt = validate_policy_retuning_run_v3(source)
    source_metadata = json.loads((source / "stage_metadata.json").read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for output_name, source_name in DIRECT_EXPORTS.items():
            shutil.copyfile(source / source_name, staging / output_name)
        headline = pd.read_csv(staging / "headline_policy_comparison.csv")
        _write_bytes(
            staging / "README.md",
            _markdown_summary(headline, run_id=run_receipt["run_id"]).encode("utf-8"),
        )
        source_files = {
            path.name: {
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in sorted(source.iterdir())
            if path.is_file()
        }
        provenance = {
            "schema_version": 1,
            "package_kind": "phase1d_policy_retuning_compact_evidence",
            "source_run": source.as_posix(),
            "source_run_id": run_receipt["run_id"],
            "source_generation_commit": run_receipt["generation_commit"],
            "policy_contract_sha256": run_receipt["policy_contract_sha256"],
            "scientific_input_sha256": run_receipt["scientific_input_sha256"],
            "source_created_at_utc": source_metadata["created_at_utc"],
            "independent_run_validation": run_receipt,
            "source_files": source_files,
            "included_files": sorted(DIRECT_EXPORTS),
            "excluded_source_files": sorted(set(source_files) - set(DIRECT_EXPORTS)),
            "publication_controls": {
                "employee_level_oof_rows_included": False,
                "fold_assignments_included": False,
                "fold_metrics_included": False,
                "candidate_search_rows_included": False,
                "selected_fold_hyperparameters_included": False,
                "raw_data_included": False,
                "fitted_models_included": False,
                "fixed_and_retuned_estimands_separately_labelled": True,
                "inferential_claim_from_point_difference_allowed": False,
                "universal_best_policy_claim_allowed": False,
            },
            "row_counts": {
                filename: int(len(pd.read_csv(staging / filename)))
                for filename in DIRECT_EXPORTS
            },
        }
        _write_bytes(staging / "provenance_receipt.json", _json_bytes(provenance))
        manifest_records = [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in sorted(staging.iterdir())
            if path.is_file()
        ]
        manifest = {
            "schema_version": 1,
            "package_kind": "phase1d_policy_retuning_compact_evidence",
            "source_run_id": run_receipt["run_id"],
            "file_count_excluding_manifest": len(manifest_records),
            "files": manifest_records,
        }
        _write_bytes(staging / MANIFEST_NAME, _json_bytes(manifest))
        _require({path.name for path in staging.iterdir() if path.is_file()} == EXPECTED_EXPORT_FILES, "Compact staging inventory drifted.")
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            for child in sorted(staging.iterdir(), reverse=True):
                if child.is_file():
                    child.unlink()
            staging.rmdir()
        raise
    return validate_policy_retuning_compact_v3(destination, source_run=source)


def validate_policy_retuning_compact_v3(
    package_dir: Path | str = DEFAULT_OUTPUT,
    *,
    source_run: Path | str = DEFAULT_POLICY_RETUNING_RUN,
) -> dict[str, Any]:
    """Validate compact contents, source equivalence, exclusions, and hashes."""

    package = Path(package_dir)
    source = Path(source_run)
    _require(package.is_dir(), f"Compact package is absent: {package.as_posix()}.")
    inventory = {path.name for path in package.iterdir() if path.is_file()}
    _require(inventory == EXPECTED_EXPORT_FILES, f"Compact closed-world inventory drifted: {sorted(inventory ^ EXPECTED_EXPORT_FILES)}.")
    _require(not any(path.is_dir() for path in package.iterdir()), "Compact package contains an unexpected directory.")
    run_receipt = validate_policy_retuning_run_v3(source)

    manifest = json.loads((package / MANIFEST_NAME).read_text(encoding="utf-8"))
    _require(manifest.get("source_run_id") == run_receipt["run_id"], "Compact manifest run id drifted.")
    records = manifest.get("files")
    _require(isinstance(records, list), "Compact manifest records are absent.")
    _require(manifest.get("file_count_excluding_manifest") == len(EXPECTED_EXPORT_FILES) - 1, "Compact manifest count drifted.")
    _require({record.get("path") for record in records} == EXPECTED_EXPORT_FILES - {MANIFEST_NAME}, "Compact manifest inventory drifted.")
    for record in records:
        path = package / str(record["path"])
        _require(path.stat().st_size == int(record["size_bytes"]), f"Compact size drifted for {path.name}.")
        _require(sha256_file(path) == record["sha256"], f"Compact hash drifted for {path.name}.")

    provenance = json.loads((package / "provenance_receipt.json").read_text(encoding="utf-8"))
    _require(provenance.get("source_run_id") == run_receipt["run_id"], "Compact provenance run id drifted.")
    _require(provenance.get("source_generation_commit") == run_receipt["generation_commit"], "Compact provenance commit drifted.")
    _require(provenance.get("independent_run_validation") == run_receipt, "Compact provenance validation receipt drifted.")
    controls = provenance.get("publication_controls")
    _require(isinstance(controls, Mapping), "Compact publication controls are absent.")
    for field in (
        "employee_level_oof_rows_included",
        "fold_assignments_included",
        "fold_metrics_included",
        "candidate_search_rows_included",
        "selected_fold_hyperparameters_included",
        "raw_data_included",
        "fitted_models_included",
        "inferential_claim_from_point_difference_allowed",
        "universal_best_policy_claim_allowed",
    ):
        _require(controls.get(field) is False, f"Compact control {field} drifted.")

    row_counts: dict[str, int] = {}
    for output_name, source_name in DIRECT_EXPORTS.items():
        compact_path = package / output_name
        source_path = source / source_name
        _require(sha256_file(compact_path) == sha256_file(source_path), f"Compact/source bytes drifted for {output_name}.")
        frame = pd.read_csv(compact_path)
        row_counts[output_name] = len(frame)
        normalized_columns = [str(column).casefold() for column in frame.columns]
        for token in FORBIDDEN_PUBLIC_COLUMN_TOKENS:
            _require(not any(token in column for column in normalized_columns), f"Compact {output_name} exposes forbidden column token {token}.")
    _require(row_counts == {"aggregate_metrics.csv": 192, "metric_comparison.csv": 96, "headline_policy_comparison.csv": 6, "selected_candidate_frequency.csv": 21}, "Compact row counts drifted.")
    _require(provenance.get("row_counts") == row_counts, "Compact provenance row counts drifted.")

    readme = (package / "README.md").read_text(encoding="utf-8")
    for required in (
        "Raw differences are independently retuned minus fixed",
        "P3 is an exact replay control",
        "not confidence intervals or significance tests",
        "No universally best policy",
        "Employee-level OOF predictions",
    ):
        _require(required in readme, f"Compact README boundary is absent: {required}.")
    return {
        "status": "passed",
        "source_run_id": run_receipt["run_id"],
        "source_generation_commit": run_receipt["generation_commit"],
        "file_count": len(EXPECTED_EXPORT_FILES),
        "total_size_bytes": sum(path.stat().st_size for path in package.iterdir() if path.is_file()),
        "manifest_sha256": sha256_file(package / MANIFEST_NAME),
        "row_counts": row_counts,
        "employee_level_rows_included": False,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_POLICY_RETUNING_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.validate_only:
        receipt = validate_policy_retuning_compact_v3(args.output_dir, source_run=args.source_run)
    else:
        receipt = export_policy_retuning_compact_v3(args.source_run, args.output_dir)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
