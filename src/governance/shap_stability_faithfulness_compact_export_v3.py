"""Create and validate a publication-safe compact Phase 2A evidence package."""

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
from src.governance.shap_stability_faithfulness_run_validator_v3 import (
    DEFAULT_SHAP_STABILITY_FAITHFULNESS_RUN,
    validate_shap_stability_faithfulness_run_v3,
)


DEFAULT_OUTPUT = Path(
    "reports/research_log/major_revision_v3/phase2a_shap_stability_faithfulness"
)
MANIFEST_NAME = "manifest.json"
DIRECT_EXPORTS = {
    "aggregation_receipt.json": "aggregation_receipt.json",
    "stability_summary.csv": "stability_summary.csv",
    "faithfulness_summary.csv": "faithfulness_summary.csv",
    "faithfulness_contrasts.csv": "faithfulness_contrasts.csv",
    "deletion_auc_summary.csv": "deletion_auc_summary.csv",
}
EXPECTED_EXPORT_FILES = frozenset(
    {"README.md", "provenance_receipt.json", MANIFEST_NAME, *DIRECT_EXPORTS}
)
CSV_ROW_COUNTS = {
    "stability_summary.csv": 9,
    "faithfulness_summary.csv": 63,
    "faithfulness_contrasts.csv": 6,
    "deletion_auc_summary.csv": 21,
}
FORBIDDEN_PUBLIC_COLUMN_TOKENS = (
    "sample_index",
    "employee",
    "empnumber",
    "y_true",
    "y_pred",
    "prob_class_",
    "sample_key",
    "outer_fold",
    "training_membership",
    "deleted_features",
)


class V3ShapStabilityFaithfulnessCompactExportError(RuntimeError):
    """Raised when a compact Phase 2A export is unsafe or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V3ShapStabilityFaithfulnessCompactExportError(message)


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


def _markdown_summary(
    stability: pd.DataFrame,
    contrasts: pd.DataFrame,
    auc_summary: pd.DataFrame,
    *,
    run_id: str,
) -> str:
    stability_values = stability.set_index(["stability_type", "top_k"])
    contrast_values = contrasts.set_index(["deleted_feature_count", "metric"])
    guided_auc = float(
        auc_summary.loc[
            auc_summary["method"] == "shap_guided", "mean_deletion_auc"
        ].iloc[0]
    )
    random_auc = auc_summary.loc[
        auc_summary["method"] == "random", "mean_deletion_auc"
    ].astype(float)
    random_auc_mean = float(random_auc.mean())
    return "\n".join(
        [
            "# Phase 2A SHAP Stability and Model-Level Faithfulness — Compact Evidence",
            "",
            f"Source run: `{run_id}`",
            "",
            "This package contains only contract-approved aggregate evidence. Per-sample SHAP values, perturbation rows, fold/resample memberships, raw data, and fitted models are deliberately excluded.",
            "",
            "## Ranking stability",
            "",
            "| Comparison | Pairs | Top-5 Jaccard | Top-10 Jaccard | Top-15 Jaccard | All-feature Spearman |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *[
                "| "
                + " | ".join(
                    [
                        label,
                        str(int(stability_values.loc[(kind, 5), "n_pairs"])),
                        f"{stability_values.loc[(kind, 5), 'jaccard_mean']:.4f}",
                        f"{stability_values.loc[(kind, 10), 'jaccard_mean']:.4f}",
                        f"{stability_values.loc[(kind, 15), 'jaccard_mean']:.4f}",
                        f"{stability_values.loc[(kind, 5), 'spearman_mean']:.4f}",
                    ]
                )
                + " |"
                for kind, label in (
                    ("canonical_outer_fold_pair", "Canonical outer-fold pairs"),
                    ("model_seed", "Model seeds"),
                    ("outer_train_resample", "80% outer-train resamples"),
                )
            ],
            "",
            "The 45 canonical fold pairs, 15 model-seed pairs, and 10 resampling pairs share data and/or protocol components. They are not independent observations, so no confidence interval or significance claim is attached to these descriptive summaries.",
            "",
            "## Deletion faithfulness",
            "",
            "| Deleted features | Guided probability drop | Random-repetition mean | Guided − random |",
            "| ---: | ---: | ---: | ---: |",
            *[
                "| "
                + " | ".join(
                    [
                        str(count),
                        f"{contrast_values.loc[(count, 'mean_probability_drop'), 'guided_value']:.4f}",
                        f"{contrast_values.loc[(count, 'mean_probability_drop'), 'random_repetition_mean']:.4f}",
                        f"{contrast_values.loc[(count, 'mean_probability_drop'), 'guided_minus_random_mean']:+.4f}",
                    ]
                )
                + " |"
                for count in (1, 3, 5)
            ],
            "",
            f"The mean probability-drop deletion AUC is {guided_auc:.4f} for SHAP-guided deletion and {random_auc_mean:.4f} across the 20 random-repetition means (difference {guided_auc - random_auc_mean:+.4f}). These are descriptive model-level perturbation results, not inferential tests.",
            "",
            "## Interpretation boundaries",
            "",
            "- Global importance uses TreeSHAP raw margins, signed grouping from encoded columns to raw feature families, then absolute values and averaging across classes and exactly-once OOF samples.",
            "- Stability and faithfulness are different properties. Ranking agreement alone does not establish faithfulness; deletion behavior does not establish explanation robustness.",
            "- Median/mode masking can create out-of-distribution hybrid records. The deletion results therefore diagnose this fitted model under this masking intervention, not real employee outcomes.",
            "- Model attribution is not a causal feature effect, prescriptive HR advice, fairness evidence, prospective validation, or evidence of human explanation usefulness.",
            "- The evidence does not justify claims of a leakage-free or deployment-ready HR decision system.",
            "",
            "## Files",
            "",
            "- `aggregation_receipt.json`: SHAP library, output-space, axis, grouping, averaging, and additivity details.",
            "- `stability_summary.csv`: top-5/10/15 Jaccard and all-feature Spearman summaries.",
            "- `faithfulness_summary.csv`: probability/margin deletion summaries by method, repetition, and deletion count.",
            "- `faithfulness_contrasts.csv`: guided-minus-random descriptive contrasts for probability and raw margin.",
            "- `deletion_auc_summary.csv`: sample-aggregated probability-drop deletion AUC by method and repetition.",
            "- `provenance_receipt.json` and `manifest.json`: independent validation, source identities, exclusions, and byte hashes.",
        ]
    ) + "\n"


def export_shap_stability_faithfulness_compact_v3(
    source_run: Path | str = DEFAULT_SHAP_STABILITY_FAITHFULNESS_RUN,
    output_dir: Path | str = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Validate the full run and atomically export only compact safe evidence."""

    source = Path(source_run)
    destination = Path(output_dir)
    _require(
        not destination.exists(),
        f"Compact destination already exists: {destination.as_posix()}.",
    )
    run_receipt = validate_shap_stability_faithfulness_run_v3(source)
    source_metadata = json.loads(
        (source / "stage_metadata.json").read_text(encoding="utf-8")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for output_name, source_name in DIRECT_EXPORTS.items():
            shutil.copyfile(source / source_name, staging / output_name)
        readme = _markdown_summary(
            pd.read_csv(staging / "stability_summary.csv"),
            pd.read_csv(staging / "faithfulness_contrasts.csv"),
            pd.read_csv(staging / "deletion_auc_summary.csv"),
            run_id=run_receipt["run_id"],
        )
        _write_bytes(staging / "README.md", readme.encode("utf-8"))
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
            "package_kind": "phase2a_shap_stability_faithfulness_compact_evidence",
            "source_run": source.as_posix(),
            "source_run_id": run_receipt["run_id"],
            "source_generation_commit": run_receipt["generation_commit"],
            "contract_sha256": run_receipt["contract_sha256"],
            "scientific_input_sha256": run_receipt["scientific_input_sha256"],
            "source_created_at_utc": source_metadata["created_at_utc"],
            "independent_run_validation": run_receipt,
            "source_files": source_files,
            "included_files": sorted(DIRECT_EXPORTS),
            "excluded_source_files": sorted(set(source_files) - set(DIRECT_EXPORTS)),
            "publication_controls": {
                "per_sample_shap_rows_included": False,
                "per_sample_perturbation_rows_included": False,
                "fold_or_resample_memberships_included": False,
                "fold_feature_importance_rows_included": False,
                "pairwise_rows_included": False,
                "raw_data_included": False,
                "fitted_models_included": False,
                "pair_independence_assumed": False,
                "inferential_claim_allowed": False,
                "causal_feature_effect_claim_allowed": False,
                "human_usefulness_claim_allowed": False,
            },
            "row_counts": {
                filename: int(len(pd.read_csv(staging / filename)))
                for filename in CSV_ROW_COUNTS
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
            "package_kind": "phase2a_shap_stability_faithfulness_compact_evidence",
            "source_run_id": run_receipt["run_id"],
            "file_count_excluding_manifest": len(manifest_records),
            "files": manifest_records,
        }
        _write_bytes(staging / MANIFEST_NAME, _json_bytes(manifest))
        _require(
            {path.name for path in staging.iterdir() if path.is_file()}
            == EXPECTED_EXPORT_FILES,
            "Compact staging inventory drifted.",
        )
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            for child in sorted(staging.iterdir(), reverse=True):
                if child.is_file():
                    child.unlink()
            staging.rmdir()
        raise
    return validate_shap_stability_faithfulness_compact_v3(
        destination, source_run=source
    )


def validate_shap_stability_faithfulness_compact_v3(
    package_dir: Path | str = DEFAULT_OUTPUT,
    *,
    source_run: Path | str = DEFAULT_SHAP_STABILITY_FAITHFULNESS_RUN,
) -> dict[str, Any]:
    """Validate compact contents, source equivalence, exclusions, and hashes."""

    package = Path(package_dir)
    source = Path(source_run)
    _require(package.is_dir(), f"Compact package is absent: {package.as_posix()}.")
    inventory = {path.name for path in package.iterdir() if path.is_file()}
    _require(
        inventory == EXPECTED_EXPORT_FILES,
        f"Compact closed-world inventory drifted: {sorted(inventory ^ EXPECTED_EXPORT_FILES)}.",
    )
    _require(
        not any(path.is_dir() for path in package.iterdir()),
        "Compact package contains an unexpected directory.",
    )
    run_receipt = validate_shap_stability_faithfulness_run_v3(source)

    manifest = json.loads((package / MANIFEST_NAME).read_text(encoding="utf-8"))
    _require(
        manifest.get("source_run_id") == run_receipt["run_id"],
        "Compact manifest run id drifted.",
    )
    records = manifest.get("files")
    _require(isinstance(records, list), "Compact manifest records are absent.")
    _require(
        manifest.get("file_count_excluding_manifest")
        == len(EXPECTED_EXPORT_FILES) - 1,
        "Compact manifest count drifted.",
    )
    _require(
        {record.get("path") for record in records}
        == EXPECTED_EXPORT_FILES - {MANIFEST_NAME},
        "Compact manifest inventory drifted.",
    )
    for record in records:
        path = package / str(record["path"])
        _require(
            path.stat().st_size == int(record["size_bytes"]),
            f"Compact size drifted for {path.name}.",
        )
        _require(
            sha256_file(path) == record["sha256"],
            f"Compact hash drifted for {path.name}.",
        )

    provenance = json.loads(
        (package / "provenance_receipt.json").read_text(encoding="utf-8")
    )
    _require(
        provenance.get("source_run_id") == run_receipt["run_id"],
        "Compact provenance run id drifted.",
    )
    _require(
        provenance.get("source_generation_commit")
        == run_receipt["generation_commit"],
        "Compact provenance commit drifted.",
    )
    _require(
        provenance.get("independent_run_validation") == run_receipt,
        "Compact provenance validation receipt drifted.",
    )
    controls = provenance.get("publication_controls")
    _require(isinstance(controls, Mapping), "Compact publication controls are absent.")
    for field in (
        "per_sample_shap_rows_included",
        "per_sample_perturbation_rows_included",
        "fold_or_resample_memberships_included",
        "fold_feature_importance_rows_included",
        "pairwise_rows_included",
        "raw_data_included",
        "fitted_models_included",
        "pair_independence_assumed",
        "inferential_claim_allowed",
        "causal_feature_effect_claim_allowed",
        "human_usefulness_claim_allowed",
    ):
        _require(controls.get(field) is False, f"Compact control {field} drifted.")

    row_counts: dict[str, int] = {}
    for output_name, source_name in DIRECT_EXPORTS.items():
        compact_path = package / output_name
        source_path = source / source_name
        _require(
            sha256_file(compact_path) == sha256_file(source_path),
            f"Compact/source bytes drifted for {output_name}.",
        )
        if output_name.endswith(".csv"):
            frame = pd.read_csv(compact_path)
            row_counts[output_name] = len(frame)
            normalized_columns = [str(column).casefold() for column in frame.columns]
            for token in FORBIDDEN_PUBLIC_COLUMN_TOKENS:
                _require(
                    not any(token in column for column in normalized_columns),
                    f"Compact {output_name} exposes forbidden column token {token}.",
                )
    _require(row_counts == CSV_ROW_COUNTS, "Compact row counts drifted.")
    _require(
        provenance.get("row_counts") == row_counts,
        "Compact provenance row counts drifted.",
    )

    readme = (package / "README.md").read_text(encoding="utf-8")
    for required in (
        "They are not independent observations",
        "no confidence interval or significance claim",
        "Stability and faithfulness are different properties",
        "out-of-distribution hybrid records",
        "Model attribution is not a causal feature effect",
        "human explanation usefulness",
        "deployment-ready HR decision system",
        "Per-sample SHAP values",
    ):
        _require(
            required in readme,
            f"Compact README interpretation boundary is absent: {required}.",
        )
    return {
        "status": "passed",
        "source_run_id": run_receipt["run_id"],
        "source_generation_commit": run_receipt["generation_commit"],
        "file_count": len(EXPECTED_EXPORT_FILES),
        "total_size_bytes": sum(
            path.stat().st_size for path in package.iterdir() if path.is_file()
        ),
        "manifest_sha256": sha256_file(package / MANIFEST_NAME),
        "row_counts": row_counts,
        "per_sample_rows_included": False,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        type=Path,
        default=DEFAULT_SHAP_STABILITY_FAITHFULNESS_RUN,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.validate_only:
        receipt = validate_shap_stability_faithfulness_compact_v3(
            args.output_dir, source_run=args.source_run
        )
    else:
        receipt = export_shap_stability_faithfulness_compact_v3(
            args.source_run, args.output_dir
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
