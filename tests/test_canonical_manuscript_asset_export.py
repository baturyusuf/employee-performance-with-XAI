from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from tools.canonical_manuscript_asset_export import (
    GENERATION_COMMIT,
    MAIN_FIGURE_EXPORTS,
    RUN_ID,
    SOURCE_TREE_HASH,
    validate_manuscript_asset_package,
)


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "manuscript/mdpi_information/assets"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_compact_asset_package_passes_closed_world_validation() -> None:
    result = validate_manuscript_asset_package(ROOT, ASSETS)

    assert result == {
        "status": "passed",
        "file_count_including_manifest": 109,
        "main_png": 7,
        "main_svg": 7,
        "supplementary_png": 3,
        "supplementary_svg": 3,
        "main_tables": 8,
        "total_bytes": 10_338_351,
        "manifest_sha256": "fbe7355b956df01ad9817f27b42dc13c0f3e0e33e7f0e5c42a2477beb9d001e1",
    }


def test_manifest_identity_inventory_and_source_lineage_are_complete() -> None:
    manifest = json.loads(
        (ASSETS / "manifests/manuscript_asset_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["status"] == "passed"
    assert manifest["manifest_kind"] == "compact_manuscript_support_assets"
    assert manifest["run_id"] == RUN_ID
    assert manifest["generation_commit"] == GENERATION_COMMIT
    assert manifest["source_tree_hash"] == SOURCE_TREE_HASH
    assert manifest["file_count_excluding_manifest"] == 108
    assert len(manifest["files"]) == 108
    assert all(record["source_paths"] for record in manifest["files"])
    assert all(
        len(record["source_paths"]) == len(record["source_sha256_values"])
        for record in manifest["files"]
    )


def test_manuscript_figure_numbering_is_explicit_and_gap_free() -> None:
    with (ASSETS / "manifests/figure_number_mapping.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert [int(row["manuscript_figure_number"]) for row in rows] == list(range(1, 8))
    assert [int(row["canonical_figure_number"]) for row in rows] == [
        1,
        3,
        2,
        4,
        5,
        6,
        7,
    ]
    assert [item["canonical_number"] for item in MAIN_FIGURE_EXPORTS] == [
        1,
        3,
        2,
        4,
        5,
        6,
        7,
    ]


def test_additive_v3_work_does_not_modify_frozen_v2_scientific_sources() -> None:
    result = _git(
        "diff",
        "--name-status",
        f"{GENERATION_COMMIT}..HEAD",
        "--",
        "src",
        "configs",
        "pyproject.toml",
        "requirements.txt",
        "requirements-lock.txt",
    )

    assert result.returncode == 0, result.stderr
    changes = [line.split("\t", maxsplit=1) for line in result.stdout.splitlines()]
    assert changes
    phase_1a_exception = {
        "configs/feature_availability_v3.json",
        "src/governance/feature_availability_contract.py",
    }
    assert all(status == "A" for status, _ in changes)
    assert all(
        path in phase_1a_exception or "_v3." in Path(path).name
        for _, path in changes
    )


def test_tracking_policy_keeps_full_evidence_and_sensitive_outputs_ignored() -> None:
    ignored = (
        f"reports/manuscript_final/{RUN_ID}/core/fold_models/model.joblib",
        f"reports/manuscript_final/{RUN_ID}/core/oof_predictions.parquet",
        "data/raw/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls",
        "data/external/hrdataset_v14/raw.csv",
    )
    for path in ignored:
        result = _git("check-ignore", "--no-index", path)
        assert result.returncode == 0, f"Expected ignored path: {path}\n{result.stderr}"

    tracked_export = _git(
        "check-ignore",
        "--no-index",
        "manuscript/mdpi_information/assets/README.md",
    )
    assert tracked_export.returncode == 1


def test_package_contains_no_row_level_or_model_binary_artifacts() -> None:
    forbidden_suffixes = {
        ".joblib",
        ".pkl",
        ".pickle",
        ".parquet",
        ".npy",
        ".npz",
        ".cbm",
        ".xlsx",
        ".xls",
    }
    files = [path for path in ASSETS.rglob("*") if path.is_file()]

    assert not [path for path in files if path.suffix.casefold() in forbidden_suffixes]
    assert not [path for path in files if "oof_prediction" in path.name.casefold()]
    assert not [path for path in files if "fold_model" in path.as_posix().casefold()]
