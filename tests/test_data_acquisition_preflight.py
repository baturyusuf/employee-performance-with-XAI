from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.data.canonical_loader import (
    AcquisitionNotApprovedError,
    DataIntegrityError,
    load_canonical_dataset,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_contract(
    tmp_path: Path,
    *,
    destination: Path,
    expected_source: Path,
    approved_url: str | None,
    allowed: bool,
) -> tuple[Path, Path]:
    config = tmp_path / "config.yaml"
    manifest = tmp_path / "acquisition.yaml"
    config.write_text(
        json.dumps(
            {
                "manuscript_final": {
                    "datasets": {"dataset": {"path": str(destination), "target": "target"}}
                }
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "data_acquisition": {
                    "schema_version": 1,
                    "physical_datasets": {
                        "physical": {
                            "local_path": str(destination),
                            "expected_sha256": _sha(expected_source),
                            "expected_rows": 2,
                            "expected_column_count": 2,
                            "expected_columns": ["feature", "target"],
                            "format": "csv",
                            "delimiter": ",",
                            "encoding": "utf-8",
                            "target_profiles": {
                                "main": {
                                    "raw_target": "target",
                                    "expected_distribution": {"2": 1, "3": 1},
                                }
                            },
                            "approved_download_url": approved_url,
                            "automatic_download_allowed": allowed,
                        }
                    },
                    "logical_bindings": {
                        "dataset": {
                            "physical_dataset": "physical",
                            "target_profile": "main",
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return config, manifest


def test_missing_file_without_approved_url_fails_closed(tmp_path: Path) -> None:
    expected = tmp_path / "expected.csv"
    expected.write_text("feature,target\n1,2\n2,3\n", encoding="utf-8")
    destination = tmp_path / "missing.csv"
    config, manifest = _write_contract(
        tmp_path,
        destination=destination,
        expected_source=expected,
        approved_url=None,
        allowed=False,
    )

    with pytest.raises(AcquisitionNotApprovedError, match="no approved automatic acquisition"):
        load_canonical_dataset(config, "dataset", manifest, allow_download=True)
    assert not destination.exists()


def test_approved_download_is_installed_only_after_full_validation(tmp_path: Path) -> None:
    remote = tmp_path / "remote.csv"
    remote.write_text("feature,target\n1,2\n2,3\n", encoding="utf-8")
    destination = tmp_path / "installed" / "dataset.csv"
    config, manifest = _write_contract(
        tmp_path,
        destination=destination,
        expected_source=remote,
        approved_url=remote.as_uri(),
        allowed=True,
    )

    loaded = load_canonical_dataset(config, "dataset", manifest, allow_download=True)

    assert destination.read_bytes() == remote.read_bytes()
    assert loaded.receipt["actual_sha256"] == _sha(remote)
    assert loaded.receipt["acquisition_method"] == "approved_manifest_download"
    assert loaded.receipt["schema_status"] == "valid"


def test_download_hash_mismatch_is_not_admitted_and_writes_report(tmp_path: Path) -> None:
    expected = tmp_path / "expected.csv"
    expected.write_text("feature,target\n1,2\n2,3\n", encoding="utf-8")
    remote = tmp_path / "unexpected.csv"
    remote.write_text("feature,target\n9,2\n8,3\n", encoding="utf-8")
    destination = tmp_path / "installed" / "dataset.csv"
    report = tmp_path / "reports" / "comparison.json"
    config, manifest = _write_contract(
        tmp_path,
        destination=destination,
        expected_source=expected,
        approved_url=remote.as_uri(),
        allowed=True,
    )

    with pytest.raises(DataIntegrityError, match="pinned acquisition contract"):
        load_canonical_dataset(
            config,
            "dataset",
            manifest,
            allow_download=True,
            mismatch_report_path=report,
        )

    assert not destination.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "sha256" in payload["differences"]
    assert payload["expected_sha256"] == _sha(expected)
    assert payload["actual_sha256"] == _sha(remote)


def test_existing_file_schema_or_target_drift_fails_closed(tmp_path: Path) -> None:
    expected = tmp_path / "expected.csv"
    expected.write_text("feature,target\n1,2\n2,3\n", encoding="utf-8")
    destination = tmp_path / "dataset.csv"
    destination.write_text("target,feature\n2,1\n3,2\n", encoding="utf-8")
    config, manifest = _write_contract(
        tmp_path,
        destination=destination,
        expected_source=destination,
        approved_url=None,
        allowed=False,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    record = payload["data_acquisition"]["physical_datasets"]["physical"]
    record["expected_columns"] = ["feature", "target"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataIntegrityError, match="ordered_schema"):
        load_canonical_dataset(config, "dataset", manifest)
