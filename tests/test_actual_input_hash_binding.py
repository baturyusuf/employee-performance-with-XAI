from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.canonical_loader import CanonicalDataError, load_canonical_dataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_contract(tmp_path: Path, source: Path) -> tuple[Path, Path]:
    config = tmp_path / "config.yaml"
    manifest = tmp_path / "acquisition.yaml"
    config.write_text(
        json.dumps(
            {
                "manuscript_final": {
                    "datasets": {
                        "declared": {"path": str(source), "target": "target"}
                    }
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
                            "local_path": str(source),
                            "expected_sha256": _sha(source),
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
                            "approved_download_url": None,
                            "automatic_download_allowed": False,
                        }
                    },
                    "logical_bindings": {
                        "declared": {
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


def test_configured_path_is_actual_input_even_when_unrelated_interim_exists(tmp_path: Path) -> None:
    configured = tmp_path / "configured.csv"
    configured.write_text("feature,target\n11,2\n12,3\n", encoding="utf-8")
    unrelated_interim = tmp_path / "inx_employee_performance_validated.csv"
    unrelated_interim.write_text("feature,target\n999,2\n998,3\n", encoding="utf-8")
    config, manifest = _write_contract(tmp_path, configured)

    loaded = load_canonical_dataset(config, "declared", manifest)

    assert loaded.frame["feature"].tolist() == [11, 12]
    assert loaded.receipt["actual_sha256"] == _sha(configured)
    assert loaded.receipt["row_count"] == 2
    assert loaded.receipt["column_count"] == 2
    assert loaded.receipt["schema_status"] == "valid"
    assert loaded.receipt["schema_columns"] == ["feature", "target"]
    assert loaded.receipt["target_distribution"] == {"2": 1, "3": 1}
    assert loaded.receipt["acquisition_method"] == "existing_local_file"
    assert 999 not in loaded.frame["feature"].tolist()


def test_dataset_not_declared_by_config_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "configured.csv"
    source.write_text("feature,target\n11,2\n12,3\n", encoding="utf-8")
    config, manifest = _write_contract(tmp_path, source)

    with pytest.raises(CanonicalDataError, match="not declared"):
        load_canonical_dataset(config, "not_declared", manifest)


def test_config_and_acquisition_paths_must_identify_same_file(tmp_path: Path) -> None:
    configured = tmp_path / "configured.csv"
    configured.write_text("feature,target\n11,2\n12,3\n", encoding="utf-8")
    other = tmp_path / "other.csv"
    other.write_text(configured.read_text(encoding="utf-8"), encoding="utf-8")
    config, manifest = _write_contract(tmp_path, configured)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["data_acquisition"]["physical_datasets"]["physical"]["local_path"] = str(other)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CanonicalDataError, match="path mismatch"):
        load_canonical_dataset(config, "declared", manifest)


def test_receipt_hash_is_for_bytes_actually_loaded(tmp_path: Path) -> None:
    source = tmp_path / "configured.csv"
    source.write_text("feature,target\n11,2\n12,3\n", encoding="utf-8")
    config, manifest = _write_contract(tmp_path, source)
    loaded = load_canonical_dataset(config, "declared", manifest)

    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert loaded.receipt["actual_sha256"] == expected
    pd.testing.assert_frame_equal(
        loaded.frame.reset_index(drop=True),
        pd.DataFrame({"feature": [11, 12], "target": [2, 3]}),
    )
