from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.data.canonical_loader import load_canonical_dataset
from src.experiments import build_manuscript_evidence as manuscript_build
from src.experiments.build_manuscript_evidence import StageContext
from src.governance.manuscript_contract import (
    RunManifestError,
    create_run_manifest,
    load_manuscript_config,
    validate_run_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _project_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    """Build a complete config contract around tiny, test-only real bytes."""

    dataset = tmp_path / "data" / "raw" / "sample.csv"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text(
        "feature,PerformanceRating\n10,2\n20,3\n30,4\n",
        encoding="utf-8",
    )

    acquisition_path = tmp_path / "configs" / "data_acquisition.yaml"
    _write_json(
        acquisition_path,
        {
            "data_acquisition": {
                "schema_version": 1,
                "physical_datasets": {
                    "sample_physical": {
                        "local_path": "data/raw/sample.csv",
                        "expected_sha256": _sha256(dataset),
                        "expected_rows": 3,
                        "expected_column_count": 2,
                        "expected_columns": ["feature", "PerformanceRating"],
                        "format": "csv",
                        "delimiter": ",",
                        "encoding": "utf-8",
                        "target_profiles": {
                            key: {
                                "raw_target": "PerformanceRating",
                                "expected_distribution": {"2": 1, "3": 1, "4": 1},
                            }
                            for key in (
                                "inx_primary",
                                "hrdataset_v14",
                                "ibm_hr_analytics",
                                "ibm_hr_analytics_attrition",
                                "employee_turnover",
                            )
                        },
                        "acquisition_method": "user_provided_local_file",
                        "approved_download_url": None,
                        "automatic_download_allowed": False,
                    }
                },
                "logical_bindings": {
                    key: {
                        "physical_dataset": "sample_physical",
                        "target_profile": key,
                    }
                    for key in (
                        "inx_primary",
                        "hrdataset_v14",
                        "ibm_hr_analytics",
                        "ibm_hr_analytics_attrition",
                        "employee_turnover",
                    )
                },
            }
        },
    )

    side_paths = {
        "data_acquisition_contract": "configs/data_acquisition.yaml",
        "dataset_provenance": "data/contracts/dataset_provenance.yaml",
        "feature_taxonomy": "data/contracts/feature_taxonomy.yaml",
        "model_search_space": "data/contracts/model_search_space.yaml",
        "external_hrdataset_v14_schema_mapping": "data/contracts/hr_schema_mapping.json",
        "external_ibm_hr_analytics_schema_mapping": "data/contracts/ibm_schema_mapping.json",
        "external_employee_turnover_schema_mapping": "data/contracts/turnover_schema_mapping.json",
    }
    for logical_name, relative_path in side_paths.items():
        if logical_name == "data_acquisition_contract":
            continue
        _write_json(
            tmp_path / relative_path,
            {"logical_name": logical_name, "schema_version": 1},
        )

    config = copy.deepcopy(load_manuscript_config())
    settings = config["manuscript_final"]
    for definition in settings["datasets"].values():
        definition["path"] = "data/raw/sample.csv"
    settings["datasets"]["hrdataset_v14"]["schema_mapping_path"] = side_paths[
        "external_hrdataset_v14_schema_mapping"
    ]
    settings["datasets"]["ibm_hr_analytics"]["schema_mapping_path"] = side_paths[
        "external_ibm_hr_analytics_schema_mapping"
    ]
    settings["datasets"]["ibm_hr_analytics_attrition"]["schema_mapping_path"] = side_paths[
        "external_ibm_hr_analytics_schema_mapping"
    ]
    settings["datasets"]["employee_turnover"]["schema_mapping_path"] = side_paths[
        "external_employee_turnover_schema_mapping"
    ]
    settings["provenance"]["data_acquisition_manifest"] = "configs/data_acquisition.yaml"
    settings["provenance"]["scientific_side_inputs"] = side_paths
    settings["evidence_scopes"] = {
        "core": {
            "dataset_keys": ["inx_primary", "hrdataset_v14"],
            "side_input_keys": [
                "data_acquisition_contract",
                "dataset_provenance",
                "feature_taxonomy",
                "model_search_space",
                "external_hrdataset_v14_schema_mapping",
            ],
            "stages": ["core_fixture"],
        },
        "supplementary": {
            "dataset_keys": [
                "inx_primary",
                "ibm_hr_analytics",
                "ibm_hr_analytics_attrition",
                "employee_turnover",
            ],
            "side_input_keys": [
                "data_acquisition_contract",
                "dataset_provenance",
                "feature_taxonomy",
                "model_search_space",
                "external_ibm_hr_analytics_schema_mapping",
                "external_employee_turnover_schema_mapping",
            ],
            "stages": ["supplementary_fixture"],
        },
    }
    config_path = _write_json(tmp_path / "configs" / "manuscript_final.yaml", config)
    return config_path, acquisition_path, dataset, side_paths


def test_manifest_actual_input_receipt_matches_loader_consumption(tmp_path: Path) -> None:
    config_path, acquisition_path, _, _ = _project_fixture(tmp_path)

    loaded = load_canonical_dataset(
        config_path,
        "inx_primary",
        acquisition_path,
        project_root=tmp_path,
    )
    manifest = create_run_manifest(
        config_path,
        evidence_scope="core",
        project_root=tmp_path,
        run_id="unit_1b_actual_receipt",
    )

    receipt = manifest["actual_input_receipts"]["inx_primary"]
    for field in (
        "actual_path",
        "actual_sha256",
        "row_count",
        "column_count",
        "schema_status",
        "schema_columns",
        "target_column",
        "target_distribution",
        "acquisition_manifest_sha256",
    ):
        assert receipt[field] == loaded.receipt[field]
    assert manifest["dataset_hashes"]["inx_primary"]["path"] == receipt["actual_path"]
    assert manifest["dataset_hashes"]["inx_primary"]["sha256"] == receipt["actual_sha256"]


def test_every_declared_side_input_is_hashed_with_portable_identity(tmp_path: Path) -> None:
    config_path, _, _, declared = _project_fixture(tmp_path)

    manifest = create_run_manifest(
        config_path,
        evidence_scope="core",
        project_root=tmp_path,
        run_id="unit_1b_sides",
    )
    observed = manifest["side_input_hashes"]

    assert set(observed) == set(manifest["scope_contract"]["side_input_keys"])
    for logical_name in observed:
        relative_path = declared[logical_name]
        source = tmp_path / relative_path
        assert observed[logical_name] == {
            "path": Path(relative_path).as_posix(),
            "sha256": _sha256(source),
            "size_bytes": source.stat().st_size,
        }
    assert len(manifest["scientific_input_hash"]) == 64


def test_side_input_change_alone_changes_aggregate_scientific_identity(tmp_path: Path) -> None:
    config_path, _, _, declared = _project_fixture(tmp_path)
    first = create_run_manifest(
        config_path,
        evidence_scope="core",
        project_root=tmp_path,
        run_id="unit_1b_before",
    )

    changed_side = tmp_path / declared["feature_taxonomy"]
    changed_side.write_text('{"schema_version": 2, "changed": true}\n', encoding="utf-8")
    second = create_run_manifest(
        config_path,
        evidence_scope="core",
        project_root=tmp_path,
        run_id="unit_1b_after",
    )

    assert second["config_hash"] == first["config_hash"]
    assert second["source_tree_hash"] == first["source_tree_hash"]
    assert second["dataset_hashes"] == first["dataset_hashes"]
    assert second["actual_input_receipts"] == first["actual_input_receipts"]
    assert (
        second["side_input_hashes"]["feature_taxonomy"]["sha256"]
        != first["side_input_hashes"]["feature_taxonomy"]["sha256"]
    )
    assert second["scientific_input_hash"] != first["scientific_input_hash"]


@pytest.mark.parametrize("failure_mode", ["missing", "hash_mismatch"])
def test_manifest_validation_fails_when_declared_side_input_drifts(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    config_path, _, _, declared = _project_fixture(tmp_path)
    manifest = create_run_manifest(
        config_path,
        evidence_scope="core",
        project_root=tmp_path,
        run_id="unit_1b_drift",
    )
    side_input = tmp_path / declared["dataset_provenance"]
    if failure_mode == "missing":
        side_input.unlink()
    else:
        side_input.write_text('{"changed_after_manifest": true}\n', encoding="utf-8")

    with pytest.raises(RunManifestError, match="side input"):
        validate_run_manifest(manifest, project_root=tmp_path)


def test_manifest_validation_rejects_tampered_actual_receipt(tmp_path: Path) -> None:
    config_path, _, _, _ = _project_fixture(tmp_path)
    manifest = create_run_manifest(
        config_path,
        evidence_scope="core",
        project_root=tmp_path,
        run_id="unit_1b_receipt_tamper",
    )
    manifest["actual_input_receipts"]["inx_primary"]["actual_sha256"] = "0" * 64

    with pytest.raises(RunManifestError, match="actual input"):
        validate_run_manifest(manifest, project_root=tmp_path)


def test_run_input_snapshots_preserve_every_side_input_and_reject_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _, _, declared = _project_fixture(tmp_path)
    for required in ("requirements.txt", "requirements-dev.txt", "environment.yml"):
        (tmp_path / required).write_text(f"# fixture {required}\n", encoding="utf-8")
    manifest = create_run_manifest(
        config_path,
        evidence_scope="core",
        project_root=tmp_path,
        run_id="unit_1b_snapshots",
    )
    config = load_manuscript_config(config_path)
    context = StageContext(
        config_path=config_path,
        config=config,
        settings=config["manuscript_final"],
        run_dir=tmp_path / "reports" / "manuscript_final" / "unit_1b_snapshots",
        run_id="unit_1b_snapshots",
        config_hash=manifest["config_hash"],
        manifest=manifest,
    )
    monkeypatch.setattr(manuscript_build, "PROJECT_ROOT", tmp_path)

    manuscript_build._write_input_snapshots(context)
    contract_path = context.run_dir / "run_inputs" / "input_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    side_rows = {
        row["logical_name"]: row
        for row in contract["snapshots"]
        if row["input_kind"] == "scientific_side_input"
    }
    assert set(side_rows) == set(manifest["scope_contract"]["side_input_keys"])
    assert contract["actual_input_receipts"] == manifest["actual_input_receipts"]
    assert contract["side_input_hashes"] == manifest["side_input_hashes"]
    assert contract["scientific_input_hash"] == manifest["scientific_input_hash"]
    for logical_name, row in side_rows.items():
        source = tmp_path / declared[logical_name]
        snapshot = tmp_path / row["snapshot_path"]
        assert snapshot.read_bytes() == source.read_bytes()
        assert row["sha256"] == _sha256(source)

    drifted = tmp_path / declared["feature_taxonomy"]
    drifted.write_text('{"changed_after_snapshot": true}\n', encoding="utf-8")
    with pytest.raises(manuscript_build.ManuscriptBuildError, match="source hash mismatch"):
        manuscript_build._write_input_snapshots(context)
