from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.governance.manuscript_contract import (
    MANIFEST_SCHEMA_VERSION,
    ManuscriptConfigError,
    RunManifestError,
    create_run_manifest,
    evidence_scope_contract,
    load_manuscript_config,
    validate_run_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _scoped_project(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    dataset = tmp_path / "data" / "raw" / "sample.csv"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text("feature,target\n1,2\n2,3\n3,4\n", encoding="utf-8")

    config = copy.deepcopy(load_manuscript_config())
    settings = config["manuscript_final"]
    for definition in settings["datasets"].values():
        definition["path"] = "data/raw/sample.csv"

    acquisition_path = tmp_path / "configs" / "data_acquisition.yaml"
    dataset_keys = list(settings["datasets"])
    _write_json(
        acquisition_path,
        {
            "data_acquisition": {
                "schema_version": 1,
                "physical_datasets": {
                    "fixture": {
                        "local_path": "data/raw/sample.csv",
                        "expected_sha256": _sha256(dataset),
                        "expected_rows": 3,
                        "expected_column_count": 2,
                        "expected_columns": ["feature", "target"],
                        "format": "csv",
                        "delimiter": ",",
                        "encoding": "utf-8",
                        "target_profiles": {
                            key: {
                                "raw_target": "target",
                                "expected_distribution": {"2": 1, "3": 1, "4": 1},
                            }
                            for key in dataset_keys
                        },
                        "automatic_download_allowed": False,
                    }
                },
                "logical_bindings": {
                    key: {"physical_dataset": "fixture", "target_profile": key}
                    for key in dataset_keys
                },
            }
        },
    )

    side_inputs = dict(settings["provenance"]["scientific_side_inputs"])
    for logical_name, relative_path in side_inputs.items():
        if logical_name == "data_acquisition_contract":
            continue
        _write_json(
            tmp_path / relative_path,
            {"logical_name": logical_name, "schema_version": 1},
        )
    settings["provenance"]["data_acquisition_manifest"] = "configs/data_acquisition.yaml"
    config_path = _write_json(tmp_path / "configs" / "manuscript_final.yaml", config)
    return config_path, side_inputs


def test_only_fixed_named_evidence_scopes_are_accepted() -> None:
    config = load_manuscript_config()
    assert set(evidence_scope_contract(config, "core")["dataset_keys"]) == {
        "inx_primary",
        "hrdataset_v14",
    }
    with pytest.raises(ManuscriptConfigError, match="Unknown evidence scope"):
        evidence_scope_contract(config, "arbitrary_subset")

    changed = copy.deepcopy(config)
    changed["manuscript_final"]["evidence_scopes"]["core"]["dataset_keys"] = [
        "inx_primary"
    ]
    with pytest.raises(ManuscriptConfigError, match="non-canonical dataset set"):
        evidence_scope_contract(changed, "core")


def test_core_scope_rejects_prohibited_stage_dependencies() -> None:
    config = copy.deepcopy(load_manuscript_config())
    config["manuscript_final"]["evidence_scopes"]["core"]["stages"].append(
        "counterfactual"
    )
    with pytest.raises(ManuscriptConfigError, match="prohibited scope dependencies"):
        evidence_scope_contract(config, "core")


def test_scoped_manifests_hash_only_the_exact_declared_inputs(tmp_path: Path) -> None:
    config_path, _ = _scoped_project(tmp_path)
    core = create_run_manifest(
        config_path,
        evidence_scope="core",
        project_root=tmp_path,
        run_id="scoped_core",
    )
    supplementary = create_run_manifest(
        config_path,
        evidence_scope="supplementary",
        project_root=tmp_path,
        run_id="scoped_supplementary",
    )

    assert MANIFEST_SCHEMA_VERSION == 3
    assert core["manifest_schema_version"] == 3
    assert core["evidence_scope"] == "core"
    assert set(core["dataset_hashes"]) == {"inx_primary", "hrdataset_v14"}
    assert set(core["actual_input_receipts"]) == {"inx_primary", "hrdataset_v14"}
    assert set(core["side_input_hashes"]) == set(core["scope_contract"]["side_input_keys"])
    assert "external_ibm_hr_analytics_schema_mapping" not in core["side_input_hashes"]

    assert supplementary["evidence_scope"] == "supplementary"
    assert set(supplementary["dataset_hashes"]) == {
        "inx_primary",
        "ibm_hr_analytics",
        "ibm_hr_analytics_attrition",
        "employee_turnover",
    }
    assert "hrdataset_v14" not in supplementary["dataset_hashes"]
    assert "external_hrdataset_v14_schema_mapping" not in supplementary["side_input_hashes"]
    assert core["scope_contract_hash"] != supplementary["scope_contract_hash"]
    assert core["scientific_input_hash"] != supplementary["scientific_input_hash"]


def test_dataset_path_assertion_must_match_exact_scope(tmp_path: Path) -> None:
    config_path, _ = _scoped_project(tmp_path)
    with pytest.raises(RunManifestError, match="exactly every dataset"):
        create_run_manifest(
            config_path,
            evidence_scope="core",
            project_root=tmp_path,
            dataset_paths={"inx_primary": "data/raw/sample.csv"},
        )


def test_validation_rejects_cross_scope_and_scope_contract_tampering(tmp_path: Path) -> None:
    config_path, _ = _scoped_project(tmp_path)
    manifest = create_run_manifest(
        config_path,
        evidence_scope="core",
        project_root=tmp_path,
        run_id="scope_validation",
    )
    validate_run_manifest(
        manifest,
        project_root=tmp_path,
        expected_evidence_scope="core",
    )
    with pytest.raises(RunManifestError, match="does not equal expected"):
        validate_run_manifest(
            manifest,
            project_root=tmp_path,
            expected_evidence_scope="supplementary",
        )

    tampered = copy.deepcopy(manifest)
    tampered["scope_contract"]["dataset_keys"] = ["inx_primary"]
    with pytest.raises(RunManifestError, match="scope_contract"):
        validate_run_manifest(tampered, project_root=tmp_path)


def test_unselected_side_input_drift_does_not_invalidate_core_scope(tmp_path: Path) -> None:
    config_path, side_inputs = _scoped_project(tmp_path)
    core = create_run_manifest(
        config_path,
        evidence_scope="core",
        project_root=tmp_path,
        run_id="core_side_scope",
    )
    supplementary = create_run_manifest(
        config_path,
        evidence_scope="supplementary",
        project_root=tmp_path,
        run_id="supplementary_side_scope",
    )
    ibm_mapping = tmp_path / side_inputs["external_ibm_hr_analytics_schema_mapping"]
    ibm_mapping.write_text('{"changed": true}\n', encoding="utf-8")

    validate_run_manifest(core, project_root=tmp_path, expected_evidence_scope="core")
    with pytest.raises(RunManifestError, match="side input"):
        validate_run_manifest(
            supplementary,
            project_root=tmp_path,
            expected_evidence_scope="supplementary",
        )


def test_scope_must_include_selected_dataset_schema_mapping() -> None:
    config = copy.deepcopy(load_manuscript_config())
    config["manuscript_final"]["evidence_scopes"]["core"]["side_input_keys"].remove(
        "external_hrdataset_v14_schema_mapping"
    )
    with pytest.raises(ManuscriptConfigError, match="schema-mapping side inputs"):
        evidence_scope_contract(config, "core")
