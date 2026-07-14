from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data.canonical_loader import CanonicalDataset
from src.data.external_adapters import load_external_config, load_external_dataset
from src.experiments import manuscript_external_evidence as external
from src.governance.manuscript_contract import canonical_config_hash
from src.utils.config_loader import load_config


CONFIG_PATH = Path("configs/manuscript_final.yaml")


def _write_minimal_mapping(path: Path, *, dataset_name: str) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "target": {
                    "raw_column": "Score",
                    "canonical_column": "PerformanceRating",
                    "task_type": "ordinal_multiclass_performance",
                    "mapping": {"low": 2, "middle": 3, "high": 4},
                },
                "rename_columns": {},
                "id_columns": [],
                "leakage_risk_columns": [],
                "sensitive_audit_only_columns": [],
                "proxy_risk_columns": [],
                "feature_policy_variants": {"primary": {"exclude_columns": []}},
            }
        ),
        encoding="utf-8",
    )


def test_adapter_accepts_verified_frame_and_explicit_mapping_without_path_discovery(tmp_path: Path) -> None:
    mapping = tmp_path / "explicit_mapping.json"
    _write_minimal_mapping(mapping, dataset_name="fixture_external")
    frame = pd.DataFrame({"Score": ["low", "middle", "high"], "SafeFeature": [1, 2, 3]})

    dataset = load_external_dataset(
        "fixture_external",
        raw_frame=frame,
        schema_mapping_path=mapping,
    )

    assert dataset.labels == [2, 3, 4]
    assert dataset.config.schema_mapping_path == mapping.resolve()
    assert dataset.raw is not frame
    assert "ExternalSampleId" not in frame.columns


def test_explicit_mapping_must_declare_the_requested_dataset_identity(tmp_path: Path) -> None:
    mapping = tmp_path / "wrong_mapping.json"
    _write_minimal_mapping(mapping, dataset_name="different_dataset")

    with pytest.raises(ValueError, match="identity mismatch"):
        load_external_config("requested_dataset", schema_mapping_path=mapping)


@pytest.mark.parametrize(
    ("scope", "expected_loader_keys", "expected_task_keys"),
    [
        ("core", ["inx_primary", "hrdataset_v14"], ["hrdataset_v14"]),
        (
            "supplementary",
            ["ibm_hr_analytics", "ibm_hr_analytics_attrition", "employee_turnover"],
            ["ibm_performance", "ibm_attrition", "employee_turnover"],
        ),
    ],
)
def test_canonical_stage_binds_only_the_selected_scope_through_verified_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    expected_loader_keys: list[str],
    expected_task_keys: list[str],
) -> None:
    config = copy.deepcopy(load_config(CONFIG_PATH))
    settings = config["manuscript_final"]
    for spec in external.RUN_SPECS:
        mapping = tmp_path / f"{spec.key}_mapping.json"
        mapping.write_text("{}\n", encoding="utf-8")
        settings["datasets"][spec.config_dataset_key]["schema_mapping_path"] = str(mapping.resolve())
    config_path = tmp_path / "canonical.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_hash = canonical_config_hash(config)

    loader_calls: list[tuple[str, bool]] = []
    adapter_calls: list[tuple[str, str, Path]] = []
    transport_calls: list[str] = []

    def fake_loader(config_path, dataset_key, *, allow_download, mismatch_report_path):
        loader_calls.append((dataset_key, allow_download))
        frame = pd.DataFrame({f"verified_{dataset_key}": [1]})
        return CanonicalDataset(
            frame=frame,
            receipt={
                "dataset_key": dataset_key,
                "actual_path": f"data/{dataset_key}.csv",
                "actual_sha256": "a" * 64,
                "row_count": 1,
                "column_count": 1,
                "schema_status": "valid",
            },
        )

    def fake_adapter(dataset_name, target_kind="primary", *, raw_frame, schema_mapping_path):
        adapter_calls.append((dataset_name, target_kind, Path(schema_mapping_path)))
        assert any(column.startswith("verified_") for column in raw_frame.columns)
        return SimpleNamespace(task_type=next(spec.task_type for spec in external.RUN_SPECS if spec.dataset_name == dataset_name and spec.target_kind == target_kind))

    def fake_task(spec, *, output_dir, settings, run_id, config_hash):
        binding = settings[external._CANONICAL_EXTERNAL_INPUTS_KEY][spec.key]
        assert isinstance(binding, external.CanonicalExternalInput)
        assert binding.receipt["dataset_key"] == spec.config_dataset_key
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "policy_summary.csv"
        pd.DataFrame(
            [{
                "run_id": run_id,
                "config_hash": config_hash,
                "dataset_key": spec.key,
                "task_type": spec.task_type,
                "role": spec.role,
                "policy": spec.policies[0],
            }]
        ).to_csv(path, index=False)
        return {"policy_summary": path}

    def fake_transport(config, *, run_id, config_hash):
        transport_calls.append(run_id)
        runtime = external._settings(config)[external._CANONICAL_EXTERNAL_INPUTS_KEY]
        assert isinstance(runtime["inx_primary"], CanonicalDataset)
        assert isinstance(runtime["hrdataset_v14"], external.CanonicalExternalInput)
        return (
            pd.DataFrame(
                [{
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "feature": "safe",
                    "in_inx_canonical_primary": True,
                    "in_hrdataset_department_free": True,
                    "common_safe_feature": True,
                }]
            ),
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "status": "infeasible_or_too_limited",
                "locked_inx_model_transported": False,
                "n_common_safe_features": 1,
                "common_safe_features": ["safe"],
                "minimum_feature_gate": 5,
            },
        )

    monkeypatch.setattr(external, "load_canonical_dataset", fake_loader)
    monkeypatch.setattr(external, "load_external_dataset", fake_adapter)
    monkeypatch.setattr(external, "_run_dataset_task", fake_task)
    monkeypatch.setattr(external, "compute_transport_assessment", fake_transport)

    external.run(
        config_path,
        scope=scope,
        output_dir=tmp_path / "external",
        run_id="bound-input-test",
        config_hash=config_hash,
    )

    assert loader_calls == [(key, True) for key in expected_loader_keys]
    assert [
        spec.key
        for dataset_name, target_kind, _ in adapter_calls
        for spec in external.RUN_SPECS
        if spec.dataset_name == dataset_name and spec.target_kind == target_kind
    ] == expected_task_keys
    assert all(path.is_file() for _, _, path in adapter_calls)
    assert transport_calls == (["bound-input-test"] if scope == "core" else [])
