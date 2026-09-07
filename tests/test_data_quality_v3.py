from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.data_quality_v3 import (
    EXPECTED_LOCAL_FILES,
    _identifier_rows,
    evaluate_data_quality_v3,
    preflight_data_quality_v3,
)
from src.utils.config_loader import load_config


CONTRACT = Path("configs/data_quality_v3.json")


def test_preflight_is_exactly_core_offline_and_fit_free() -> None:
    receipt = preflight_data_quality_v3()
    assert receipt["status"] == "passed"
    assert receipt["dataset_keys"] == ["inx_primary", "hrdataset_v14"]
    assert receipt["planned_dataset_count"] == 2
    assert receipt["planned_raw_column_profiles"] == 64
    assert receipt["model_fit_calls"] == 0
    assert receipt["network_calls"] == receipt["paid_api_calls"] == 0


def test_real_audit_reconstructs_profiles_schemas_and_declared_findings() -> None:
    result = evaluate_data_quality_v3(load_config(CONTRACT))
    assert len(result.dataset_summary) == 2
    assert len(result.column_profiles) == 64
    assert len(result.raw_schema) == 64
    assert len(result.cleaned_schema) == 67
    assert len(result.target_distribution) == 10
    assert len(result.manuscript_ready_data_quality) == 2
    summary = result.dataset_summary.set_index("dataset_key")
    assert summary.loc["inx_primary", "effective_missing_cell_count"] == 0
    assert summary.loc["hrdataset_v14", "effective_missing_cell_count"] == 215
    assert summary.loc["inx_primary", "exact_duplicate_extra_rows"] == 0
    assert summary.loc["hrdataset_v14", "exact_duplicate_extra_rows"] == 0
    assert summary["identifier_candidates_in_primary_features"].sum() == 0
    findings = result.rule_anomaly_audit.loc[
        result.rule_anomaly_audit["anomaly_count"] > 0,
        ["dataset_key", "rule_id", "anomaly_count"],
    ]
    assert findings.to_dict("records") == [
        {"dataset_key": "hrdataset_v14", "rule_id": "review_on_or_after_hire", "anomaly_count": 2},
        {"dataset_key": "hrdataset_v14", "rule_id": "performance_text_matches_code", "anomaly_count": 2},
        {"dataset_key": "hrdataset_v14", "rule_id": "department_id_functionally_maps_to_text", "anomaly_count": 2},
    ]


def test_identifier_detection_does_not_treat_high_cardinality_measure_as_identity() -> None:
    frame = pd.DataFrame(
        {
            "EmpID": [101, 102, 103, 104],
            "Salary": [51_001, 61_003, 72_007, 83_009],
            "PerformanceScore": ["A", "B", "A", "B"],
        }
    )
    rows = _identifier_rows(
        "hrdataset_v14",
        frame,
        {"declared_identifier_columns": ["EmpID"]},
        primary_features=("Salary",),
        tokens=("id", "identifier", "name", "number"),
        unique_threshold=0.98,
    )
    assert {row["column_name"] for row in rows} == {"EmpID", "ExternalSampleId"}
    assert all(row["leakage_audit_status"] == "passed_excluded" for row in rows)


def test_local_inventory_excludes_raw_data_and_row_level_values() -> None:
    assert EXPECTED_LOCAL_FILES == {
        "categorical_cardinality.csv",
        "cleaned_schema.csv",
        "column_profiles.csv",
        "dataset_summary.csv",
        "duplicate_audit.csv",
        "identifier_audit.csv",
        "manuscript_ready_data_quality.csv",
        "raw_schema.csv",
        "rule_anomaly_audit.csv",
        "stage_metadata.json",
        "target_distribution.csv",
    }
    assert not any("raw.csv" in name or "row_level" in name for name in EXPECTED_LOCAL_FILES)
