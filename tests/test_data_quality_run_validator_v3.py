from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.governance.data_quality_run_validator_v3 import (
    DEFAULT_DATA_QUALITY_RUN,
    DataQualityRunValidationV3Error,
    validate_data_quality_run_v3,
)


def _copy_run(tmp_path: Path) -> Path:
    destination = tmp_path / DEFAULT_DATA_QUALITY_RUN.parent.name / "data_quality"
    shutil.copytree(DEFAULT_DATA_QUALITY_RUN, destination)
    return destination


@pytest.mark.skipif(not DEFAULT_DATA_QUALITY_RUN.is_dir(), reason="local aggregate Phase 3B run is intentionally ignored")
def test_real_run_is_independently_reconstructed() -> None:
    receipt = validate_data_quality_run_v3()
    assert receipt["status"] == "passed"
    assert receipt["file_count"] == 11
    assert receipt["table_count"] == 10
    assert receipt["raw_column_profile_count"] == 64
    assert receipt["cleaned_schema_row_count"] == 67
    assert receipt["identifier_candidates_in_primary_features"] == 0
    assert receipt["findings"] == [
        {"dataset_key": "hrdataset_v14", "rule_id": "review_on_or_after_hire", "anomaly_count": 2},
        {"dataset_key": "hrdataset_v14", "rule_id": "performance_text_matches_code", "anomaly_count": 2},
        {"dataset_key": "hrdataset_v14", "rule_id": "department_id_functionally_maps_to_text", "anomaly_count": 2},
    ]
    assert receipt["model_fit_calls"] == 0
    assert receipt["network_calls"] == receipt["paid_api_calls"] == 0


def test_validator_does_not_import_phase3b_runner() -> None:
    source = Path("src/governance/data_quality_run_validator_v3.py").read_text(encoding="utf-8")
    assert "from src.experiments.data_quality_v3 import" not in source
    assert "import src.experiments.data_quality_v3" not in source


@pytest.mark.skipif(not DEFAULT_DATA_QUALITY_RUN.is_dir(), reason="local aggregate Phase 3B run is intentionally ignored")
def test_validator_rejects_unexpected_file(tmp_path: Path) -> None:
    copied = _copy_run(tmp_path)
    (copied / "unexpected.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(DataQualityRunValidationV3Error, match="closed-world inventory drifted"):
        validate_data_quality_run_v3(copied)


@pytest.mark.skipif(not DEFAULT_DATA_QUALITY_RUN.is_dir(), reason="local aggregate Phase 3B run is intentionally ignored")
def test_independent_recomputation_rejects_rehashed_table_tampering(tmp_path: Path) -> None:
    copied = _copy_run(tmp_path)
    table_path = copied / "dataset_summary.csv"
    table = pd.read_csv(table_path)
    table.loc[table["dataset_key"] == "inx_primary", "effective_missing_cell_count"] = 1
    table.to_csv(table_path, index=False, lineterminator="\n")
    metadata_path = copied / "stage_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["output_hashes"]["dataset_summary.csv"] = hashlib.sha256(table_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(DataQualityRunValidationV3Error, match="independent recomputation"):
        validate_data_quality_run_v3(copied)
