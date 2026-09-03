from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.governance.ordinal_benchmark_compact_export_v3 import (
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE_RUN,
    EXPECTED_EXPORT_FILES,
    V3OrdinalCompactExportError,
    export_ordinal_benchmark_compact_v3,
    validate_ordinal_benchmark_compact_v3,
)


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="compact package not generated yet")
def test_tracked_compact_ordinal_package_is_closed_world_and_safe() -> None:
    receipt = validate_ordinal_benchmark_compact_v3()
    assert receipt["status"] == "passed"
    assert receipt["file_count_including_manifest"] == len(EXPECTED_EXPORT_FILES) == 9
    assert receipt["aggregate_metric_rows"] == 144
    assert receipt["per_class_rows"] == 27
    assert receipt["confusion_rows"] == 81
    assert receipt["fold_metric_rows"] == 50
    assert receipt["selection_rows"] == 50
    assert receipt["candidate_summary_rows"] == 14


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="compact package not generated yet")
def test_compact_package_contains_no_employee_level_columns_or_artifacts() -> None:
    forbidden_names = ("oof", "model.joblib", "raw.csv")
    assert not [
        path
        for path in DEFAULT_OUTPUT.rglob("*")
        if path.is_file() and any(token in path.name.casefold() for token in forbidden_names)
    ]
    for path in DEFAULT_OUTPUT.glob("*.csv"):
        columns = [column.casefold() for column in pd.read_csv(path, nrows=0).columns]
        assert not [
            column
            for column in columns
            if any(
                token in column
                for token in ("sample_index", "empnumber", "y_true", "y_pred", "prob_class_")
            )
        ]


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="compact package not generated yet")
def test_compact_package_is_byte_preserved_by_git_attributes() -> None:
    completed = subprocess.run(
        ["git", "check-attr", "text", "--", str(DEFAULT_OUTPUT / "aggregate_metrics.csv")],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.stdout.rstrip().endswith(": text: unset")


@pytest.mark.skipif(
    not DEFAULT_SOURCE_RUN.is_dir(), reason="local row-level v3 source is intentionally ignored"
)
def test_export_is_deterministic_from_validated_local_source(tmp_path: Path) -> None:
    output = tmp_path / "compact"
    receipt = export_ordinal_benchmark_compact_v3(DEFAULT_SOURCE_RUN, output)
    assert receipt["status"] == "passed"
    if DEFAULT_OUTPUT.is_dir():
        assert {
            path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
        } == {
            path.name: path.read_bytes() for path in DEFAULT_OUTPUT.iterdir() if path.is_file()
        }


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="compact package not generated yet")
def test_compact_validator_rejects_manifested_file_tampering(tmp_path: Path) -> None:
    copied = tmp_path / "compact"
    shutil.copytree(DEFAULT_OUTPUT, copied)
    with (copied / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    with pytest.raises(V3OrdinalCompactExportError, match="hash drifted"):
        validate_ordinal_benchmark_compact_v3(copied)
