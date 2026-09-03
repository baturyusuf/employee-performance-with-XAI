from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.governance.repeated_nested_cv_compact_export_v3 import (
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE_RUN,
    EXPECTED_EXPORT_FILES,
    V3RepeatedNestedCVCompactExportError,
    export_repeated_nested_cv_compact_v3,
    validate_repeated_nested_cv_compact_v3,
)


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="compact package not generated yet")
def test_tracked_compact_repeated_cv_package_is_closed_world_and_safe() -> None:
    receipt = validate_repeated_nested_cv_compact_v3()
    assert receipt["status"] == "passed"
    assert receipt["file_count_including_manifest"] == len(EXPECTED_EXPORT_FILES) == 9
    assert receipt["repetition_metric_rows"] == 720
    assert receipt["variability_rows"] == 36
    assert receipt["rank_rows"] == 120
    assert receipt["rank_summary_rows"] == 24
    assert receipt["ordering_stability_rows"] == 4
    assert receipt["candidate_frequency_rows"] == 22


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="compact package not generated yet")
def test_compact_package_contains_no_employee_or_fold_level_artifacts() -> None:
    forbidden_names = ("oof", "fold_metrics", "fold_assignments", "model.joblib", "raw.csv")
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
                for token in ("sample_index", "empnumber", "y_true", "y_pred", "prob_class_", "sample_key")
            )
        ]


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="compact package not generated yet")
def test_compact_package_is_byte_preserved_by_git_attributes() -> None:
    completed = subprocess.run(
        ["git", "check-attr", "text", "--", str(DEFAULT_OUTPUT / "repetition_metrics.csv")],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.stdout.rstrip().endswith(": text: unset")


@pytest.mark.skipif(not DEFAULT_SOURCE_RUN.is_dir(), reason="local row-level v3 source is intentionally ignored")
def test_export_is_deterministic_from_validated_local_source(tmp_path: Path) -> None:
    output = tmp_path / "compact"
    receipt = export_repeated_nested_cv_compact_v3(DEFAULT_SOURCE_RUN, output)
    assert receipt["status"] == "passed"
    if DEFAULT_OUTPUT.is_dir():
        assert {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()} == {
            path.name: path.read_bytes() for path in DEFAULT_OUTPUT.iterdir() if path.is_file()
        }


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="compact package not generated yet")
def test_compact_validator_rejects_manifested_file_tampering(tmp_path: Path) -> None:
    copied = tmp_path / "compact"
    shutil.copytree(DEFAULT_OUTPUT, copied)
    with (copied / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    with pytest.raises(V3RepeatedNestedCVCompactExportError, match="hash drifted"):
        validate_repeated_nested_cv_compact_v3(copied)
