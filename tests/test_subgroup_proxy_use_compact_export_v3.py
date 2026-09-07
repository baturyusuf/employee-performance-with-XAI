from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.governance import subgroup_proxy_use_compact_export_v3 as compact
from src.governance.subgroup_proxy_use_compact_export_v3 import (
    DEFAULT_OUTPUT,
    EXPECTED_EXPORT_FILES,
    V3SubgroupProxyUseCompactExportError,
    export_subgroup_proxy_use_compact_v3,
    validate_subgroup_proxy_use_compact_v3,
)
from src.governance.subgroup_proxy_use_run_validator_v3 import (
    DEFAULT_SUBGROUP_PROXY_USE_RUN,
    validate_subgroup_proxy_use_run_v3,
)


@pytest.fixture(scope="module")
def run_receipt():
    return validate_subgroup_proxy_use_run_v3()


def test_two_compact_exports_are_byte_identical(
    run_receipt, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compact, "validate_subgroup_proxy_use_run_v3", lambda _path: run_receipt)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_receipt = export_subgroup_proxy_use_compact_v3(
        DEFAULT_SUBGROUP_PROXY_USE_RUN, first
    )
    second_receipt = export_subgroup_proxy_use_compact_v3(
        DEFAULT_SUBGROUP_PROXY_USE_RUN, second
    )
    assert first_receipt["status"] == second_receipt["status"] == "passed"
    assert first_receipt["manifest_sha256"] == second_receipt["manifest_sha256"]
    assert {path.name for path in first.iterdir()} == EXPECTED_EXPORT_FILES
    for filename in EXPECTED_EXPORT_FILES:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_compact_export_excludes_employee_level_sources(
    run_receipt, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compact, "validate_subgroup_proxy_use_run_v3", lambda _path: run_receipt)
    output = tmp_path / "compact"
    receipt = export_subgroup_proxy_use_compact_v3(DEFAULT_SUBGROUP_PROXY_USE_RUN, output)
    assert receipt["employee_level_rows_included"] is False
    assert "proxy_prediction_change_sample.csv" not in {path.name for path in output.iterdir()}
    assert "jobrole_permutation_sample.csv" not in {path.name for path in output.iterdir()}
    assert "stage_metadata.json" not in {path.name for path in output.iterdir()}


def test_compact_validator_rejects_tampering(
    run_receipt, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compact, "validate_subgroup_proxy_use_run_v3", lambda _path: run_receipt)
    output = tmp_path / "compact"
    export_subgroup_proxy_use_compact_v3(DEFAULT_SUBGROUP_PROXY_USE_RUN, output)
    with (output / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    with pytest.raises(V3SubgroupProxyUseCompactExportError, match="size drifted|hash drifted"):
        validate_subgroup_proxy_use_compact_v3(
            output, source_run=DEFAULT_SUBGROUP_PROXY_USE_RUN
        )


def test_tracked_compact_package_when_present() -> None:
    if not DEFAULT_OUTPUT.is_dir():
        pytest.skip("Tracked Phase 2C compact package has not been published yet.")
    receipt = validate_subgroup_proxy_use_compact_v3()
    assert receipt["status"] == "passed"
    assert receipt["row_counts"]["subgroup_metric_grid.csv"] == 2025
    assert receipt["employee_level_rows_included"] is False
