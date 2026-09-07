from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.governance.subgroup_proxy_use_run_validator_v3 import (
    DEFAULT_SUBGROUP_PROXY_USE_RUN,
    V3SubgroupProxyUseRunValidationError,
    validate_subgroup_proxy_use_run_v3,
)


@pytest.fixture(scope="module")
def validation_receipt():
    return validate_subgroup_proxy_use_run_v3()


def test_independent_validator_recomputes_complete_phase2c_run(validation_receipt) -> None:
    assert validation_receipt["status"] == "passed"
    assert validation_receipt["generation_commit"] == (
        "e314bb5d1935da6a23e26b2085330dacb468fe32"
    )
    assert validation_receipt["file_count"] == 12
    assert validation_receipt["subgroup_metric_rows"] == 2025
    assert validation_receipt["subgroup_gap_rows"] == 486
    assert validation_receipt["primary_gap_interval_rows"] == 162
    assert validation_receipt["bootstrap_resample_hash"] == (
        "d16aadb56f2dde124df62387447bb01ddf51236f7f9e9e01b6218e8f6265f646"
    )
    assert validation_receipt["proxy_overall_prediction_change_rate"] == pytest.approx(
        0.1075
    )
    assert validation_receipt["new_model_fit_calls"] == 0
    assert validation_receipt["network_calls"] == validation_receipt["paid_api_calls"] == 0


def test_validator_does_not_import_phase2c_runner_calculations() -> None:
    source = Path(
        "src/governance/subgroup_proxy_use_run_validator_v3.py"
    ).read_text(encoding="utf-8")
    assert "from src.experiments.subgroup_proxy_use_v3" not in source
    assert "import subgroup_proxy_use_v3" not in source


def test_validator_rejects_tampered_scientific_output(tmp_path: Path) -> None:
    copied = tmp_path / "phase2c_v3_20260907T070905Z_e314bb5" / "subgroup_proxy_use"
    shutil.copytree(DEFAULT_SUBGROUP_PROXY_USE_RUN, copied)
    path = copied / "subgroup_gap_sensitivity.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(V3SubgroupProxyUseRunValidationError, match="output hash drifted"):
        validate_subgroup_proxy_use_run_v3(copied)


def test_validator_rejects_extra_file_before_recomputation(tmp_path: Path) -> None:
    copied = tmp_path / "phase2c_v3_20260907T070905Z_e314bb5" / "subgroup_proxy_use"
    shutil.copytree(DEFAULT_SUBGROUP_PROXY_USE_RUN, copied)
    (copied / "unexpected.csv").write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(V3SubgroupProxyUseRunValidationError, match="closed-world"):
        validate_subgroup_proxy_use_run_v3(copied)
