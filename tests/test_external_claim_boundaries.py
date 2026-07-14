from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.external_adapters import load_external_dataset
from src.governance.external_claims import HRDATASET_REPLICATION_CLAIM, external_allowed_claim
from src.governance.external_validation_reports import (
    DATASET_RUNS,
    build_tables,
    governance_markdown,
    manuscript_markdown,
    summary_markdown,
)
from src.models.task_schema import (
    BINARY_ATTRITION_TRANSFER,
    BINARY_TURNOVER_TRANSFER,
    RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS,
)
from src.utils.config import SETTINGS


HR_RAW = Path(__file__).resolve().parents[1] / "data/external/hrdataset_v14/raw.csv"


def test_dataset_claims_use_registered_conservative_labels() -> None:
    assert external_allowed_claim("hrdataset_v14") == HRDATASET_REPLICATION_CLAIM
    assert external_allowed_claim("ibm_hr_analytics") == "restricted-target performance robustness"
    assert external_allowed_claim("ibm_hr_analytics", "attrition") == "related HR attrition task transfer"
    assert external_allowed_claim("employee_turnover") == "related HR turnover task transfer"


def test_external_config_claims_and_task_types_match_the_registered_contract() -> None:
    config_path = SETTINGS.project_root / "configs" / "external_validation.yaml"
    rows = json.loads(config_path.read_text(encoding="utf-8"))["external_validation"]["datasets"]
    by_name = {row["dataset_name"]: row for row in rows}

    assert by_name["hrdataset_v14"]["role_in_manuscript"] == HRDATASET_REPLICATION_CLAIM
    assert by_name["hrdataset_v14"]["locked_inx_model_transported"] is False
    assert by_name["ibm_hr_analytics"]["task_type"] == RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS
    assert by_name["ibm_hr_analytics_attrition"]["task_type"] == BINARY_ATTRITION_TRANSFER
    assert by_name["employee_turnover"]["task_type"] == BINARY_TURNOVER_TRANSFER


def test_report_tables_separate_non_comparable_task_families_and_sanitize_legacy_metrics() -> None:
    tables = build_tables()

    assert set(tables["performance_ordinal"]["dataset"]) == {"INX primary model", "HRDataset_v14"}
    assert set(tables["performance_restricted"]["dataset"]) == {"IBM HR Analytics performance"}
    assert set(tables["performance_binary_transfer"]["dataset"]) == {
        "IBM HR Analytics attrition",
        "Employee Turnover",
    }

    restricted = tables["performance_restricted"]
    binary = tables["performance_binary_transfer"]
    for frame in (restricted, binary):
        assert frame["severe_error_rate"].isna().all()
        assert frame["ordinal_mae"].isna().all()
        assert frame["qwk"].isna().all()
    assert binary["brier"].isna().all()


def test_report_renderers_do_not_emit_prohibited_external_validation_labels() -> None:
    tables = build_tables()
    reports = "\n".join(
        [
            summary_markdown(tables),
            manuscript_markdown(tables),
            governance_markdown(tables),
        ]
    ).lower()

    assert HRDATASET_REPLICATION_CLAIM in reports
    assert "direct external performance validation" not in reports
    assert "direct employee-performance external validation" not in reports
    assert "performance metrics across datasets" not in reports
    assert "related binary task transfer" in reports


@pytest.mark.skipif(not HR_RAW.is_file(), reason="requires the ignored local HRDataset_v14 dataset")
def test_hrdataset_mapping_support_and_three_safe_feature_transport_gate_are_preserved() -> None:
    dataset = load_external_dataset("hrdataset_v14")
    assert dataset.canonical[dataset.target_column].value_counts().sort_index().to_dict() == {2: 31, 3: 243, 4: 37}

    overlap_path = (
        SETTINGS.reports_dir
        / "external_validation"
        / "hrdataset_v14"
        / "cross_dataset_inx_to_hrdataset"
        / "feature_overlap.csv"
    )
    overlap = pd.read_csv(overlap_path)
    common = overlap[overlap["common"].astype(str).str.lower() == "true"]["feature"].tolist()
    assert common == ["EmpJobRole", "EmpJobSatisfaction", "ExperienceYearsAtThisCompany"]


def test_dataset_run_registry_has_no_cross_task_role_drift() -> None:
    by_key = {row["key"]: row for row in DATASET_RUNS}
    assert by_key["hrdataset_v14"]["role"] == HRDATASET_REPLICATION_CLAIM
    assert by_key["ibm_hr_analytics"]["task_type"] == RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS
    assert by_key["employee_turnover"]["task_type"] == BINARY_TURNOVER_TRANSFER
