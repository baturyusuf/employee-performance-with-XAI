from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.governance import inx_workbook_equivalence as workbook_equivalence
from src.governance.inx_workbook_equivalence import (
    NORMALIZATION_CONTRACT,
    WorkbookEquivalenceError,
    WorkbookSnapshot,
    compare_matrices,
    normalize_cell,
    run,
    validate_receipt,
)
from src.governance.manuscript_contract import sha256_file
from src.utils.config_loader import PROJECT_ROOT, load_config


DATA_SHEET = "INX_Future_Inc_Employee_Perform"
DEFINITION_SHEET = "Data Definitions"
OUTPUT = Path("reports/research_log/finalization_v2/11_inx_workbook_equivalence_receipt.json")
LOCAL_WORKBOOK = PROJECT_ROOT / "data/raw/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fixture_project(tmp_path: Path, *, valid_mapping: bool = True) -> tuple[Path, WorkbookSnapshot]:
    workbook = tmp_path / "data/raw/source.xls"
    csv_path = tmp_path / "data/raw/canonical.csv"
    workbook.parent.mkdir(parents=True)
    workbook.write_bytes(b"BIFF8 fixture bytes; reader is injected only inside this test")
    csv_path.write_text(
        "EmpNumber;EmpRelationshipSatisfaction;PerformanceRating\n"
        "EMPLOYEE_SECRET_001;3;4\n"
        "EMPLOYEE_SECRET_002;2;3\n",
        encoding="utf-8-sig",
    )
    expected_columns = ["EmpNumber", "EmpRelationshipSatisfaction", "PerformanceRating"]
    mapping = {
        "RelationshipSatisfaction": "EmpRelationshipSatisfaction",
        "PerformanceRating": "PerformanceRating",
    }
    if not valid_mapping:
        mapping.pop("RelationshipSatisfaction")
    acquisition = {
        "data_acquisition": {
            "physical_datasets": {
                "inx_employee_performance": {
                    "local_path": "data/raw/canonical.csv",
                    "expected_sha256": sha256_file(csv_path),
                    "expected_rows": 2,
                    "expected_column_count": 3,
                    "expected_columns": expected_columns,
                    "delimiter": ";",
                    "encoding": "utf-8-sig",
                    "source_workbook_provenance": {
                        "local_path": "data/raw/source.xls",
                        "expected_sha256": sha256_file(workbook),
                        "format": "xls_biff8_ole2",
                        "expected_sheet_order": [DATA_SHEET, DEFINITION_SHEET],
                        "data_sheet_name": DATA_SHEET,
                        "data_definitions_sheet_name": DEFINITION_SHEET,
                        "expected_data_rows_excluding_header": 2,
                        "expected_data_column_count": 3,
                        "expected_defined_columns": [
                            "RelationshipSatisfaction",
                            "PerformanceRating",
                        ],
                        "definition_to_data_column_mapping": mapping,
                        "canonical_experiment_input": "data/raw/canonical.csv",
                        "comparison_normalization": NORMALIZATION_CONTRACT,
                        "reader_policy": "xlrd_if_available_else_explicit_windows_excel_com",
                        "macros_and_links": "disabled_no_update_read_only",
                        "equivalence_receipt": OUTPUT.as_posix(),
                    },
                }
            }
        }
    }
    provenance = {
        "dataset_provenance": {
            "physical_sources": {
                "inx_employee_performance": {
                    "source_workbook_path": "data/raw/source.xls",
                    "source_workbook_sha256": sha256_file(workbook),
                    "workbook_csv_equivalence_receipt": OUTPUT.as_posix(),
                    "source_authenticity_verification_status": "manual_review_required",
                    "licence_verification_status": "manual_review_required",
                    "citation_verification_status": "manual_review_required",
                }
            }
        }
    }
    _write_json(tmp_path / "configs/data_acquisition.yaml", acquisition)
    _write_json(tmp_path / "configs/dataset_provenance.yaml", provenance)
    snapshot = WorkbookSnapshot(
        engine="test_fixture_reader",
        engine_version="1",
        sheets={
            DATA_SHEET: (
                tuple(expected_columns),
                ("EMPLOYEE_SECRET_001", 3.0, 4.0),
                ("EMPLOYEE_SECRET_002", 2.0, 3.0),
            ),
            DEFINITION_SHEET: (
                ("RelationshipSatisfaction", ""),
                ("", "1 Low"),
                ("", "2 Medium"),
                ("PerformanceRating", ""),
                ("", "3 Good"),
                ("", "4 Excellent"),
            ),
        },
    )
    return tmp_path, snapshot


def test_numeric_normalization_preserves_identifier_like_leading_zero_text() -> None:
    assert normalize_cell(3.0) == "3"
    assert normalize_cell(" 3.000 ") == "3"
    assert normalize_cell("0012") == "0012"
    assert normalize_cell("  staff  ") == "staff"
    assert normalize_cell(None) == ""


def test_cli_defaults_are_portable_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["inx-workbook-equivalence"])
    arguments = workbook_equivalence.parse_args()
    assert arguments.acquisition_config == "configs/data_acquisition.yaml"
    assert arguments.provenance_config == "configs/dataset_provenance.yaml"
    assert arguments.output == OUTPUT.as_posix()
    assert "\\" not in arguments.output


def test_matrix_mismatch_receipt_exposes_coordinates_but_not_values() -> None:
    comparison = compare_matrices(
        (("id", "rating"), ("EMPLOYEE_SECRET_A", 3)),
        (("id", "rating"), ("EMPLOYEE_SECRET_B", 3)),
        expected_columns=("id", "rating"),
    )
    serialized = json.dumps(comparison)
    assert comparison["equivalent"] is False
    assert comparison["mismatch_count"] == 1
    assert comparison["mismatch_coordinates"] == [
        {"row_1_based": 2, "column_1_based": 1, "column_name": "id"}
    ]
    assert "EMPLOYEE_SECRET_A" not in serialized
    assert "EMPLOYEE_SECRET_B" not in serialized


def test_fixture_run_is_exact_but_explicitly_noncanonical(tmp_path: Path) -> None:
    root, snapshot = _fixture_project(tmp_path)
    receipt = run(
        project_root=root,
        output_path=OUTPUT,
        require_clean=False,
        workbook_reader=lambda _: snapshot,
    )
    receipt_path = root / OUTPUT
    serialized = receipt_path.read_text(encoding="utf-8")
    assert receipt["status"] == "passed"
    assert receipt["first_sheet_csv_equivalent"] is True
    assert receipt["comparison"]["mismatch_count"] == 0
    assert receipt["comparison"]["workbook_normalized_content_sha256"] == receipt["comparison"][
        "csv_normalized_content_sha256"
    ]
    assert receipt["data_definitions"]["definition_to_data_column_mapping"] == {
        "RelationshipSatisfaction": "EmpRelationshipSatisfaction",
        "PerformanceRating": "PerformanceRating",
    }
    assert receipt["data_definitions"]["all_definition_blocks_bound_to_data_columns"] is True
    assert receipt["data_definitions"]["complete_data_dictionary"] is False
    assert receipt["raw_employee_values_in_receipt"] is False
    assert receipt["canonical_eligible"] is False
    assert "EMPLOYEE_SECRET_001" not in serialized
    assert "EMPLOYEE_SECRET_002" not in serialized
    validate_receipt(receipt_path, project_root=root, require_canonical=False)
    with pytest.raises(WorkbookEquivalenceError, match="diagnostic-only"):
        validate_receipt(receipt_path, project_root=root)


def test_missing_explicit_definition_alias_writes_failure_receipt(tmp_path: Path) -> None:
    root, snapshot = _fixture_project(tmp_path, valid_mapping=False)
    with pytest.raises(WorkbookEquivalenceError, match="equivalence failed"):
        run(
            project_root=root,
            output_path=OUTPUT,
            require_clean=False,
            workbook_reader=lambda _: snapshot,
        )
    receipt = json.loads((root / OUTPUT).read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["data_definitions"]["defined_columns_exact"] is True
    assert receipt["data_definitions"]["definition_mapping_exact"] is False
    assert receipt["data_definitions"]["all_definition_blocks_bound_to_data_columns"] is False


def test_receipt_validation_detects_tampering_and_config_drift(tmp_path: Path) -> None:
    root, snapshot = _fixture_project(tmp_path)
    run(
        project_root=root,
        output_path=OUTPUT,
        require_clean=False,
        workbook_reader=lambda _: snapshot,
    )
    receipt_path = root / OUTPUT
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["raw_employee_values_in_receipt"] = True
    _write_json(receipt_path, payload)
    with pytest.raises(WorkbookEquivalenceError, match="must not contain employee values"):
        validate_receipt(receipt_path, project_root=root, require_canonical=False)

    payload["raw_employee_values_in_receipt"] = False
    _write_json(receipt_path, payload)
    acquisition_path = root / "configs/data_acquisition.yaml"
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    acquisition["data_acquisition"]["physical_datasets"]["inx_employee_performance"][
        "expected_rows"
    ] = 99
    _write_json(acquisition_path, acquisition)
    with pytest.raises(WorkbookEquivalenceError, match="acquisition config changed"):
        validate_receipt(receipt_path, project_root=root, require_canonical=False)


def test_atomic_writer_reports_secondary_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_replace(_: str, __: Path) -> None:
        raise RuntimeError("replace failed")

    def fail_unlink(_: str) -> None:
        raise OSError("cleanup failed")

    monkeypatch.setattr(workbook_equivalence.os, "replace", fail_replace)
    monkeypatch.setattr(workbook_equivalence.os, "unlink", fail_unlink)
    with pytest.raises(RuntimeError, match="replace failed") as error:
        workbook_equivalence._atomic_write_json(tmp_path / "receipt.json", {"status": "passed"})
    assert any("Atomic receipt cleanup also failed" in note for note in error.value.__notes__)


@pytest.mark.skipif(not LOCAL_WORKBOOK.is_file(), reason="requires the ignored local INX workbook")
def test_repository_contract_binds_tracked_workbook_csv_and_explicit_alias() -> None:
    acquisition = load_config(PROJECT_ROOT / "configs/data_acquisition.yaml")
    provenance = load_config(PROJECT_ROOT / "configs/dataset_provenance.yaml")
    physical = acquisition["data_acquisition"]["physical_datasets"]["inx_employee_performance"]
    contract = physical["source_workbook_provenance"]
    source = provenance["dataset_provenance"]["physical_sources"]["inx_employee_performance"]
    workbook = PROJECT_ROOT / contract["local_path"]
    csv_path = PROJECT_ROOT / physical["local_path"]
    assert workbook.read_bytes()[:8] == bytes.fromhex("d0cf11e0a1b11ae1")
    assert sha256_file(workbook) == contract["expected_sha256"] == source["source_workbook_sha256"]
    assert sha256_file(csv_path) == physical["expected_sha256"]
    assert contract["canonical_experiment_input"] == physical["local_path"]
    assert contract["definition_to_data_column_mapping"]["RelationshipSatisfaction"] == (
        "EmpRelationshipSatisfaction"
    )
    assert source["data_definitions_semantic_authority_status"] == "manual_review_required"
