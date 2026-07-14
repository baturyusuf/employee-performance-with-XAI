"""Executable INX BIFF8-workbook to canonical-CSV provenance validation.

The scientific pipeline consumes the verified semicolon CSV.  This validator
does not make the workbook a model input.  It independently binds the tracked
BIFF8 workbook, compares its first worksheet to the CSV after one explicit
cell-normalization contract, hashes the partial ``Data Definitions`` sheet,
and writes a compact receipt containing no employee cell values.

``xlrd`` is the portable preferred reader.  On Windows, an explicit isolated
Excel COM instance is the accepted fallback when ``xlrd`` is unavailable.  COM
opens the workbook read-only with macros, events, alerts, and link updates
disabled, then quits only the instance created by this validator.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from src.governance.manuscript_contract import sha256_file, source_tree_hash
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_ACQUISITION_CONFIG = Path("configs/data_acquisition.yaml")
DEFAULT_PROVENANCE_CONFIG = Path("configs/dataset_provenance.yaml")
DEFAULT_OUTPUT = Path(
    "reports/research_log/finalization_v2/11_inx_workbook_equivalence_receipt.json"
)
VALIDATOR_SOURCE_PATH = Path("src/governance/inx_workbook_equivalence.py")
DATASET_KEY = "inx_employee_performance"
NORMALIZATION_CONTRACT = "trim_strings_blank_to_empty_numeric_decimal_canonical_dates_iso_v1"
MAX_MISMATCH_COORDINATES = 25
_NUMERIC_TEXT = re.compile(r"^[+-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$")


class WorkbookEquivalenceError(RuntimeError):
    """Raised when the workbook provenance contract cannot be validated."""


Matrix = tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class WorkbookSnapshot:
    engine: str
    engine_version: str
    sheets: Mapping[str, Matrix]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _portable_path(project_root: Path, value: Any, *, field: str) -> tuple[str, Path]:
    raw = value.as_posix() if isinstance(value, Path) else str(value)
    if not raw or raw != raw.strip() or "\\" in raw:
        raise WorkbookEquivalenceError(f"{field} must be a portable repository-relative path.")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise WorkbookEquivalenceError(f"{field} is unsafe: {raw!r}.")
    resolved = (project_root / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise WorkbookEquivalenceError(f"{field} escapes the repository root.") from exc
    return relative.as_posix(), resolved


def _canonical_decimal(value: Any) -> str:
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise WorkbookEquivalenceError(f"Invalid numeric workbook/CSV cell: {value!r}.") from exc
    if not decimal.is_finite():
        raise WorkbookEquivalenceError("Workbook/CSV numeric cells must be finite.")
    if decimal == 0:
        return "0"
    normalized = format(decimal.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def normalize_cell(value: Any) -> str:
    """Normalize one cell without retaining a Python type distinction."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise WorkbookEquivalenceError("Workbook/CSV numeric cells must be finite.")
        return _canonical_decimal(value)
    text = str(value).strip()
    if not text:
        return ""
    if _NUMERIC_TEXT.fullmatch(text):
        unsigned = text.lstrip("+-")
        integer_part = unsigned.split(".", 1)[0].split("e", 1)[0].split("E", 1)[0]
        if len(integer_part) == 1 or not integer_part.startswith("0"):
            return _canonical_decimal(text)
    return text


def normalize_matrix(rows: Sequence[Sequence[Any]], *, name: str) -> tuple[tuple[str, ...], ...]:
    materialized = [tuple(row) for row in rows]
    if not materialized:
        raise WorkbookEquivalenceError(f"{name} is empty.")
    width = len(materialized[0])
    if width < 1 or any(len(row) != width for row in materialized):
        raise WorkbookEquivalenceError(f"{name} is not a non-empty rectangular matrix.")
    return tuple(tuple(normalize_cell(value) for value in row) for row in materialized)


def matrix_sha256(matrix: Sequence[Sequence[str]]) -> str:
    digest = hashlib.sha256()
    for row in matrix:
        digest.update(json.dumps(list(row), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _excel_value_to_matrix(value: Any, rows: int, columns: int) -> Matrix:
    if rows == 1 and columns == 1:
        return ((value,),)
    if rows == 1:
        if not isinstance(value, tuple):
            return ((value,),)
        if value and isinstance(value[0], tuple):
            return tuple(tuple(row) for row in value)
        return (tuple(value),)
    if not isinstance(value, tuple):
        raise WorkbookEquivalenceError("Excel COM returned a non-matrix UsedRange.")
    return tuple(tuple(row) if isinstance(row, tuple) else (row,) for row in value)


def _read_with_excel_com(path: Path) -> WorkbookSnapshot:
    if os.name != "nt":
        raise WorkbookEquivalenceError("Excel COM fallback is available only on Windows.")
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise WorkbookEquivalenceError("Excel COM fallback requires pywin32.") from exc

    pythoncom.CoInitialize()
    excel = None
    workbook = None
    worksheets: list[Any] = []
    snapshot: WorkbookSnapshot | None = None
    primary_error: WorkbookEquivalenceError | None = None
    try:
        excel = win32com.client.DispatchEx("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False
        excel.EnableEvents = False
        excel.AutomationSecurity = 3  # msoAutomationSecurityForceDisable
        workbook = excel.Workbooks.Open(
            str(path),
            UpdateLinks=0,
            ReadOnly=True,
            IgnoreReadOnlyRecommended=True,
            AddToMru=False,
        )
        sheets: dict[str, Matrix] = {}
        for index in range(1, int(workbook.Worksheets.Count) + 1):
            worksheet = workbook.Worksheets(index)
            worksheets.append(worksheet)
            name = str(worksheet.Name)
            used = worksheet.UsedRange
            rows = int(used.Rows.Count)
            columns = int(used.Columns.Count)
            sheets[name] = _excel_value_to_matrix(used.Value2, rows, columns)
        snapshot = WorkbookSnapshot(
            engine="excel_com_windows",
            engine_version=str(excel.Version),
            sheets=sheets,
        )
    except Exception as exc:
        if isinstance(exc, WorkbookEquivalenceError):
            primary_error = exc
        else:
            primary_error = WorkbookEquivalenceError(
                f"Excel COM workbook read failed: {type(exc).__name__}: {exc}"
            )
            primary_error.__cause__ = exc
    finally:
        cleanup_errors: list[str] = []
        if workbook is not None:
            try:
                workbook.Close(SaveChanges=False)
            except Exception as exc:
                cleanup_errors.append(f"workbook.Close: {type(exc).__name__}: {exc}")
        if excel is not None:
            try:
                excel.Quit()
            except Exception as exc:
                cleanup_errors.append(f"excel.Quit: {type(exc).__name__}: {exc}")
        worksheets.clear()
        workbook = None
        excel = None
        gc.collect()
        pythoncom.CoUninitialize()
        if cleanup_errors:
            detail = "; ".join(cleanup_errors)
            if primary_error is None:
                primary_error = WorkbookEquivalenceError(f"Excel COM cleanup failed: {detail}")
            else:
                primary_error.add_note(f"Excel COM cleanup also failed: {detail}")
    if primary_error is not None:
        raise primary_error
    if snapshot is None:  # pragma: no cover - defensive invariant
        raise WorkbookEquivalenceError("Excel COM completed without a workbook snapshot.")
    return snapshot


def _read_with_xlrd(path: Path) -> WorkbookSnapshot:
    try:
        import xlrd
    except ImportError as exc:
        raise WorkbookEquivalenceError("xlrd is not installed.") from exc
    workbook = None
    try:
        workbook = xlrd.open_workbook(str(path), on_demand=True)
        sheets: dict[str, Matrix] = {}
        for raw_name in workbook.sheet_names():
            sheet = workbook.sheet_by_name(raw_name)
            sheets[str(raw_name)] = tuple(
                tuple(sheet.cell_value(row, column) for column in range(sheet.ncols))
                for row in range(sheet.nrows)
            )
    except Exception as exc:
        raise WorkbookEquivalenceError(f"xlrd workbook read failed: {type(exc).__name__}: {exc}") from exc
    finally:
        if workbook is not None:
            workbook.release_resources()
    return WorkbookSnapshot(
        engine="xlrd",
        engine_version=str(getattr(xlrd, "__version__", "unknown")),
        sheets=sheets,
    )


def read_workbook(path: Path, *, engine: str = "auto") -> WorkbookSnapshot:
    if engine not in {"auto", "xlrd", "excel_com_windows"}:
        raise WorkbookEquivalenceError(f"Unknown workbook reader engine: {engine!r}.")
    if engine == "xlrd":
        return _read_with_xlrd(path)
    if engine == "excel_com_windows":
        return _read_with_excel_com(path)
    if importlib.util.find_spec("xlrd") is not None:
        return _read_with_xlrd(path)
    return _read_with_excel_com(path)


def _read_csv(path: Path, *, delimiter: str, encoding: str) -> Matrix:
    with path.open("r", encoding=encoding, newline="") as stream:
        return tuple(tuple(row) for row in csv.reader(stream, delimiter=delimiter))


def compare_matrices(
    workbook: Sequence[Sequence[Any]],
    csv_rows: Sequence[Sequence[Any]],
    *,
    expected_columns: Sequence[str],
) -> Mapping[str, Any]:
    workbook_normalized = normalize_matrix(workbook, name="workbook data sheet")
    csv_normalized = normalize_matrix(csv_rows, name="canonical CSV")
    workbook_shape = (len(workbook_normalized), len(workbook_normalized[0]))
    csv_shape = (len(csv_normalized), len(csv_normalized[0]))
    expected_header = tuple(str(value) for value in expected_columns)
    header_exact = (
        workbook_normalized[0] == expected_header
        and csv_normalized[0] == expected_header
    )
    mismatch_count = 0
    coordinates: list[dict[str, Any]] = []
    max_rows = max(workbook_shape[0], csv_shape[0])
    max_columns = max(workbook_shape[1], csv_shape[1])
    for row in range(max_rows):
        for column in range(max_columns):
            workbook_value = (
                workbook_normalized[row][column]
                if row < workbook_shape[0] and column < workbook_shape[1]
                else None
            )
            csv_value = (
                csv_normalized[row][column]
                if row < csv_shape[0] and column < csv_shape[1]
                else None
            )
            if workbook_value == csv_value:
                continue
            mismatch_count += 1
            if len(coordinates) < MAX_MISMATCH_COORDINATES:
                coordinates.append(
                    {
                        "row_1_based": row + 1,
                        "column_1_based": column + 1,
                        "column_name": expected_header[column]
                        if column < len(expected_header)
                        else "__OUT_OF_SCHEMA__",
                    }
                )
    workbook_hash = matrix_sha256(workbook_normalized)
    csv_hash = matrix_sha256(csv_normalized)
    return {
        "normalization_contract": NORMALIZATION_CONTRACT,
        "workbook_shape_including_header": list(workbook_shape),
        "csv_shape_including_header": list(csv_shape),
        "expected_header_sha256": _sha256_json(list(expected_header)),
        "header_exact": header_exact,
        "workbook_normalized_content_sha256": workbook_hash,
        "csv_normalized_content_sha256": csv_hash,
        "normalized_content_hash_equal": workbook_hash == csv_hash,
        "mismatch_count": mismatch_count,
        "mismatch_coordinates": coordinates,
        "mismatch_coordinates_truncated": mismatch_count > len(coordinates),
        "equivalent": (
            workbook_shape == csv_shape
            and header_exact
            and mismatch_count == 0
            and workbook_hash == csv_hash
        ),
    }


def _definition_summary(
    matrix: Sequence[Sequence[Any]],
    *,
    expected_defined_columns: Sequence[str],
    definition_to_data_column: Mapping[str, str],
    data_headers: Sequence[str],
) -> Mapping[str, Any]:
    normalized = normalize_matrix(matrix, name="Data Definitions sheet")
    if len(normalized[0]) != 2:
        raise WorkbookEquivalenceError("Data Definitions sheet must have exactly two used columns.")
    blocks: dict[str, list[tuple[str, str]]] = {}
    current: str | None = None
    for row in normalized:
        if row[0]:
            current = row[0]
            if current in blocks:
                raise WorkbookEquivalenceError(
                    f"Data Definitions contains a duplicate block: {current!r}."
                )
            blocks[current] = []
        if current is not None and row[1]:
            blocks[current].append(row)
    observed = list(blocks)
    expected = [str(value) for value in expected_defined_columns]
    mapping = {str(key): str(value) for key, value in definition_to_data_column.items()}
    mapping_exact = list(mapping) == expected and len(set(mapping.values())) == len(mapping)
    mapped_data_columns = [mapping.get(column, "__UNMAPPED__") for column in observed]
    block_rows = [
        {
            "column": column,
            "definition_entry_count": len(blocks[column]),
            "definition_block_sha256": _sha256_json(blocks[column]),
        }
        for column in observed
    ]
    return {
        "sheet_shape": [len(normalized), len(normalized[0])],
        "sheet_normalized_content_sha256": matrix_sha256(normalized),
        "nonempty_cell_count": sum(bool(value) for row in normalized for value in row),
        "defined_columns": observed,
        "expected_defined_columns": expected,
        "defined_columns_exact": observed == expected,
        "definition_to_data_column_mapping": mapping,
        "definition_mapping_sha256": _sha256_json(mapping),
        "definition_mapping_exact": mapping_exact,
        "mapped_data_columns": mapped_data_columns,
        "all_definition_blocks_bound_to_data_columns": (
            mapping_exact and set(mapped_data_columns).issubset(data_headers)
        ),
        "documented_column_count": len(observed),
        "data_column_count": len(data_headers),
        "complete_data_dictionary": set(observed) == set(data_headers),
        "definition_blocks": block_rows,
        "claim_boundary": (
            "Partial codebook hash and column-name coverage only; this does not verify semantic "
            "authority, source authenticity, licence, or a complete data dictionary."
        ),
    }


def _git_identity(project_root: Path, *, require_clean: bool) -> tuple[str, bool]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        if require_clean:
            raise WorkbookEquivalenceError("A clean Git identity is required for the receipt.") from exc
        return "unavailable", True
    dirty = bool(status)
    if require_clean and dirty:
        raise WorkbookEquivalenceError("Workbook equivalence receipt requires a clean worktree.")
    return head, dirty


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception as exc:
        try:
            os.unlink(temporary)
        except OSError as cleanup_exc:
            exc.add_note(
                "Atomic receipt cleanup also failed: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            )
        raise


def validate_receipt(
    receipt_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    require_canonical: bool = True,
) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    path = Path(receipt_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        receipt_relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise WorkbookEquivalenceError("Receipt path escapes the repository root.") from exc
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_true = (
        "workbook_byte_hash_verified",
        "csv_byte_hash_verified",
        "sheet_order_exact",
        "first_sheet_csv_equivalent",
        "canonical_experiment_consumes_csv_only",
    )
    if payload.get("status") != "passed" or payload.get("contract_kind") != "inx_workbook_csv_equivalence":
        raise WorkbookEquivalenceError("Workbook equivalence receipt is not a passing supported contract.")
    if any(payload.get(field) is not True for field in required_true):
        raise WorkbookEquivalenceError("Workbook equivalence receipt boolean invariants failed.")
    if payload.get("raw_employee_values_in_receipt") is not False:
        raise WorkbookEquivalenceError("Workbook equivalence receipt must not contain employee values.")
    if payload.get("network_calls") != 0 or payload.get("paid_api_calls") != 0:
        raise WorkbookEquivalenceError("Workbook equivalence validation must be offline and API-free.")
    if require_canonical and (
        payload.get("canonical_eligible") is not True
        or payload.get("git_worktree_dirty") is not False
        or not re.fullmatch(r"[0-9a-f]{40}", str(payload.get("git_commit", "")))
    ):
        raise WorkbookEquivalenceError("Receipt is diagnostic-only, dirty, or lacks a Git commit identity.")
    config_paths: dict[str, Path] = {}
    for config_field in ("acquisition", "provenance"):
        relative, config_path = _portable_path(
            root,
            payload.get(f"{config_field}_config_path"),
            field=f"{config_field}_config_path",
        )
        if relative != payload.get(f"{config_field}_config_path") or sha256_file(config_path) != payload.get(
            f"{config_field}_config_sha256"
        ):
            raise WorkbookEquivalenceError(f"{config_field} config changed after receipt generation.")
        config_paths[config_field] = config_path
    acquisition = load_config(config_paths["acquisition"])
    physical = acquisition.get("data_acquisition", {}).get("physical_datasets", {}).get(DATASET_KEY)
    if not isinstance(physical, Mapping):
        raise WorkbookEquivalenceError("Current INX acquisition contract is unavailable.")
    contract = physical.get("source_workbook_provenance")
    if not isinstance(contract, Mapping):
        raise WorkbookEquivalenceError("Current workbook provenance contract is unavailable.")
    for prefix in ("workbook", "csv"):
        relative, source = _portable_path(root, payload.get(f"{prefix}_path"), field=f"{prefix}_path")
        if relative != payload.get(f"{prefix}_path") or sha256_file(source) != payload.get(f"{prefix}_sha256"):
            raise WorkbookEquivalenceError(f"{prefix} bytes changed after receipt generation.")
    comparison = payload.get("comparison")
    definitions = payload.get("data_definitions")
    normalized_hashes = (
        str(comparison.get("workbook_normalized_content_sha256", ""))
        if isinstance(comparison, Mapping)
        else "",
        str(comparison.get("csv_normalized_content_sha256", ""))
        if isinstance(comparison, Mapping)
        else "",
    )
    if (
        not isinstance(comparison, Mapping)
        or comparison.get("equivalent") is not True
        or comparison.get("mismatch_count") != 0
        or comparison.get("mismatch_coordinates") != []
        or comparison.get("normalized_content_hash_equal") is not True
        or normalized_hashes[0] != normalized_hashes[1]
        or not re.fullmatch(r"[0-9a-f]{64}", normalized_hashes[0])
        or not isinstance(definitions, Mapping)
        or definitions.get("defined_columns_exact") is not True
        or definitions.get("definition_mapping_exact") is not True
        or definitions.get("all_definition_blocks_bound_to_data_columns") is not True
        or definitions.get("definition_mapping_sha256")
        != _sha256_json(definitions.get("definition_to_data_column_mapping"))
    ):
        raise WorkbookEquivalenceError("Workbook comparison or codebook coverage is invalid.")
    if payload.get("source_tree_hash") != source_tree_hash(root):
        raise WorkbookEquivalenceError("Scientific source tree changed after receipt generation.")
    if (
        payload.get("workbook_path") != contract.get("local_path")
        or payload.get("csv_path") != physical.get("local_path")
        or payload.get("workbook_sha256") != contract.get("expected_sha256")
        or payload.get("csv_sha256") != physical.get("expected_sha256")
        or payload.get("expected_sheet_order") != contract.get("expected_sheet_order")
        or definitions.get("definition_to_data_column_mapping")
        != contract.get("definition_to_data_column_mapping")
        or receipt_relative != contract.get("equivalence_receipt")
        or (
            require_canonical
            and payload.get("reader_engine") not in {"xlrd", "excel_com_windows"}
        )
    ):
        raise WorkbookEquivalenceError("Receipt no longer matches the current workbook contract.")
    if (
        payload.get("validator_source_path") != VALIDATOR_SOURCE_PATH.as_posix()
        or payload.get("validator_source_sha256") != sha256_file(Path(__file__))
    ):
        raise WorkbookEquivalenceError("Workbook validator source changed after receipt generation.")
    if require_canonical:
        try:
            subprocess.run(
                ["git", "cat-file", "-e", f"{payload['git_commit']}^{{commit}}"],
                cwd=root,
                check=True,
                capture_output=True,
            )
        except (KeyError, OSError, subprocess.CalledProcessError) as exc:
            raise WorkbookEquivalenceError("Receipt Git commit is not present in this repository.") from exc
    return payload


def run(
    acquisition_config: str | Path = DEFAULT_ACQUISITION_CONFIG,
    provenance_config: str | Path = DEFAULT_PROVENANCE_CONFIG,
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    project_root: str | Path = PROJECT_ROOT,
    engine: str = "auto",
    require_clean: bool = True,
    workbook_reader: Callable[[Path], WorkbookSnapshot] | None = None,
) -> Mapping[str, Any]:
    """Validate exact workbook/CSV equivalence and atomically write a compact receipt."""

    root = Path(project_root).resolve()
    acquisition_path = Path(acquisition_config)
    if not acquisition_path.is_absolute():
        acquisition_path = root / acquisition_path
    provenance_path = Path(provenance_config)
    if not provenance_path.is_absolute():
        provenance_path = root / provenance_path
    acquisition = load_config(acquisition_path)
    provenance = load_config(provenance_path)
    physical = acquisition.get("data_acquisition", {}).get("physical_datasets", {}).get(DATASET_KEY)
    provenance_entry = provenance.get("dataset_provenance", {}).get("physical_sources", {}).get(DATASET_KEY)
    if not isinstance(physical, Mapping) or not isinstance(provenance_entry, Mapping):
        raise WorkbookEquivalenceError("INX acquisition/provenance entries are missing.")
    contract = physical.get("source_workbook_provenance")
    if not isinstance(contract, Mapping):
        raise WorkbookEquivalenceError("INX source_workbook_provenance contract is missing.")
    if contract.get("comparison_normalization") != NORMALIZATION_CONTRACT:
        raise WorkbookEquivalenceError("Workbook comparison normalization contract drifted.")
    workbook_relative, workbook_path = _portable_path(root, contract.get("local_path"), field="workbook path")
    csv_relative, csv_path = _portable_path(root, physical.get("local_path"), field="CSV path")
    output_relative, output = _portable_path(root, output_path, field="receipt path")
    if output_relative != contract.get("equivalence_receipt"):
        raise WorkbookEquivalenceError("Requested receipt path differs from the acquisition contract.")
    if (
        contract.get("canonical_experiment_input") != csv_relative
        or contract.get("reader_policy") != "xlrd_if_available_else_explicit_windows_excel_com"
        or contract.get("macros_and_links") != "disabled_no_update_read_only"
        or int(contract.get("expected_data_rows_excluding_header", -1))
        != int(physical.get("expected_rows", -2))
        or int(contract.get("expected_data_column_count", -1))
        != int(physical.get("expected_column_count", -2))
    ):
        raise WorkbookEquivalenceError("Workbook/CSV execution contract drifted.")
    if output.exists():
        raise WorkbookEquivalenceError(f"Receipt path already exists: {output_relative}")
    if not workbook_path.is_file() or not csv_path.is_file():
        raise WorkbookEquivalenceError("Workbook and canonical CSV must both exist locally.")
    workbook_sha = sha256_file(workbook_path)
    csv_sha = sha256_file(csv_path)
    if workbook_sha != contract.get("expected_sha256") or csv_sha != physical.get("expected_sha256"):
        raise WorkbookEquivalenceError("Workbook or CSV byte SHA-256 differs from its contract.")
    if (
        provenance_entry.get("source_workbook_path") != workbook_relative
        or provenance_entry.get("source_workbook_sha256") != workbook_sha
        or provenance_entry.get("workbook_csv_equivalence_receipt") != output_relative
    ):
        raise WorkbookEquivalenceError("Dataset provenance workbook binding drifted.")
    git_head, git_dirty = _git_identity(root, require_clean=require_clean)
    snapshot = workbook_reader(workbook_path) if workbook_reader is not None else read_workbook(workbook_path, engine=engine)
    expected_sheet_order = [str(value) for value in contract.get("expected_sheet_order", [])]
    observed_sheet_order = list(snapshot.sheets)
    data_sheet_name = str(contract.get("data_sheet_name", ""))
    definitions_sheet_name = str(contract.get("data_definitions_sheet_name", ""))
    if data_sheet_name not in snapshot.sheets or definitions_sheet_name not in snapshot.sheets:
        raise WorkbookEquivalenceError("Required workbook sheets are missing.")
    expected_columns = [str(value) for value in physical.get("expected_columns", [])]
    csv_rows = _read_csv(
        csv_path,
        delimiter=str(physical.get("delimiter", ";")),
        encoding=str(physical.get("encoding", "utf-8-sig")),
    )
    comparison = compare_matrices(
        snapshot.sheets[data_sheet_name],
        csv_rows,
        expected_columns=expected_columns,
    )
    definitions = _definition_summary(
        snapshot.sheets[definitions_sheet_name],
        expected_defined_columns=[str(value) for value in contract.get("expected_defined_columns", [])],
        definition_to_data_column=contract.get("definition_to_data_column_mapping", {}),
        data_headers=expected_columns,
    )
    expected_shape = [int(physical.get("expected_rows", -1)) + 1, int(physical.get("expected_column_count", -1))]
    passed = (
        observed_sheet_order == expected_sheet_order
        and comparison["equivalent"] is True
        and comparison["workbook_shape_including_header"] == expected_shape
        and comparison["csv_shape_including_header"] == expected_shape
        and definitions["defined_columns_exact"] is True
        and definitions["definition_mapping_exact"] is True
        and definitions["all_definition_blocks_bound_to_data_columns"] is True
    )
    payload = {
        "schema_version": 1,
        "contract_kind": "inx_workbook_csv_equivalence",
        "status": "passed" if passed else "failed",
        "dataset_key": DATASET_KEY,
        "git_commit": git_head,
        "git_worktree_dirty": git_dirty,
        "source_tree_hash": source_tree_hash(root),
        "acquisition_config_path": acquisition_path.relative_to(root).as_posix(),
        "acquisition_config_sha256": sha256_file(acquisition_path),
        "provenance_config_path": provenance_path.relative_to(root).as_posix(),
        "provenance_config_sha256": sha256_file(provenance_path),
        "validator_source_path": VALIDATOR_SOURCE_PATH.as_posix(),
        "validator_source_sha256": sha256_file(Path(__file__)),
        "workbook_path": workbook_relative,
        "workbook_sha256": workbook_sha,
        "workbook_size_bytes": workbook_path.stat().st_size,
        "workbook_format": contract.get("format"),
        "workbook_byte_hash_verified": True,
        "csv_path": csv_relative,
        "csv_sha256": csv_sha,
        "csv_size_bytes": csv_path.stat().st_size,
        "csv_delimiter": physical.get("delimiter"),
        "csv_encoding": physical.get("encoding"),
        "csv_byte_hash_verified": True,
        "reader_engine": snapshot.engine,
        "reader_engine_version": snapshot.engine_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "expected_sheet_order": expected_sheet_order,
        "observed_sheet_order": observed_sheet_order,
        "sheet_order_exact": observed_sheet_order == expected_sheet_order,
        "data_sheet_name": data_sheet_name,
        "data_definitions_sheet_name": definitions_sheet_name,
        "comparison": comparison,
        "first_sheet_csv_equivalent": comparison["equivalent"],
        "data_definitions": definitions,
        "canonical_experiment_consumes_csv_only": True,
        "raw_employee_values_in_receipt": False,
        "source_authenticity_status": provenance_entry.get("source_authenticity_verification_status"),
        "licence_verification_status": provenance_entry.get("licence_verification_status"),
        "citation_verification_status": provenance_entry.get("citation_verification_status"),
        "canonical_eligible": workbook_reader is None and require_clean and passed,
        "network_calls": 0,
        "paid_api_calls": 0,
        "validated_at": _utc_now(),
    }
    _atomic_write_json(output, payload)
    if not passed:
        raise WorkbookEquivalenceError(
            f"Workbook/CSV equivalence failed; compact failure receipt written to {output_relative}."
        )
    validate_receipt(
        output,
        project_root=root,
        require_canonical=workbook_reader is None and require_clean,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate INX workbook/CSV provenance equivalence.")
    parser.add_argument("--acquisition-config", default=DEFAULT_ACQUISITION_CONFIG.as_posix())
    parser.add_argument("--provenance-config", default=DEFAULT_PROVENANCE_CONFIG.as_posix())
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--engine", choices=("auto", "xlrd", "excel_com_windows"), default="auto")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    payload = run(
        arguments.acquisition_config,
        arguments.provenance_config,
        output_path=arguments.output,
        engine=arguments.engine,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "receipt": str(arguments.output),
                "reader_engine": payload["reader_engine"],
                "first_sheet_csv_equivalent": payload["first_sheet_csv_equivalent"],
                "data_definitions_complete_dictionary": payload["data_definitions"][
                    "complete_data_dictionary"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT",
    "NORMALIZATION_CONTRACT",
    "WorkbookEquivalenceError",
    "WorkbookSnapshot",
    "compare_matrices",
    "matrix_sha256",
    "normalize_cell",
    "normalize_matrix",
    "read_workbook",
    "run",
    "validate_receipt",
]
