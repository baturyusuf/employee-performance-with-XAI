from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from src.utils.config import SETTINGS


@dataclass
class ValidationReport:
    row_count: int
    column_count: int
    missing_required_columns: List[str]
    unexpected_columns: List[str]
    duplicate_id_count: int
    null_target_count: int
    target_labels_found: List[int]

    @property
    def is_valid(self) -> bool:
        return (
            len(self.missing_required_columns) == 0
            and self.duplicate_id_count == 0
            and self.null_target_count == 0
        )


def normalize_key(name: str) -> str:
    """
    Column name normalization for robust matching.
    Example:
    ' Emp Job Role ' -> 'empjobrole'
    """
    cleaned = str(name).replace("\ufeff", "").strip()
    cleaned = re.sub(r"[\s_\-]+", "", cleaned)
    return cleaned.lower()


def build_rename_map(columns: List[str]) -> Dict[str, str]:
    expected_by_key = {normalize_key(col): col for col in SETTINGS.expected_columns}
    aliases_by_key = {normalize_key(src): dst for src, dst in SETTINGS.column_aliases.items()}

    rename_map: Dict[str, str] = {}

    for col in columns:
        key = normalize_key(col)

        if key in expected_by_key:
            rename_map[col] = expected_by_key[key]
        elif key in aliases_by_key:
            rename_map[col] = aliases_by_key[key]
        else:
            rename_map[col] = str(col).strip()

    return rename_map


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = build_rename_map(list(df.columns))
    df = df.rename(columns=rename_map)
    return df


def coerce_target_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if SETTINGS.target_col not in df.columns:
        raise ValueError(
            f"Target column '{SETTINGS.target_col}' not found in dataframe.\n"
            f"Available columns: {df.columns.tolist()}"
        )
    df[SETTINGS.target_col] = pd.to_numeric(df[SETTINGS.target_col], errors="coerce")

    return df


def validate_required_columns(df: pd.DataFrame) -> List[str]:
    missing = [col for col in SETTINGS.expected_columns if col not in df.columns]
    return missing


def validate_duplicate_ids(df: pd.DataFrame) -> int:
    if SETTINGS.id_col not in df.columns:
        return 0
    return int(df[SETTINGS.id_col].duplicated().sum())


def validate_target_values(df: pd.DataFrame) -> tuple[int, List[int]]:
    if SETTINGS.target_col not in df.columns:
        return 0, []

    null_target_count = int(df[SETTINGS.target_col].isna().sum())

    target_non_null = df[SETTINGS.target_col].dropna().astype(int)
    labels_found = sorted(target_non_null.unique().tolist())

    invalid_labels = sorted(set(labels_found) - SETTINGS.allowed_target_labels)
    if invalid_labels:
        raise ValueError(
            f"Unexpected target labels found: {invalid_labels}. "
            f"Expected labels: {sorted(SETTINGS.allowed_target_labels)}"
        )

    return null_target_count, labels_found


def validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Returns:
        validated_df: standardized dataframe
        report: validation report
    Raises:
        ValueError if target labels are invalid
    """
    df = standardize_columns(df)
    df = coerce_target_column(df)

    missing_required_columns = validate_required_columns(df)
    duplicate_id_count = validate_duplicate_ids(df)
    null_target_count, target_labels_found = validate_target_values(df)

    unexpected_columns = [
        col for col in df.columns
        if col not in SETTINGS.expected_columns
    ]

    report = ValidationReport(
        row_count=len(df),
        column_count=len(df.columns),
        missing_required_columns=missing_required_columns,
        unexpected_columns=unexpected_columns,
        duplicate_id_count=duplicate_id_count,
        null_target_count=null_target_count,
        target_labels_found=target_labels_found,
    )

    return df, report


def print_validation_report(report: ValidationReport) -> None:
    print("\n=== VALIDATION REPORT ===")
    print(f"Rows: {report.row_count}")
    print(f"Columns: {report.column_count}")
    print(f"Missing required columns: {report.missing_required_columns}")
    print(f"Unexpected columns: {report.unexpected_columns}")
    print(f"Duplicate ID count: {report.duplicate_id_count}")
    print(f"Null target count: {report.null_target_count}")
    print(f"Target labels found: {report.target_labels_found}")
    print(f"Schema valid: {report.is_valid}")