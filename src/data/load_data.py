from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.validate_schema import print_validation_report, validate_dataframe
from src.utils.config import SETTINGS


def _read_csv_with_best_effort(csv_path: Path) -> pd.DataFrame:
    """
    Tries to read CSV robustly.
    Handles common delimiters such as comma, semicolon, and tab.
    """
    tried = []

    # 1) Let pandas infer separator
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
        if df.shape[1] > 1:
            return df
        tried.append("sep=None")
    except Exception as exc:
        tried.append(f"sep=None failed: {exc}")

    # 2) Try common delimiters manually
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(csv_path, sep=sep)
            if df.shape[1] > 1:
                return df
            tried.append(f"sep='{sep}' -> 1 column")
        except Exception as exc:
            tried.append(f"sep='{sep}' failed: {exc}")

    raise ValueError(
        f"Could not parse dataset correctly from: {csv_path}\n"
        f"Tried methods: {tried}"
    )


def load_raw_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    file_path = Path(path) if path is not None else SETTINGS.raw_data_path

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            f"Please place your real dataset at this location."
        )

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = _read_csv_with_best_effort(file_path)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Please use .csv, .xlsx, or .xls"
        )

    return df


def load_and_validate_data(
    path: Optional[str | Path] = None,
    save_interim: bool = True,
) -> pd.DataFrame:
    df = load_raw_data(path)

    print("\n=== RAW COLUMNS ===")
    print(df.columns.tolist())

    validated_df, report = validate_dataframe(df)

    print_validation_report(report)

    if report.missing_required_columns:
        raise ValueError(
            "Dataset schema is incomplete. Missing required columns: "
            f"{report.missing_required_columns}"
        )

    if report.duplicate_id_count > 0:
        raise ValueError(
            f"Duplicate '{SETTINGS.id_col}' values detected: {report.duplicate_id_count}"
        )

    if report.null_target_count > 0:
        raise ValueError(
            f"Null values found in target column '{SETTINGS.target_col}': {report.null_target_count}"
        )

    if save_interim:
        SETTINGS.interim_data_path.parent.mkdir(parents=True, exist_ok=True)
        validated_df.to_csv(SETTINGS.interim_data_path, index=False)
        print(f"\nValidated dataset saved to: {SETTINGS.interim_data_path}")

    return validated_df


if __name__ == "__main__":
    df = load_and_validate_data()
    print("\nPreview:")
    print(df.head())