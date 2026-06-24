from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from src.core.io_utils import ensure_dir


def write_dataframe(path: Path, df: pd.DataFrame) -> Path:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path


def write_markdown(path: Path, lines: Iterable[str]) -> Path:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def markdown_table(df: pd.DataFrame, columns: List[str] | None = None) -> List[str]:
    if columns is not None:
        df = df.loc[:, columns]
    if df.empty:
        return ["No rows available."]
    headers = [str(col) for col in df.columns]
    rows = []
    for row in df.itertuples(index=False):
        rows.append([_format_cell(value) for value in row])
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return out


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")

