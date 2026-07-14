from __future__ import annotations

from pathlib import Path

import pytest

from src.governance.unit2g_stage_validator import (
    Unit2GValidationError,
    _safe_stage_path,
    canonical_csv_content_sha256,
)


def test_canonical_csv_content_hash_ignores_bom_and_line_endings(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_bytes(b"\xef\xbb\xbfname,value\r\nAlice,1\r\nBob,2\r\n")
    second.write_bytes(b"name,value\nAlice,1\nBob,2\n")

    first_hash, first_rows, first_columns = canonical_csv_content_sha256(first)
    second_hash, second_rows, second_columns = canonical_csv_content_sha256(second)

    assert first_hash == second_hash
    assert (first_rows, first_columns) == (second_rows, second_columns) == (2, 2)


@pytest.mark.parametrize("unsafe", ["../escape.csv", "nested\\windows.csv", "/absolute.csv"])
def test_safe_stage_path_rejects_nonportable_or_escaping_paths(
    tmp_path: Path, unsafe: str
) -> None:
    with pytest.raises(Unit2GValidationError):
        _safe_stage_path(tmp_path, unsafe)


def test_safe_stage_path_accepts_contained_posix_path(tmp_path: Path) -> None:
    expected = (tmp_path / "nested" / "artifact.csv").resolve()
    assert _safe_stage_path(tmp_path, "nested/artifact.csv") == expected
