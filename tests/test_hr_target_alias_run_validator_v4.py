from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.governance.hr_target_alias_compact_export_v4 import (
    EXPECTED,
    HRTargetAliasCompactExportV4Error,
    validate_hr_target_alias_compact_v4,
)
from src.governance.hr_target_alias_run_validator_v4 import _select


def test_independent_alias_selector_uses_inclusive_tolerance_and_qwk() -> None:
    rows = pd.DataFrame(
        {
            "candidate_index": [0, 1, 2],
            "inner_macro_f1_mean": [0.8, 0.799, 0.799],
            "inner_qwk_mean": [0.2, 0.9, 0.9],
        }
    )
    assert _select(rows) == 1


def test_alias_compact_inventory_excludes_row_level_files() -> None:
    assert not any("oof" in name.lower() or "candidate_search" in name.lower() for name in EXPECTED)
    assert "population_receipt.json" in EXPECTED
    assert "comparison_deltas.csv" in EXPECTED


def test_alias_compact_validator_rejects_extra_oof_file(tmp_path: Path) -> None:
    for name in EXPECTED:
        (tmp_path / name).write_text("{}\n" if name.endswith(".json") else "x\n", encoding="utf-8")
    (tmp_path / "oof_predictions.csv").write_text("sample_index\n1\n", encoding="utf-8")
    with pytest.raises(HRTargetAliasCompactExportV4Error, match="closed-world"):
        validate_hr_target_alias_compact_v4(tmp_path)
