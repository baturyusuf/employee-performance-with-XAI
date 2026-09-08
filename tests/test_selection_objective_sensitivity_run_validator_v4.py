from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.governance.selection_objective_sensitivity_compact_export_v4 import (
    EXPECTED_FILES,
    SelectionObjectiveCompactExportV4Error,
    validate_selection_objective_sensitivity_compact_v4,
)
from src.governance.selection_objective_sensitivity_run_validator_v4 import _select


def test_independent_selector_uses_inclusive_pool_secondary_then_index() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_index": [0, 1, 2],
            "primary": [0.8, 0.799, 0.799],
            "secondary": [0.1, 0.9, 0.9],
        }
    )
    assert int(_select(candidates, "primary", "secondary")["candidate_index"]) == 1


def test_compact_validator_rejects_oof_file(tmp_path: Path) -> None:
    for name in EXPECTED_FILES:
        (tmp_path / name).write_text("{}\n" if name.endswith(".json") else "x\n", encoding="utf-8")
    (tmp_path / "oof_predictions.csv").write_text("sample_index\n1\n", encoding="utf-8")
    with pytest.raises(SelectionObjectiveCompactExportV4Error, match="closed-world"):
        validate_selection_objective_sensitivity_compact_v4(tmp_path)


def test_compact_expected_inventory_has_no_row_level_outputs() -> None:
    lowered = {name.lower() for name in EXPECTED_FILES}
    assert not any("oof" in name for name in lowered)
    assert "ranking_changes.csv" in EXPECTED_FILES
    assert "metric_effect_magnitudes.csv" in EXPECTED_FILES
    assert "empirical_prior_metrics.csv" in EXPECTED_FILES
