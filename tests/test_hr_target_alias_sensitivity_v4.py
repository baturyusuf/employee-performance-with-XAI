from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.hr_target_alias_sensitivity_v4 import (
    build_comparison_deltas,
    identify_disagreement_indices,
)


def _contract() -> dict:
    return json.loads(Path("configs/hr_target_alias_sensitivity_v4.json").read_text(encoding="utf-8"))


def test_contract_freezes_matched_primary_and_fit_free_sample_removal_comparisons() -> None:
    contract = _contract()
    assert contract["comparisons"]["sample_removal_effect"] == {
        "left": "historical_canonical_311",
        "right": "restricted_canonical_309",
        "interpretation": "fit_free_population_change_only",
    }
    assert contract["comparisons"]["matched_refit_effect"] == {
        "left": "restricted_canonical_309",
        "right": "exclusion_refit_309",
        "interpretation": "primary_training_and_data_rule_sensitivity_on_identical_population",
    }
    assert contract["target"]["canonical_rule_changes"] is False
    assert contract["excluded_scope"] == ["sigmoid_calibration", "naive_baselines", "additional_repetitions"]


def test_disagreement_identification_uses_audited_text_to_code_mapping() -> None:
    raw = pd.DataFrame(
        {
            "PerformanceScore": ["Fully Meets", " PIP ", "Needs Improvement", "Exceeds"],
            "PerfScoreID": [3, 3, 2, 1],
        }
    )
    contract = _contract()
    contract["target"]["expected_disagreement_count"] = 2
    assert identify_disagreement_indices(raw, contract) == (1, 3)


def test_comparison_deltas_keep_population_change_and_refit_effect_separate() -> None:
    rows = []
    for metric, values in {
        "macro_f1": (0.60, 0.61, 0.64),
        "ordinal_mae": (0.20, 0.19, 0.18),
    }.items():
        for arm, value in zip(
            ("historical_canonical_311", "restricted_canonical_309", "exclusion_refit_309"), values
        ):
            rows.append({"arm": arm, "metric": metric, "value": value, "sample_count": 0})
    result = build_comparison_deltas(pd.DataFrame(rows), _contract())
    removal = result.loc[result["comparison_id"] == "sample_removal_effect"]
    refit = result.loc[result["comparison_id"] == "matched_refit_effect"]
    assert removal["evaluation_population_matched"].eq(False).all()
    assert refit["evaluation_population_matched"].eq(True).all()
    assert np.isclose(
        removal.loc[removal["metric"] == "macro_f1", "right_minus_left"].item(), 0.01
    )
    assert np.isclose(
        refit.loc[refit["metric"] == "macro_f1", "right_minus_left"].item(), 0.03
    )
