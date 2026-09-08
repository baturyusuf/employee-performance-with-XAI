from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.governance.selection_objective_sensitivity_compact_export_v4 import (
    validate_selection_objective_sensitivity_compact_v4,
)


ROOT = Path("reports/research_log/major_revision_round2")
PACKAGE = ROOT / "selection_objective_sensitivity"


def test_round2_selection_compact_package_is_closed_and_valid() -> None:
    receipt = validate_selection_objective_sensitivity_compact_v4(PACKAGE)
    assert receipt["status"] == "passed"
    assert receipt["file_count"] == 12
    assert receipt["employee_level_oof_published"] is False


def test_selection_report_replays_key_rounded_values_and_separate_changes() -> None:
    report = (ROOT / "SELECTION_OBJECTIVE_SENSITIVITY.md").read_text(encoding="utf-8")
    aggregate = pd.read_csv(PACKAGE / "aggregate_metrics.csv")
    per_class = pd.read_csv(PACKAGE / "per_class_metrics.csv")
    candidate_changes = pd.read_csv(PACKAGE / "selected_candidate_changes.csv")
    rankings = pd.read_csv(PACKAGE / "ranking_changes.csv")
    assert int(candidate_changes["selected_candidate_changed"].sum()) == 37
    assert int(rankings["leader_changed"].sum()) == 6
    assert int(rankings["full_ordering_changed"].sum()) == 9
    xgb_qwk = aggregate.loc[
        (aggregate["selection_objective"] == "qwk")
        & (aggregate["model"] == "xgboost")
        & (aggregate["metric"] == "quadratic_weighted_kappa"),
        "value",
    ].item()
    xgb_class4 = per_class.loc[
        (per_class["selection_objective"] == "qwk")
        & (per_class["model_name"] == "xgboost")
        & (per_class["class_label"] == 4),
        "recall",
    ].item()
    assert f"{xgb_qwk:.6f}" in report
    assert f"{xgb_class4:.6f}" in report
    assert "one binary material-dependence classification" in report


def test_probability_report_replays_probability_metrics() -> None:
    report = (ROOT / "PROBABILITY_BASELINE_REPORT.md").read_text(encoding="utf-8")
    metrics = pd.read_csv(PACKAGE / "empirical_prior_metrics.csv").set_index("metric")["value"]
    for metric in (
        "nll_log_loss",
        "multiclass_brier",
        "ranked_probability_score",
        "ece_confidence",
    ):
        assert f"{metrics.loc[metric]:.6f}" in report
    assert "outer-training" in report
    assert "performs no model fit" in report


def test_extreme_class_report_retains_random_forest_zero() -> None:
    report = (ROOT / "PER_CLASS_EXTREME_CLASS_REPORT.md").read_text(encoding="utf-8")
    per_class = pd.read_csv(PACKAGE / "per_class_metrics.csv")
    row = per_class.loc[
        (per_class["selection_objective"] == "macro_f1")
        & (per_class["model_name"] == "random_forest")
        & (per_class["class_label"] == 4)
    ].iloc[0]
    assert row["recall"] == 0.0
    assert row["f1"] == 0.0
    assert "assigning no OOF case to rating 4" in report
    assert "0.000000 | 0.000000 | 0.000000 | 132" in report
