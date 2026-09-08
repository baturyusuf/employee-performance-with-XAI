from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.selection_objective_sensitivity_v4 import (
    MODEL_NAMES,
    _comparison_tables,
    _empirical_prior,
    build_selection_schedule,
)


def _candidate_fixture() -> pd.DataFrame:
    rows = []
    for model in MODEL_NAMES:
        for outer_fold in range(1, 11):
            rows.extend(
                [
                    {
                        "outer_fold": outer_fold,
                        "model": model,
                        "candidate_index": 0,
                        "parameters_json": '{"candidate":0}',
                        "inner_macro_f1_mean": 0.7000,
                        "inner_qwk_mean": 0.6000,
                        "n_inner_folds": 5,
                        "candidate_status": "complete",
                        "outer_test_used_for_selection": False,
                        "historical_selected_by_protocol": True,
                        "evidence_source": "fixture",
                    },
                    {
                        "outer_fold": outer_fold,
                        "model": model,
                        "candidate_index": 1,
                        "parameters_json": '{"candidate":1}',
                        "inner_macro_f1_mean": 0.6980,
                        "inner_qwk_mean": 0.8000,
                        "n_inner_folds": 5,
                        "candidate_status": "complete",
                        "outer_test_used_for_selection": False,
                        "historical_selected_by_protocol": False,
                        "evidence_source": "fixture",
                    },
                ]
            )
    return pd.DataFrame(rows)


def test_contract_prespecifies_separate_nonbinary_ranking_outputs() -> None:
    contract = json.loads(
        Path("configs/selection_objective_sensitivity_v4.json").read_text(encoding="utf-8")
    )
    ranking = contract["ranking_report"]
    assert ranking["composite_material_dependence_flag_allowed"] is False
    assert ranking["separate_fields"] == [
        "leader_changed",
        "full_ordering_changed",
        "selected_candidate_changes",
        "metric_effect_magnitudes",
    ]
    assert contract["selection_regimes"]["qwk"]["practical_tie_tolerance"] == 0.001
    assert contract["empirical_prior_baseline"]["enters_hyperparameter_selection"] is False


def test_schedule_swaps_primary_and_tie_break_and_reports_every_fold() -> None:
    schedule, changes = build_selection_schedule(_candidate_fixture())
    assert len(schedule) == 120
    assert len(changes) == 60
    assert schedule["outer_test_used_for_selection"].eq(False).all()
    assert schedule.loc[
        schedule["selection_objective"] == "macro_f1", "selected_candidate_index"
    ].eq(0).all()
    assert schedule.loc[
        schedule["selection_objective"] == "qwk", "selected_candidate_index"
    ].eq(1).all()
    assert changes["selected_candidate_changed"].eq(True).all()


def test_schedule_inclusive_primary_pool_uses_secondary_metric() -> None:
    evidence = _candidate_fixture()
    evidence.loc[evidence["candidate_index"] == 1, "inner_macro_f1_mean"] = 0.699
    schedule, _ = build_selection_schedule(evidence)
    assert schedule.loc[
        schedule["selection_objective"] == "macro_f1", "selected_candidate_index"
    ].eq(1).all()
    assert np.allclose(
        schedule.loc[
            schedule["selection_objective"] == "macro_f1", "primary_gap_from_best"
        ],
        0.001,
    )


def test_schedule_uses_lowest_index_after_exact_primary_and_secondary_ties() -> None:
    evidence = _candidate_fixture()
    evidence["inner_macro_f1_mean"] = 0.7
    evidence["inner_qwk_mean"] = 0.6
    schedule, changes = build_selection_schedule(evidence)
    assert schedule["selected_candidate_index"].eq(0).all()
    assert changes["selected_candidate_changed"].eq(False).all()


def test_empirical_prior_uses_complement_of_each_outer_test_fold() -> None:
    sample_index = np.arange(30)
    target = pd.Series(np.tile([2, 3, 4], 10), index=sample_index)
    outer = pd.DataFrame(
        {
            "sample_index": sample_index,
            "outer_fold": np.repeat(np.arange(1, 11), 3),
            "y_true": target.to_numpy(),
        }
    )
    parameters, predictions, metrics = _empirical_prior(target, outer)
    assert len(parameters) == 10
    assert parameters["outer_train_count"].eq(27).all()
    assert parameters["outer_test_count"].eq(3).all()
    assert parameters["derived_from_outer_training_labels_only"].eq(True).all()
    assert np.allclose(
        parameters[["prior_prob_class_2", "prior_prob_class_3", "prior_prob_class_4"]],
        1.0 / 3.0,
    )
    assert len(predictions) == 30
    assert not predictions["sample_index"].duplicated().any()
    assert {
        "nll_log_loss",
        "multiclass_brier",
        "ranked_probability_score",
        "ece_confidence",
    }.issubset(set(metrics["metric"]))


def test_ranking_outputs_keep_leader_and_full_ordering_changes_separate() -> None:
    rows = []
    for index, model in enumerate(MODEL_NAMES):
        rows.append(
            {"selection_objective": "macro_f1", "model": model, "metric": "macro_f1", "value": 1.0 - index / 10}
        )
        qwk_value = 1.0 - index / 10
        if index == 4:
            qwk_value = 0.49
        if index == 5:
            qwk_value = 0.50
        rows.append(
            {"selection_objective": "qwk", "model": model, "metric": "macro_f1", "value": qwk_value}
        )
    effects, changes = _comparison_tables(pd.DataFrame(rows), {"macro_f1": "higher"})
    assert len(effects) == len(MODEL_NAMES)
    assert changes.loc[0, "leader_changed"] in (False, np.bool_(False))
    assert changes.loc[0, "full_ordering_changed"] in (True, np.bool_(True))
    assert "material" not in " ".join(changes.columns).lower()
    assert {"qwk_minus_macro_f1_selection", "absolute_effect_magnitude", "rank_position_change"}.issubset(effects.columns)
