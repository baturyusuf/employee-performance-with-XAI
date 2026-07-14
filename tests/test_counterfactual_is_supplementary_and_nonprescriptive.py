from __future__ import annotations

import copy
import inspect
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments import manuscript_counterfactual_search as search_stage
from src.governance.manuscript_contract import (
    ManuscriptConfigError,
    expected_counterfactual_search_contract,
    load_manuscript_config,
    validate_manuscript_config,
    canonical_config_hash,
)


class _Predictor:
    named_steps = {"model": SimpleNamespace(classes_=np.asarray([2, 3, 4]))}

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        success = pd.to_numeric(frame["TrainingTimesLastYear"]).to_numpy() >= 2
        return np.asarray(
            [[0.1, 0.8, 0.1] if value else [0.8, 0.1, 0.1] for value in success],
            dtype=float,
        )


def test_config_freezes_distinct_scopes_nested_budgets_and_supplementary_role() -> None:
    config = load_manuscript_config("configs/manuscript_final.yaml")
    settings = config["manuscript_final"]
    protocol = settings["counterfactuals"]
    assert protocol == expected_counterfactual_search_contract()
    assert "heuristic_counterfactual" in settings["evidence_scopes"]["supplementary"]["stages"]
    assert "heuristic_counterfactual" not in settings["evidence_scopes"]["core"]["stages"]
    scope_sets = [tuple(values) for values in protocol["candidate_feature_scopes"].values()]
    assert len(scope_sets) == len(set(scope_sets)) == 4
    assert "no_salary" not in protocol["candidate_feature_scopes"]
    bounds = [
        (row["max_prototypes"], row["max_features_changed"])
        for row in protocol["search_budgets"]
    ]
    assert bounds == [(50, 2), (100, 3), (250, 3)]


def test_config_drift_or_redundant_mode_fails_closed() -> None:
    config = load_manuscript_config("configs/manuscript_final.yaml")
    drifted = copy.deepcopy(config)
    drifted["manuscript_final"]["counterfactuals"]["search_budgets"][1][
        "max_prototypes"
    ] = 49
    with pytest.raises(ManuscriptConfigError, match="heuristic-search contract"):
        validate_manuscript_config(drifted)


def test_nested_budgets_share_one_candidate_pool_and_preserve_inclusion() -> None:
    training = pd.DataFrame(
        {
            "TrainingTimesLastYear": [1, 2, 3],
            "EmpJobInvolvement": [1, 2, 3],
        }
    )
    sample = training.iloc[0]
    budgets = [
        {
            "budget_id": "restricted",
            "role": "sensitivity",
            "max_prototypes": 1,
            "max_features_changed": 1,
        },
        {
            "budget_id": "expanded",
            "role": "sensitivity",
            "max_prototypes": 2,
            "max_features_changed": 2,
        },
    ]
    results = search_stage.find_search_scenarios(
        _Predictor(),
        sample,
        training.iloc[1:],
        list(training.columns),
        search_stage.training_scales(training),
        [2, 3, 4],
        3,
        np.asarray([0.8, 0.1, 0.1]),
        {
            "TrainingTimesLastYear": {"control_type": "employee_controllable"},
            "EmpJobInvolvement": {"control_type": "employee_controllable"},
        },
        search_budgets=budgets,
    )
    assert results["restricted"]["candidates_within_budget"] <= results["expanded"][
        "candidates_within_budget"
    ]
    assert results["restricted"]["actual_probability_evaluations"] == results[
        "expanded"
    ]["actual_probability_evaluations"]
    assert results["expanded"]["search_success"] is True
    assert int(results["restricted"]["search_success"]) <= int(
        results["expanded"]["search_success"]
    )


def test_production_module_and_outputs_use_nonprescriptive_search_terminology() -> None:
    source = inspect.getsource(search_stage).casefold()
    for prohibited in (
        "actionability",
        "validity_rate",
        "n_valid_counterfactuals",
        "intervention_mode",
        "actionability_summary.csv",
    ):
        assert prohibited not in source
    for required in (
        "heuristic_search_success_rate",
        "heuristic_search_budget_sensitivity.csv",
        "search_failure_reason",
        "candidate_pool_size",
        "normalized_search_cost",
    ):
        assert required in source


def test_bounded_production_path_writes_identity_bound_closed_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_manuscript_config("configs/manuscript_final.yaml")
    target = np.tile(np.asarray([2, 3, 4]), 10)
    X = pd.DataFrame(
        {
            "TrainingTimesLastYear": np.tile([1, 2, 3], 10),
            "EmpEnvironmentSatisfaction": np.tile([1, 2, 3], 10),
            "EmpJobRole": np.tile(["A", "B", "C"], 10),
            "TotalWorkExperienceInYears": np.tile([2, 4, 6], 10),
        }
    )
    frame = X.copy()
    frame["PerformanceRating"] = target
    monkeypatch.setattr(
        search_stage,
        "load_canonical_dataset",
        lambda *_args, **_kwargs: SimpleNamespace(
            frame=frame.copy(), receipt={"actual_sha256": "d" * 64}
        ),
    )
    monkeypatch.setattr(
        search_stage,
        "exact_policy_frame",
        lambda *_args, **_kwargs: (X.copy(), ["PerformanceRating"]),
    )
    monkeypatch.setattr(
        search_stage,
        "taxonomy_by_feature",
        lambda: {
            "TrainingTimesLastYear": {"control_type": "employee_controllable"},
            "EmpEnvironmentSatisfaction": {"control_type": "manager_controllable"},
            "EmpJobRole": {"control_type": "organisation_controllable"},
            "TotalWorkExperienceInYears": {"control_type": "immutable"},
        },
    )
    monkeypatch.setattr(
        search_stage,
        "_fit_supplementary_pipeline",
        lambda *_args, **_kwargs: _Predictor(),
    )
    output = tmp_path / "heuristic_counterfactual"
    paths = search_stage.run(
        "configs/manuscript_final.yaml",
        output_dir=output,
        run_id="bounded-test",
        config_hash=canonical_config_hash(config),
        scientific_input_hash="b" * 64,
        source_tree_hash="c" * 64,
        max_cases=2,
    )
    assert set(paths) == {
        "protocol",
        "oof_predictions",
        "fold_model_receipts",
        "feature_scopes",
        "by_case",
        "summary",
        "budget_sensitivity",
        "failures",
        "uncertainty",
        "examples",
        "interpretation",
        "inventory",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    inventory = json.loads(paths["inventory"].read_text(encoding="utf-8"))
    assert inventory["status"] == "complete"
    assert inventory["artifact_count"] == 11
    assert len(pd.read_csv(paths["summary"])) == 4
    assert len(pd.read_csv(paths["budget_sensitivity"])) == 12
    cases = pd.read_csv(paths["by_case"])
    predictions = pd.read_csv(paths["oof_predictions"])
    receipts = pd.read_csv(paths["fold_model_receipts"])
    assert receipts["model_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert receipts["model_set_sha256"].nunique() == 1
    assert cases["model_set_sha256"].nunique() == 1
    assert predictions["model_set_sha256"].nunique() == 1
    model_hash_by_fold = receipts.set_index("outer_fold")["model_sha256"]
    assert cases["source_outer_model_sha256"].equals(
        cases["outer_fold"].map(model_hash_by_fold)
    )
    assert predictions["source_outer_model_sha256"].equals(
        predictions["outer_fold"].map(model_hash_by_fold)
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths.values())
    for prohibited in ("actionability", "validity_rate", "intervention_mode"):
        assert prohibited not in combined.casefold()
