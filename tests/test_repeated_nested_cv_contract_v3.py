from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.governance.repeated_nested_cv_contract_v3 import (
    DEFAULT_REPEATED_CV_CONTRACT_PATH,
    RepeatedNestedCVContractError,
    validate_repeated_nested_cv_contract_v3,
)


def _contract() -> dict[str, object]:
    return json.loads(DEFAULT_REPEATED_CV_CONTRACT_PATH.read_text(encoding="utf-8"))


def _write_contract(tmp_path: Path, payload: dict[str, object]) -> Path:
    destination = tmp_path / "repeated_nested_cv_v3.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


def test_repeated_nested_cv_contract_is_complete_and_fit_free() -> None:
    receipt = validate_repeated_nested_cv_contract_v3()
    assert receipt["status"] == "passed"
    assert receipt["repetitions"] == 5
    assert receipt["outer_splits"] == receipt["inner_splits"] == 5
    assert receipt["outer_partition_count"] == 25
    assert receipt["planned_estimator_fit_calls"] == 5_725
    assert receipt["nominal_candidate_count"] == 30
    assert receipt["ordinal_candidate_count"] == 14
    assert receipt["tuned_model_count"] == 6
    assert receipt["total_system_count"] == 9
    assert receipt["priority_metric_count"] == 4
    assert receipt["model_fit_count"] == 0
    assert receipt["network_calls"] == receipt["paid_api_calls"] == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("repetitions", 10, "design.repetitions drifted"),
        ("outer_splits", 10, "design.outer_splits drifted"),
        ("different_fold_assignment_required_across_repetitions", False, "drifted"),
        (
            "outer_test_usage",
            "evaluation_and_selection",
            "design.outer_test_usage drifted",
        ),
    ],
)
def test_repeated_design_drift_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _contract()
    design = payload["design"]
    assert isinstance(design, dict)
    design[field] = value
    with pytest.raises(RepeatedNestedCVContractError, match=message):
        validate_repeated_nested_cv_contract_v3(_write_contract(tmp_path, payload))


def test_seed_schedule_cannot_be_repeated_or_result_selected(tmp_path: Path) -> None:
    payload = _contract()
    design = payload["design"]
    assert isinstance(design, dict)
    schedule = design["seed_schedule"]
    assert isinstance(schedule, list)
    schedule[4] = deepcopy(schedule[3])
    with pytest.raises(RepeatedNestedCVContractError, match="seed schedule drifted"):
        validate_repeated_nested_cv_contract_v3(_write_contract(tmp_path, payload))

    payload = _contract()
    selection = payload["selection"]
    assert isinstance(selection, dict)
    selection["seed_or_repetition_selected_from_results"] = True
    with pytest.raises(RepeatedNestedCVContractError, match="selection.*drifted"):
        validate_repeated_nested_cv_contract_v3(_write_contract(tmp_path, payload))


def test_v2_oof_cannot_replace_new_repeated_fits(tmp_path: Path) -> None:
    payload = _contract()
    registry = payload["model_registry"]
    assert isinstance(registry, dict)
    registry["canonical_v2_oof_reuse_allowed"] = True
    with pytest.raises(RepeatedNestedCVContractError, match="cannot stand in"):
        validate_repeated_nested_cv_contract_v3(_write_contract(tmp_path, payload))


def test_priority_metric_direction_and_interval_language_are_frozen(
    tmp_path: Path,
) -> None:
    payload = _contract()
    evaluation = payload["evaluation"]
    assert isinstance(evaluation, dict)
    directions = evaluation["metric_directions"]
    assert isinstance(directions, dict)
    directions["ordinal_mae"] = "higher"
    with pytest.raises(RepeatedNestedCVContractError, match="directions drifted"):
        validate_repeated_nested_cv_contract_v3(_write_contract(tmp_path, payload))

    payload = _contract()
    evaluation = payload["evaluation"]
    assert isinstance(evaluation, dict)
    evaluation["interval_interpretation"] = "confidence_interval"
    with pytest.raises(RepeatedNestedCVContractError, match="cannot be labelled"):
        validate_repeated_nested_cv_contract_v3(_write_contract(tmp_path, payload))


def test_bound_source_hash_drift_fails_before_execution(tmp_path: Path) -> None:
    payload = _contract()
    sources = payload["source_contracts"]
    assert isinstance(sources, dict)
    ordinal = sources["ordinal_benchmark"]
    assert isinstance(ordinal, dict)
    ordinal["sha256"] = "0" * 64
    with pytest.raises(RepeatedNestedCVContractError, match="Source hash drifted"):
        validate_repeated_nested_cv_contract_v3(_write_contract(tmp_path, payload))


def test_contract_rejects_unexpected_top_level_fields(tmp_path: Path) -> None:
    payload = _contract()
    payload["post_hoc_seed"] = 999
    with pytest.raises(RepeatedNestedCVContractError, match="top-level inventory"):
        validate_repeated_nested_cv_contract_v3(_write_contract(tmp_path, payload))
