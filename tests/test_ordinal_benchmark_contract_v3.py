from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.governance.ordinal_benchmark_contract_v3 import (
    DEFAULT_BENCHMARK_CONTRACT_PATH,
    EXPECTED_AGGREGATE_METRICS,
    OrdinalBenchmarkContractError,
    validate_ordinal_benchmark_contract_v3,
)


def _contract() -> dict:
    return json.loads(DEFAULT_BENCHMARK_CONTRACT_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "ordinal_benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_real_v3_ordinal_benchmark_contract_is_complete_and_fit_free() -> None:
    receipt = validate_ordinal_benchmark_contract_v3()
    assert receipt["status"] == "passed"
    assert receipt["ordered_labels"] == [2, 3, 4]
    assert receipt["nominal_model_count"] == 4
    assert receipt["ordinal_model_count"] == 2
    assert receipt["naive_baseline_count"] == 3
    assert receipt["ordinal_candidate_counts"] == {
        "proportional_odds_logistic": 6,
        "cumulative_threshold_xgboost": 8,
    }
    assert receipt["aggregate_metric_count"] == len(EXPECTED_AGGREGATE_METRICS) == 16
    assert receipt["model_fit_count"] == 0


def test_contract_binds_information_and_nominal_registry_hashes(tmp_path: Path) -> None:
    contract = _contract()
    contract["information_contract"]["sha256"] = "0" * 64
    with pytest.raises(OrdinalBenchmarkContractError, match="byte hash drifted"):
        validate_ordinal_benchmark_contract_v3(_write(tmp_path, contract))

    contract = _contract()
    contract["nominal_model_registry"]["sha256"] = "0" * 64
    with pytest.raises(OrdinalBenchmarkContractError, match="registry hash drifted"):
        validate_ordinal_benchmark_contract_v3(_write(tmp_path, contract))


def test_contract_binds_loader_and_acquisition_inputs_and_disables_download(tmp_path: Path) -> None:
    contract = _contract()
    contract["data_source"]["canonical_loader_config_sha256"] = "0" * 64
    with pytest.raises(OrdinalBenchmarkContractError, match="config_sha256.*drifted"):
        validate_ordinal_benchmark_contract_v3(_write(tmp_path, contract))

    contract = _contract()
    contract["data_source"]["automatic_download_allowed"] = True
    with pytest.raises(OrdinalBenchmarkContractError, match="automatic download"):
        validate_ordinal_benchmark_contract_v3(_write(tmp_path, contract))


def test_canonical_v2_comparison_sources_require_exact_identity_and_hash_shape(
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["canonical_v2_comparison_source"]["run_id"] = "different_run"
    with pytest.raises(OrdinalBenchmarkContractError, match="run identity"):
        validate_ordinal_benchmark_contract_v3(_write(tmp_path, contract))

    contract = _contract()
    contract["canonical_v2_comparison_source"]["fold_contract"]["sha256"] = "bad"
    with pytest.raises(OrdinalBenchmarkContractError, match="lowercase SHA-256"):
        validate_ordinal_benchmark_contract_v3(_write(tmp_path, contract))


@pytest.mark.parametrize(
    "section,key,value,message",
    [
        ("preprocessing", "outer_test_used_for_fit", True, "Outer-test preprocessing"),
        ("shared_nested_cv", "outer_test_usage", "selection", "outer_test_usage"),
        ("selection", "outer_test_used", True, "Outer test cannot"),
        ("selection", "baselines_enter_hyperparameter_selection", True, "cannot enter tuning"),
        ("xai_reference", "independent_of_predictive_ranking", False, "independent"),
        ("publication", "publish_employee_level_oof_rows", True, "cannot be published"),
    ],
)
def test_leakage_ranking_and_publication_boundaries_fail_closed(
    tmp_path: Path,
    section: str,
    key: str,
    value,
    message: str,
) -> None:
    contract = _contract()
    contract[section][key] = value
    with pytest.raises(OrdinalBenchmarkContractError, match=message):
        validate_ordinal_benchmark_contract_v3(_write(tmp_path, contract))


def test_exact_two_ordinal_models_and_three_baselines_are_required(tmp_path: Path) -> None:
    contract = _contract()
    del contract["ordinal_models"]["proportional_odds_logistic"]
    with pytest.raises(OrdinalBenchmarkContractError, match="two-model registry"):
        validate_ordinal_benchmark_contract_v3(_write(tmp_path, contract))

    contract = _contract()
    contract["naive_baselines"].pop()
    with pytest.raises(OrdinalBenchmarkContractError, match="baseline identity"):
        validate_ordinal_benchmark_contract_v3(_write(tmp_path, contract))


def test_ordinal_candidates_cannot_duplicate_or_override_fixed_params(tmp_path: Path) -> None:
    contract = _contract()
    contract["ordinal_models"]["proportional_odds_logistic"]["candidates"].append(
        copy.deepcopy(
            contract["ordinal_models"]["proportional_odds_logistic"]["candidates"][0]
        )
    )
    with pytest.raises(OrdinalBenchmarkContractError, match="duplicate candidates"):
        validate_ordinal_benchmark_contract_v3(_write(tmp_path, contract))

    contract = _contract()
    contract["ordinal_models"]["proportional_odds_logistic"]["candidates"][0][
        "max_iter"
    ] = 100
    with pytest.raises(OrdinalBenchmarkContractError, match="overwrites fixed"):
        validate_ordinal_benchmark_contract_v3(_write(tmp_path, contract))


def test_invalid_ordinal_hyperparameter_is_rejected_before_any_fit(tmp_path: Path) -> None:
    contract = _contract()
    contract["ordinal_models"]["cumulative_threshold_xgboost"]["fixed_params"][
        "n_jobs"
    ] = -1
    with pytest.raises(OrdinalBenchmarkContractError, match="n_jobs=1"):
        validate_ordinal_benchmark_contract_v3(_write(tmp_path, contract))


def test_rps_two_level_and_per_class_contracts_are_exact(tmp_path: Path) -> None:
    contract = _contract()
    metrics = contract["evaluation"]["aggregate_metrics"]
    assert "ranked_probability_score" in metrics
    assert "two_level_reversal_rate" in metrics
    assert "severe_error_rate" not in metrics

    contract["evaluation"]["aggregate_metrics"][-1] = "severe_error_rate"
    with pytest.raises(OrdinalBenchmarkContractError, match="exact ordered contract"):
        validate_ordinal_benchmark_contract_v3(_write(tmp_path, contract))


def test_validator_constructs_contracts_but_never_fits_models(monkeypatch) -> None:
    def _forbidden_fit(*args, **kwargs):
        raise AssertionError("contract validation must not fit a model")

    monkeypatch.setattr(
        "src.models.ordinal_models_v3.ProportionalOddsClassifier.fit", _forbidden_fit
    )
    monkeypatch.setattr(
        "src.models.ordinal_models_v3.CumulativeThresholdXGBClassifier.fit", _forbidden_fit
    )
    assert validate_ordinal_benchmark_contract_v3()["model_fit_count"] == 0
