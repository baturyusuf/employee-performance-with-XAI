from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.experiments import manuscript_model_benchmark as benchmark
from src.experiments.shared_folds import generate_shared_folds
from src.models.oof_bootstrap import BootstrapProtocol
from src.models.canonical_models import CANONICAL_ESTIMATOR_PATHS
from src.utils.config_loader import load_config


def _grid(*, selection_metric: str | None = "macro_f1") -> dict:
    models = {}
    for model_name, count in benchmark.EXPECTED_CANDIDATE_COUNTS.items():
        fixed = {"n_jobs": 1} if model_name != "logistic_regression" else {}
        candidate_key = {
            "logistic_regression": "C",
            "random_forest": "max_depth",
            "lightgbm": "num_leaves",
            "xgboost": "max_depth",
        }[model_name]
        models[model_name] = {
            "display_name": model_name,
            "role": "baseline" if model_name != "xgboost" else "primary",
            "estimator": CANONICAL_ESTIMATOR_PATHS[model_name],
            "fixed_params": fixed,
            "candidates": [{candidate_key: index + 1} for index in range(count)],
        }
    return {
        "model_benchmark": {
            "schema_version": 2,
            "selection_metric": selection_metric,
            "baseline_gate_metric": selection_metric,
            "candidate_failure_policy": "fail_entire_stage",
            "tie_breaking": "highest_inner_mean_then_lowest_candidate_index",
            "models": models,
        }
    }


def _fixture():
    frame = pd.DataFrame(
        {
            "EmpNumber": [f"E{index:03d}" for index in range(18)],
            "numeric": np.arange(18, dtype=float),
            "category": ["a", "b", "c"] * 6,
            "PerformanceRating": [2, 3, 4] * 6,
        },
        index=range(18),
    )
    folds = generate_shared_folds(
        frame,
        target_column="PerformanceRating",
        id_column="EmpNumber",
        run_id="benchmark-unit",
        config_hash="a" * 64,
        scientific_input_hash="b" * 64,
        dataset_key="inx_primary",
        dataset_sha256="c" * 64,
        outer_splits=2,
        inner_splits=3,
        seed=42,
        inner_seed=43,
    )
    return frame[["numeric", "category"]], frame["PerformanceRating"], folds


class _FakePipeline:
    classes_ = np.asarray([2, 3, 4])

    def __init__(self, candidate_token: int, *, fail: bool = False):
        self.candidate_token = candidate_token
        self.fail = fail

    def fit(self, X, y):
        if self.fail:
            raise RuntimeError("intentional candidate failure")
        self.fitted_indices = tuple(X.index)
        return self

    def predict(self, X):
        return np.full(len(X), 3, dtype=int)

    def predict_proba(self, X):
        return np.tile([0.2, 0.6, 0.2], (len(X), 1))


def _fake_builder(
    model_name,
    training_features,
    *,
    fixed_parameters,
    candidate_parameters,
    random_state,
    forbidden_features,
):
    return _FakePipeline(int(next(iter(candidate_parameters.values()))))


def test_actual_grid_is_predeclared_but_metric_decision_blocks_real_execution() -> None:
    actual = load_config("configs/model_grid.yaml")
    with pytest.raises(benchmark.ModelBenchmarkError, match="selection_metric is pending"):
        benchmark.validate_benchmark_config(actual)

    decided = copy.deepcopy(actual)
    decided["model_benchmark"]["selection_metric"] = "macro_f1"
    decided["model_benchmark"]["baseline_gate_metric"] = "macro_f1"
    settings = benchmark.validate_benchmark_config(decided)
    assert set(settings["models"]) == set(benchmark.CANONICAL_MODEL_NAMES)
    assert {
        name: len(definition["candidates"])
        for name, definition in settings["models"].items()
    } == benchmark.EXPECTED_CANDIDATE_COUNTS


def test_pending_metric_preflight_blocks_before_data_or_output_access(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _data_access_forbidden(*args, **kwargs):
        raise AssertionError("data loader must not be reached while metric decision is pending")

    monkeypatch.setattr(benchmark, "load_canonical_dataset", _data_access_forbidden)
    output = tmp_path / "benchmark-output"
    with pytest.raises(benchmark.ModelBenchmarkError, match="selection_metric is pending"):
        benchmark.run(
            "configs/manuscript_final.yaml",
            model_grid_path="configs/model_grid.yaml",
            shared_folds_dir=tmp_path / "missing-folds",
            output_dir=output,
            run_id="blocked",
            config_hash="a" * 64,
            scientific_input_hash="b" * 64,
            model_grid_sha256=benchmark.sha256_file("configs/model_grid.yaml"),
        )
    assert not output.exists()


def test_model_grid_hash_mismatch_blocks_before_config_or_data_use(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _access_forbidden(*args, **kwargs):
        raise AssertionError("No config/data access is allowed after side-input hash mismatch")

    monkeypatch.setattr(benchmark, "load_config", _access_forbidden)
    monkeypatch.setattr(benchmark, "load_canonical_dataset", _access_forbidden)
    output = tmp_path / "benchmark-output"
    with pytest.raises(benchmark.ModelBenchmarkError, match="search-space hash"):
        benchmark.run(
            "configs/manuscript_final.yaml",
            model_grid_path="configs/model_grid.yaml",
            shared_folds_dir=tmp_path / "missing-folds",
            output_dir=output,
            run_id="blocked",
            config_hash="a" * 64,
            scientific_input_hash="b" * 64,
            model_grid_sha256="0" * 64,
        )
    assert not output.exists()


def test_candidate_selection_is_deterministic_and_uses_lowest_index_on_tie() -> None:
    assert benchmark.select_candidate_index([0.5, 0.7, 0.7], better_direction="higher") == 1
    assert benchmark.select_candidate_index([0.5, 0.2, 0.2], better_direction="lower") == 1


def test_exact_primary_feature_frame_removes_every_declared_exclusion() -> None:
    frame = pd.DataFrame(
        {
            "EmpNumber": ["E1", "E2"],
            "Age": [30, 40],
            "PerformanceRating": [2, 3],
            "safe_feature": [1.0, 2.0],
        }
    )
    features = benchmark.exact_primary_feature_frame(
        frame,
        excluded_features=["EmpNumber", "Age", "PerformanceRating"],
    )
    assert features.columns.tolist() == ["safe_feature"]
    with pytest.raises(benchmark.ModelBenchmarkError, match="absent from the dataset"):
        benchmark.exact_primary_feature_frame(
            frame,
            excluded_features=["EmpNumber", "missing", "PerformanceRating"],
        )


def test_fit_path_enters_a_single_thread_native_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[object] = []

    class _Limit:
        def __enter__(self):
            events.append("entered")

        def __exit__(self, exc_type, exc, traceback):
            events.append("exited")

    def _limits(*, limits):
        events.append(limits)
        return _Limit()

    class _Pipeline:
        def fit(self, X, y):
            events.append("fit")
            return self

    monkeypatch.setattr(benchmark, "threadpool_limits", _limits)
    fitted = benchmark._fit_pipeline_or_fail(
        _Pipeline(),
        pd.DataFrame({"x": [1.0, 2.0]}),
        pd.Series([2, 3]),
        context="thread-test",
    )
    assert isinstance(fitted, _Pipeline)
    assert events == [1, "entered", "fit", "exited"]


def test_nested_benchmark_uses_inner_validation_only_and_oof_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, target, folds = _fixture()
    observed_selection_sizes: list[int] = []
    original_metric = benchmark._metric_value

    def _metric_spy(metric, y_true, prediction, probability, labels, *, task_type):
        observed_selection_sizes.append(len(y_true))
        return original_metric(
            metric,
            y_true,
            prediction,
            probability,
            labels,
            task_type=task_type,
        )

    monkeypatch.setattr(benchmark, "build_model_pipeline", _fake_builder)
    monkeypatch.setattr(benchmark, "_metric_value", _metric_spy)
    result = benchmark.evaluate_nested_benchmark(
        features,
        target,
        folds,
        _grid(),
        labels=[2, 3, 4],
        run_id="benchmark-unit",
        config_hash="a" * 64,
        scientific_input_hash="b" * 64,
        random_state=42,
        forbidden_features=["EmpNumber", "PerformanceRating"],
        bootstrap_protocol=BootstrapProtocol(n_resamples=4, seed=42),
    )

    # Outer-test partitions contain nine samples; selection saw only the
    # three-sample inner validation partitions supplied by the fold contract.
    assert observed_selection_sizes
    assert set(observed_selection_sizes) == {3}
    assert result.candidate_search_results["outer_test_used_for_selection"].eq(False).all()
    assert result.selected_hyperparameters["selected_candidate_index"].eq(0).all()
    coverage = result.oof_predictions.groupby("model")["sample_index"].agg(["size", "nunique"])
    assert coverage["size"].eq(len(features)).all()
    assert coverage["nunique"].eq(len(features)).all()
    assert len(result.oof_predictions) == len(features) * len(benchmark.CANONICAL_MODEL_NAMES)
    assert result.oof_predictions.groupby(["model", "sample_index"]).size().eq(1).all()
    probability_columns = ["prob_class_2", "prob_class_3", "prob_class_4"]
    np.testing.assert_allclose(
        result.oof_predictions[probability_columns].sum(axis=1).to_numpy(),
        1.0,
    )
    assert result.baseline_gate["gate_metric"] == "macro_f1"


def test_any_candidate_or_inner_fold_failure_aborts_entire_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, target, folds = _fixture()

    def _failing_builder(
        model_name,
        training_features,
        *,
        fixed_parameters,
        candidate_parameters,
        random_state,
        forbidden_features,
    ):
        token = int(next(iter(candidate_parameters.values())))
        return _FakePipeline(token, fail=model_name == "logistic_regression" and token == 2)

    monkeypatch.setattr(benchmark, "build_model_pipeline", _failing_builder)
    with pytest.raises(benchmark.ModelBenchmarkError, match="intentional candidate failure"):
        benchmark.evaluate_nested_benchmark(
            features,
            target,
            folds,
            _grid(),
            labels=[2, 3, 4],
            run_id="benchmark-unit",
            config_hash="a" * 64,
            scientific_input_hash="b" * 64,
            random_state=42,
            bootstrap_protocol=BootstrapProtocol(n_resamples=2, seed=42),
        )


def test_candidate_count_or_model_family_drift_fails_closed() -> None:
    config = _grid()
    config["model_benchmark"]["models"]["logistic_regression"]["candidates"].pop()
    with pytest.raises(benchmark.ModelBenchmarkError, match="exactly 6 candidates"):
        benchmark.validate_benchmark_config(config)

    config = _grid()
    config["model_benchmark"]["models"]["catboost"] = config["model_benchmark"][
        "models"
    ]["logistic_regression"]
    with pytest.raises(benchmark.ModelBenchmarkError, match="exactly the approved models"):
        benchmark.validate_benchmark_config(config)


def test_estimator_path_and_every_merged_candidate_are_preflight_validated() -> None:
    config = _grid()
    config["model_benchmark"]["models"]["lightgbm"]["estimator"] = "different.Estimator"
    with pytest.raises(benchmark.ModelBenchmarkError, match="estimator must be"):
        benchmark.validate_benchmark_config(config)

    config = _grid()
    config["model_benchmark"]["models"]["xgboost"]["candidates"][0] = {
        "max_depth": 1,
        "random_state": 99,
    }
    with pytest.raises(benchmark.ModelBenchmarkError, match="candidate 0 fails estimator preflight"):
        benchmark.validate_benchmark_config(config)

    config = _grid()
    config["model_benchmark"]["models"]["lightgbm"]["candidates"][0] = {
        "unknown_scientific_knob": 1,
    }
    with pytest.raises(benchmark.ModelBenchmarkError, match="Unsupported parameters"):
        benchmark.validate_benchmark_config(config)

    config = _grid()
    config["model_benchmark"]["models"]["random_forest"]["candidates"][0] = {
        "n_jobs": 1,
    }
    with pytest.raises(benchmark.ModelBenchmarkError, match="cannot overwrite fixed parameters"):
        benchmark.validate_benchmark_config(config)
