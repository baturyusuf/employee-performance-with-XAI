from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.experiments import manuscript_model_benchmark as benchmark
from src.experiments.shared_folds import generate_shared_folds
from src.models.oof_bootstrap import BootstrapProtocol
from src.models.canonical_models import (
    ALIGNED_PROBABILITY_PROTOCOL,
    CANONICAL_ESTIMATOR_PATHS,
    COMMON_PREPROCESSOR_OUTPUT_CONTAINER,
)
from src.utils.config_loader import load_config


def _grid(
    *,
    selection_metric: str | None = "macro_f1",
    tie_break_metric: str | None = "quadratic_weighted_kappa",
    practical_tie_tolerance: float | None = 0.001,
    gate_metric: str | None = "macro_f1",
) -> dict:
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
            "schema_version": 3,
            "protocol_name": "restrained_nested_tuning_v2_10x5",
            "selection_metric": selection_metric,
            "selection_tie_break_metric": tie_break_metric,
            "primary_practical_tie_tolerance": practical_tie_tolerance,
            "baseline_gate_metric": gate_metric,
            "candidate_failure_policy": "fail_entire_stage",
            "tie_breaking": (
                "highest_macro_f1_within_0_001_then_highest_qwk_then_"
                "lowest_candidate_index"
            ),
            "models": models,
        }
    }


def _fixture():
    frame = pd.DataFrame(
        {
            "EmpNumber": [f"E{index:03d}" for index in range(30)],
            "numeric": np.arange(30, dtype=float),
            "category": ["a", "b", "c"] * 10,
            "PerformanceRating": [2, 3, 4] * 10,
        },
        index=range(30),
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
        inner_splits=5,
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


def test_actual_grid_freezes_the_predeclared_selection_and_gate_protocol() -> None:
    actual = load_config("configs/model_grid.yaml")
    settings = benchmark.validate_benchmark_config(actual)
    manuscript = load_config("configs/manuscript_final.yaml")
    nested = benchmark.validate_benchmark_manuscript_alignment(
        settings,
        manuscript["manuscript_final"],
    )
    assert settings["schema_version"] == benchmark.BENCHMARK_SCHEMA_VERSION
    assert settings["protocol_name"] == benchmark.BENCHMARK_PROTOCOL_NAME
    assert settings["selection_metric"] == benchmark.PRIMARY_SELECTION_METRIC
    assert settings["selection_tie_break_metric"] == benchmark.SELECTION_TIE_BREAK_METRIC
    assert settings["primary_practical_tie_tolerance"] == pytest.approx(0.001)
    assert settings["baseline_gate_metric"] == benchmark.BASELINE_GATE_METRIC
    assert nested["inner_splits"] == 5
    preprocessing = manuscript["manuscript_final"]["model"]["preprocessing"]
    assert preprocessing["output_container"] == COMMON_PREPROCESSOR_OUTPUT_CONTAINER
    assert preprocessing["probability_alignment"] == ALIGNED_PROBABILITY_PROTOCOL
    assert {"joblib", "threadpoolctl"}.issubset(
        manuscript["manuscript_final"]["provenance"]["package_names"]
    )
    assert set(settings["models"]) == set(benchmark.CANONICAL_MODEL_NAMES)
    assert {
        name: len(definition["candidates"])
        for name, definition in settings["models"].items()
    } == benchmark.EXPECTED_CANDIDATE_COUNTS


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version must be 3"),
        ("protocol_name", "restrained_nested_tuning_v1", "protocol_name must be"),
        ("selection_metric", "quadratic_weighted_kappa", "selection_metric must be"),
        ("selection_tie_break_metric", "macro_f1", "selection_tie_break_metric must be"),
        ("baseline_gate_metric", "quadratic_weighted_kappa", "strictly 'macro_f1'"),
    ],
)
def test_validator_rejects_selection_or_gate_protocol_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    config = _grid()
    config["model_benchmark"][field] = value
    with pytest.raises(benchmark.ModelBenchmarkError, match=message):
        benchmark.validate_benchmark_config(config)


@pytest.mark.parametrize("value", [None, True, float("nan"), -0.001, 1.001])
def test_validator_rejects_invalid_practical_tie_tolerance(value: object) -> None:
    config = _grid()
    config["model_benchmark"]["primary_practical_tie_tolerance"] = value
    with pytest.raises(benchmark.ModelBenchmarkError, match="finite numeric value"):
        benchmark.validate_benchmark_config(config)


def test_validator_rejects_practical_tie_tolerance_protocol_drift() -> None:
    config = _grid()
    config["model_benchmark"]["primary_practical_tie_tolerance"] = 0.002
    with pytest.raises(benchmark.ModelBenchmarkError, match="must remain exactly 0.001"):
        benchmark.validate_benchmark_config(config)

    config["model_benchmark"]["tie_breaking"] = (
        "highest_macro_f1_within_0_002_then_highest_qwk_then_lowest_candidate_index"
    )
    with pytest.raises(benchmark.ModelBenchmarkError, match="must remain exactly 0.001"):
        benchmark.validate_benchmark_config(config)


def test_invalid_selection_protocol_blocks_before_data_or_output_access(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = _grid(tie_break_metric=None)

    def _load_grid_only(*args, **kwargs):
        return invalid

    def _data_access_forbidden(*args, **kwargs):
        raise AssertionError("data loader must not be reached for an invalid selection protocol")

    monkeypatch.setattr(benchmark, "load_config", _load_grid_only)
    monkeypatch.setattr(benchmark, "load_canonical_dataset", _data_access_forbidden)
    output = tmp_path / "benchmark-output"
    with pytest.raises(benchmark.ModelBenchmarkError, match="selection_tie_break_metric"):
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


def test_run_blocks_manuscript_model_grid_protocol_mismatch_before_data_access(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_grid = load_config("configs/model_grid.yaml")
    manuscript = load_config("configs/manuscript_final.yaml")
    mismatched = copy.deepcopy(manuscript)
    mismatched["manuscript_final"]["model"]["nested_tuning"][
        "selection_tie_break_metric"
    ] = "macro_f1"

    def _load_config(path):
        if str(path).replace("\\", "/").endswith("configs/model_grid.yaml"):
            return model_grid
        return mismatched

    def _data_access_forbidden(*args, **kwargs):
        raise AssertionError("fold/data access must not follow a protocol mismatch")

    monkeypatch.setattr(benchmark, "load_config", _load_config)
    monkeypatch.setattr(benchmark, "read_shared_folds", _data_access_forbidden)
    monkeypatch.setattr(benchmark, "load_canonical_dataset", _data_access_forbidden)
    output = tmp_path / "benchmark-output"
    with pytest.raises(benchmark.ModelBenchmarkError, match="manuscript/model-grid benchmark mismatch"):
        benchmark.run(
            "configs/manuscript_final.yaml",
            model_grid_path="configs/model_grid.yaml",
            shared_folds_dir=tmp_path / "missing-folds",
            output_dir=output,
            run_id="blocked",
            config_hash=benchmark.canonical_config_hash(mismatched),
            scientific_input_hash="b" * 64,
            model_grid_sha256=benchmark.sha256_file("configs/model_grid.yaml"),
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_container", "numpy_dense"),
        ("probability_alignment", "accept_tolerance_without_normalization"),
    ],
)
def test_alignment_rejects_preprocessing_or_probability_contract_drift(
    field: str,
    value: str,
) -> None:
    settings = benchmark.validate_benchmark_config(load_config("configs/model_grid.yaml"))
    manuscript = load_config("configs/manuscript_final.yaml")
    manuscript["manuscript_final"]["model"]["preprocessing"][field] = value
    with pytest.raises(benchmark.ModelBenchmarkError, match="preprocessing contract mismatch"):
        benchmark.validate_benchmark_manuscript_alignment(
            settings,
            manuscript["manuscript_final"],
        )


def test_run_requires_canonical_ten_outer_fold_configuration_before_fold_access(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_grid = load_config("configs/model_grid.yaml")
    manuscript = load_config("configs/manuscript_final.yaml")
    mismatched = copy.deepcopy(manuscript)
    mismatched["manuscript_final"]["evaluation"]["cv"]["n_splits"] = 9

    def _load_config(path):
        if str(path).replace("\\", "/").endswith("configs/model_grid.yaml"):
            return model_grid
        return mismatched

    def _fold_or_data_access_forbidden(*args, **kwargs):
        raise AssertionError("fold/data access must not follow a 9-fold manuscript config")

    monkeypatch.setattr(benchmark, "load_config", _load_config)
    monkeypatch.setattr(benchmark, "read_shared_folds", _fold_or_data_access_forbidden)
    monkeypatch.setattr(benchmark, "load_canonical_dataset", _fold_or_data_access_forbidden)
    output = tmp_path / "benchmark-output"
    with pytest.raises(benchmark.ModelBenchmarkError, match="n_splits must be exactly 10"):
        benchmark.run(
            "configs/manuscript_final.yaml",
            model_grid_path="configs/model_grid.yaml",
            shared_folds_dir=tmp_path / "missing-folds",
            output_dir=output,
            run_id="blocked",
            config_hash=benchmark.canonical_config_hash(mismatched),
            scientific_input_hash="b" * 64,
            model_grid_sha256=benchmark.sha256_file("configs/model_grid.yaml"),
        )
    assert not output.exists()


def test_run_requires_persisted_ten_outer_fold_contract_before_data_access(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_grid = load_config("configs/model_grid.yaml")
    manuscript = load_config("configs/manuscript_final.yaml")

    def _load_config(path):
        if str(path).replace("\\", "/").endswith("configs/model_grid.yaml"):
            return model_grid
        return manuscript

    class _FoldStub:
        contract = {"outer_splits": 9, "inner_splits": 5}

    def _data_access_forbidden(*args, **kwargs):
        raise AssertionError("data access must not follow a 9-fold persisted contract")

    monkeypatch.setattr(benchmark, "load_config", _load_config)
    monkeypatch.setattr(benchmark, "read_shared_folds", lambda *args, **kwargs: _FoldStub())
    monkeypatch.setattr(benchmark, "load_canonical_dataset", _data_access_forbidden)
    output = tmp_path / "benchmark-output"
    with pytest.raises(benchmark.ModelBenchmarkError, match="outer_splits must be exactly 10"):
        benchmark.run(
            "configs/manuscript_final.yaml",
            model_grid_path="configs/model_grid.yaml",
            shared_folds_dir=tmp_path / "nine-folds",
            output_dir=output,
            run_id="blocked",
            config_hash=benchmark.canonical_config_hash(manuscript),
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


def test_candidate_selection_uses_qwk_only_inside_the_primary_tolerance() -> None:
    primary = [0.8000, 0.7995, 0.7989]
    secondary = [0.50, 0.90, 1.00]
    assert benchmark.select_candidate_index(
        primary,
        secondary,
        practical_tie_tolerance=0.001,
    ) == 1


def test_candidate_selection_includes_the_numeric_tolerance_boundary() -> None:
    assert benchmark.select_candidate_index(
        [0.8, 0.799],
        [0.2, 0.9],
        practical_tie_tolerance=0.001,
    ) == 1


def test_candidate_selection_is_deterministic_and_uses_lowest_index_on_full_tie() -> None:
    inputs = ([0.5, 0.7, 0.7], [0.1, 0.9, 0.9])
    observed = [
        benchmark.select_candidate_index(
            *inputs,
            practical_tie_tolerance=0.001,
        )
        for _ in range(10)
    ]
    assert observed == [1] * 10
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

    # Outer-test partitions contain 15 samples; selection saw only the
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
    assert result.baseline_gate["trigger_rule"] == (
        "point_estimate_gt_zero_and_paired_ci_low_gt_zero"
    )
    eligible_metrics = result.paired_model_differences.loc[
        result.paired_model_differences["gate_eligible"].astype(bool), "metric"
    ]
    assert set(eligible_metrics) == {"macro_f1"}


def test_nested_selection_uses_secondary_qwk_only_for_primary_practical_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, target, folds = _fixture()
    probability_call_sizes: list[int] = []

    class _SelectionPipeline(_FakePipeline):
        def predict(self, X):
            token = self.candidate_token
            pattern = np.asarray(
                [2 + token % 3, 2 + (token // 3) % 3, 2 + (token // 9) % 3],
                dtype=int,
            )
            return np.resize(pattern, len(X))

        def predict_proba(self, X):
            probability_call_sizes.append(len(X))
            return np.tile([0.2, 0.6, 0.2], (len(X), 1))

    def _selection_builder(
        model_name,
        training_features,
        *,
        fixed_parameters,
        candidate_parameters,
        random_state,
        forbidden_features,
    ):
        return _SelectionPipeline(int(next(iter(candidate_parameters.values()))))

    def _controlled_metric(metric, y_true, prediction, probability, labels, *, task_type):
        assert probability is None
        token = sum((int(prediction[index]) - 2) * (3**index) for index in range(3))
        if metric == "macro_f1":
            return {1: 0.8000, 2: 0.7995}.get(token, 0.7900)
        if metric == "quadratic_weighted_kappa":
            return {1: 0.5000, 2: 0.9000}.get(token, 0.9900)
        raise AssertionError(f"unexpected inner-selection metric: {metric}")

    monkeypatch.setattr(benchmark, "build_model_pipeline", _selection_builder)
    monkeypatch.setattr(benchmark, "_metric_value", _controlled_metric)
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
        bootstrap_protocol=BootstrapProtocol(n_resamples=4, seed=42),
    )

    assert result.selected_hyperparameters["selected_candidate_index"].eq(1).all()
    assert result.selected_hyperparameters["selection_metric"].eq("macro_f1").all()
    assert result.selected_hyperparameters["selection_tie_break_metric"].eq(
        "quadratic_weighted_kappa"
    ).all()
    assert result.selected_hyperparameters["primary_practical_tie_tolerance"].eq(0.001).all()
    assert probability_call_sizes
    assert 3 not in probability_call_sizes
    search = result.candidate_search_results
    candidate_two = search[search["candidate_index"].eq(1)]
    candidate_three = search[search["candidate_index"].eq(2)]
    assert candidate_two["within_primary_practical_tie"].eq(True).all()
    assert candidate_two["selected_by_protocol"].eq(True).all()
    assert candidate_three["within_primary_practical_tie"].eq(False).all()
    assert candidate_three["selected_by_protocol"].eq(False).all()


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
