from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from src.experiments import manuscript_calibration as calibration
from src.governance.manuscript_contract import canonical_config_hash
from src.utils.config_loader import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "manuscript_final.yaml"


def _settings() -> dict[str, object]:
    return load_config(CONFIG_PATH)["manuscript_final"]


def _training_evidence() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray([2, 3, 4] * 6, dtype=int)
    probabilities = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.15, 0.70, 0.15],
            [0.10, 0.25, 0.65],
            [0.55, 0.35, 0.10],
            [0.20, 0.60, 0.20],
            [0.15, 0.20, 0.65],
        ]
        * 3,
        dtype=float,
    )
    return probabilities, y_true


def test_complete_applicability_registry_does_not_expand_calibration_metrics() -> None:
    settings = _settings()

    assert calibration._configured_metrics(
        settings, calibration.PRIMARY_TASK
    ) == calibration.METRICS


def test_calibration_metric_must_remain_in_complete_applicability_registry() -> None:
    settings = _settings()
    task = settings["evaluation"]["metric_applicability"][calibration.PRIMARY_TASK]
    task["applicable"].remove("nll_log_loss")

    with pytest.raises(calibration.CalibrationContractError, match="missing_report_metrics"):
        calibration._configured_metrics(settings, calibration.PRIMARY_TASK)


def _parameter_frame(
    fitted: calibration.SigmoidCalibrator,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "class_label": item.class_label,
                "coefficient": item.coefficient,
                "intercept": item.intercept,
                "n_positive": item.n_positive,
                "n_negative": item.n_negative,
                "n_iter": item.n_iter,
                "calibration_seed": fitted.seed,
                "solver": fitted.solver,
                "regularization": fitted.regularization,
                "l1_ratio": fitted.l1_ratio,
                "C": fitted.c_value,
                "fit_intercept": fitted.fit_intercept,
                "max_iter": fitted.max_iter,
                "tol": fitted.tolerance,
                "probability_clip": fitted.probability_clip,
                "threadpool_limit": fitted.threadpool_limit,
                "training_probability_sha256": fitted.training_probability_sha256,
                "training_labels_sha256": fitted.training_labels_sha256,
                "sigmoid_parameter_sha256": fitted.parameter_sha256,
            }
            for item in fitted.class_parameters
        ]
    )


def test_reliability_bins_include_empty_bins_and_preserve_denominators() -> None:
    labels = [2, 3, 4]
    probability = np.asarray(
        [[0.0, 0.4, 0.6], [1.0, 0.0, 0.0], [0.2, 0.7, 0.1]],
        dtype=float,
    )
    identity = {
        "run_id": "run-1",
        "config_hash": "a" * 64,
        "scientific_input_hash": "b" * 64,
    }

    frame = pd.DataFrame(
        calibration.calibration_bin_rows(
            [4, 2, 3],
            probability,
            labels,
            run_id="run-1",
            config_hash="a" * 64,
            method="sigmoid",
            n_bins=10,
            identity=identity,
        )
    )

    assert len(frame) == 30
    assert frame.groupby("class_label")["n_samples"].sum().to_dict() == {
        2: 3,
        3: 3,
        4: 3,
    }
    assert set(frame["bin"]) == set(range(1, 11))
    assert (frame.loc[frame["bin_status"] == "empty", "n_samples"] == 0).all()
    assert frame.loc[
        frame["bin_status"] == "empty", "mean_predicted_probability"
    ].isna().all()
    assert frame["primary_method"].all()
    assert set(frame["scientific_input_hash"]) == {"b" * 64}


def test_sigmoid_fit_is_deterministic_and_parameter_rows_replay_exactly() -> None:
    probabilities, y_true = _training_evidence()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = calibration.fit_sigmoid_calibrator(
            probabilities,
            y_true,
            [2, 3, 4],
            seed=42,
        )
        repeated = calibration.fit_sigmoid_calibrator(
            probabilities,
            y_true,
            [2, 3, 4],
            seed=42,
        )

    assert caught == []
    reconstructed = calibration.calibrator_from_parameter_rows(
        _parameter_frame(first)
    )

    assert first.parameter_sha256 == repeated.parameter_sha256
    assert reconstructed.parameter_sha256 == first.parameter_sha256
    assert reconstructed.threadpool_limit == 1
    expected = calibration.apply_sigmoid_calibrator(first, probabilities)
    observed = calibration.apply_sigmoid_calibrator(reconstructed, probabilities)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(observed.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)


def test_parameter_tampering_is_rejected() -> None:
    probabilities, y_true = _training_evidence()
    fitted = calibration.fit_sigmoid_calibrator(
        probabilities,
        y_true,
        [2, 3, 4],
        seed=42,
    )
    rows = _parameter_frame(fitted)
    rows.loc[rows["class_label"] == 2, "coefficient"] += 0.01

    with pytest.raises(
        calibration.CalibrationContractError,
        match="parameter hash does not match",
    ):
        calibration.calibrator_from_parameter_rows(rows)


def test_persisted_probability_csv_uses_exact_round_trip_float_parsing(
    tmp_path: Path,
) -> None:
    probability = 0.05880365829597878
    path = tmp_path / "probability.csv"
    pd.DataFrame({"prob_class_2": [probability]}).to_csv(path, index=False)

    replayed = pd.read_csv(path, float_precision="round_trip").loc[
        0, "prob_class_2"
    ]

    assert replayed == probability
    validation_source = inspect.getsource(calibration._validate_persisted_outputs)
    assert 'float_precision="round_trip"' in validation_source


def test_populated_output_is_rejected_before_dataset_or_upstream_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "sigmoid_calibration"
    output.mkdir()
    (output / "historical.txt").write_text("preserve", encoding="utf-8")

    def unexpected_loader(*args, **kwargs):  # pragma: no cover - failure sentinel
        raise AssertionError("scientific input must not be accessed")

    monkeypatch.setattr(calibration, "load_canonical_dataset", unexpected_loader)

    with pytest.raises(
        calibration.CalibrationContractError,
        match="absent or an empty builder-owned directory",
    ):
        calibration.run(
            CONFIG_PATH,
            shared_folds_dir=tmp_path / "shared_folds",
            model_benchmarks_dir=tmp_path / "model_benchmarks",
            output_dir=output,
            run_id="unit-test",
            config_hash=canonical_config_hash(CONFIG_PATH),
            scientific_input_hash="b" * 64,
        )

    assert (output / "historical.txt").read_text(encoding="utf-8") == "preserve"


def test_metric_panel_accepts_percentile_interval_that_excludes_point() -> None:
    summary = pd.DataFrame(
        [
            {
                "method": method,
                "ordinal_mae_oof": 0.50,
                "ordinal_mae_ci_low": 0.30,
                "ordinal_mae_ci_high": 0.40,
            }
            for method in calibration.SYSTEM_ORDER
        ]
    )
    figure, axis = plt.subplots()
    try:
        calibration._metric_panel(axis, summary, ("ordinal_mae",), "test")
    finally:
        plt.close(figure)


def test_prepublication_revalidation_rejects_changed_upstream_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = SimpleNamespace(run_id="run", config_hash="a" * 64)
    original = SimpleNamespace(
        identity=identity,
        model_set_sha256="b" * 64,
        upstream_file_hashes={"oof_predictions": "c" * 64},
    )
    refreshed = SimpleNamespace(
        identity=identity,
        model_set_sha256="b" * 64,
        upstream_file_hashes={"oof_predictions": "d" * 64},
    )
    monkeypatch.setattr(calibration, "load_manuscript_config", lambda path: {})
    monkeypatch.setattr(calibration, "canonical_config_hash", lambda config: "a" * 64)
    monkeypatch.setattr(
        calibration,
        "load_canonical_dataset",
        lambda *args: SimpleNamespace(receipt={"actual_sha256": "e" * 64}),
    )
    monkeypatch.setattr(
        calibration,
        "read_xgboost_oof_artifacts",
        lambda *args, **kwargs: refreshed,
    )
    monkeypatch.setattr(calibration, "validate_xgboost_oof_replay", lambda *args, **kwargs: None)

    with pytest.raises(
        calibration.CalibrationContractError,
        match="changed during calibration",
    ):
        calibration._revalidate_upstreams_before_publish(
            config_path=CONFIG_PATH,
            shared_folds_dir="shared",
            model_benchmarks_dir="benchmarks",
            run_id="run",
            config_hash="a" * 64,
            scientific_input_hash="f" * 64,
            dataset_sha256="e" * 64,
            features=pd.DataFrame({"feature": [1.0]}, index=[0]),
            target=pd.Series([2], index=[0]),
            labels=(2, 3, 4),
            original_bundle=original,
        )
