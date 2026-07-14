from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.experiments import manuscript_calibration as calibration


FIXED_PARAMETERS = {"max_depth": 2}
CANDIDATE_PARAMETERS = {"learning_rate": 0.1}
OUTER_TEST_IDS = set(range(5))
OUTER_TRAIN_IDS = set(range(5, 15))


class _FakeModel:
    def __init__(self, parameters: dict[str, object]):
        self._parameters = parameters

    def get_params(self, deep: bool = False) -> dict[str, object]:
        return dict(self._parameters)


class _FakePipeline:
    fit_indices: list[set[int]] = []
    prediction_indices: list[set[int]] = []

    def __init__(self, feature_columns: tuple[str, ...], model_seed: int):
        self.feature_names_in_ = np.asarray(feature_columns, dtype=object)
        self.classes_ = np.asarray([2, 3, 4], dtype=int)
        self.named_steps = {
            "preprocessor": SimpleNamespace(
                feature_names_in_=np.asarray(feature_columns, dtype=object)
            ),
            "model": _FakeModel(
                {
                    **FIXED_PARAMETERS,
                    **CANDIDATE_PARAMETERS,
                    "random_state": model_seed,
                    "n_jobs": 1,
                }
            ),
        }

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "_FakePipeline":
        assert features.index.equals(target.index)
        self.fit_indices.append(set(int(value) for value in features.index))
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.prediction_indices.append(set(int(value) for value in features.index))
        rows = []
        for sample_index in features.index:
            preferred = int(sample_index) % 3
            row = np.full(3, 0.1, dtype=float)
            row[preferred] = 0.8
            rows.append(row)
        return np.asarray(rows, dtype=float)


def _fixture() -> tuple[pd.DataFrame, pd.Series, SimpleNamespace]:
    sample_indices = list(range(15))
    outer_folds = [1] * 5 + [2, 3, 4, 5, 6, 7, 8, 9, 10, 2]
    target = pd.Series(
        [2 + (index % 3) for index in sample_indices],
        index=sample_indices,
        name="PerformanceRating",
        dtype=int,
    )
    features = pd.DataFrame(
        {
            "feature_numeric": np.arange(15, dtype=float),
            "feature_other": np.arange(15, dtype=float) / 10.0,
        },
        index=sample_indices,
    )
    outer = pd.DataFrame(
        {
            "sample_index": sample_indices,
            "outer_fold": outer_folds,
            "y_true": target.to_numpy(),
        }
    )
    inner = pd.DataFrame(
        {
            "outer_fold": [1] * len(OUTER_TRAIN_IDS),
            "inner_fold": [((value - 5) % 5) + 1 for value in sorted(OUTER_TRAIN_IDS)],
            "sample_index": sorted(OUTER_TRAIN_IDS),
            "y_true": target.loc[sorted(OUTER_TRAIN_IDS)].to_numpy(),
        }
    )
    selected = pd.DataFrame(
        [
            {
                "outer_fold": 1,
                "selected_candidate_index": 7,
                "fixed_parameters_json": json.dumps(FIXED_PARAMETERS),
                "selected_candidate_parameters_json": json.dumps(
                    CANDIDATE_PARAMETERS
                ),
            }
        ]
    )
    bundle = SimpleNamespace(
        folds=SimpleNamespace(outer_assignments=outer, inner_assignments=inner),
        selected_hyperparameters=selected,
        fold_models={
            1: SimpleNamespace(
                selected_candidate_index=7,
                sha256="d" * 64,
            )
        },
        identity=SimpleNamespace(fold_contract_hash="c" * 64),
        labels=(2, 3, 4),
    )
    return features, target, bundle


def _identity() -> dict[str, str]:
    values = {field: "a" * 64 for field in calibration.IDENTITY_FIELDS}
    values["run_id"] = "cross-fit-unit-test"
    return values


@pytest.fixture(autouse=True)
def _reset_fake_pipeline() -> None:
    _FakePipeline.fit_indices = []
    _FakePipeline.prediction_indices = []


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build(
        model_name: str,
        features: pd.DataFrame,
        *,
        fixed_parameters: dict,
        candidate_parameters: dict,
        random_state: int,
        forbidden_features: tuple[str, ...] | list[str],
    ) -> _FakePipeline:
        assert model_name == "xgboost"
        assert fixed_parameters == FIXED_PARAMETERS
        assert candidate_parameters == CANDIDATE_PARAMETERS
        assert set(features.index).isdisjoint(OUTER_TEST_IDS)
        assert tuple(forbidden_features) == ("forbidden",)
        return _FakePipeline(tuple(str(value) for value in features.columns), random_state)

    monkeypatch.setattr(calibration, "build_model_pipeline", fake_build)


def test_cross_fit_uses_five_inner_fits_and_never_touches_outer_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, target, bundle = _fixture()
    _patch_pipeline(monkeypatch)

    predictions, receipts = calibration.cross_fit_outer_training(
        features=features,
        target=target,
        bundle=bundle,
        outer_fold=1,
        forbidden_features=("forbidden",),
        model_seed=42,
        identity=_identity(),
    )

    assert len(_FakePipeline.fit_indices) == 5
    assert len(_FakePipeline.prediction_indices) == 5
    assert all(indices.isdisjoint(OUTER_TEST_IDS) for indices in _FakePipeline.fit_indices)
    assert all(
        indices.isdisjoint(OUTER_TEST_IDS)
        for indices in _FakePipeline.prediction_indices
    )
    assert set().union(*_FakePipeline.prediction_indices) == OUTER_TRAIN_IDS
    assert sum(len(indices) for indices in _FakePipeline.prediction_indices) == len(
        OUTER_TRAIN_IDS
    )
    assert set(predictions["sample_index"]) == OUTER_TRAIN_IDS
    assert predictions["sample_index"].is_unique
    assert set(predictions["sample_index"]).isdisjoint(OUTER_TEST_IDS)
    assert len(receipts) == 5
    assert not receipts["outer_test_used_for_fit"].any()
    assert (receipts["n_outer_test"] == len(OUTER_TEST_IDS)).all()
    assert (receipts["threadpool_limit"] == 1).all()


def test_outer_test_label_changes_cannot_change_crossfit_training_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, target, bundle = _fixture()
    _patch_pipeline(monkeypatch)
    baseline_predictions, baseline_receipts = calibration.cross_fit_outer_training(
        features=features,
        target=target,
        bundle=bundle,
        outer_fold=1,
        forbidden_features=("forbidden",),
        model_seed=42,
        identity=_identity(),
    )

    changed_target = target.copy()
    changed_target.loc[sorted(OUTER_TEST_IDS)] = [4, 4, 4, 2, 2]
    _FakePipeline.fit_indices = []
    _FakePipeline.prediction_indices = []
    changed_predictions, changed_receipts = calibration.cross_fit_outer_training(
        features=features,
        target=changed_target,
        bundle=bundle,
        outer_fold=1,
        forbidden_features=("forbidden",),
        model_seed=42,
        identity=_identity(),
    )

    pd.testing.assert_frame_equal(baseline_predictions, changed_predictions)
    pd.testing.assert_frame_equal(baseline_receipts, changed_receipts)


def test_outer_test_injected_into_inner_membership_fails_before_model_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, target, bundle = _fixture()
    tampered = bundle.folds.inner_assignments.copy()
    tampered.loc[tampered.index[0], "sample_index"] = min(OUTER_TEST_IDS)
    tampered.loc[tampered.index[0], "y_true"] = target.loc[min(OUTER_TEST_IDS)]
    bundle.folds.inner_assignments = tampered

    def unexpected_build(*args, **kwargs):  # pragma: no cover - failure sentinel
        raise AssertionError("model build must not occur")

    monkeypatch.setattr(calibration, "build_model_pipeline", unexpected_build)

    with pytest.raises(
        calibration.CalibrationContractError,
        match="membership does not equal the outer-training partition",
    ):
        calibration.cross_fit_outer_training(
            features=features,
            target=target,
            bundle=bundle,
            outer_fold=1,
            forbidden_features=("forbidden",),
            model_seed=42,
            identity=_identity(),
        )
