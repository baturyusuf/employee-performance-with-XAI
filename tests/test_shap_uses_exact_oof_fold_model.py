from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.experiments import manuscript_shap_evidence as shap_stage
from src.experiments.benchmark_artifact_contract import (
    BenchmarkArtifactIdentity,
    XGBoostFoldModel,
    XGBoostOOFArtifacts,
)
from src.experiments.shared_folds import SharedFoldArtifacts
from src.models.canonical_models import CanonicalXGBClassifier, build_common_preprocessor


CONFIG_HASH = "c" * 64
SCIENTIFIC_INPUT_HASH = "b" * 64
FOLD_HASH = "d" * 64
MODEL_SET_HASH = "e" * 64
DATASET_HASH = "a" * 64
RUN_ID = "shap-exact-fold-test"
POLICY = "no_salary_hike_no_attrition_no_department"
EXCLUDED = [
    "Age",
    "Gender",
    "MaritalStatus",
    "EmpDepartment",
    "EmpLastSalaryHikePercent",
    "Attrition",
    "EmpNumber",
    "PerformanceRating",
]


def _config() -> dict:
    return {
        "manuscript_final": {
            "feature_policies": {
                "primary_policy": POLICY,
                "definitions": {
                    POLICY: {
                        "excluded_features": list(EXCLUDED),
                        "role": "canonical_primary",
                        "audit_only": False,
                    }
                },
            },
            "target": {"column": "PerformanceRating", "labels": [2, 3, 4]},
            "evaluation": {"cv": {"n_splits": 10}},
            "shap": {
                "model_source": "model_benchmarks.persisted_selected_xgboost_outer_fold_pipelines",
                "model_refit_in_shap_stage": False,
                "oof_prediction_replay_required": True,
                "global_top_n": 2,
                "local_top_k": 2,
                "top_k_values": [1, 2],
                "stability": {
                    "top_k": 2,
                    "uncertainty_type": "descriptive_dependent_fold_pairs",
                    "confidence_interval_applicable": False,
                },
                "local": {"top_k_reason_codes": 2},
            },
        }
    }


def _data() -> pd.DataFrame:
    n_rows = 60
    return pd.DataFrame(
        {
            "Signal": np.linspace(-2.0, 2.0, n_rows),
            "Category": np.resize(np.asarray(["A", "B", "C"]), n_rows),
            "Age": np.resize(np.asarray([25, 35, 45]), n_rows),
            "Gender": np.resize(np.asarray(["Male", "Female"]), n_rows),
            "MaritalStatus": np.resize(np.asarray(["Single", "Married"]), n_rows),
            "EmpDepartment": np.resize(np.asarray(["Sales", "HR"]), n_rows),
            "EmpLastSalaryHikePercent": np.resize(np.asarray([10, 12, 15]), n_rows),
            "Attrition": np.resize(np.asarray(["Yes", "No"]), n_rows),
            "EmpNumber": [f"E{index:03d}" for index in range(n_rows)],
            "PerformanceRating": np.resize(np.asarray([2, 3, 4]), n_rows),
        }
    )


def _bundle(frame: pd.DataFrame) -> XGBoostOOFArtifacts:
    raw_features = frame[["Signal", "Category"]]
    preprocessor = build_common_preprocessor(raw_features).fit(raw_features)
    transformed_names = tuple(str(value) for value in preprocessor.get_feature_names_out())
    folds = np.resize(np.arange(1, 11), len(frame))
    predictions = frame["PerformanceRating"].to_numpy(dtype=int).copy()
    predictions[np.arange(len(frame)) % 7 == 0] = 3
    probabilities = np.full((len(frame), 3), 0.1, dtype=np.float64)
    for position, prediction in enumerate(predictions):
        probabilities[position, [2, 3, 4].index(int(prediction))] = 0.8

    outer = pd.DataFrame(
        {
            "sample_index": frame.index.astype(int),
            "outer_fold": folds,
            "y_true": frame["PerformanceRating"].astype(int),
        }
    )
    shared = SharedFoldArtifacts(
        outer_assignments=outer,
        inner_assignments=pd.DataFrame(),
        contract={
            "run_id": RUN_ID,
            "config_hash": CONFIG_HASH,
            "scientific_input_hash": SCIENTIFIC_INPUT_HASH,
            "fold_contract_hash": FOLD_HASH,
            "dataset_sha256": DATASET_HASH,
            "outer_splits": 10,
            "inner_splits": 5,
        },
    )
    identity = BenchmarkArtifactIdentity(
        run_id=RUN_ID,
        config_hash=CONFIG_HASH,
        scientific_input_hash=SCIENTIFIC_INPUT_HASH,
        fold_contract_hash=FOLD_HASH,
    )
    fold_models = {}
    lineage_rows = []
    model_rows = []
    selected_rows = []
    for outer_fold in range(1, 11):
        model_sha = f"{outer_fold:02d}" * 32
        classifier = SimpleNamespace(
            model_=SimpleNamespace(
                n_features_in_=len(transformed_names),
                fold_marker=float(outer_fold),
            )
        )
        pipeline = SimpleNamespace(
            named_steps={"preprocessor": preprocessor, "model": classifier}
        )
        test_ids = tuple(int(value) for value in frame.index[folds == outer_fold])
        fold_models[outer_fold] = XGBoostFoldModel(
            outer_fold=outer_fold,
            pipeline=pipeline,
            path=Path(f"models/xgboost/outer_fold_{outer_fold:02d}.joblib"),
            sha256=model_sha,
            size_bytes=100 + outer_fold,
            selected_candidate_index=outer_fold % 4,
            transformed_feature_names=transformed_names,
            test_sample_indices=test_ids,
        )
        model_rows.append({"outer_fold": outer_fold})
        selected_rows.append({"outer_fold": outer_fold})
        for index, name in enumerate(transformed_names):
            lineage_rows.append(
                {
                    "outer_fold": outer_fold,
                    "transformed_feature_index": index,
                    "transformed_feature_name": name,
                }
            )

    oof = pd.DataFrame(
        {
            "sample_index": frame.index.astype(int),
            "outer_fold": folds,
            "y_true": frame["PerformanceRating"].astype(int),
            "y_pred": predictions,
            "selected_candidate_index": [fold_models[int(fold)].selected_candidate_index for fold in folds],
            "prob_class_2": probabilities[:, 0],
            "prob_class_3": probabilities[:, 1],
            "prob_class_4": probabilities[:, 2],
        }
    )
    return XGBoostOOFArtifacts(
        identity=identity,
        folds=shared,
        oof_predictions=oof,
        selected_hyperparameters=pd.DataFrame(selected_rows),
        model_index=pd.DataFrame(model_rows),
        transformed_lineage=pd.DataFrame(lineage_rows),
        fold_models=MappingProxyType(fold_models),
        model_set_sha256=MODEL_SET_HASH,
        upstream_file_hashes=MappingProxyType({"fold_contract": "9" * 64}),
        baseline_gate=MappingProxyType({"gate_triggered": False}),
        labels=(2, 3, 4),
        raw_feature_order=("Signal", "Category"),
        benchmark_dir=Path("model_benchmarks"),
    )


def test_shap_stage_uses_each_exact_oof_fold_model_without_refit(
    tmp_path,
    monkeypatch,
) -> None:
    import shap

    frame = _data()
    bundle = _bundle(frame)
    reader_calls = []
    replay_calls = []

    def fake_reader(shared_folds_dir, model_benchmarks_dir, **kwargs):
        reader_calls.append((Path(shared_folds_dir), Path(model_benchmarks_dir), kwargs))
        return bundle

    def fake_replay(artifacts, features, target, **kwargs):
        replay_calls.append((artifacts, features.copy(), target.copy(), kwargs))

    class FakeTreeExplainer:
        def __init__(self, model):
            self.fold_marker = float(model.fold_marker)

        def shap_values(self, transformed):
            n_samples, n_features = transformed.shape
            values = np.empty((n_samples, n_features, 3), dtype=np.float64)
            for class_index in range(3):
                values[:, :, class_index] = self.fold_marker + class_index / 10.0
            return values

    def fail_fit(*args, **kwargs):
        raise AssertionError("The OOF SHAP stage must not fit any model or preprocessor")

    monkeypatch.setattr(shap_stage, "load_config", lambda path: _config())
    monkeypatch.setattr(shap_stage, "canonical_config_hash", lambda config: CONFIG_HASH)
    monkeypatch.setattr(
        shap_stage,
        "load_canonical_dataset",
        lambda path, key: SimpleNamespace(
            frame=frame.copy(), receipt={"actual_sha256": DATASET_HASH}
        ),
    )
    monkeypatch.setattr(shap_stage, "read_xgboost_oof_artifacts", fake_reader)
    monkeypatch.setattr(shap_stage, "validate_xgboost_oof_replay", fake_replay)
    monkeypatch.setattr(
        shap_stage,
        "taxonomy_by_feature",
        lambda: {
            feature: {
                "control_type": "diagnostic",
                "sensitive_or_proxy": False,
                "leakage_risk": "low",
                "allowed_for_final_model": True,
                "notes": "test taxonomy",
            }
            for feature in ("Signal", "Category")
        },
    )
    monkeypatch.setattr(shap, "TreeExplainer", FakeTreeExplainer)
    monkeypatch.setattr(Pipeline, "fit", fail_fit)
    monkeypatch.setattr(CanonicalXGBClassifier, "fit", fail_fit)

    failure_output = tmp_path / "failed_oof_shap"
    failure_output.mkdir()
    original_write_json = shap_stage.write_json

    def fail_after_staged_files(path, value, *, indent=2):
        if Path(path).name == "shap_metadata.json":
            raise RuntimeError("injected late metadata failure")
        return original_write_json(path, value, indent=indent)

    monkeypatch.setattr(shap_stage, "write_json", fail_after_staged_files)
    with pytest.raises(RuntimeError, match="injected late metadata failure"):
        shap_stage.run(
            "configs/manuscript_final.yaml",
            shared_folds_dir=tmp_path / "shared_folds",
            model_benchmarks_dir=tmp_path / "model_benchmarks",
            output_dir=failure_output,
            run_id=RUN_ID,
            config_hash=CONFIG_HASH,
            scientific_input_hash=SCIENTIFIC_INPUT_HASH,
        )
    assert failure_output.is_dir()
    assert not any(failure_output.iterdir())
    monkeypatch.setattr(shap_stage, "write_json", original_write_json)

    output = tmp_path / "oof_shap"
    output.mkdir()
    paths = shap_stage.run(
        "configs/manuscript_final.yaml",
        shared_folds_dir=tmp_path / "shared_folds",
        model_benchmarks_dir=tmp_path / "model_benchmarks",
        output_dir=output,
        run_id=RUN_ID,
        config_hash=CONFIG_HASH,
        scientific_input_hash=SCIENTIFIC_INPUT_HASH,
    )

    assert len(reader_calls) == 2
    assert all(call[2]["expected_feature_columns"] == ["Signal", "Category"] for call in reader_calls)
    assert all(call[2]["expected_labels"] == [2, 3, 4] for call in reader_calls)
    assert len(replay_calls) == 2
    assert all(
        call[3] == {"labels": [2, 3, 4], "probability_atol": 1e-12}
        for call in replay_calls
    )
    assert all(path.is_file() for path in paths.values())

    local = pd.read_csv(paths["local_values"], dtype={"model_sha256": str})
    assert set(local["outer_fold"]) == set(range(1, 11))
    expected_sha = {fold: f"{fold:02d}" * 32 for fold in range(1, 11)}
    assert all(
        set(group["model_sha256"]) == {expected_sha[int(fold)]}
        for fold, group in local.groupby("outer_fold")
    )
    assert {
        "run_id",
        "config_hash",
        "scientific_input_hash",
        "fold_contract_hash",
        "policy",
        "model",
        "model_set_sha256",
        "outer_fold",
        "model_sha256",
        "selected_candidate_index",
    }.issubset(local.columns)
    pairwise = pd.read_csv(paths["pairwise"])
    assert len(pairwise) == 90
    stability = pd.read_csv(paths["stability"])
    assert stability["n_fold_pairs"].eq(45).all()
    assert stability["confidence_interval_applicable"].eq(False).all()
    assert not any("ci_low" in column or "ci_high" in column for column in stability.columns)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["protocol"]["model_refit_in_shap_stage"] is False
    assert metadata["model_set_sha256"] == MODEL_SET_HASH
    assert all(not Path(value).is_absolute() for value in metadata["outputs"].values())
