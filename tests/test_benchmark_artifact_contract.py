from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest

from src.experiments.benchmark_artifact_contract import (
    BenchmarkArtifactContractError,
    XGBoostOOFArtifacts,
    read_xgboost_oof_artifacts,
    validate_xgboost_oof_replay,
)
from src.experiments.shared_folds import generate_shared_folds, write_shared_folds
from src.governance.manuscript_contract import sha256_file
from src.models.canonical_models import (
    CanonicalXGBClassifier,
    aligned_predict_proba,
    build_model_pipeline,
)


RUN_ID = "artifact-contract-test"
CONFIG_HASH = "a" * 64
SCIENTIFIC_INPUT_HASH = "b" * 64
LABELS = (2, 3, 4)
FIXED_PARAMETERS = {
    "n_estimators": 2,
    "learning_rate": 0.1,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "n_jobs": 1,
    "verbosity": 0,
}
CANDIDATE_PARAMETERS = {
    "max_depth": 1,
    "min_child_weight": 1.0,
    "class_weight": None,
}


@dataclass(frozen=True)
class _ArtifactFixture:
    shared_folds_dir: Path
    benchmark_dir: Path
    features: pd.DataFrame
    target: pd.Series


def _source_frame() -> pd.DataFrame:
    row_count = 60
    index = pd.Index(range(1000, 1000 + row_count), name="source_position")
    return pd.DataFrame(
        {
            "EmpNumber": [f"E{offset:04d}" for offset in range(row_count)],
            "numeric": [float((offset * 7) % 19) for offset in range(row_count)],
            "category": [f"group_{offset % 4}" for offset in range(row_count)],
            "PerformanceRating": [LABELS[offset % len(LABELS)] for offset in range(row_count)],
        },
        index=index,
    )


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _build_artifacts(tmp_path: Path) -> _ArtifactFixture:
    source = _source_frame()
    features = source[["numeric", "category"]].copy()
    target = source["PerformanceRating"].astype(int)
    folds = generate_shared_folds(
        source,
        target_column="PerformanceRating",
        id_column="EmpNumber",
        run_id=RUN_ID,
        config_hash=CONFIG_HASH,
        scientific_input_hash=SCIENTIFIC_INPUT_HASH,
        dataset_key="inx_primary",
        dataset_sha256="c" * 64,
        outer_splits=10,
        inner_splits=5,
        seed=42,
        inner_seed=43,
    )
    shared_dir = tmp_path / "shared_folds"
    benchmark_dir = tmp_path / "model_benchmarks"
    write_shared_folds(folds, shared_dir)
    benchmark_dir.mkdir(parents=True)

    pipeline = build_model_pipeline(
        "xgboost",
        features,
        fixed_parameters=FIXED_PARAMETERS,
        candidate_parameters=CANDIDATE_PARAMETERS,
        random_state=42,
    ).fit(features, target)
    prediction = np.asarray(pipeline.predict(features), dtype=int)
    probability = aligned_predict_proba(pipeline, features, labels=LABELS)
    transformed_names = tuple(
        str(value) for value in pipeline.named_steps["preprocessor"].get_feature_names_out()
    )
    identity = {
        "run_id": RUN_ID,
        "config_hash": CONFIG_HASH,
        "scientific_input_hash": SCIENTIFIC_INPUT_HASH,
        "fold_contract_hash": folds.contract["fold_contract_hash"],
    }
    baseline_gate = {
        **identity,
        "comparison_direction": "baseline_improvement_over_xgboost",
        "gate_metric": "macro_f1",
        "gate_triggered": False,
        "n_resamples": 5000,
        "resample_hash": "d" * 64,
        "trigger_rule": "point_estimate_gt_zero_and_paired_ci_low_gt_zero",
        "triggered_comparisons": [],
        "user_decision_required_if_triggered": True,
    }

    candidate_rows = []
    selected_rows = []
    model_rows = []
    lineage_rows = []
    for outer_fold in range(1, 11):
        candidate_rows.append(
            {
                **identity,
                "model": "xgboost",
                "outer_fold": outer_fold,
                "candidate_index": 0,
                "parameters_json": _json(CANDIDATE_PARAMETERS),
                "selected_by_protocol": True,
                "outer_test_used_for_selection": False,
            }
        )
        selected_rows.append(
            {
                **identity,
                "model": "xgboost",
                "outer_fold": outer_fold,
                "selected_candidate_index": 0,
                "selected_candidate_parameters_json": _json(CANDIDATE_PARAMETERS),
                "fixed_parameters_json": _json(FIXED_PARAMETERS),
                "outer_test_used_for_selection": False,
            }
        )
        relative = Path("models") / "xgboost" / f"outer_fold_{outer_fold:02d}.joblib"
        model_path = benchmark_dir / relative
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        model_rows.append(
            {
                **identity,
                "model": "xgboost",
                "outer_fold": outer_fold,
                "path": relative.as_posix(),
                "sha256": sha256_file(model_path),
                "size_bytes": model_path.stat().st_size,
            }
        )
        lineage_rows.extend(
            {
                **identity,
                "model": "xgboost",
                "outer_fold": outer_fold,
                "transformed_feature_index": feature_index,
                "transformed_feature_name": feature_name,
            }
            for feature_index, feature_name in enumerate(transformed_names)
        )

    outer = folds.outer_assignments.set_index("sample_index")
    oof_rows = []
    for row_position, sample_index in enumerate(features.index):
        outer_fold = int(outer.loc[sample_index, "outer_fold"])
        row = {
            **identity,
            "system_id": "xgboost",
            "model": "xgboost",
            "sample_index": int(sample_index),
            "outer_fold": outer_fold,
            "y_true": int(target.loc[sample_index]),
            "y_pred": int(prediction[row_position]),
            "selected_candidate_index": 0,
        }
        row.update(
            {
                f"prob_class_{label}": float(probability[row_position, label_index])
                for label_index, label in enumerate(LABELS)
            }
        )
        oof_rows.append(row)

    pd.DataFrame(candidate_rows).to_csv(
        benchmark_dir / "candidate_search_results.csv", index=False
    )
    pd.DataFrame(selected_rows).to_csv(
        benchmark_dir / "selected_hyperparameters.csv", index=False
    )
    pd.DataFrame(oof_rows).to_csv(benchmark_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(model_rows).to_csv(benchmark_dir / "fitted_model_index.csv", index=False)
    pd.DataFrame(lineage_rows).to_csv(
        benchmark_dir / "transformed_feature_lineage.csv", index=False
    )
    paired_rows = [
        {
            **identity,
            "comparison_id": f"{baseline}_minus_xgboost",
            "metric": "macro_f1",
            "improvement_oriented_difference": -0.02,
            "improvement_ci_low": -0.04,
            "gate_eligible": True,
            "gate_triggered": False,
            "resample_hash": baseline_gate["resample_hash"],
        }
        for baseline in ("logistic_regression", "random_forest", "lightgbm")
    ]
    pd.DataFrame(paired_rows).to_csv(
        benchmark_dir / "paired_model_differences.csv", index=False
    )
    (benchmark_dir / "baseline_xgboost_gate.json").write_text(
        json.dumps(baseline_gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (benchmark_dir / "stage_metadata.json").write_text(
        json.dumps(
            {
                **identity,
                "stage": "model_benchmarks",
                "status": "complete",
                "benchmark_schema_version": 3,
                "benchmark_protocol_name": "restrained_nested_tuning_v2_10x5",
                "selection_metric": "macro_f1",
                "selection_tie_break_metric": "quadratic_weighted_kappa",
                "models": ["xgboost"],
                "baseline_gate": baseline_gate,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return _ArtifactFixture(shared_dir, benchmark_dir, features, target)


def _read(fixture: _ArtifactFixture) -> XGBoostOOFArtifacts:
    return read_xgboost_oof_artifacts(
        fixture.shared_folds_dir,
        fixture.benchmark_dir,
        expected_run_id=RUN_ID,
        expected_config_hash=CONFIG_HASH,
        expected_scientific_input_hash=SCIENTIFIC_INPUT_HASH,
        expected_feature_columns=fixture.features.columns,
        expected_labels=LABELS,
    )


def _rewrite_csv(path: Path, update) -> None:
    frame = pd.read_csv(path)
    update(frame)
    frame.to_csv(path, index=False)


def test_reader_loads_ten_exact_hash_bound_xgboost_pipelines(tmp_path: Path) -> None:
    fixture = _build_artifacts(tmp_path)
    artifacts = _read(fixture)

    assert artifacts.identity.run_id == RUN_ID
    assert artifacts.labels == LABELS
    assert artifacts.raw_feature_order == tuple(fixture.features.columns)
    assert set(artifacts.fold_models) == set(range(1, 11))
    assert len(artifacts.oof_predictions) == len(fixture.features)
    assert artifacts.oof_predictions["sample_index"].is_unique
    assert len(artifacts.model_set_sha256) == 64
    assert set(artifacts.upstream_file_hashes) == {
        "baseline_xgboost_gate",
        "benchmark_stage_metadata",
        "candidate_search_results",
        "fitted_model_index",
        "oof_predictions",
        "paired_model_differences",
        "selected_hyperparameters",
        "shared_fold_contract",
        "shared_inner_assignments",
        "shared_outer_assignments",
        "transformed_feature_lineage",
    }
    for outer_fold, model in artifacts.fold_models.items():
        assert model.outer_fold == outer_fold
        assert not model.path.is_absolute()
        assert model.path.as_posix().startswith("models/xgboost/")
        assert isinstance(model.pipeline.named_steps["model"], CanonicalXGBClassifier)
        assert model.selected_candidate_index == 0
        assert model.test_sample_indices
        assert model.transformed_feature_names
    with pytest.raises(TypeError):
        artifacts.fold_models[1] = artifacts.fold_models[1]  # type: ignore[index]


def test_model_set_hash_is_deterministic_and_replay_never_fits(tmp_path: Path) -> None:
    fixture = _build_artifacts(tmp_path)
    first = _read(fixture)
    second = _read(fixture)

    assert first.model_set_sha256 == second.model_set_sha256
    with patch.object(
        CanonicalXGBClassifier,
        "fit",
        side_effect=AssertionError("replay must never fit"),
    ):
        result = validate_xgboost_oof_replay(
            first,
            fixture.features,
            fixture.target,
            labels=LABELS,
        )
    assert result is None


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("identity", "identity"),
        ("missing_fold", "outer folds 1..10"),
        ("path_escape", "non-portable"),
        ("model_bytes", "hash/size"),
        ("selected_index", "selected candidate indices"),
        ("oof_fold", "folds differ"),
        ("oof_probability", "normalized probabilities"),
        ("lineage", "lineage differs"),
        ("model_classes", "model classes"),
        ("selected_parameters", "parameter 'max_depth' differs"),
        ("gate_true", "gate_triggered must be False"),
        ("paired_gate", "point-plus-CI rule"),
    ],
)
def test_reader_fails_closed_on_cross_artifact_tampering(
    tmp_path: Path,
    defect: str,
    message: str,
) -> None:
    fixture = _build_artifacts(tmp_path)
    selected_path = fixture.benchmark_dir / "selected_hyperparameters.csv"
    oof_path = fixture.benchmark_dir / "oof_predictions.csv"
    model_index_path = fixture.benchmark_dir / "fitted_model_index.csv"
    lineage_path = fixture.benchmark_dir / "transformed_feature_lineage.csv"

    if defect == "identity":
        _rewrite_csv(selected_path, lambda frame: frame.__setitem__("config_hash", "d" * 64))
    elif defect == "missing_fold":
        _rewrite_csv(
            model_index_path,
            lambda frame: frame.drop(frame.index[-1], inplace=True),
        )
    elif defect == "path_escape":
        _rewrite_csv(
            model_index_path,
            lambda frame: frame.loc.__setitem__((frame.index[0], "path"), "../escape.joblib"),
        )
    elif defect == "model_bytes":
        first_model = fixture.benchmark_dir / pd.read_csv(model_index_path).iloc[0]["path"]
        with first_model.open("ab") as stream:
            stream.write(b"tamper")
    elif defect == "selected_index":
        _rewrite_csv(
            oof_path,
            lambda frame: frame.loc.__setitem__((frame.index[0], "selected_candidate_index"), 1),
        )
    elif defect == "oof_fold":
        def _change_fold(frame: pd.DataFrame) -> None:
            current = int(frame.loc[0, "outer_fold"])
            frame.loc[0, "outer_fold"] = 1 if current != 1 else 2

        _rewrite_csv(oof_path, _change_fold)
    elif defect == "oof_probability":
        _rewrite_csv(
            oof_path,
            lambda frame: frame.loc.__setitem__((frame.index[0], "prob_class_2"), 0.9),
        )
    elif defect == "lineage":
        _rewrite_csv(
            lineage_path,
            lambda frame: frame.loc.__setitem__(
                (frame.index[0], "transformed_feature_name"), "numeric__different"
            ),
        )
    elif defect == "selected_parameters":
        def _change_parameters(frame: pd.DataFrame) -> None:
            changed = dict(CANDIDATE_PARAMETERS)
            changed["max_depth"] = 9
            frame["selected_candidate_parameters_json"] = _json(changed)

        _rewrite_csv(selected_path, _change_parameters)
        candidate_path = fixture.benchmark_dir / "candidate_search_results.csv"
        _rewrite_csv(
            candidate_path,
            lambda frame: frame.__setitem__(
                "parameters_json", _json({**CANDIDATE_PARAMETERS, "max_depth": 9})
            ),
        )
    elif defect == "gate_true":
        gate_path = fixture.benchmark_dir / "baseline_xgboost_gate.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        gate["gate_triggered"] = True
        gate["triggered_comparisons"] = ["lightgbm_minus_xgboost"]
        gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        metadata_path = fixture.benchmark_dir / "stage_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["baseline_gate"] = gate
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif defect == "paired_gate":
        paired_path = fixture.benchmark_dir / "paired_model_differences.csv"
        _rewrite_csv(
            paired_path,
            lambda frame: frame.loc.__setitem__((frame.index[0], "gate_triggered"), True),
        )
    else:
        index = pd.read_csv(model_index_path)
        first_path = fixture.benchmark_dir / index.loc[0, "path"]
        pipeline = joblib.load(first_path)
        pipeline.named_steps["model"].classes_ = np.asarray([2, 3])
        joblib.dump(pipeline, first_path)
        index.loc[0, "sha256"] = sha256_file(first_path)
        index.loc[0, "size_bytes"] = first_path.stat().st_size
        index.to_csv(model_index_path, index=False)

    with pytest.raises(BenchmarkArtifactContractError, match=message):
        _read(fixture)


def test_reader_rejects_wrong_declared_raw_order_or_labels(tmp_path: Path) -> None:
    fixture = _build_artifacts(tmp_path)

    with pytest.raises(BenchmarkArtifactContractError, match="raw feature order"):
        read_xgboost_oof_artifacts(
            fixture.shared_folds_dir,
            fixture.benchmark_dir,
            expected_run_id=RUN_ID,
            expected_config_hash=CONFIG_HASH,
            expected_scientific_input_hash=SCIENTIFIC_INPUT_HASH,
            expected_feature_columns=["category", "numeric"],
            expected_labels=LABELS,
        )
    with pytest.raises(BenchmarkArtifactContractError, match="target-label order"):
        read_xgboost_oof_artifacts(
            fixture.shared_folds_dir,
            fixture.benchmark_dir,
            expected_run_id=RUN_ID,
            expected_config_hash=CONFIG_HASH,
            expected_scientific_input_hash=SCIENTIFIC_INPUT_HASH,
            expected_feature_columns=fixture.features.columns,
            expected_labels=[4, 3, 2],
        )


def test_replay_rejects_target_feature_oof_or_post_read_model_tampering(tmp_path: Path) -> None:
    fixture = _build_artifacts(tmp_path)
    artifacts = _read(fixture)

    changed_target = fixture.target.copy()
    changed_target.iloc[0] = 4 if int(changed_target.iloc[0]) != 4 else 2
    with pytest.raises(BenchmarkArtifactContractError, match="target values differ"):
        validate_xgboost_oof_replay(
            artifacts,
            fixture.features,
            changed_target,
            labels=LABELS,
        )
    with pytest.raises(BenchmarkArtifactContractError, match="raw feature order"):
        validate_xgboost_oof_replay(
            artifacts,
            fixture.features[["category", "numeric"]],
            fixture.target,
            labels=LABELS,
        )

    artifacts.oof_predictions.loc[0, "y_pred"] = (
        4 if int(artifacts.oof_predictions.loc[0, "y_pred"]) != 4 else 2
    )
    with pytest.raises(BenchmarkArtifactContractError, match="prediction mismatches"):
        validate_xgboost_oof_replay(
            artifacts,
            fixture.features,
            fixture.target,
            labels=LABELS,
        )

    artifacts = _read(fixture)
    first_model = artifacts.fold_models[1]
    with (fixture.benchmark_dir / first_model.path).open("ab") as stream:
        stream.write(b"post-read tamper")
    with pytest.raises(BenchmarkArtifactContractError, match="bytes changed"):
        validate_xgboost_oof_replay(
            artifacts,
            fixture.features,
            fixture.target,
            labels=LABELS,
        )
