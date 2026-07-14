from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.experiments.leakage_safe_cv import LabelEncodedXGBClassifier
from src.experiments.hrdataset_replication_diagnostics import (
    ATTRIBUTION_WARNING,
    FoldModelReference,
    HRDatasetDiagnosticsError,
    ReplicationIdentity,
    SHAP_ADDITIVITY_OUTPUT_SPACE,
    SHAP_ATTRIBUTION_UNIT,
    ShapComputation,
    _aligned_probabilities,
    canonicalize_multiclass_shap,
    compute_exact_oof_grouped_shap,
    feature_policy_contract_sha256,
    model_set_sha256,
)
from src.experiments.manuscript_hrdataset_replication import _write_local_reason_codes
from src.models.canonical_models import aligned_predict_proba


LABELS = (2, 3, 4)
PRIMARY_POLICY = "hrdataset_leakage_aware_primary"


def test_fold_probability_replay_uses_the_canonical_normalized_simplex() -> None:
    class Float32LikeEstimator:
        classes_ = np.asarray(LABELS)

        def predict_proba(self, frame):
            assert len(frame) == 2
            return np.asarray(
                [
                    [0.10000000, 0.20000000, 0.69999994],
                    [0.24999999, 0.25000000, 0.50000000],
                ],
                dtype=np.float32,
            )

    replay = _aligned_probabilities(
        Float32LikeEstimator(),
        pd.DataFrame({"feature": [1.0, 2.0]}),
        LABELS,
    )
    assert replay.dtype == np.float64
    assert np.allclose(replay.sum(axis=1), 1.0, rtol=0.0, atol=np.finfo(np.float64).eps * 3)
    assert not np.array_equal(
        replay,
        Float32LikeEstimator().predict_proba(pd.DataFrame({"feature": [1.0, 2.0]})),
    )


def _pipeline(frame: pd.DataFrame, target: np.ndarray) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["numeric_feature"],
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ["category_feature"],
            ),
        ],
        sparse_threshold=0.0,
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(solver="lbfgs", max_iter=2000, random_state=17),
            ),
        ]
    )
    pipeline.fit(frame[["numeric_feature", "category_feature"]], target)
    return pipeline


def _xgboost_pipeline(frame: pd.DataFrame, target: np.ndarray) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["numeric_feature"],
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ["category_feature"],
            ),
        ],
        sparse_threshold=0.0,
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LabelEncodedXGBClassifier(
                    n_estimators=5,
                    max_depth=2,
                    learning_rate=0.1,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=17,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(frame[["numeric_feature", "category_feature"]], target)
    return pipeline


def _canonical_provider(
    pipeline: Pipeline,
    transformed: np.ndarray,
    labels: tuple[int, ...],
) -> ShapComputation:
    probabilities = pipeline.named_steps["model"].predict_proba(transformed)
    margins = np.log(np.clip(probabilities, 1e-12, 1.0))
    values = np.zeros((len(transformed), transformed.shape[1], len(labels)), dtype=float)
    values[:, 0, :] = margins
    return ShapComputation(
        values=values,
        base_values=np.zeros(len(labels), dtype=float),
        margins=margins,
        attribution_unit=SHAP_ATTRIBUTION_UNIT,
        additivity_output_space=SHAP_ADDITIVITY_OUTPUT_SPACE,
        axis_source="synthetic_canonical_sample_feature_class",
    )


def _fixture(pipeline_builder=_pipeline) -> dict[str, object]:
    n_samples = 18
    sample_ids = np.arange(100, 100 + n_samples)
    features = pd.DataFrame(
        {
            "sample_index": sample_ids,
            "numeric_feature": np.linspace(-2.0, 2.0, n_samples),
            "category_feature": np.where(np.arange(n_samples) % 2 == 0, "A", "B"),
        }
    )
    outer_fold = np.repeat([1, 2, 3], 6)
    folds = pd.DataFrame({"sample_index": sample_ids, "outer_fold": outer_fold})
    training_target = np.tile(np.asarray(LABELS, dtype=int), 6)
    references: list[FoldModelReference] = []
    prediction_rows: list[dict[str, object]] = []
    for fold in (1, 2, 3):
        train = outer_fold != fold
        test = outer_fold == fold
        pipeline = pipeline_builder(features.loc[train], training_target[train])
        buffer = io.BytesIO()
        joblib.dump(pipeline, buffer, compress=0, protocol=4)
        model_hash = hashlib.sha256(buffer.getvalue()).hexdigest()
        references.append(
            FoldModelReference(outer_fold=fold, model_sha256=model_hash, pipeline=pipeline)
        )
        X_test = features.loc[test, ["numeric_feature", "category_feature"]]
        probabilities = aligned_predict_proba(pipeline, X_test, labels=LABELS)
        predictions = np.asarray(LABELS, dtype=int)[np.argmax(probabilities, axis=1)]
        for offset, sample_position in enumerate(np.flatnonzero(test)):
            # Guarantee both correct and incorrect representative-case strata.
            true_label = (
                int(predictions[offset])
                if sample_position % 2 == 0
                else int(LABELS[(LABELS.index(int(predictions[offset])) + 1) % len(LABELS)])
            )
            prediction_rows.append(
                {
                    "sample_index": int(sample_ids[sample_position]),
                    "outer_fold": fold,
                    "policy": PRIMARY_POLICY,
                    "source_outer_model_sha256": model_hash,
                    "y_true": true_label,
                    "y_pred": int(predictions[offset]),
                    **{
                        f"prob_class_{label}": float(probabilities[offset, label_index])
                        for label_index, label in enumerate(LABELS)
                    },
                }
            )
    policies = {PRIMARY_POLICY: ["numeric_feature", "category_feature"]}
    identity = ReplicationIdentity(
        run_id="external-shap-test",
        config_hash="a" * 64,
        scientific_input_hash="b" * 64,
        dataset_sha256="c" * 64,
        schema_mapping_sha256="d" * 64,
        fold_contract_hash="e" * 64,
        feature_policy_contract_sha256=feature_policy_contract_sha256(policies),
        model_set_sha256=model_set_sha256(references),
    )
    predictions = pd.DataFrame(prediction_rows)
    for field_name, value in identity.as_dict().items():
        predictions[field_name] = value
    return {
        "features": features,
        "folds": folds,
        "predictions": predictions,
        "references": references,
        "policies": policies,
        "identity": identity,
    }


def test_exact_fold_reference_rejects_in_memory_hash_mismatch() -> None:
    fixture = _fixture()
    reference = fixture["references"][0]
    bad = FoldModelReference(
        outer_fold=reference.outer_fold,
        model_sha256="f" * 64,
        pipeline=reference.pipeline,
    )
    with pytest.raises(HRDatasetDiagnosticsError, match="In-memory fold model hash mismatch"):
        bad.load()


def test_canonicalize_multiclass_shap_accepts_known_axes_and_rejects_ambiguity() -> None:
    canonical = np.arange(5 * 4 * 3, dtype=float).reshape(5, 4, 3)
    observed, source = canonicalize_multiclass_shap(
        canonical,
        n_samples=5,
        n_transformed_features=4,
        n_classes=3,
    )
    assert source == "sample_feature_class"
    np.testing.assert_array_equal(observed, canonical)

    class_list = [canonical[:, :, class_index] for class_index in range(3)]
    observed, source = canonicalize_multiclass_shap(
        class_list,
        n_samples=5,
        n_transformed_features=4,
        n_classes=3,
    )
    assert source == "class_list__sample_feature"
    np.testing.assert_array_equal(observed, canonical)

    with pytest.raises(HRDatasetDiagnosticsError, match="ambiguous"):
        canonicalize_multiclass_shap(
            np.zeros((3, 3, 3)),
            n_samples=3,
            n_transformed_features=3,
            n_classes=3,
        )


def test_exact_fold_shap_replays_predictions_groups_axes_and_reports_descriptive_stability(
    tmp_path: Path,
) -> None:
    fixture = _fixture()
    evidence = compute_exact_oof_grouped_shap(
        features=fixture["features"],
        fold_assignments=fixture["folds"],
        oof_predictions=fixture["predictions"],
        fold_models=fixture["references"],
        policy_features=fixture["policies"],
        primary_policy=PRIMARY_POLICY,
        forbidden_features=["EmpDepartment", "DeptID", "PerformanceScore", "Gender"],
        identity=fixture["identity"],
        labels=LABELS,
        feature_governance={
            "numeric_feature": {
                "governance_category": "operational",
                "temporality_status": "predeclared_available",
            },
            "category_feature": {
                "governance_category": "proxy_watchlist",
                "proxy_watchlist": True,
            },
        },
        top_k=2,
        shap_provider=_canonical_provider,
    )

    assert len(evidence.local_values) == 18 * 2 * 3
    assert set(evidence.local_values["feature"]) == {"numeric_feature", "category_feature"}
    assert set(evidence.local_values["outer_fold"]) == {1, 2, 3}
    assert evidence.local_values["prediction_replay_max_abs_error"].max() <= 1e-12
    assert evidence.local_values["shap_additivity_max_abs_error"].max() <= 1e-12
    assert set(evidence.local_values["attribution_warning"]) == {ATTRIBUTION_WARNING}
    assert evidence.local_values["noncausality_warning"].str.contains("not causality").all()
    assert evidence.local_values["temporality_warning"].str.contains("timing").all()
    assert evidence.local_values["model_sha256"].str.len().eq(64).all()
    assert evidence.local_values["transformed_lineage_sha256"].str.len().eq(64).all()
    assert set(evidence.local_values["attribution_unit"]) == {SHAP_ATTRIBUTION_UNIT}
    assert set(evidence.local_values["additivity_output_space"]) == {
        SHAP_ADDITIVITY_OUTPUT_SPACE
    }

    assert len(evidence.global_importance) == 2
    assert len(evidence.class_importance) == 3 * 2
    assert len(evidence.fold_rankings) == 3 * 2
    for frame in (
        evidence.global_importance,
        evidence.class_importance,
        evidence.fold_rankings,
    ):
        assert set(frame["attribution_unit"]) == {SHAP_ATTRIBUTION_UNIT}
    assert len(evidence.stability_pairwise) == 3
    assert set(evidence.stability_summary["confidence_interval_applicable"]) == {False}
    assert set(evidence.stability_pairwise["uncertainty_scope"]) == {
        "descriptive_dependent_fold_pairs_no_confidence_interval"
    }
    assert set(evidence.representative_cases["case_type"]) == {
        "correct_high_confidence",
        "correct_low_confidence",
        "incorrect_high_confidence",
        "incorrect_low_confidence",
        "minority_true_class",
        "most_uncertain",
    }
    assert evidence.metadata["model_refit_in_diagnostic"] is False
    assert evidence.metadata["n_samples"] == 18
    assert len(evidence.metadata["fold_receipts"]) == 3
    assert evidence.metadata["attribution_unit"] == SHAP_ATTRIBUTION_UNIT
    assert evidence.metadata["additivity_output_space"] == SHAP_ADDITIVITY_OUTPUT_SPACE
    assert {receipt["attribution_unit"] for receipt in evidence.metadata["fold_receipts"]} == {
        SHAP_ATTRIBUTION_UNIT
    }
    assert {
        receipt["additivity_output_space"] for receipt in evidence.metadata["fold_receipts"]
    } == {SHAP_ADDITIVITY_OUTPUT_SPACE}

    reason_paths = _write_local_reason_codes(tmp_path, evidence)
    json_path = next(path for path in reason_paths if path.suffix == ".json")
    markdown_path = next(path for path in reason_paths if path.suffix == ".md")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["attribution_unit"] == SHAP_ATTRIBUTION_UNIT
    assert payload["additivity_output_space"] == SHAP_ADDITIVITY_OUTPUT_SPACE
    markdown = markdown_path.read_text(encoding="utf-8")
    assert SHAP_ATTRIBUTION_UNIT in markdown
    assert "Grouped SHAP (XGBoost raw-margin score)" in markdown


def test_default_tree_provider_validates_real_xgboost_axes_and_additivity() -> None:
    fixture = _fixture(_xgboost_pipeline)
    evidence = compute_exact_oof_grouped_shap(
        features=fixture["features"],
        fold_assignments=fixture["folds"],
        oof_predictions=fixture["predictions"],
        fold_models=fixture["references"],
        policy_features=fixture["policies"],
        primary_policy=PRIMARY_POLICY,
        forbidden_features=["EmpDepartment", "DeptID", "PerformanceScore", "Gender"],
        identity=fixture["identity"],
        labels=LABELS,
        top_k=2,
    )

    assert set(evidence.local_values["shap_axis_source"]) == {
        "tree_explainer_sample_feature_output"
    }
    assert evidence.local_values["shap_additivity_max_abs_error"].max() < 1e-6
    assert evidence.local_values["prediction_replay_max_abs_error"].max() <= 1e-12
    assert set(evidence.local_values["attribution_unit"]) == {SHAP_ATTRIBUTION_UNIT}


def test_exact_fold_shap_fails_for_forbidden_raw_feature_or_prediction_drift() -> None:
    fixture = _fixture()
    with pytest.raises(HRDatasetDiagnosticsError, match="Forbidden"):
        compute_exact_oof_grouped_shap(
            features=fixture["features"],
            fold_assignments=fixture["folds"],
            oof_predictions=fixture["predictions"],
            fold_models=fixture["references"],
            policy_features=fixture["policies"],
            primary_policy=PRIMARY_POLICY,
            forbidden_features=["category_feature"],
            identity=fixture["identity"],
            labels=LABELS,
            shap_provider=_canonical_provider,
        )

    drifted = copy.deepcopy(fixture["predictions"])
    first = drifted.index[0]
    drifted.loc[first, ["prob_class_2", "prob_class_3", "prob_class_4"]] = [0.8, 0.1, 0.1]
    drifted.loc[first, "y_pred"] = 2
    with pytest.raises(HRDatasetDiagnosticsError, match="replay"):
        compute_exact_oof_grouped_shap(
            features=fixture["features"],
            fold_assignments=fixture["folds"],
            oof_predictions=drifted,
            fold_models=fixture["references"],
            policy_features=fixture["policies"],
            primary_policy=PRIMARY_POLICY,
            forbidden_features=[],
            identity=fixture["identity"],
            labels=LABELS,
            shap_provider=_canonical_provider,
        )

    identity_drifted = copy.deepcopy(fixture["predictions"])
    identity_drifted.loc[
        identity_drifted["outer_fold"].astype(int) == 1,
        "source_outer_model_sha256",
    ] = "0" * 64
    with pytest.raises(HRDatasetDiagnosticsError, match="source model hash differs"):
        compute_exact_oof_grouped_shap(
            features=fixture["features"],
            fold_assignments=fixture["folds"],
            oof_predictions=identity_drifted,
            fold_models=fixture["references"],
            policy_features=fixture["policies"],
            primary_policy=PRIMARY_POLICY,
            forbidden_features=[],
            identity=fixture["identity"],
            labels=LABELS,
            shap_provider=_canonical_provider,
        )


def test_exact_fold_shap_fails_closed_on_additivity_error() -> None:
    fixture = _fixture()

    def invalid_provider(
        pipeline: Pipeline,
        transformed: np.ndarray,
        labels: tuple[int, ...],
    ) -> ShapComputation:
        return ShapComputation(
            values=np.zeros((len(transformed), transformed.shape[1], len(labels))),
            base_values=np.zeros(len(labels)),
            margins=np.ones((len(transformed), len(labels))),
            attribution_unit=SHAP_ATTRIBUTION_UNIT,
            additivity_output_space=SHAP_ADDITIVITY_OUTPUT_SPACE,
            axis_source="synthetic_canonical_sample_feature_class",
        )

    with pytest.raises(HRDatasetDiagnosticsError, match="additivity"):
        compute_exact_oof_grouped_shap(
            features=fixture["features"],
            fold_assignments=fixture["folds"],
            oof_predictions=fixture["predictions"],
            fold_models=fixture["references"],
            policy_features=fixture["policies"],
            primary_policy=PRIMARY_POLICY,
            forbidden_features=[],
            identity=fixture["identity"],
            labels=LABELS,
            shap_provider=invalid_provider,
        )


def test_exact_fold_shap_rejects_provider_or_config_unit_drift() -> None:
    fixture = _fixture()

    def wrong_unit_provider(
        pipeline: Pipeline,
        transformed: np.ndarray,
        labels: tuple[int, ...],
    ) -> ShapComputation:
        result = _canonical_provider(pipeline, transformed, labels)
        return ShapComputation(
            values=result.values,
            base_values=result.base_values,
            margins=result.margins,
            attribution_unit="probability",
            additivity_output_space=result.additivity_output_space,
            axis_source=result.axis_source,
        )

    common = {
        "features": fixture["features"],
        "fold_assignments": fixture["folds"],
        "oof_predictions": fixture["predictions"],
        "fold_models": fixture["references"],
        "policy_features": fixture["policies"],
        "primary_policy": PRIMARY_POLICY,
        "forbidden_features": [],
        "identity": fixture["identity"],
        "labels": LABELS,
    }
    with pytest.raises(HRDatasetDiagnosticsError, match="provider attribution unit"):
        compute_exact_oof_grouped_shap(**common, shap_provider=wrong_unit_provider)
    with pytest.raises(HRDatasetDiagnosticsError, match="attribution_unit"):
        compute_exact_oof_grouped_shap(
            **common,
            attribution_unit="probability",
            shap_provider=_canonical_provider,
        )

    def wrong_space_provider(
        pipeline: Pipeline,
        transformed: np.ndarray,
        labels: tuple[int, ...],
    ) -> ShapComputation:
        result = _canonical_provider(pipeline, transformed, labels)
        return ShapComputation(
            values=result.values,
            base_values=result.base_values,
            margins=result.margins,
            attribution_unit=result.attribution_unit,
            additivity_output_space="probability",
            axis_source=result.axis_source,
        )

    with pytest.raises(HRDatasetDiagnosticsError, match="additivity output space"):
        compute_exact_oof_grouped_shap(**common, shap_provider=wrong_space_provider)
