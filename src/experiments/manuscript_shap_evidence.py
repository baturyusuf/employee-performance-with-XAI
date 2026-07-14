from __future__ import annotations

import argparse
import itertools
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.core.atomic_publish import atomic_replace_directory, cleanup_temporary_directory
from src.core.io_utils import ensure_dir, write_json
from src.data.canonical_loader import load_canonical_dataset
from src.experiments.benchmark_artifact_contract import (
    XGBoostOOFArtifacts,
    read_xgboost_oof_artifacts,
    validate_xgboost_oof_replay,
)
from src.experiments.manuscript_model_benchmark import exact_primary_feature_frame
from src.explainability.canonical_shap_axis import (
    build_canonical_shap_axis,
    group_canonical_shap_values,
    normalize_multiclass_shap_values,
)
from src.features.feature_sets import taxonomy_by_feature
from src.governance.manuscript_contract import canonical_config_hash, primary_excluded_features
from src.utils.config_loader import load_config


MANDATORY_WARNINGS = (
    "SHAP values are model attributions, not causal effects.",
    "Removing sensitive or organisational fields does not prove fairness; proxy risk may remain.",
    "This is research-grade analysis, not an autonomous HR decision system.",
)

COMMON_IDENTITY_FIELDS = (
    "run_id",
    "config_hash",
    "scientific_input_hash",
    "fold_contract_hash",
    "policy",
    "model",
    "model_set_sha256",
)


class ShapEvidenceError(RuntimeError):
    """Raised when canonical SHAP evidence violates its run or feature policy."""


def _identity_payload(
    *,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    fold_contract_hash: str,
    policy: str,
    model_set_sha256: str,
) -> Dict[str, str]:
    payload = {
        "run_id": str(run_id),
        "config_hash": str(config_hash),
        "scientific_input_hash": str(scientific_input_hash),
        "fold_contract_hash": str(fold_contract_hash),
        "policy": str(policy),
        "model": "xgboost",
        "model_set_sha256": str(model_set_sha256),
    }
    for field, value in payload.items():
        if not value.strip():
            raise ShapEvidenceError(f"SHAP identity field {field!r} must be non-empty.")
    return payload


def _attach_identity(frame: pd.DataFrame, identity: Mapping[str, str]) -> pd.DataFrame:
    attached = frame.copy()
    for field in COMMON_IDENTITY_FIELDS:
        expected = identity.get(field)
        if expected is None:
            raise ShapEvidenceError(f"SHAP identity is missing {field!r}.")
        attached[field] = str(expected)
    leading = list(COMMON_IDENTITY_FIELDS)
    return attached.loc[:, [*leading, *[column for column in attached.columns if column not in leading]]]


def _assert_frame_identity(frame: pd.DataFrame, identity: Mapping[str, str], *, name: str) -> None:
    for field in COMMON_IDENTITY_FIELDS:
        if field not in frame.columns:
            raise ShapEvidenceError(f"{name} is missing identity field {field!r}.")
        observed = set(frame[field].dropna().astype(str).unique())
        if observed != {str(identity[field])}:
            raise ShapEvidenceError(
                f"{name} has inconsistent {field}; expected={identity[field]!r}, observed={sorted(observed)}."
            )


def reorder_grouped_shap_to_feature_order(
    grouped_values: np.ndarray,
    grouped_feature_names: Sequence[str],
    raw_feature_order: Sequence[str],
) -> np.ndarray:
    """Align grouped SHAP values to the canonical raw dataframe column order.

    ``ColumnTransformer`` emits numeric and categorical blocks, so its grouped
    feature-family order can differ from the source dataframe's interleaved
    order. The feature sets must be identical; only an explicit axis
    permutation is allowed.
    """

    grouped_names = [str(value) for value in grouped_feature_names]
    raw_names = [str(value) for value in raw_feature_order]
    if len(grouped_names) != len(set(grouped_names)) or len(raw_names) != len(set(raw_names)):
        raise ShapEvidenceError("Grouped and raw feature-family names must be unique.")
    if set(grouped_names) != set(raw_names):
        missing = sorted(set(raw_names) - set(grouped_names))
        extra = sorted(set(grouped_names) - set(raw_names))
        raise ShapEvidenceError(
            f"Raw feature families changed across SHAP preprocessing; missing={missing}, extra={extra}."
        )
    if grouped_values.shape[-1] != len(grouped_names):
        raise ShapEvidenceError(
            "Grouped SHAP feature axis does not match the grouped feature-name count."
        )
    positions = {name: index for index, name in enumerate(grouped_names)}
    return grouped_values[..., [positions[name] for name in raw_names]]


def feature_governance(feature: str, taxonomy: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    row = taxonomy.get(feature, {})
    return {
        "control_type": row.get("control_type", "unknown"),
        "sensitive_or_proxy": row.get("sensitive_or_proxy", "unknown"),
        "leakage_risk": row.get("leakage_risk", "unknown"),
        "allowed_for_final_model": row.get("allowed_for_final_model", "unknown"),
        "governance_notes": row.get("notes", ""),
    }


def assert_feature_names_allowed(features: Iterable[str], forbidden_features: Iterable[str]) -> None:
    forbidden = {str(value).casefold() for value in forbidden_features}
    present = sorted({str(feature) for feature in features if str(feature).casefold() in forbidden})
    if present:
        raise ShapEvidenceError(f"Forbidden primary-policy features found in SHAP evidence: {present}")


def select_representative_cases(predictions: pd.DataFrame, labels: Sequence[int]) -> pd.DataFrame:
    required = {
        *COMMON_IDENTITY_FIELDS,
        "sample_index",
        "outer_fold",
        "model_sha256",
        "selected_candidate_index",
        "y_true",
        "y_pred",
        "confidence",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ShapEvidenceError(f"Representative-case predictions lack columns: {missing}")
    for identity_field in COMMON_IDENTITY_FIELDS:
        observed = set(predictions[identity_field].dropna().astype(str).unique())
        if len(observed) != 1:
            raise ShapEvidenceError(
                "Representative-case predictions require one canonical "
                f"{identity_field}; observed={sorted(observed)}."
            )
    candidates: list[Dict[str, Any]] = []
    seen: set[int] = set()

    def add(
        rows: pd.DataFrame,
        case_type: str,
        reason: str,
        *,
        ascending: bool,
        allow_duplicate_sample: bool = False,
    ) -> None:
        if rows.empty:
            return
        ordered = rows.sort_values(["confidence", "sample_index"], ascending=[ascending, True])
        for row in ordered.itertuples(index=False):
            sample_index = int(row.sample_index)
            if sample_index in seen and not allow_duplicate_sample:
                continue
            seen.add(sample_index)
            candidates.append(
                {
                    **{
                        field: str(getattr(row, field))
                        for field in COMMON_IDENTITY_FIELDS
                    },
                    "case_type": case_type,
                    "sample_index": sample_index,
                    "sampling_reason": reason,
                    "true_class": int(row.y_true),
                    "predicted_class": int(row.y_pred),
                    "confidence": float(row.confidence),
                    "correct": bool(row.y_true == row.y_pred),
                    "outer_fold": int(row.outer_fold),
                    "model_sha256": str(row.model_sha256),
                    "selected_candidate_index": int(row.selected_candidate_index),
                }
            )
            return

    correct = predictions[predictions["y_true"] == predictions["y_pred"]]
    incorrect = predictions[predictions["y_true"] != predictions["y_pred"]]
    add(correct, "correct_high_confidence", "correct and highest OOF confidence", ascending=False)
    add(correct, "correct_low_confidence", "correct and lowest OOF confidence", ascending=True)
    add(incorrect, "incorrect_high_confidence", "incorrect and highest OOF confidence", ascending=False)
    add(incorrect, "incorrect_low_confidence", "incorrect and lowest OOF confidence", ascending=True)
    add(predictions, "most_uncertain", "lowest maximum OOF probability", ascending=True)

    distribution = predictions["y_true"].value_counts()
    minority_label = int(distribution.sort_values().index[0])
    minority = predictions[predictions["y_true"] == minority_label]
    add(
        minority[minority["y_true"] == minority["y_pred"]],
        f"minority_class_{minority_label}_correct",
        f"correct case from least-supported true class {minority_label}",
        ascending=False,
        allow_duplicate_sample=True,
    )
    for label in labels:
        rows = predictions[(predictions["y_true"] == int(label)) & (predictions["y_pred"] == int(label))]
        add(
            rows,
            f"correct_class_{label}",
            f"correct high-confidence coverage for class {label}",
            ascending=False,
            allow_duplicate_sample=True,
        )
    return pd.DataFrame(candidates)


def _importance_table(
    values: np.ndarray,
    features: Sequence[str],
    *,
    identity: Mapping[str, str],
    taxonomy: Mapping[str, Mapping[str, Any]],
    n_outer_models: int,
) -> pd.DataFrame:
    importance = np.mean(np.abs(values), axis=tuple(range(values.ndim - 1)))
    rows = []
    for feature, value in zip(features, importance):
        rows.append(
            {
                "feature": feature,
                "mean_abs_grouped_shap": float(value),
                **feature_governance(feature, taxonomy),
            }
        )
    frame = pd.DataFrame(rows).sort_values("mean_abs_grouped_shap", ascending=False).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    frame["n_outer_models"] = int(n_outer_models)
    return _attach_identity(frame, identity)


def _fold_rankings(
    grouped_values: np.ndarray,
    folds: np.ndarray,
    features: Sequence[str],
    *,
    identity: Mapping[str, str],
    fold_models: Mapping[int, Any],
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        importance = np.mean(np.abs(grouped_values[folds == fold]), axis=(0, 1))
        order = np.argsort(-importance)
        for rank, feature_index in enumerate(order, start=1):
            rows.append(
                {
                    "outer_fold": int(fold),
                    "model_sha256": str(fold_models[int(fold)].sha256),
                    "selected_candidate_index": int(
                        fold_models[int(fold)].selected_candidate_index
                    ),
                    "feature": features[int(feature_index)],
                    "rank": rank,
                    "mean_abs_grouped_shap": float(importance[int(feature_index)]),
                }
            )
    return _attach_identity(pd.DataFrame(rows), identity)


def shap_stability_pairwise(rankings: pd.DataFrame, top_k_values: Sequence[int]) -> pd.DataFrame:
    if not top_k_values or any(int(value) < 1 for value in top_k_values):
        raise ShapEvidenceError("SHAP stability top-k values must be positive integers.")
    identity = {field: str(rankings[field].iloc[0]) for field in COMMON_IDENTITY_FIELDS}
    _assert_frame_identity(rankings, identity, name="Fold feature rankings")
    rows: list[Dict[str, Any]] = []
    fold_rankings = {
        int(fold): group.sort_values("rank")["feature"].tolist()
        for fold, group in rankings.groupby("outer_fold")
    }
    if sorted(fold_rankings) != list(range(1, 11)):
        raise ShapEvidenceError("Canonical SHAP stability requires ten contiguous outer folds.")
    features = sorted(rankings["feature"].unique())
    for fold_a, fold_b in itertools.combinations(sorted(fold_rankings), 2):
        rank_a = {feature: index + 1 for index, feature in enumerate(fold_rankings[fold_a])}
        rank_b = {feature: index + 1 for index, feature in enumerate(fold_rankings[fold_b])}
        correlation = spearmanr(
            [rank_a[feature] for feature in features],
            [rank_b[feature] for feature in features],
        ).statistic
        for top_k in top_k_values:
            set_a = set(fold_rankings[fold_a][: int(top_k)])
            set_b = set(fold_rankings[fold_b][: int(top_k)])
            union = set_a.union(set_b)
            row_a = rankings[rankings["outer_fold"].eq(fold_a)].iloc[0]
            row_b = rankings[rankings["outer_fold"].eq(fold_b)].iloc[0]
            rows.append(
                {
                    "outer_fold_a": fold_a,
                    "outer_fold_b": fold_b,
                    "model_sha256_a": str(row_a["model_sha256"]),
                    "model_sha256_b": str(row_b["model_sha256"]),
                    "top_k": int(top_k),
                    "top_k_jaccard": float(len(set_a.intersection(set_b)) / len(union)) if union else 1.0,
                    "spearman_all_features": float(correlation),
                }
            )
    result = _attach_identity(pd.DataFrame(rows), identity)
    expected_rows = 45 * len({int(value) for value in top_k_values})
    if len(result) != expected_rows:
        raise ShapEvidenceError(
            f"SHAP stability expected {expected_rows} pair/top-k rows; observed {len(result)}."
        )
    return result


def summarize_stability(pairwise: pd.DataFrame) -> pd.DataFrame:
    identity = {field: str(pairwise[field].iloc[0]) for field in COMMON_IDENTITY_FIELDS}
    _assert_frame_identity(pairwise, identity, name="Pairwise SHAP stability")
    rows: list[Dict[str, Any]] = []
    for top_k, group in pairwise.groupby("top_k"):
        if len(group) != 45:
            raise ShapEvidenceError(
                f"Ten-fold SHAP stability requires 45 fold pairs per top-k; observed {len(group)}."
            )
        jaccard = group["top_k_jaccard"].astype(float)
        spearman = group["spearman_all_features"].astype(float)
        rows.append(
            {
                "top_k": int(top_k),
                "n_outer_folds": 10,
                "n_fold_pairs": len(group),
                "jaccard_mean": float(jaccard.mean()),
                "jaccard_std": float(jaccard.std(ddof=1)),
                "jaccard_median": float(jaccard.median()),
                "jaccard_min": float(jaccard.min()),
                "jaccard_max": float(jaccard.max()),
                "spearman_mean": float(spearman.mean()),
                "spearman_std": float(spearman.std(ddof=1)),
                "spearman_median": float(spearman.median()),
                "spearman_min": float(spearman.min()),
                "spearman_max": float(spearman.max()),
                "fold_pair_independence_assumed": False,
                "confidence_interval_applicable": False,
                "uncertainty_type": "descriptive_dependent_fold_pairs",
            }
        )
    return _attach_identity(pd.DataFrame(rows), identity)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def write_local_reason_codes(
    representative_cases: pd.DataFrame,
    local_values: pd.DataFrame,
    predictions: pd.DataFrame,
    raw_features: pd.DataFrame,
    labels: Sequence[int],
    output_dir: Path,
    *,
    top_k: int,
    identity: Mapping[str, str],
    taxonomy: Mapping[str, Mapping[str, Any]],
) -> list[Path]:
    generated: list[Path] = []
    prediction_index = predictions.set_index("sample_index")
    for case in representative_cases.itertuples(index=False):
        sample_index = int(case.sample_index)
        prediction = prediction_index.loc[sample_index]
        predicted_class = int(prediction["y_pred"])
        rows = local_values[
            (local_values["sample_index"] == sample_index)
            & (local_values["class_label"] == predicted_class)
        ].copy()
        rows = rows.sort_values("abs_grouped_shap_value", ascending=False)
        rows["feature_value"] = rows["feature"].map(
            {feature: _jsonable_value(raw_features.loc[sample_index, feature]) for feature in raw_features.columns}
        )
        for key in ("control_type", "sensitive_or_proxy", "leakage_risk", "allowed_for_final_model", "governance_notes"):
            rows[key] = rows["feature"].map(lambda feature, k=key: feature_governance(str(feature), taxonomy)[k])
        rows["case_type"] = case.case_type
        rows["sampling_reason"] = case.sampling_reason
        slug = f"{_safe_slug(str(case.case_type))}_{sample_index}"
        csv_path = output_dir / f"local_reason_code_{slug}.csv"
        json_path = output_dir / f"local_reason_code_{slug}.json"
        md_path = output_dir / f"local_reason_code_{slug}.md"
        rows.to_csv(csv_path, index=False)

        supporting = rows[rows["grouped_shap_value"] > 0].head(top_k)
        opposing = rows[rows["grouped_shap_value"] < 0].head(top_k)
        probabilities = {str(label): float(prediction[f"prob_class_{label}"]) for label in labels}
        payload = {
            **dict(identity),
            "case_type": case.case_type,
            "sampling_reason": case.sampling_reason,
            "sample_index": sample_index,
            "outer_fold": int(prediction["outer_fold"]),
            "model_sha256": str(prediction["model_sha256"]),
            "selected_candidate_index": int(prediction["selected_candidate_index"]),
            "true_class": int(prediction["y_true"]),
            "predicted_class": predicted_class,
            "confidence": float(prediction["confidence"]),
            "probabilities": probabilities,
            "top_supporting_features": supporting.to_dict(orient="records"),
            "top_opposing_features": opposing.to_dict(orient="records"),
            "warnings": list(MANDATORY_WARNINGS),
            "interpretation_boundary": "OOF model attribution for the predicted class; not a causal or prescriptive explanation.",
        }
        write_json(json_path, payload)
        lines = [
            f"# Local Reason Code: {case.case_type}",
            "",
            f"Run ID: `{identity['run_id']}`  ",
            f"Config hash: `{identity['config_hash']}`  ",
            f"Scientific input hash: `{identity['scientific_input_hash']}`  ",
            f"Fold contract hash: `{identity['fold_contract_hash']}`  ",
            f"Model-set hash: `{identity['model_set_sha256']}`  ",
            f"Sample index: `{sample_index}`; OOF fold: `{int(prediction['outer_fold'])}`; model SHA-256: `{prediction['model_sha256']}`  ",
            f"True class: `{int(prediction['y_true'])}`; predicted class: `{predicted_class}`; confidence: `{float(prediction['confidence']):.4f}`",
            "",
            "## Top Supporting Attributions",
            "",
        ]
        for row in supporting.itertuples(index=False):
            lines.append(f"- `{row.feature}` = `{row.feature_value}`; grouped SHAP `{row.grouped_shap_value:.5f}`.")
        lines.extend(["", "## Top Opposing Attributions", ""])
        for row in opposing.itertuples(index=False):
            lines.append(f"- `{row.feature}` = `{row.feature_value}`; grouped SHAP `{row.grouped_shap_value:.5f}`.")
        lines.extend(["", "## Warnings", ""] + [f"- {warning}" for warning in MANDATORY_WARNINGS])
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        generated.extend([json_path, csv_path, md_path])
    return generated


def validate_shap_artifacts(
    *,
    global_tables: Sequence[pd.DataFrame],
    rankings: pd.DataFrame,
    local_values: pd.DataFrame,
    additional_frames: Sequence[pd.DataFrame] = (),
    forbidden_features: Sequence[str],
    identity: Mapping[str, str],
) -> Dict[str, Any]:
    frames = [*global_tables, rankings, local_values, *additional_frames]
    checked = 0
    for frame in frames:
        if "feature" in frame.columns:
            assert_feature_names_allowed(frame["feature"].astype(str), forbidden_features)
        _assert_frame_identity(frame, identity, name="SHAP artifact frame")
        checked += 1
    return {
        "status": "passed",
        **dict(identity),
        "frames_checked": checked,
        "forbidden_features_checked": list(forbidden_features),
    }


def run(
    config_path: str | Path,
    *,
    shared_folds_dir: str | Path,
    model_benchmarks_dir: str | Path,
    output_dir: str | Path,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
) -> Dict[str, Path]:
    """Build grouped SHAP evidence from exact prediction-producing OOF models.

    This stage deliberately has no model-fitting fallback.  It accepts only the
    hash-verified XGBoost pipelines emitted by the same benchmark run, replays
    their OOF predictions, and explains each sample with its assigned fold
    model.
    """

    import shap

    raw_config = load_config(config_path)
    settings = raw_config.get("manuscript_final", raw_config)
    observed_config_hash = canonical_config_hash(raw_config)
    if observed_config_hash != str(config_hash):
        raise ShapEvidenceError(
            "Supplied config_hash does not match the canonical manuscript configuration."
        )
    if not isinstance(scientific_input_hash, str) or len(scientific_input_hash) != 64:
        raise ShapEvidenceError("scientific_input_hash must be a lowercase SHA-256 digest.")
    if any(character not in "0123456789abcdef" for character in scientific_input_hash):
        raise ShapEvidenceError("scientific_input_hash must be a lowercase SHA-256 digest.")

    primary_policy = str(settings["feature_policies"]["primary_policy"])
    definition = settings["feature_policies"]["definitions"][primary_policy]
    forbidden = list(primary_excluded_features(raw_config))
    configured_forbidden = tuple(str(value) for value in definition["excluded_features"])
    if tuple(forbidden) != configured_forbidden:
        raise ShapEvidenceError(
            "The canonical primary feature-policy exclusions do not match their resolved contract."
        )
    target = settings.get("target", {})
    target_column = str(target.get("column", "PerformanceRating"))
    labels = [int(value) for value in target.get("labels", [2, 3, 4])]
    if labels != [2, 3, 4]:
        raise ShapEvidenceError("Canonical INX SHAP evidence requires labels [2, 3, 4].")
    canonical = load_canonical_dataset(config_path, "inx_primary")
    data = canonical.frame
    features = exact_primary_feature_frame(data, excluded_features=forbidden)
    assert_feature_names_allowed(features.columns, forbidden)
    target_values = data[target_column].astype(int)

    bundle: XGBoostOOFArtifacts = read_xgboost_oof_artifacts(
        shared_folds_dir,
        model_benchmarks_dir,
        expected_run_id=run_id,
        expected_config_hash=config_hash,
        expected_scientific_input_hash=scientific_input_hash,
        expected_feature_columns=features.columns.tolist(),
        expected_labels=labels,
    )
    validate_xgboost_oof_replay(
        bundle,
        features,
        target_values,
        labels=labels,
        probability_atol=1e-12,
    )
    if str(bundle.folds.contract.get("dataset_sha256")) != str(
        canonical.receipt.get("actual_sha256")
    ):
        raise ShapEvidenceError(
            "The exact-model SHAP fold contract does not match the canonical input bytes."
        )

    identity = _identity_payload(
        run_id=run_id,
        config_hash=config_hash,
        scientific_input_hash=scientific_input_hash,
        fold_contract_hash=bundle.identity.fold_contract_hash,
        policy=primary_policy,
        model_set_sha256=bundle.model_set_sha256,
    )

    protocol = settings.get("shap", {})
    stability_protocol = protocol.get("stability", {}) if isinstance(protocol.get("stability", {}), Mapping) else {}
    local_protocol = protocol.get("local", {}) if isinstance(protocol.get("local", {}), Mapping) else {}
    n_splits = int(settings.get("evaluation", {}).get("cv", {}).get("n_splits", 10))
    if n_splits != 10 or len(bundle.fold_models) != n_splits:
        raise ShapEvidenceError("Canonical OOF SHAP requires exactly ten outer-fold XGBoost models.")
    configured_top_k = stability_protocol.get("top_k")
    default_top_k = sorted({5, int(configured_top_k or 10), 15})
    top_k_values = [int(value) for value in protocol.get("top_k_values", default_top_k)]
    if len(top_k_values) != len(set(top_k_values)):
        raise ShapEvidenceError("SHAP stability top-k values must be unique.")
    local_top_k = int(protocol.get("local_top_k", local_protocol.get("top_k_reason_codes", 5)))
    if local_top_k < 1:
        raise ShapEvidenceError("local_top_k must be positive.")

    feature_names = features.columns.tolist()
    position_by_sample = {int(sample): position for position, sample in enumerate(features.index)}
    if len(position_by_sample) != len(features):
        raise ShapEvidenceError("Canonical feature rows require unique integer sample indices.")
    grouped_oof = np.full(
        (len(features), len(labels), len(feature_names)),
        np.nan,
        dtype=np.float64,
    )
    fold_oof = np.zeros(len(features), dtype=int)
    model_sha_oof = np.empty(len(features), dtype=object)
    candidate_oof = np.full(len(features), -1, dtype=int)
    assigned = np.zeros(len(features), dtype=bool)

    for outer_fold in sorted(bundle.fold_models):
        fold_model = bundle.fold_models[outer_fold]
        test_ids = [int(value) for value in fold_model.test_sample_indices]
        try:
            positions = np.asarray([position_by_sample[value] for value in test_ids], dtype=int)
        except KeyError as exc:
            raise ShapEvidenceError(
                f"Outer fold {outer_fold} refers to an unknown canonical sample index."
            ) from exc
        if assigned[positions].any():
            raise ShapEvidenceError("A canonical sample was assigned to multiple SHAP fold models.")
        test_features = features.loc[test_ids]
        pipeline = fold_model.pipeline
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["model"]
        axis = build_canonical_shap_axis(
            preprocessor,
            raw_feature_order=feature_names,
            forbidden_features=forbidden,
        )
        if tuple(axis.transformed_feature_names) != tuple(fold_model.transformed_feature_names):
            raise ShapEvidenceError(
                f"Outer fold {outer_fold} transformed lineage differs from benchmark evidence."
            )
        transformed = preprocessor.transform(test_features)
        axis.validate_transformed_matrix(transformed)
        native_feature_count = int(getattr(classifier.model_, "n_features_in_", -1))
        if native_feature_count != len(axis.transformed_feature_names):
            raise ShapEvidenceError(
                f"Outer fold {outer_fold} native model feature count does not match lineage."
            )
        raw_values = shap.TreeExplainer(classifier.model_).shap_values(transformed)
        normalized = normalize_multiclass_shap_values(
            raw_values,
            n_samples=len(test_ids),
            n_classes=len(labels),
            n_transformed_features=len(axis.transformed_feature_names),
        )
        grouped = group_canonical_shap_values(normalized, axis)
        grouped_oof[positions] = grouped
        fold_oof[positions] = int(outer_fold)
        model_sha_oof[positions] = str(fold_model.sha256)
        candidate_oof[positions] = int(fold_model.selected_candidate_index)
        assigned[positions] = True

    if not assigned.all() or np.isnan(grouped_oof).any():
        missing = [int(features.index[index]) for index in np.flatnonzero(~assigned)[:10]]
        raise ShapEvidenceError(f"OOF SHAP coverage is incomplete; sample_indices={missing}.")

    taxonomy = taxonomy_by_feature()
    missing_taxonomy = sorted(set(feature_names).difference(taxonomy))
    if missing_taxonomy:
        raise ShapEvidenceError(
            f"Canonical SHAP governance taxonomy is missing features: {missing_taxonomy}."
        )
    global_importance = _importance_table(
        grouped_oof,
        feature_names,
        identity=identity,
        taxonomy=taxonomy,
        n_outer_models=n_splits,
    )
    class_tables: Dict[int, pd.DataFrame] = {}
    for class_index, label in enumerate(labels):
        class_tables[label] = _importance_table(
            grouped_oof[:, class_index, :],
            feature_names,
            identity=identity,
            taxonomy=taxonomy,
            n_outer_models=n_splits,
        )
        class_tables[label]["class_label"] = label

    rankings = _fold_rankings(
        grouped_oof,
        fold_oof,
        feature_names,
        identity=identity,
        fold_models=bundle.fold_models,
    )
    pairwise = shap_stability_pairwise(rankings, top_k_values)
    stability = summarize_stability(pairwise)

    persisted_oof = bundle.oof_predictions.sort_values("sample_index").reset_index(drop=True)
    expected_samples = [int(value) for value in features.index]
    if persisted_oof["sample_index"].astype(int).tolist() != expected_samples:
        raise ShapEvidenceError("Benchmark XGBoost OOF rows are not in canonical sample order.")
    prediction_rows: list[Dict[str, Any]] = []
    local_rows: list[Dict[str, Any]] = []
    for position, sample_index in enumerate(features.index):
        persisted = persisted_oof.iloc[position]
        if int(persisted["outer_fold"]) != int(fold_oof[position]):
            raise ShapEvidenceError("SHAP fold assignment differs from benchmark OOF evidence.")
        prediction_row: Dict[str, Any] = {
            **identity,
            "system_id": "xgboost",
            "sample_index": int(sample_index),
            "outer_fold": int(fold_oof[position]),
            "model_sha256": str(model_sha_oof[position]),
            "selected_candidate_index": int(candidate_oof[position]),
            "y_true": int(persisted["y_true"]),
            "y_pred": int(persisted["y_pred"]),
        }
        probabilities = np.asarray(
            [float(persisted[f"prob_class_{label}"]) for label in labels],
            dtype=np.float64,
        )
        prediction_row["confidence"] = float(np.max(probabilities))
        prediction_row["correct"] = bool(prediction_row["y_true"] == prediction_row["y_pred"])
        for label_index, label in enumerate(labels):
            prediction_row[f"prob_class_{label}"] = float(probabilities[label_index])
            for feature_index, feature in enumerate(feature_names):
                value = float(grouped_oof[position, label_index, feature_index])
                local_rows.append(
                    {
                        **identity,
                        "sample_index": int(sample_index),
                        "outer_fold": int(fold_oof[position]),
                        "model_sha256": str(model_sha_oof[position]),
                        "selected_candidate_index": int(candidate_oof[position]),
                        "class_label": int(label),
                        "feature": feature,
                        "grouped_shap_value": value,
                        "abs_grouped_shap_value": abs(value),
                    }
                )
        prediction_rows.append(prediction_row)
    predictions = pd.DataFrame(prediction_rows)
    local_values = pd.DataFrame(local_rows)
    representative = select_representative_cases(predictions, labels)

    validation = validate_shap_artifacts(
        global_tables=[global_importance, *class_tables.values()],
        rankings=rankings,
        local_values=local_values,
        additional_frames=[pairwise, stability, predictions, representative],
        forbidden_features=forbidden,
        identity=identity,
    )
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ShapEvidenceError(
            f"SHAP output directory must be absent or an empty builder-owned directory: {output}"
        )
    ensure_dir(output.parent)
    temporary = tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent)
    staging = Path(temporary.name)
    local_dir = ensure_dir(staging / "local_reason_codes")
    paths: Dict[str, Path] = {
        "global": staging / "global_grouped_shap_importance.csv",
        "rankings": staging / "fold_feature_rankings.csv",
        "pairwise": staging / "shap_stability_pairwise.csv",
        "stability": staging / "shap_stability_summary.csv",
        "representative": staging / "representative_cases.csv",
        "predictions": staging / "shap_oof_predictions.csv",
        "local_values": staging / "local_grouped_shap_values.csv",
        "validation": staging / "shap_artifact_validation.json",
        "metadata": staging / "shap_metadata.json",
    }
    try:
        global_importance.to_csv(paths["global"], index=False)
        rankings.to_csv(paths["rankings"], index=False)
        pairwise.to_csv(paths["pairwise"], index=False)
        stability.to_csv(paths["stability"], index=False)
        representative.to_csv(paths["representative"], index=False)
        predictions.to_csv(paths["predictions"], index=False)
        local_values.to_csv(paths["local_values"], index=False)
        for label, table in class_tables.items():
            path = staging / f"class_{label}_grouped_shap_importance.csv"
            table.to_csv(path, index=False)
            paths[f"class_{label}"] = path
        reason_code_paths = write_local_reason_codes(
            representative,
            local_values,
            predictions,
            features,
            labels,
            local_dir,
            top_k=local_top_k,
            identity=identity,
            taxonomy=taxonomy,
        )
        write_json(
            paths["validation"],
            {**validation, "reason_code_files_checked": len(reason_code_paths)},
        )
        write_json(
            paths["metadata"],
            {
                "stage": "oof_shap",
                **identity,
                "dataset_sha256": canonical.receipt.get("actual_sha256"),
                "excluded_features": forbidden,
                "protocol": {
                    "evaluation": "exact_prediction_producing_outer_fold_models",
                    "n_outer_models": n_splits,
                    "model_refit_in_shap_stage": False,
                    "grouping": "one-hot SHAP summed exactly once to canonical raw feature families",
                    "stability_uncertainty": "descriptive_dependent_fold_pairs_no_confidence_interval",
                    "top_k_values": top_k_values,
                    "local_top_k": local_top_k,
                },
                "upstream_file_hashes": dict(sorted(bundle.upstream_file_hashes.items())),
                "fold_models": {
                    str(fold): {
                        "sha256": model.sha256,
                        "size_bytes": model.size_bytes,
                        "selected_candidate_index": model.selected_candidate_index,
                    }
                    for fold, model in sorted(bundle.fold_models.items())
                },
                "warnings": list(MANDATORY_WARNINGS),
                "outputs": {
                    key: value.relative_to(staging).as_posix()
                    for key, value in paths.items()
                    if key != "metadata"
                },
                "local_reason_code_files": [
                    path.relative_to(staging).as_posix() for path in reason_code_paths
                ],
            },
        )
        relative_paths = {
            key: path.relative_to(staging)
            for key, path in paths.items()
        }
        if output.exists():
            # The canonical stage orchestrator creates an empty stage directory
            # before invoking its runner.  Remove only that verified-empty shell
            # immediately before atomic publication; never delete a populated
            # or incompatible scientific directory.
            output.rmdir()
        atomic_replace_directory(staging, output)
        cleanup_temporary_directory(temporary)
    except Exception as error:
        cleanup_temporary_directory(temporary, primary_error=error)
        raise
    return {key: output / relative for key, relative in relative_paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical OOF grouped SHAP and local reason-code evidence.")
    parser.add_argument("--config", default="configs/manuscript_final.yaml")
    parser.add_argument("--shared-folds-dir", required=True)
    parser.add_argument("--model-benchmarks-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--scientific-input-hash", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            {
                key: str(value)
                for key, value in run(
                    arguments.config,
                    shared_folds_dir=arguments.shared_folds_dir,
                    model_benchmarks_dir=arguments.model_benchmarks_dir,
                    output_dir=arguments.output_dir,
                    run_id=arguments.run_id,
                    config_hash=arguments.config_hash,
                    scientific_input_hash=arguments.scientific_input_hash,
                ).items()
            },
            indent=2,
            sort_keys=True,
        )
    )
