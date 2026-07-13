"""Neutral diagnostics for an independently trained HRDataset replication.

The functions in this module are deliberately report-root agnostic.  They do
not discover historical artifacts, fit a replacement performance model, or
import the legacy fairness implementation.  Callers must provide the exact
outer-fold pipelines that produced the supplied OOF predictions together with
complete scientific identity and explicit feature/audit contracts.

The module exposes three bounded evidence builders:

* exact-fold grouped multiclass SHAP and descriptive fold stability;
* support-aware subgroup gaps with pointwise sample-level bootstrap intervals;
* department (or another nominal target) reconstructability, which fails
  closed to a machine-readable ``not_estimated`` result when an outer training
  fold lacks any observed target class.

All outputs are descriptive research evidence.  They do not establish causal
effects, fairness, discrimination, employee actionability, or suitability for
autonomous HR decisions.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src.models.canonical_models import aligned_predict_proba
from src.models.oof_bootstrap import (
    BootstrapProtocol,
    OOFBootstrapError,
    generate_stratified_resample_indices,
)


PRIMARY_TASK = "ordinal_multiclass_performance"
PROXY_TASK = "nominal_multiclass_proxy_diagnostic"
REQUIRED_BOOTSTRAP_RESAMPLES = 5000
IDENTITY_FIELDS = (
    "run_id",
    "config_hash",
    "scientific_input_hash",
    "dataset_sha256",
    "schema_mapping_sha256",
    "fold_contract_hash",
    "feature_policy_contract_sha256",
    "model_set_sha256",
)
SUBGROUP_CATEGORIES = frozenset({"protected_sensitive", "exploratory_operational"})
SUBGROUP_OVERALL_METRICS = ("accuracy", "macro_f1")
SUBGROUP_CLASS_METRICS = (
    "positive_prediction_rate",
    "true_positive_rate",
    "false_positive_rate",
    "precision",
    "mean_predicted_probability",
)
PROXY_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
_PROBABILITY_SUM_ATOL = 1e-6

ATTRIBUTION_WARNING = "Grouped SHAP is model attribution for an exact OOF model, not causality."
TEMPORALITY_WARNING = (
    "Feature availability and timing must be verified for the intended prediction time; "
    "an attribution does not establish temporal or intervention validity."
)
RESEARCH_USE_WARNING = (
    "Research-grade descriptive evidence only; no autonomous HR decision or employee prescription."
)
SUBGROUP_LIMITATION = (
    "Pointwise descriptive OOF subgroup estimate conditional on the observed samples, fixed folds, "
    "and fitted models; no multiplicity adjustment, causal interpretation, discrimination finding, "
    "or fairness guarantee."
)
PROXY_LIMITATION = (
    "Nominal target reconstructability is proxy-risk evidence only; it does not establish that the "
    "performance model uses the target causally or discriminatorily, and it is not a fairness guarantee."
)


class HRDatasetDiagnosticsError(RuntimeError):
    """Raised when an external diagnostic violates its scientific contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    observed = str(value)
    if len(observed) != 64 or any(character not in "0123456789abcdef" for character in observed):
        raise HRDatasetDiagnosticsError(f"{name} must be a lowercase SHA-256 digest.")
    return observed


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _normalise_name(value: Any) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _set_unique_index(frame: pd.DataFrame, column: str, name: str) -> pd.DataFrame:
    indexed = frame.set_index(column)
    if not indexed.index.is_unique:
        raise HRDatasetDiagnosticsError(f"{name} must have a unique {column} index.")
    return indexed


@dataclass(frozen=True)
class ReplicationIdentity:
    """Complete identity attached to every diagnostic evidence row."""

    run_id: str
    config_hash: str
    scientific_input_hash: str
    dataset_sha256: str
    schema_mapping_sha256: str
    fold_contract_hash: str
    feature_policy_contract_sha256: str
    model_set_sha256: str

    def __post_init__(self) -> None:
        if not str(self.run_id).strip():
            raise HRDatasetDiagnosticsError("run_id must be non-empty.")
        for field_name in IDENTITY_FIELDS[1:]:
            _require_sha256(field_name, getattr(self, field_name))

    def as_dict(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in IDENTITY_FIELDS}


@dataclass(frozen=True)
class FoldModelReference:
    """One exact prediction-producing outer-fold pipeline."""

    outer_fold: int
    model_sha256: str
    pipeline: Any | None = field(default=None, repr=False, compare=False)
    model_path: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.outer_fold, int) or self.outer_fold < 1:
            raise HRDatasetDiagnosticsError("outer_fold must be a positive integer.")
        _require_sha256("model_sha256", self.model_sha256)
        if (self.pipeline is None) == (self.model_path is None):
            raise HRDatasetDiagnosticsError(
                "Exactly one of pipeline or model_path must be supplied for each fold model."
            )

    def load(self) -> Any:
        if self.pipeline is not None:
            buffer = io.BytesIO()
            joblib.dump(self.pipeline, buffer, compress=0, protocol=4)
            observed = hashlib.sha256(buffer.getvalue()).hexdigest()
            if observed != self.model_sha256:
                raise HRDatasetDiagnosticsError(
                    f"In-memory fold model hash mismatch for outer fold {self.outer_fold}."
                )
            return self.pipeline
        assert self.model_path is not None
        path = Path(self.model_path)
        if not path.is_file() or path.stat().st_size <= 0:
            raise HRDatasetDiagnosticsError(f"Persisted fold model is missing or empty: {path}")
        if _sha256_file(path) != self.model_sha256:
            raise HRDatasetDiagnosticsError(f"Persisted fold model hash mismatch: {path}")
        return joblib.load(path)


def model_set_sha256(fold_models: Sequence[FoldModelReference]) -> str:
    rows = [
        {"outer_fold": int(reference.outer_fold), "model_sha256": reference.model_sha256}
        for reference in sorted(fold_models, key=lambda item: item.outer_fold)
    ]
    if not rows or len({row["outer_fold"] for row in rows}) != len(rows):
        raise HRDatasetDiagnosticsError("Fold model references must be non-empty and fold-unique.")
    return _sha256_json(rows)


def feature_policy_contract_sha256(policy_features: Mapping[str, Sequence[str]]) -> str:
    if not policy_features:
        raise HRDatasetDiagnosticsError("At least one named feature policy is required.")
    payload: dict[str, list[str]] = {}
    for policy, features in sorted(policy_features.items()):
        values = [str(feature) for feature in features]
        if not str(policy).strip() or not values or len(set(values)) != len(values):
            raise HRDatasetDiagnosticsError("Feature policy names/features must be non-empty and unique.")
        payload[str(policy)] = values
    return _sha256_json(payload)


def outer_fold_assignment_sha256(
    folds: pd.DataFrame,
    *,
    sample_id_column: str = "sample_index",
    fold_column: str = "outer_fold",
) -> str:
    required = {sample_id_column, fold_column}
    missing = sorted(required.difference(folds.columns))
    if missing:
        raise HRDatasetDiagnosticsError(f"Fold assignments are missing columns: {missing}")
    scoped = folds[[sample_id_column, fold_column]].copy()
    if scoped.empty or scoped[sample_id_column].duplicated().any() or scoped.isna().any().any():
        raise HRDatasetDiagnosticsError("Fold assignments must contain one complete row per sample.")
    scoped = scoped.sort_values(sample_id_column, kind="stable")
    return hashlib.sha256(scoped.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def _validate_identity_columns(frame: pd.DataFrame, identity: ReplicationIdentity, name: str) -> None:
    missing = sorted(set(IDENTITY_FIELDS).difference(frame.columns))
    if missing:
        raise HRDatasetDiagnosticsError(f"{name} is missing complete identity columns: {missing}")
    expected = identity.as_dict()
    for field_name, value in expected.items():
        observed = set(frame[field_name].astype(str))
        if observed != {str(value)}:
            raise HRDatasetDiagnosticsError(
                f"{name} {field_name} differs from the active scientific identity: {observed}"
            )


def _validate_forbidden_features(features: Iterable[str], forbidden: Iterable[str]) -> None:
    forbidden_by_key = {_normalise_name(value): str(value) for value in forbidden}
    observed = [str(value) for value in features]
    collisions = sorted(
        value for value in observed if _normalise_name(value) in forbidden_by_key
    )
    if collisions:
        raise HRDatasetDiagnosticsError(f"Forbidden features enter diagnostic evidence: {collisions}")


def canonicalize_multiclass_shap(
    values: Any,
    *,
    n_samples: int,
    n_transformed_features: int,
    n_classes: int,
) -> tuple[np.ndarray, str]:
    """Return SHAP as ``(sample, transformed_feature, class)`` or fail."""

    if hasattr(values, "values") and not isinstance(values, np.ndarray):
        values = values.values
    if isinstance(values, (list, tuple)):
        if len(values) != n_classes:
            raise HRDatasetDiagnosticsError(
                f"SHAP class list length {len(values)} differs from n_classes={n_classes}."
            )
        arrays = [np.asarray(item, dtype=float) for item in values]
        if any(item.shape != (n_samples, n_transformed_features) for item in arrays):
            raise HRDatasetDiagnosticsError("SHAP class-list matrices have incompatible axes.")
        return np.stack(arrays, axis=2), "class_list__sample_feature"

    array = np.asarray(values, dtype=float)
    if array.ndim != 3:
        raise HRDatasetDiagnosticsError(
            f"Multiclass SHAP must be three-dimensional; observed shape={array.shape}."
        )
    candidates: list[tuple[str, np.ndarray]] = []
    if array.shape == (n_samples, n_transformed_features, n_classes):
        candidates.append(("sample_feature_class", array))
    if array.shape == (n_samples, n_classes, n_transformed_features):
        candidates.append(("sample_class_feature", np.transpose(array, (0, 2, 1))))
    if array.shape == (n_classes, n_samples, n_transformed_features):
        candidates.append(("class_sample_feature", np.transpose(array, (1, 2, 0))))
    if len(candidates) != 1:
        raise HRDatasetDiagnosticsError(
            "SHAP axes are incompatible or ambiguous for the declared sample/feature/class sizes: "
            f"shape={array.shape}, candidates={[name for name, _ in candidates]}."
        )
    canonical = candidates[0][1]
    if not np.isfinite(canonical).all():
        raise HRDatasetDiagnosticsError("SHAP values contain non-finite entries.")
    return canonical, candidates[0][0]


def _transformed_feature_groups(
    preprocessor: Any,
    raw_features: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], str]:
    try:
        transformed_names = tuple(str(value) for value in preprocessor.get_feature_names_out())
    except Exception as exc:  # pragma: no cover - library-specific failure detail
        raise HRDatasetDiagnosticsError(
            f"Preprocessor does not expose deterministic transformed feature names: {exc}"
        ) from exc
    if not transformed_names:
        raise HRDatasetDiagnosticsError("Preprocessor transformed feature lineage is empty.")

    raw = tuple(str(value) for value in raw_features)
    raw_keys = {_normalise_name(value): value for value in raw}
    group_index: list[int] = []
    for transformed in transformed_names:
        suffix = transformed.split("__", 1)[-1]
        matching = [
            feature
            for feature in raw
            if suffix == feature or suffix.startswith(f"{feature}_")
        ]
        if not matching:
            normalised_suffix = _normalise_name(suffix)
            matching = [
                raw_keys[key]
                for key in raw_keys
                if normalised_suffix == key or normalised_suffix.startswith(key)
            ]
        if not matching:
            raise HRDatasetDiagnosticsError(
                f"Cannot map transformed feature {transformed!r} to one raw feature family."
            )
        selected = sorted(set(matching), key=lambda value: (-len(value), value))[0]
        group_index.append(raw.index(selected))
    missing_groups = [feature for index, feature in enumerate(raw) if index not in group_index]
    if missing_groups:
        raise HRDatasetDiagnosticsError(
            f"Preprocessor dropped declared raw feature families: {missing_groups}"
        )
    lineage_hash = _sha256_json(
        [
            {"transformed_feature": name, "raw_feature": raw[group]}
            for name, group in zip(transformed_names, group_index)
        ]
    )
    return raw, transformed_names, tuple(group_index), lineage_hash


def _group_transformed_shap(
    canonical: np.ndarray,
    group_index: Sequence[int],
    n_groups: int,
) -> np.ndarray:
    grouped = np.zeros((canonical.shape[0], n_groups, canonical.shape[2]), dtype=float)
    for transformed_index, raw_index in enumerate(group_index):
        grouped[:, int(raw_index), :] += canonical[:, transformed_index, :]
    if not np.allclose(grouped.sum(axis=1), canonical.sum(axis=1), rtol=0.0, atol=1e-12):
        raise HRDatasetDiagnosticsError("Grouped SHAP does not preserve transformed attribution sums.")
    return grouped


@dataclass(frozen=True)
class ShapComputation:
    """Provider-neutral transformed SHAP result for one fold."""

    values: Any
    base_values: np.ndarray
    margins: np.ndarray
    axis_source: str | None = None


ShapProvider = Callable[[Any, np.ndarray, Sequence[int]], ShapComputation]


def _unwrap_estimator(pipeline: Any) -> Any:
    if not hasattr(pipeline, "named_steps"):
        raise HRDatasetDiagnosticsError("Fold model must be a fitted sklearn-style pipeline.")
    for step_name in ("model", "classifier"):
        if step_name in pipeline.named_steps:
            estimator = pipeline.named_steps[step_name]
            return getattr(estimator, "model_", estimator)
    raise HRDatasetDiagnosticsError("Fold pipeline has no model/classifier step.")


def _default_tree_shap_provider(
    pipeline: Any,
    transformed: np.ndarray,
    labels: Sequence[int],
) -> ShapComputation:
    import shap

    estimator = _unwrap_estimator(pipeline)
    explainer = shap.TreeExplainer(estimator)
    try:
        values = explainer.shap_values(transformed, check_additivity=False)
    except TypeError:  # pragma: no cover - old SHAP compatibility
        values = explainer.shap_values(transformed)
    # TreeExplainer's current ndarray contract is sample x feature x output.
    # Resolve it here, where the producing API is known, so the generic axis
    # normalizer can continue to reject genuinely ambiguous anonymous arrays.
    if isinstance(values, (list, tuple)):
        canonical_values, source = canonicalize_multiclass_shap(
            values,
            n_samples=len(transformed),
            n_transformed_features=transformed.shape[1],
            n_classes=len(labels),
        )
    else:
        array = np.asarray(values, dtype=float)
        if array.shape == (len(transformed), transformed.shape[1], len(labels)):
            canonical_values = array
            source = "tree_explainer_sample_feature_output"
        else:
            canonical_values, source = canonicalize_multiclass_shap(
                array,
                n_samples=len(transformed),
                n_transformed_features=transformed.shape[1],
                n_classes=len(labels),
            )
    try:
        margins = np.asarray(estimator.predict(transformed, output_margin=True), dtype=float)
    except TypeError as exc:
        raise HRDatasetDiagnosticsError(
            "Exact-fold estimator cannot expose raw multiclass margins for SHAP additivity."
        ) from exc
    return ShapComputation(
        values=canonical_values,
        base_values=np.asarray(explainer.expected_value, dtype=float),
        margins=margins,
        axis_source=source,
    )


def _normalise_base_values(values: np.ndarray, n_classes: int) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape != (n_classes,) or not np.isfinite(array).all():
        raise HRDatasetDiagnosticsError(
            f"SHAP base values must have one finite value per class; shape={array.shape}."
        )
    return array


def _normalise_margins(values: np.ndarray, n_samples: int, n_classes: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (n_samples, n_classes) or not np.isfinite(array).all():
        raise HRDatasetDiagnosticsError(
            f"Raw model margins must have shape {(n_samples, n_classes)}; observed={array.shape}."
        )
    return array


def _aligned_probabilities(pipeline: Any, frame: pd.DataFrame, labels: Sequence[int]) -> np.ndarray:
    # Replay must use the identical clipping, class alignment and float64
    # simplex normalization used when the OOF probabilities were generated.
    # Direct ``predict_proba`` replay differs by roughly float32 softmax error
    # for XGBoost and would make the exact-fold identity check self-contradictory.
    try:
        return aligned_predict_proba(pipeline, frame, labels=labels)
    except Exception as exc:
        raise HRDatasetDiagnosticsError(
            f"Fold pipeline probability replay failed: {type(exc).__name__}: {exc}"
        ) from exc


def _prediction_frame(
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    features: pd.DataFrame,
    *,
    identity: ReplicationIdentity,
    primary_policy: str,
    labels: Sequence[int],
    sample_id_column: str,
    fold_column: str,
    probability_columns: Mapping[int, str],
) -> pd.DataFrame:
    _validate_identity_columns(predictions, identity, "OOF predictions")
    required = {
        sample_id_column,
        fold_column,
        "policy",
        "y_true",
        "y_pred",
        *probability_columns.values(),
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise HRDatasetDiagnosticsError(f"OOF predictions are missing columns: {missing}")
    scoped = predictions[predictions["policy"].astype(str) == str(primary_policy)].copy()
    if scoped.empty or scoped[sample_id_column].duplicated().any():
        raise HRDatasetDiagnosticsError("Primary-policy OOF predictions must be exactly once per sample.")
    fold_map = folds[[sample_id_column, fold_column]].copy()
    if fold_map[sample_id_column].duplicated().any() or fold_map.empty:
        raise HRDatasetDiagnosticsError("Fold assignments must be exactly once per sample.")
    feature_ids = features[sample_id_column]
    if feature_ids.duplicated().any() or features.empty:
        raise HRDatasetDiagnosticsError("Feature rows must be exactly once per sample.")
    expected_ids = set(feature_ids.tolist())
    if set(scoped[sample_id_column].tolist()) != expected_ids or set(fold_map[sample_id_column]) != expected_ids:
        raise HRDatasetDiagnosticsError("Feature, fold and OOF prediction sample sets differ.")
    merged = scoped.merge(
        fold_map.rename(columns={fold_column: "_declared_outer_fold"}),
        on=sample_id_column,
        how="left",
        validate="one_to_one",
    )
    if not np.array_equal(
        merged[fold_column].to_numpy(int), merged["_declared_outer_fold"].to_numpy(int)
    ):
        raise HRDatasetDiagnosticsError("OOF prediction fold identities differ from assignments.")
    probability = merged[[probability_columns[int(label)] for label in labels]].to_numpy(float)
    if not np.isfinite(probability).all() or (probability < 0.0).any():
        raise HRDatasetDiagnosticsError("OOF probabilities are non-finite or negative.")
    if not np.allclose(
        probability.sum(axis=1), 1.0, rtol=0.0, atol=_PROBABILITY_SUM_ATOL
    ):
        raise HRDatasetDiagnosticsError("OOF probabilities do not sum to one.")
    expected_prediction = np.asarray(labels, dtype=int)[np.argmax(probability, axis=1)]
    if not np.array_equal(expected_prediction, merged["y_pred"].to_numpy(int)):
        raise HRDatasetDiagnosticsError("OOF labels disagree with probability argmax.")
    merged["confidence"] = probability.max(axis=1)
    return merged.drop(columns="_declared_outer_fold").sort_values(sample_id_column).reset_index(drop=True)


def _rank_importance(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    grouped: Iterable[tuple[Any, pd.DataFrame]]
    if group_columns:
        grouped = frame.groupby(list(group_columns), sort=True, dropna=False)
    else:
        grouped = [((), frame)]
    for _, scoped in grouped:
        ordered = scoped.sort_values(
            ["mean_abs_grouped_shap", "feature"], ascending=[False, True], kind="stable"
        ).copy()
        ordered["rank"] = np.arange(1, len(ordered) + 1, dtype=int)
        rows.append(ordered)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _representative_cases(
    predictions: pd.DataFrame,
    *,
    identity: ReplicationIdentity,
    sample_id_column: str,
    fold_column: str,
) -> pd.DataFrame:
    correct = predictions[predictions["y_true"].astype(int) == predictions["y_pred"].astype(int)]
    incorrect = predictions[predictions["y_true"].astype(int) != predictions["y_pred"].astype(int)]
    counts = predictions["y_true"].astype(int).value_counts().sort_values(kind="stable")
    minority_label = int(counts.index[0])
    candidates: list[tuple[str, pd.DataFrame, list[str], list[bool], str]] = [
        (
            "correct_high_confidence",
            correct,
            ["confidence", sample_id_column],
            [False, True],
            "correct case with highest OOF confidence",
        ),
        (
            "correct_low_confidence",
            correct,
            ["confidence", sample_id_column],
            [True, True],
            "correct case with lowest OOF confidence",
        ),
        (
            "incorrect_high_confidence",
            incorrect,
            ["confidence", sample_id_column],
            [False, True],
            "incorrect case with highest OOF confidence",
        ),
        (
            "incorrect_low_confidence",
            incorrect,
            ["confidence", sample_id_column],
            [True, True],
            "incorrect case with lowest OOF confidence",
        ),
        (
            "minority_true_class",
            predictions[predictions["y_true"].astype(int) == minority_label],
            ["confidence", sample_id_column],
            [False, True],
            f"highest-confidence case in least-supported true class {minority_label}",
        ),
        (
            "most_uncertain",
            predictions,
            ["confidence", sample_id_column],
            [True, True],
            "lowest OOF confidence across the primary policy",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for case_type, subset, columns, ascending, rule in candidates:
        if subset.empty:
            continue
        item = subset.sort_values(columns, ascending=ascending, kind="stable").iloc[0]
        rows.append(
            {
                **identity.as_dict(),
                "case_type": case_type,
                "selection_rule": rule,
                "sample_index": _json_scalar(item[sample_id_column]),
                "outer_fold": int(item[fold_column]),
                "y_true": int(item["y_true"]),
                "y_pred": int(item["y_pred"]),
                "confidence": float(item["confidence"]),
                "correct": bool(int(item["y_true"]) == int(item["y_pred"])),
                "minority_true_label": minority_label,
                "attribution_warning": ATTRIBUTION_WARNING,
                "temporality_warning": TEMPORALITY_WARNING,
                "research_use_warning": RESEARCH_USE_WARNING,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class OOFShapEvidence:
    local_values: pd.DataFrame = field(repr=False, compare=False)
    global_importance: pd.DataFrame = field(repr=False, compare=False)
    class_importance: pd.DataFrame = field(repr=False, compare=False)
    fold_rankings: pd.DataFrame = field(repr=False, compare=False)
    stability_pairwise: pd.DataFrame = field(repr=False, compare=False)
    stability_summary: pd.DataFrame = field(repr=False, compare=False)
    representative_cases: pd.DataFrame = field(repr=False, compare=False)
    metadata: Mapping[str, Any]


def compute_exact_oof_grouped_shap(
    *,
    features: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    fold_models: Sequence[FoldModelReference],
    policy_features: Mapping[str, Sequence[str]],
    primary_policy: str,
    forbidden_features: Iterable[str],
    identity: ReplicationIdentity,
    labels: Sequence[int] = (2, 3, 4),
    feature_governance: Mapping[str, Mapping[str, Any]] | None = None,
    sample_id_column: str = "sample_index",
    fold_column: str = "outer_fold",
    top_k: int = 10,
    probability_atol: float = 1e-12,
    additivity_atol: float = 1e-4,
    shap_provider: ShapProvider | None = None,
) -> OOFShapEvidence:
    """Explain each sample only with its exact prediction-producing fold model."""

    if primary_policy not in policy_features:
        raise HRDatasetDiagnosticsError(f"Primary policy {primary_policy!r} has no feature contract.")
    if feature_policy_contract_sha256(policy_features) != identity.feature_policy_contract_sha256:
        raise HRDatasetDiagnosticsError("Named feature policies differ from the identity-bound contract.")
    if model_set_sha256(fold_models) != identity.model_set_sha256:
        raise HRDatasetDiagnosticsError("Fold model hashes differ from model_set_sha256.")
    if not isinstance(top_k, int) or top_k < 1:
        raise HRDatasetDiagnosticsError("top_k must be a positive integer.")
    labels = tuple(int(value) for value in labels)
    if len(labels) < 3 or len(set(labels)) != len(labels):
        raise HRDatasetDiagnosticsError("External performance SHAP requires unique multiclass labels.")
    raw_features = tuple(str(value) for value in policy_features[primary_policy])
    _validate_forbidden_features(raw_features, forbidden_features)
    missing_features = sorted(set(raw_features).difference(features.columns))
    if missing_features:
        raise HRDatasetDiagnosticsError(f"Primary SHAP features are absent: {missing_features}")

    probability_columns = {label: f"prob_class_{label}" for label in labels}
    predictions = _prediction_frame(
        oof_predictions,
        fold_assignments,
        features,
        identity=identity,
        primary_policy=primary_policy,
        labels=labels,
        sample_id_column=sample_id_column,
        fold_column=fold_column,
        probability_columns=probability_columns,
    )
    fold_lookup = {reference.outer_fold: reference for reference in fold_models}
    observed_folds = tuple(sorted(int(value) for value in predictions[fold_column].unique()))
    if tuple(sorted(fold_lookup)) != observed_folds:
        raise HRDatasetDiagnosticsError(
            f"Fold model set differs from prediction folds: models={sorted(fold_lookup)}, OOF={observed_folds}."
        )
    provider = shap_provider or _default_tree_shap_provider
    feature_by_id = _set_unique_index(features, sample_id_column, "SHAP feature rows")
    governance = feature_governance or {}
    local_rows: list[dict[str, Any]] = []
    fold_receipts: list[dict[str, Any]] = []

    for outer_fold in observed_folds:
        reference = fold_lookup[outer_fold]
        pipeline = reference.load()
        fold_predictions = predictions[predictions[fold_column].astype(int) == outer_fold].copy()
        if "source_outer_model_sha256" not in fold_predictions.columns:
            raise HRDatasetDiagnosticsError(
                "OOF predictions lack source_outer_model_sha256 exact-model identity."
            )
        observed_source_hashes = set(fold_predictions["source_outer_model_sha256"].astype(str))
        if observed_source_hashes != {reference.model_sha256}:
            raise HRDatasetDiagnosticsError(
                f"Fold {outer_fold} OOF source model hash differs from FoldModelReference: "
                f"{sorted(observed_source_hashes)}."
            )
        sample_ids = fold_predictions[sample_id_column].tolist()
        X_test = feature_by_id.loc[sample_ids, list(raw_features)].copy()
        fitted_names = getattr(pipeline, "feature_names_in_", None)
        if fitted_names is None and hasattr(pipeline, "named_steps"):
            fitted_names = getattr(pipeline.named_steps.get("preprocessor"), "feature_names_in_", None)
        if fitted_names is None or tuple(str(value) for value in fitted_names) != raw_features:
            raise HRDatasetDiagnosticsError(
                f"Fold {outer_fold} fitted raw feature order differs from the primary policy contract."
            )
        replay = _aligned_probabilities(pipeline, X_test, labels)
        expected = fold_predictions[[probability_columns[label] for label in labels]].to_numpy(float)
        replay_error = float(np.max(np.abs(replay - expected)))
        if replay_error > probability_atol:
            raise HRDatasetDiagnosticsError(
                f"Fold {outer_fold} model does not replay its OOF probabilities; max error={replay_error}."
            )
        predicted = np.asarray(labels, dtype=int)[np.argmax(replay, axis=1)]
        if not np.array_equal(predicted, fold_predictions["y_pred"].to_numpy(int)):
            raise HRDatasetDiagnosticsError(f"Fold {outer_fold} model replay labels differ from OOF rows.")

        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is None:
            raise HRDatasetDiagnosticsError("Fold pipeline has no fitted preprocessor step.")
        transformed = preprocessor.transform(X_test)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed, dtype=float)
        if transformed.ndim != 2 or not np.isfinite(transformed).all():
            raise HRDatasetDiagnosticsError("Transformed fold features are not a finite two-dimensional array.")
        groups, transformed_names, group_index, lineage_hash = _transformed_feature_groups(
            preprocessor, raw_features
        )
        _validate_forbidden_features(groups, forbidden_features)
        computation = provider(pipeline, transformed, labels)
        if computation.axis_source is None:
            canonical, axis_source = canonicalize_multiclass_shap(
                computation.values,
                n_samples=len(X_test),
                n_transformed_features=transformed.shape[1],
                n_classes=len(labels),
            )
        else:
            canonical = np.asarray(computation.values, dtype=float)
            if canonical.shape != (len(X_test), transformed.shape[1], len(labels)):
                raise HRDatasetDiagnosticsError(
                    "Provider-declared canonical SHAP has incompatible sample/feature/class axes."
                )
            axis_source = str(computation.axis_source)
        base_values = _normalise_base_values(computation.base_values, len(labels))
        margins = _normalise_margins(computation.margins, len(X_test), len(labels))
        reconstructed = base_values.reshape(1, -1) + canonical.sum(axis=1)
        additivity_error = float(np.max(np.abs(reconstructed - margins)))
        if additivity_error > additivity_atol:
            raise HRDatasetDiagnosticsError(
                f"Fold {outer_fold} SHAP additivity failed; max error={additivity_error}."
            )
        grouped = _group_transformed_shap(canonical, group_index, len(groups))
        fold_receipts.append(
            {
                "outer_fold": outer_fold,
                "model_sha256": reference.model_sha256,
                "transformed_lineage_sha256": lineage_hash,
                "n_test": len(X_test),
                "n_raw_features": len(groups),
                "n_transformed_features": len(transformed_names),
                "axis_source": axis_source,
                "prediction_replay_max_abs_error": replay_error,
                "shap_additivity_max_abs_error": additivity_error,
            }
        )
        for sample_offset, sample_id in enumerate(sample_ids):
            prediction = fold_predictions.iloc[sample_offset]
            for class_index, class_label in enumerate(labels):
                absolute = np.abs(grouped[sample_offset, :, class_index])
                order = sorted(range(len(groups)), key=lambda index: (-absolute[index], groups[index]))
                ranks = {feature_index: rank for rank, feature_index in enumerate(order, start=1)}
                for feature_index, feature in enumerate(groups):
                    feature_meta = governance.get(feature, {})
                    value = float(grouped[sample_offset, feature_index, class_index])
                    local_rows.append(
                        {
                            **identity.as_dict(),
                            "task_type": PRIMARY_TASK,
                            "role": "independent external performance-target replication",
                            "policy": primary_policy,
                            "sample_index": _json_scalar(sample_id),
                            "outer_fold": outer_fold,
                            "model_sha256": reference.model_sha256,
                            "transformed_lineage_sha256": lineage_hash,
                            "shap_axis_contract": "sample_raw_feature_class",
                            "shap_axis_source": axis_source,
                            "prediction_replay_max_abs_error": replay_error,
                            "shap_additivity_max_abs_error": additivity_error,
                            "y_true": int(prediction["y_true"]),
                            "y_pred": int(prediction["y_pred"]),
                            "confidence": float(prediction["confidence"]),
                            "class_label": int(class_label),
                            "is_predicted_class": bool(int(prediction["y_pred"]) == class_label),
                            "feature": feature,
                            "feature_value": _json_scalar(X_test.iloc[sample_offset][feature]),
                            "grouped_shap_value": value,
                            "abs_grouped_shap_value": abs(value),
                            "within_case_class_abs_rank": ranks[feature_index],
                            "governance_category": str(
                                feature_meta.get("governance_category", "external_context_dependent")
                            ),
                            "proxy_watchlist": bool(feature_meta.get("proxy_watchlist", False)),
                            "temporality_status": str(
                                feature_meta.get("temporality_status", "requires_context_verification")
                            ),
                            "attribution_warning": ATTRIBUTION_WARNING,
                            "noncausality_warning": ATTRIBUTION_WARNING,
                            "temporality_warning": TEMPORALITY_WARNING,
                            "research_use_warning": RESEARCH_USE_WARNING,
                        }
                    )

    local = pd.DataFrame(local_rows)
    expected_rows = len(features) * len(raw_features) * len(labels)
    if len(local) != expected_rows:
        raise HRDatasetDiagnosticsError(
            f"Incomplete exact-fold SHAP coverage: expected {expected_rows}, observed {len(local)}."
        )
    if local[["sample_index", "class_label", "feature"]].duplicated().any():
        raise HRDatasetDiagnosticsError("Exact-fold SHAP contains duplicate sample/class/feature rows.")
    _validate_forbidden_features(local["feature"].astype(str).unique(), forbidden_features)

    global_frame = (
        local.groupby("feature", as_index=False)["abs_grouped_shap_value"]
        .mean()
        .rename(columns={"abs_grouped_shap_value": "mean_abs_grouped_shap"})
    )
    global_frame = _rank_importance(global_frame, ())
    class_frame = (
        local.groupby(["class_label", "feature"], as_index=False)["abs_grouped_shap_value"]
        .mean()
        .rename(columns={"abs_grouped_shap_value": "mean_abs_grouped_shap"})
    )
    class_frame = _rank_importance(class_frame, ("class_label",))
    fold_frame = (
        local.groupby(["outer_fold", "feature"], as_index=False)["abs_grouped_shap_value"]
        .mean()
        .rename(columns={"abs_grouped_shap_value": "mean_abs_grouped_shap"})
    )
    fold_frame = _rank_importance(fold_frame, ("outer_fold",))
    for frame in (global_frame, class_frame, fold_frame):
        for offset, (field_name, value) in enumerate(reversed(list(identity.as_dict().items()))):
            frame.insert(0, field_name, value)
        frame["task_type"] = PRIMARY_TASK
        frame["policy"] = primary_policy
        frame["attribution_warning"] = ATTRIBUTION_WARNING
        frame["temporality_warning"] = TEMPORALITY_WARNING

    stability_rows: list[dict[str, Any]] = []
    effective_top_k = min(top_k, len(raw_features))
    by_fold = {
        int(fold): scoped.set_index("feature")["rank"].astype(int).to_dict()
        for fold, scoped in fold_frame.groupby("outer_fold", sort=True)
    }
    for fold_a, fold_b in itertools.combinations(sorted(by_fold), 2):
        ranks_a, ranks_b = by_fold[fold_a], by_fold[fold_b]
        features_ordered = sorted(set(ranks_a) | set(ranks_b))
        if set(ranks_a) != set(ranks_b):
            raise HRDatasetDiagnosticsError("Fold SHAP rankings cover different raw features.")
        top_a = {feature for feature, rank in ranks_a.items() if rank <= effective_top_k}
        top_b = {feature for feature, rank in ranks_b.items() if rank <= effective_top_k}
        union = top_a | top_b
        jaccard = 1.0 if not union else float(len(top_a & top_b) / len(union))
        if len(features_ordered) == 1:
            correlation = 1.0
        else:
            correlation = float(
                spearmanr(
                    [ranks_a[feature] for feature in features_ordered],
                    [ranks_b[feature] for feature in features_ordered],
                ).statistic
            )
        if not math.isfinite(correlation):
            raise HRDatasetDiagnosticsError("Fold SHAP Spearman stability is non-finite.")
        stability_rows.append(
            {
                **identity.as_dict(),
                "task_type": PRIMARY_TASK,
                "policy": primary_policy,
                "fold_a": fold_a,
                "fold_b": fold_b,
                "top_k": effective_top_k,
                "top_k_jaccard": jaccard,
                "spearman_all_feature_ranks": correlation,
                "uncertainty_scope": "descriptive_dependent_fold_pairs_no_confidence_interval",
                "attribution_warning": ATTRIBUTION_WARNING,
            }
        )
    pairwise = pd.DataFrame(stability_rows)
    summary_rows: list[dict[str, Any]] = []
    for metric in ("top_k_jaccard", "spearman_all_feature_ranks"):
        values = pairwise[metric].to_numpy(float) if not pairwise.empty else np.asarray([], dtype=float)
        summary_rows.append(
            {
                **identity.as_dict(),
                "task_type": PRIMARY_TASK,
                "policy": primary_policy,
                "metric": metric,
                "top_k": effective_top_k if metric == "top_k_jaccard" else None,
                "n_dependent_fold_pairs": len(values),
                "mean": float(values.mean()) if len(values) else None,
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else None,
                "minimum": float(values.min()) if len(values) else None,
                "maximum": float(values.max()) if len(values) else None,
                "confidence_interval_applicable": False,
                "uncertainty_scope": "descriptive_dependent_fold_pairs_no_confidence_interval",
                "attribution_warning": ATTRIBUTION_WARNING,
            }
        )
    representatives = _representative_cases(
        predictions,
        identity=identity,
        sample_id_column=sample_id_column,
        fold_column=fold_column,
    )
    metadata = MappingProxyType(
        {
            **identity.as_dict(),
            "task_type": PRIMARY_TASK,
            "role": "independent external performance-target replication",
            "policy": primary_policy,
            "labels": list(labels),
            "n_samples": len(features),
            "n_outer_folds": len(observed_folds),
            "n_raw_features": len(raw_features),
            "n_local_rows": len(local),
            "top_k": effective_top_k,
            "outer_fold_assignment_sha256": outer_fold_assignment_sha256(
                fold_assignments,
                sample_id_column=sample_id_column,
                fold_column=fold_column,
            ),
            "fold_receipts": fold_receipts,
            "prediction_replay_atol": probability_atol,
            "shap_additivity_atol": additivity_atol,
            "model_refit_in_diagnostic": False,
            "confidence_interval_for_fold_pairs": False,
            "attribution_warning": ATTRIBUTION_WARNING,
            "temporality_warning": TEMPORALITY_WARNING,
            "research_use_warning": RESEARCH_USE_WARNING,
        }
    )
    return OOFShapEvidence(
        local_values=local,
        global_importance=global_frame,
        class_importance=class_frame,
        fold_rankings=fold_frame,
        stability_pairwise=pairwise,
        stability_summary=pd.DataFrame(summary_rows),
        representative_cases=representatives,
        metadata=metadata,
    )


@dataclass(frozen=True)
class AuditAttributeSpec:
    name: str
    category: Literal["protected_sensitive", "exploratory_operational"]
    transform: Literal["categorical", "numeric_bins"] = "categorical"
    bin_edges: tuple[float, ...] = ()
    bin_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or self.category not in SUBGROUP_CATEGORIES:
            raise HRDatasetDiagnosticsError("Audit attribute name/category is invalid.")
        if self.transform == "numeric_bins":
            if len(self.bin_edges) < 3 or len(self.bin_labels) != len(self.bin_edges) - 1:
                raise HRDatasetDiagnosticsError(
                    "numeric_bins requires ordered edges and exactly one label per interval."
                )
            if any(left >= right for left, right in zip(self.bin_edges, self.bin_edges[1:])):
                raise HRDatasetDiagnosticsError("numeric bin edges must be strictly increasing.")
        elif self.transform != "categorical":
            raise HRDatasetDiagnosticsError(f"Unknown audit transform: {self.transform}")


def _audit_values(frame: pd.DataFrame, spec: AuditAttributeSpec) -> pd.Series:
    if spec.name not in frame.columns:
        raise HRDatasetDiagnosticsError(f"Audit attribute is absent: {spec.name}")
    values = frame[spec.name]
    if spec.transform == "categorical":
        return values.astype("string").fillna("__MISSING__").astype(str)
    numeric = pd.to_numeric(values, errors="coerce")
    binned = pd.cut(
        numeric,
        bins=list(spec.bin_edges),
        labels=list(spec.bin_labels),
        include_lowest=True,
        right=True,
    )
    return binned.astype("string").fillna("__MISSING__").astype(str)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _point_group_metric_rows(
    predictions: pd.DataFrame,
    audit: pd.DataFrame,
    attributes: Sequence[AuditAttributeSpec],
    labels: Sequence[int],
    *,
    identity: ReplicationIdentity,
    sample_id_column: str,
    policy_column: str,
    minimum_group_support: int,
    minimum_metric_denominator: int,
) -> pd.DataFrame:
    audit_by_id = _set_unique_index(audit, sample_id_column, "Subgroup audit rows")
    rows: list[dict[str, Any]] = []
    for policy, scoped in predictions.groupby(policy_column, sort=True):
        scoped = scoped.sort_values(sample_id_column, kind="stable").copy()
        for attribute in attributes:
            group_values = _audit_values(audit_by_id.loc[scoped[sample_id_column]], attribute).to_numpy()
            enriched = scoped.assign(_audit_group=group_values)
            for group_value, group in enriched.groupby("_audit_group", sort=True, dropna=False):
                y_true = group["y_true"].to_numpy(int)
                y_pred = group["y_pred"].to_numpy(int)
                group_n = len(group)
                base = {
                    **identity.as_dict(),
                    "task_type": PRIMARY_TASK,
                    "policy": str(policy),
                    "attribute": attribute.name,
                    "interpretation_category": attribute.category,
                    "group_value": str(group_value),
                    "group_n": group_n,
                    "minimum_group_support_threshold": minimum_group_support,
                    "group_support_eligible": bool(group_n >= minimum_group_support),
                    "limitations": SUBGROUP_LIMITATION,
                }
                for metric, value in (
                    ("accuracy", float(accuracy_score(y_true, y_pred))),
                    (
                        "macro_f1",
                        float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
                    ),
                ):
                    rows.append(
                        {
                            **base,
                            "metric": metric,
                            "class_label": None,
                            "metric_denominator_kind": "subgroup_rows",
                            "metric_denominator": group_n,
                            "minimum_metric_denominator_threshold": minimum_metric_denominator,
                            "metric_denominator_eligible": True,
                            "eligible_for_gap": bool(group_n >= minimum_group_support),
                            "point_estimate": value,
                        }
                    )
                for label in labels:
                    label = int(label)
                    true_positive = int(np.sum((y_true == label) & (y_pred == label)))
                    false_positive = int(np.sum((y_true != label) & (y_pred == label)))
                    false_negative = int(np.sum((y_true == label) & (y_pred != label)))
                    true_negative = int(np.sum((y_true != label) & (y_pred != label)))
                    actual_support = true_positive + false_negative
                    predicted_support = true_positive + false_positive
                    metric_values = {
                        "positive_prediction_rate": (
                            _safe_ratio(predicted_support, group_n),
                            group_n,
                            "subgroup_rows",
                        ),
                        "true_positive_rate": (
                            _safe_ratio(true_positive, actual_support),
                            actual_support,
                            "actual_class_rows",
                        ),
                        "false_positive_rate": (
                            _safe_ratio(false_positive, false_positive + true_negative),
                            false_positive + true_negative,
                            "actual_nonclass_rows",
                        ),
                        "precision": (
                            _safe_ratio(true_positive, predicted_support),
                            predicted_support,
                            "predicted_class_rows",
                        ),
                        "mean_predicted_probability": (
                            float(group[f"prob_class_{label}"].mean()),
                            group_n,
                            "subgroup_rows",
                        ),
                    }
                    for metric, (value, denominator, denominator_kind) in metric_values.items():
                        denominator_eligible = denominator >= minimum_metric_denominator
                        rows.append(
                            {
                                **base,
                                "metric": metric,
                                "class_label": label,
                                "metric_denominator_kind": denominator_kind,
                                "metric_denominator": int(denominator),
                                "minimum_metric_denominator_threshold": minimum_metric_denominator,
                                "metric_denominator_eligible": bool(denominator_eligible),
                                "eligible_for_gap": bool(
                                    group_n >= minimum_group_support
                                    and denominator_eligible
                                    and math.isfinite(value)
                                ),
                                "point_estimate": value if math.isfinite(value) else None,
                            }
                        )
    return pd.DataFrame(rows)


def _bootstrap_group_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability: np.ndarray,
    group_values: np.ndarray,
    sample_positions: np.ndarray,
    *,
    group_value: str,
    metric: str,
    class_label: int | None,
    labels: Sequence[int],
    minimum_group_support: int,
    minimum_metric_denominator: int,
) -> np.ndarray:
    true_draw = y_true[sample_positions]
    pred_draw = y_pred[sample_positions]
    group_draw = group_values[sample_positions]
    mask = group_draw == group_value
    group_n = mask.sum(axis=1)
    output = np.full(len(sample_positions), np.nan, dtype=float)
    support_valid = group_n >= minimum_group_support
    if metric == "accuracy":
        numerator = ((true_draw == pred_draw) & mask).sum(axis=1)
        valid = support_valid & (group_n > 0)
        output[valid] = numerator[valid] / group_n[valid]
        return output
    if metric == "macro_f1":
        values = np.zeros((len(sample_positions), len(labels)), dtype=float)
        for label_index, label in enumerate(labels):
            tp = ((true_draw == label) & (pred_draw == label) & mask).sum(axis=1)
            fp = ((true_draw != label) & (pred_draw == label) & mask).sum(axis=1)
            fn = ((true_draw == label) & (pred_draw != label) & mask).sum(axis=1)
            denominator = 2 * tp + fp + fn
            values[:, label_index] = np.divide(
                2 * tp,
                denominator,
                out=np.zeros(len(sample_positions), dtype=float),
                where=denominator > 0,
            )
        output[support_valid] = values[support_valid].mean(axis=1)
        return output

    if class_label is None:
        raise HRDatasetDiagnosticsError(f"Class-specific metric {metric} has no class label.")
    label_index = list(labels).index(int(class_label))
    tp = ((true_draw == class_label) & (pred_draw == class_label) & mask).sum(axis=1)
    fp = ((true_draw != class_label) & (pred_draw == class_label) & mask).sum(axis=1)
    fn = ((true_draw == class_label) & (pred_draw != class_label) & mask).sum(axis=1)
    tn = ((true_draw != class_label) & (pred_draw != class_label) & mask).sum(axis=1)
    if metric == "positive_prediction_rate":
        numerator, denominator = tp + fp, group_n
    elif metric == "true_positive_rate":
        numerator, denominator = tp, tp + fn
    elif metric == "false_positive_rate":
        numerator, denominator = fp, fp + tn
    elif metric == "precision":
        numerator, denominator = tp, tp + fp
    elif metric == "mean_predicted_probability":
        probability_draw = probability[sample_positions, label_index]
        numerator, denominator = (probability_draw * mask).sum(axis=1), group_n
    else:
        raise HRDatasetDiagnosticsError(f"Unknown subgroup metric: {metric}")
    valid = support_valid & (denominator >= minimum_metric_denominator)
    output[valid] = numerator[valid] / denominator[valid]
    return output


@dataclass(frozen=True)
class SubgroupDiagnosticsEvidence:
    group_metrics: pd.DataFrame = field(repr=False, compare=False)
    disparity_intervals: pd.DataFrame = field(repr=False, compare=False)
    metadata: Mapping[str, Any]


def compute_support_aware_subgroup_diagnostics(
    *,
    oof_predictions: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    audit_frame: pd.DataFrame,
    attributes: Sequence[AuditAttributeSpec],
    identity: ReplicationIdentity,
    labels: Sequence[int] = (2, 3, 4),
    sample_id_column: str = "sample_index",
    fold_column: str = "outer_fold",
    policy_column: str = "policy",
    minimum_group_support: int = 30,
    minimum_metric_denominator: int = 10,
    n_resamples: int = REQUIRED_BOOTSTRAP_RESAMPLES,
    confidence_level: float = 0.95,
    seed: int = 42,
    batch_size: int = 200,
    minimum_valid_fraction: float = 0.8,
    wide_interval_threshold: float = 0.25,
) -> SubgroupDiagnosticsEvidence:
    """Compute support-aware external OOF subgroup gaps without fairness claims."""

    if n_resamples != REQUIRED_BOOTSTRAP_RESAMPLES:
        raise HRDatasetDiagnosticsError(
            f"Canonical subgroup diagnostics require exactly {REQUIRED_BOOTSTRAP_RESAMPLES} resamples."
        )
    if minimum_group_support < 1 or minimum_metric_denominator < 1:
        raise HRDatasetDiagnosticsError("Support thresholds must be positive integers.")
    if batch_size < 1 or not 0.0 < minimum_valid_fraction <= 1.0:
        raise HRDatasetDiagnosticsError("Bootstrap batch/valid-fraction settings are invalid.")
    if not attributes or len({attribute.name for attribute in attributes}) != len(attributes):
        raise HRDatasetDiagnosticsError("Audit attribute specifications must be non-empty and unique.")
    _validate_identity_columns(oof_predictions, identity, "Subgroup OOF predictions")
    labels = tuple(int(value) for value in labels)
    required = {
        sample_id_column,
        fold_column,
        policy_column,
        "y_true",
        "y_pred",
        *(f"prob_class_{label}" for label in labels),
    }
    missing = sorted(required.difference(oof_predictions.columns))
    if missing:
        raise HRDatasetDiagnosticsError(f"Subgroup OOF predictions are missing columns: {missing}")
    folds = fold_assignments[[sample_id_column, fold_column]].copy()
    if folds.empty or folds[sample_id_column].duplicated().any():
        raise HRDatasetDiagnosticsError("Subgroup fold assignments must be exactly once per sample.")
    if audit_frame.empty or audit_frame[sample_id_column].duplicated().any():
        raise HRDatasetDiagnosticsError("Audit data must be exactly once per sample.")
    missing_attributes = sorted({attribute.name for attribute in attributes}.difference(audit_frame.columns))
    if missing_attributes:
        raise HRDatasetDiagnosticsError(f"Audit data are missing declared fields: {missing_attributes}")
    expected_ids = set(folds[sample_id_column].tolist())
    if set(audit_frame[sample_id_column].tolist()) != expected_ids:
        raise HRDatasetDiagnosticsError("Audit and fold sample sets differ.")
    policies = tuple(sorted(oof_predictions[policy_column].astype(str).unique()))
    if not policies:
        raise HRDatasetDiagnosticsError("No subgroup policies are present.")
    base: pd.DataFrame | None = None
    for policy, scoped in oof_predictions.groupby(policy_column, sort=True):
        if scoped[sample_id_column].duplicated().any() or set(scoped[sample_id_column]) != expected_ids:
            raise HRDatasetDiagnosticsError(f"Policy {policy!r} is not exactly-once OOF.")
        merged = scoped.merge(folds, on=sample_id_column, suffixes=("", "_declared"), validate="one_to_one")
        if not np.array_equal(
            merged[fold_column].to_numpy(int), merged[f"{fold_column}_declared"].to_numpy(int)
        ):
            raise HRDatasetDiagnosticsError(f"Policy {policy!r} fold identity differs from assignments.")
        aligned = scoped.sort_values(sample_id_column, kind="stable")
        if base is None:
            base = aligned[[sample_id_column, fold_column, "y_true"]].copy()
        elif not aligned[[sample_id_column, fold_column, "y_true"]].reset_index(drop=True).equals(
            base.reset_index(drop=True)
        ):
            raise HRDatasetDiagnosticsError("Subgroup policies do not share sample/fold/target identity.")
        probability = aligned[[f"prob_class_{label}" for label in labels]].to_numpy(float)
        if not np.isfinite(probability).all() or not np.allclose(
            probability.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=_PROBABILITY_SUM_ATOL,
        ):
            raise HRDatasetDiagnosticsError(f"Policy {policy!r} probabilities are invalid.")
    assert base is not None
    point = _point_group_metric_rows(
        oof_predictions,
        audit_frame,
        attributes,
        labels,
        identity=identity,
        sample_id_column=sample_id_column,
        policy_column=policy_column,
        minimum_group_support=minimum_group_support,
        minimum_metric_denominator=minimum_metric_denominator,
    )
    protocol = BootstrapProtocol(
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        strata_columns=("outer_fold", "y_true"),
        method="paired_stratified_percentile",
        quantile_method="linear",
    )
    try:
        plan = generate_stratified_resample_indices(base, protocol, sample_id_column=sample_id_column)
    except OOFBootstrapError as exc:
        raise HRDatasetDiagnosticsError(f"Cannot generate subgroup bootstrap plan: {exc}") from exc
    sorted_ids = list(plan.sorted_sample_ids)
    audit_by_id = _set_unique_index(
        audit_frame, sample_id_column, "Subgroup audit rows"
    ).loc[sorted_ids]
    attribute_values = {attribute.name: _audit_values(audit_by_id, attribute).to_numpy() for attribute in attributes}
    alpha = 1.0 - confidence_level
    interval_rows: list[dict[str, Any]] = []
    grouping = [policy_column, "attribute", "interpretation_category", "metric", "class_label"]
    for keys, scoped_point in point.groupby(grouping, sort=True, dropna=False):
        policy, attribute_name, category, metric, class_label_value = keys
        class_label = None if pd.isna(class_label_value) else int(class_label_value)
        eligible = scoped_point[scoped_point["eligible_for_gap"].astype(bool)].copy()
        groups = sorted(eligible["group_value"].astype(str).unique())
        base_row = {
            **identity.as_dict(),
            "task_type": PRIMARY_TASK,
            "policy": str(policy),
            "attribute": str(attribute_name),
            "interpretation_category": str(category),
            "metric": str(metric),
            "class_label": class_label,
            "n_groups_observed": int(scoped_point["group_value"].nunique()),
            "n_groups_eligible": len(groups),
            "eligible_groups_json": _canonical_json(groups),
            "minimum_group_support_threshold": minimum_group_support,
            "minimum_metric_denominator_threshold": minimum_metric_denominator,
            "n_resamples": n_resamples,
            "confidence_level": confidence_level,
            "bootstrap_method": "paired_stratified_sample_level_percentile",
            "bootstrap_batch_size": batch_size,
            "resample_hash": plan.resample_hash,
            "inference_scope": "pointwise_descriptive",
            "multiplicity_adjustment": "none",
            "limitations": SUBGROUP_LIMITATION,
        }
        if len(groups) < 2:
            interval_rows.append(
                {
                    **base_row,
                    "point_estimate_gap": None,
                    "min_group_value": None,
                    "max_group_value": None,
                    "minimum_eligible_group_support": (
                        int(eligible["group_n"].min()) if not eligible.empty else None
                    ),
                    "minimum_eligible_metric_denominator": (
                        int(eligible["metric_denominator"].min()) if not eligible.empty else None
                    ),
                    "ci_low": None,
                    "ci_high": None,
                    "interval_width": None,
                    "n_valid_bootstrap": 0,
                    "valid_bootstrap_fraction": 0.0,
                    "estimate_status": "insufficient_subgroup_or_metric_support",
                    "headline_eligible": False,
                }
            )
            continue
        point_by_group = eligible.set_index("group_value")["point_estimate"].astype(float)
        min_group = str(point_by_group.idxmin())
        max_group = str(point_by_group.idxmax())
        point_gap = float(point_by_group.max() - point_by_group.min())
        prediction = _set_unique_index(
            oof_predictions[oof_predictions[policy_column].astype(str) == str(policy)],
            sample_id_column,
            f"Subgroup OOF policy {policy}",
        ).loc[sorted_ids]
        y_true = prediction["y_true"].to_numpy(int)
        y_pred = prediction["y_pred"].to_numpy(int)
        probability = prediction[[f"prob_class_{label}" for label in labels]].to_numpy(float)
        draws = np.full(n_resamples, np.nan, dtype=float)
        values_for_attribute = attribute_values[str(attribute_name)]
        for start in range(0, n_resamples, batch_size):
            stop = min(start + batch_size, n_resamples)
            indices = plan.indices[start:stop]
            group_draws = np.column_stack(
                [
                    _bootstrap_group_metric(
                        y_true,
                        y_pred,
                        probability,
                        values_for_attribute,
                        indices,
                        group_value=group,
                        metric=str(metric),
                        class_label=class_label,
                        labels=labels,
                        minimum_group_support=minimum_group_support,
                        minimum_metric_denominator=minimum_metric_denominator,
                    )
                    for group in groups
                ]
            )
            valid = np.isfinite(group_draws).all(axis=1)
            draws[start:stop][valid] = (
                group_draws[valid].max(axis=1) - group_draws[valid].min(axis=1)
            )
        valid_draws = draws[np.isfinite(draws)]
        n_valid = len(valid_draws)
        valid_fraction = float(n_valid / n_resamples)
        if n_valid:
            low, high = np.quantile(
                valid_draws,
                [alpha / 2.0, 1.0 - alpha / 2.0],
                method="linear",
            )
            low = float(np.clip(low, 0.0, 1.0))
            high = float(np.clip(high, 0.0, 1.0))
            width = high - low
        else:
            low = high = width = None
        if valid_fraction < minimum_valid_fraction:
            status = "unstable_insufficient_valid_bootstrap_replicates"
        elif width is not None and width > wide_interval_threshold:
            status = "support_sufficient_but_interval_wide"
        else:
            status = "support_sufficient_descriptive_estimate"
        interval_rows.append(
            {
                **base_row,
                "point_estimate_gap": point_gap,
                "min_group_value": min_group,
                "max_group_value": max_group,
                "minimum_eligible_group_support": int(eligible["group_n"].min()),
                "minimum_eligible_metric_denominator": int(
                    eligible["metric_denominator"].min()
                ),
                "ci_low": low,
                "ci_high": high,
                "interval_width": width,
                "n_valid_bootstrap": n_valid,
                "valid_bootstrap_fraction": valid_fraction,
                "estimate_status": status,
                "headline_eligible": bool(status == "support_sufficient_descriptive_estimate"),
            }
        )
    intervals = pd.DataFrame(interval_rows)
    finite = intervals.dropna(subset=["point_estimate_gap", "ci_low", "ci_high"])
    if not finite.empty:
        for column in ("point_estimate_gap", "ci_low", "ci_high"):
            values = finite[column].to_numpy(float)
            if (values < 0.0).any() or (values > 1.0).any() or not np.isfinite(values).all():
                raise HRDatasetDiagnosticsError(f"Subgroup {column} lies outside [0, 1].")
        if (finite["ci_low"] > finite["ci_high"]).any():
            raise HRDatasetDiagnosticsError("Subgroup bootstrap interval bounds are reversed.")
    metadata = MappingProxyType(
        {
            **identity.as_dict(),
            "task_type": PRIMARY_TASK,
            "analysis_type": "support_aware_subgroup_diagnostics_not_fairness_proof",
            "policies": list(policies),
            "attributes": [
                {
                    "name": attribute.name,
                    "category": attribute.category,
                    "transform": attribute.transform,
                }
                for attribute in attributes
            ],
            "minimum_group_support": minimum_group_support,
            "minimum_metric_denominator": minimum_metric_denominator,
            "n_resamples": n_resamples,
            "confidence_level": confidence_level,
            "seed": seed,
            "batch_size": batch_size,
            "resample_hash": plan.resample_hash,
            "stratum_counts": dict(plan.stratum_counts),
            "strata_columns": ["outer_fold", "y_true"],
            "minimum_valid_fraction": minimum_valid_fraction,
            "wide_interval_threshold": wide_interval_threshold,
            "inference_scope": "pointwise_descriptive",
            "multiplicity_adjustment": "none",
            "limitations": SUBGROUP_LIMITATION,
        }
    )
    return SubgroupDiagnosticsEvidence(
        group_metrics=point,
        disparity_intervals=intervals,
        metadata=metadata,
    )


def _proxy_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame.columns if column not in numeric]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise HRDatasetDiagnosticsError("Proxy predictor contract has no usable columns.")
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def _proxy_pipeline(frame: pd.DataFrame, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _proxy_preprocessor(frame)),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    tol=1e-4,
                    random_state=seed,
                ),
            ),
        ]
    )


def _proxy_metric_values(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
    }


def _proxy_metric_draws(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int],
    indices: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    output = {metric: np.empty(len(indices), dtype=float) for metric in PROXY_METRICS}
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        positions = indices[start:stop]
        true_draw = y_true[positions]
        pred_draw = y_pred[positions]
        output["accuracy"][start:stop] = (true_draw == pred_draw).mean(axis=1)
        recall = np.zeros((len(positions), len(labels)), dtype=float)
        f1 = np.zeros_like(recall)
        for offset, label in enumerate(labels):
            tp = ((true_draw == label) & (pred_draw == label)).sum(axis=1)
            fp = ((true_draw != label) & (pred_draw == label)).sum(axis=1)
            fn = ((true_draw == label) & (pred_draw != label)).sum(axis=1)
            actual = tp + fn
            f1_denominator = 2 * tp + fp + fn
            recall[:, offset] = np.divide(
                tp,
                actual,
                out=np.zeros(len(positions), dtype=float),
                where=actual > 0,
            )
            f1[:, offset] = np.divide(
                2 * tp,
                f1_denominator,
                out=np.zeros(len(positions), dtype=float),
                where=f1_denominator > 0,
            )
        output["balanced_accuracy"][start:stop] = recall.mean(axis=1)
        output["macro_f1"][start:stop] = f1.mean(axis=1)
    return output


@dataclass(frozen=True)
class ProxyReconstructabilityEvidence:
    status: pd.DataFrame = field(repr=False, compare=False)
    feature_contracts: pd.DataFrame = field(repr=False, compare=False)
    oof_predictions: pd.DataFrame = field(repr=False, compare=False)
    fold_metrics: pd.DataFrame = field(repr=False, compare=False)
    metric_intervals: pd.DataFrame = field(repr=False, compare=False)
    paired_differences: pd.DataFrame = field(repr=False, compare=False)
    metadata: Mapping[str, Any]


def compute_proxy_reconstructability(
    *,
    features: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    audit_frame: pd.DataFrame,
    predictor_sets: Mapping[str, Sequence[str]],
    proxy_target: str,
    proxy_aliases: Iterable[str],
    identity: ReplicationIdentity,
    sample_id_column: str = "sample_index",
    fold_column: str = "outer_fold",
    seed: int = 42,
    n_resamples: int = REQUIRED_BOOTSTRAP_RESAMPLES,
    confidence_level: float = 0.95,
    batch_size: int = 200,
) -> ProxyReconstructabilityEvidence:
    """Estimate nominal proxy reconstructability, or fail closed to not-estimated.

    No proxy-target class is merged or dropped.  If any outer training fold
    lacks an observed full-data target class, no classifier is fitted and the
    returned status records the exact class counts and deficient folds.
    """

    if n_resamples != REQUIRED_BOOTSTRAP_RESAMPLES:
        raise HRDatasetDiagnosticsError(
            f"Canonical proxy diagnostics require exactly {REQUIRED_BOOTSTRAP_RESAMPLES} resamples."
        )
    if not predictor_sets:
        raise HRDatasetDiagnosticsError("At least one proxy predictor set is required.")
    if proxy_target not in audit_frame.columns:
        raise HRDatasetDiagnosticsError(f"Proxy target is absent from audit data: {proxy_target}")
    forbidden = {proxy_target, *[str(value) for value in proxy_aliases]}
    normalised_forbidden = {_normalise_name(value) for value in forbidden}
    contract_rows: list[dict[str, Any]] = []
    contract_hashes: dict[str, str] = {}
    for system_id, predictors in sorted(predictor_sets.items()):
        values = [str(value) for value in predictors]
        if not values or len(set(values)) != len(values):
            raise HRDatasetDiagnosticsError(f"Proxy predictor set {system_id!r} is empty or duplicated.")
        collision = sorted(value for value in values if _normalise_name(value) in normalised_forbidden)
        if collision:
            raise HRDatasetDiagnosticsError(
                f"Proxy target/aliases enter predictor set {system_id!r}: {collision}"
            )
        missing = sorted(set(values).difference(features.columns))
        if missing:
            raise HRDatasetDiagnosticsError(f"Proxy predictor set {system_id!r} is missing: {missing}")
        contract_hash = _sha256_json({"system_id": system_id, "predictors": values})
        contract_hashes[str(system_id)] = contract_hash
        contract_rows.append(
            {
                **identity.as_dict(),
                "task_type": PROXY_TASK,
                "system_id": str(system_id),
                "proxy_target": proxy_target,
                "proxy_aliases_json": _canonical_json(sorted(forbidden - {proxy_target})),
                "predictors_json": _canonical_json(values),
                "n_predictors": len(values),
                "predictor_contract_sha256": contract_hash,
                "proxy_target_and_aliases_absent": True,
                "limitations": PROXY_LIMITATION,
            }
        )
    folds = fold_assignments[[sample_id_column, fold_column]].copy()
    if folds.empty or folds[sample_id_column].duplicated().any():
        raise HRDatasetDiagnosticsError("Proxy fold assignments must be exactly once per sample.")
    if features.empty or features[sample_id_column].duplicated().any():
        raise HRDatasetDiagnosticsError("Proxy feature rows must be exactly once per sample.")
    if audit_frame.empty or audit_frame[sample_id_column].duplicated().any():
        raise HRDatasetDiagnosticsError("Proxy audit rows must be exactly once per sample.")
    sample_ids = set(folds[sample_id_column].tolist())
    if set(features[sample_id_column]) != sample_ids or set(audit_frame[sample_id_column]) != sample_ids:
        raise HRDatasetDiagnosticsError("Proxy features, audit data and folds have different sample sets.")
    audit = _set_unique_index(audit_frame, sample_id_column, "Proxy audit rows").loc[
        folds.sort_values(sample_id_column)[sample_id_column]
    ]
    target_text = audit[proxy_target].astype("string").fillna("__MISSING__").astype(str)
    target_classes = tuple(sorted(target_text.unique()))
    if len(target_classes) < 2:
        raise HRDatasetDiagnosticsError("Proxy target must have at least two observed classes.")
    target_lookup = {value: index for index, value in enumerate(target_classes)}
    target = target_text.map(target_lookup).astype(int)
    target.index = audit.index
    class_counts = {value: int((target_text == value).sum()) for value in target_classes}
    fold_by_id = _set_unique_index(folds, sample_id_column, "Proxy fold rows")[fold_column].astype(int)
    deficient: list[dict[str, Any]] = []
    for outer_fold in sorted(fold_by_id.unique()):
        train_ids = fold_by_id.index[fold_by_id != outer_fold]
        train_values = target_text.loc[train_ids]
        counts = {value: int((train_values == value).sum()) for value in target_classes}
        missing_classes = [value for value, count in counts.items() if count == 0]
        if missing_classes:
            deficient.append(
                {
                    "outer_fold": int(outer_fold),
                    "missing_training_classes": missing_classes,
                    "training_class_counts": counts,
                }
            )
    base_status = {
        **identity.as_dict(),
        "task_type": PROXY_TASK,
        "analysis_type": "nominal_department_reconstructability_proxy_risk",
        "proxy_target": proxy_target,
        "proxy_target_class_counts_json": _canonical_json(class_counts),
        "n_proxy_target_classes": len(target_classes),
        "minimum_proxy_target_class_support": min(class_counts.values()),
        "outer_training_deficiencies_json": _canonical_json(deficient),
        "n_outer_training_missing_class_cells": sum(
            len(item["missing_training_classes"]) for item in deficient
        ),
        "classes_merged_or_dropped": False,
        "headline_eligible": False,
        "limitations": PROXY_LIMITATION,
    }
    if deficient:
        status = pd.DataFrame(
            [
                {
                    **base_status,
                    "analysis_status": "not_estimated_insufficient_outer_training_class_support",
                    "reason": (
                        "At least one exact outer training fold lacks an observed full-data proxy-target "
                        "class; no class was merged/dropped and no reconstructability model was fitted."
                    ),
                    "models_fitted": 0,
                    "n_resamples": 0,
                    "resample_hash": None,
                }
            ]
        )
        metadata = MappingProxyType(
            {
                **base_status,
                "analysis_status": status.iloc[0]["analysis_status"],
                "models_fitted": 0,
                "n_resamples": 0,
                "class_labels": list(target_classes),
            }
        )
        empty = pd.DataFrame()
        return ProxyReconstructabilityEvidence(
            status=status,
            feature_contracts=pd.DataFrame(contract_rows),
            oof_predictions=empty,
            fold_metrics=empty,
            metric_intervals=empty,
            paired_differences=empty,
            metadata=metadata,
        )

    feature_by_id = _set_unique_index(features, sample_id_column, "Proxy feature rows")
    oof_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    models_fitted = 0
    for system_id, predictors in sorted(predictor_sets.items()):
        values = [str(value) for value in predictors]
        for outer_fold in sorted(fold_by_id.unique()):
            train_ids = fold_by_id.index[fold_by_id != outer_fold]
            test_ids = fold_by_id.index[fold_by_id == outer_fold]
            X_train = feature_by_id.loc[train_ids, values]
            X_test = feature_by_id.loc[test_ids, values]
            pipeline = _proxy_pipeline(X_train, seed)
            with threadpool_limits(limits=1):
                pipeline.fit(X_train, target.loc[train_ids])
            models_fitted += 1
            classifier = pipeline.named_steps["classifier"]
            classes = tuple(int(value) for value in classifier.classes_)
            if classes != tuple(range(len(target_classes))):
                raise HRDatasetDiagnosticsError(
                    f"Proxy fold {outer_fold}/{system_id} lacks a fitted target class unexpectedly."
                )
            probability = np.asarray(pipeline.predict_proba(X_test), dtype=float)
            prediction = np.argmax(probability, axis=1).astype(int)
            lineage = tuple(
                str(value) for value in pipeline.named_steps["preprocessor"].get_feature_names_out()
            )
            _validate_forbidden_features(lineage, forbidden)
            fit_hash = _sha256_json(
                {
                    "system_id": system_id,
                    "outer_fold": int(outer_fold),
                    "predictor_contract_sha256": contract_hashes[str(system_id)],
                    "classes": list(classes),
                    "coef": np.asarray(classifier.coef_, dtype=float).round(15).tolist(),
                    "intercept": np.asarray(classifier.intercept_, dtype=float).round(15).tolist(),
                }
            )
            metrics = _proxy_metric_values(
                target.loc[test_ids].to_numpy(int), prediction, range(len(target_classes))
            )
            fold_rows.append(
                {
                    **identity.as_dict(),
                    "task_type": PROXY_TASK,
                    "system_id": str(system_id),
                    "outer_fold": int(outer_fold),
                    "n_train": len(train_ids),
                    "n_test": len(test_ids),
                    "predictor_contract_sha256": contract_hashes[str(system_id)],
                    "fit_receipt_sha256": fit_hash,
                    "proxy_target_and_aliases_absent": True,
                    **metrics,
                }
            )
            for position, sample_id in enumerate(test_ids):
                row = {
                    **identity.as_dict(),
                    "task_type": PROXY_TASK,
                    "system_id": str(system_id),
                    "sample_index": _json_scalar(sample_id),
                    "outer_fold": int(outer_fold),
                    "proxy_target": str(target_text.loc[sample_id]),
                    "y_true": int(target.loc[sample_id]),
                    "y_pred": int(prediction[position]),
                    "predictor_contract_sha256": contract_hashes[str(system_id)],
                    "fit_receipt_sha256": fit_hash,
                    "proxy_target_and_aliases_absent": True,
                    "limitations": PROXY_LIMITATION,
                }
                for label in range(len(target_classes)):
                    row[f"prob_class_{label}"] = float(probability[position, label])
                oof_rows.append(row)
    oof = pd.DataFrame(oof_rows).sort_values(["system_id", "sample_index"]).reset_index(drop=True)
    expected_n = len(features)
    for system_id, scoped in oof.groupby("system_id", sort=False):
        if len(scoped) != expected_n or scoped["sample_index"].duplicated().any():
            raise HRDatasetDiagnosticsError(f"Proxy system {system_id!r} is not exactly-once OOF.")
    base = oof[oof["system_id"] == sorted(predictor_sets)[0]][
        ["sample_index", "outer_fold", "y_true"]
    ].copy()
    protocol = BootstrapProtocol(
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        strata_columns=("outer_fold", "y_true"),
        method="paired_stratified_percentile",
        quantile_method="linear",
    )
    try:
        plan = generate_stratified_resample_indices(base, protocol)
    except OOFBootstrapError as exc:
        raise HRDatasetDiagnosticsError(f"Cannot generate proxy bootstrap plan: {exc}") from exc
    alpha = 1.0 - confidence_level
    interval_rows: list[dict[str, Any]] = []
    draws_by_system: dict[str, dict[str, np.ndarray]] = {}
    points_by_system: dict[str, dict[str, float]] = {}
    for system_id in sorted(predictor_sets):
        scoped = oof[oof["system_id"] == system_id].set_index("sample_index").loc[
            list(plan.sorted_sample_ids)
        ]
        y_true = scoped["y_true"].to_numpy(int)
        y_pred = scoped["y_pred"].to_numpy(int)
        points = _proxy_metric_values(y_true, y_pred, range(len(target_classes)))
        draws = _proxy_metric_draws(
            y_true,
            y_pred,
            range(len(target_classes)),
            plan.indices,
            batch_size=batch_size,
        )
        draws_by_system[system_id] = draws
        points_by_system[system_id] = points
        for metric in PROXY_METRICS:
            values = draws[metric]
            low, high = np.quantile(
                values, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear"
            )
            interval_rows.append(
                {
                    **identity.as_dict(),
                    "task_type": PROXY_TASK,
                    "analysis_type": "nominal_department_reconstructability_proxy_risk",
                    "system_id": system_id,
                    "metric": metric,
                    "point_estimate": points[metric],
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "n_samples": len(scoped),
                    "n_resamples": n_resamples,
                    "n_valid": n_resamples,
                    "confidence_level": confidence_level,
                    "resample_hash": plan.resample_hash,
                    "uncertainty_method": "paired_stratified_sample_level_percentile_bootstrap",
                    "inference_scope": "pointwise_descriptive",
                    "multiplicity_adjustment": "none",
                    "headline_eligible": False,
                    "limitations": PROXY_LIMITATION,
                }
            )
    paired_rows: list[dict[str, Any]] = []
    for system_a, system_b in itertools.combinations(sorted(predictor_sets), 2):
        for metric in PROXY_METRICS:
            difference = draws_by_system[system_a][metric] - draws_by_system[system_b][metric]
            low, high = np.quantile(
                difference, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear"
            )
            paired_rows.append(
                {
                    **identity.as_dict(),
                    "task_type": PROXY_TASK,
                    "comparison_id": f"{system_a}__minus__{system_b}",
                    "system_a": system_a,
                    "system_b": system_b,
                    "metric": metric,
                    "difference": points_by_system[system_a][metric]
                    - points_by_system[system_b][metric],
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "n_resamples": n_resamples,
                    "n_valid": n_resamples,
                    "confidence_level": confidence_level,
                    "resample_hash": plan.resample_hash,
                    "inference_scope": "pointwise_descriptive",
                    "multiplicity_adjustment": "none",
                    "headline_eligible": False,
                    "limitations": PROXY_LIMITATION,
                }
            )
    status = pd.DataFrame(
        [
            {
                **base_status,
                "analysis_status": "estimated_descriptive_proxy_risk",
                "reason": "Every exact outer training fold contains every observed proxy-target class.",
                "models_fitted": models_fitted,
                "n_resamples": n_resamples,
                "resample_hash": plan.resample_hash,
            }
        ]
    )
    metadata = MappingProxyType(
        {
            **base_status,
            "analysis_status": "estimated_descriptive_proxy_risk",
            "models_fitted": models_fitted,
            "n_resamples": n_resamples,
            "confidence_level": confidence_level,
            "seed": seed,
            "batch_size": batch_size,
            "resample_hash": plan.resample_hash,
            "class_labels": list(target_classes),
            "strata_semantics": ["outer_fold", proxy_target],
            "internal_bootstrap_columns": ["outer_fold", "y_true"],
            "inference_scope": "pointwise_descriptive",
            "multiplicity_adjustment": "none",
            "limitations": PROXY_LIMITATION,
        }
    )
    return ProxyReconstructabilityEvidence(
        status=status,
        feature_contracts=pd.DataFrame(contract_rows),
        oof_predictions=oof,
        fold_metrics=pd.DataFrame(fold_rows),
        metric_intervals=pd.DataFrame(interval_rows),
        paired_differences=pd.DataFrame(paired_rows),
        metadata=metadata,
    )


__all__ = [
    "ATTRIBUTION_WARNING",
    "AuditAttributeSpec",
    "FoldModelReference",
    "HRDatasetDiagnosticsError",
    "OOFShapEvidence",
    "PROXY_LIMITATION",
    "ProxyReconstructabilityEvidence",
    "REQUIRED_BOOTSTRAP_RESAMPLES",
    "RESEARCH_USE_WARNING",
    "ReplicationIdentity",
    "SUBGROUP_LIMITATION",
    "ShapComputation",
    "SubgroupDiagnosticsEvidence",
    "TEMPORALITY_WARNING",
    "canonicalize_multiclass_shap",
    "compute_exact_oof_grouped_shap",
    "compute_proxy_reconstructability",
    "compute_support_aware_subgroup_diagnostics",
    "feature_policy_contract_sha256",
    "model_set_sha256",
    "outer_fold_assignment_sha256",
]
