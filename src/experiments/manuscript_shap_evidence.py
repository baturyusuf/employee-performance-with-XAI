from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

from src.core.io_utils import ensure_dir, write_json
from src.data.preprocess import load_validated_or_raw_data
from src.experiments.final_evidence_common import align_proba, predict_labels_from_proba
from src.experiments.final_shap_stability import get_group_mapping, group_shap_values, normalize_shap_values
from src.experiments.leakage_safe_cv import infer_columns, make_preprocessor
from src.experiments.manuscript_calibration import _fit_pipeline
from src.experiments.manuscript_policy_ablation import _mean_ci, _model_parameters, exact_policy_frame, resolve_seed
from src.features.feature_sets import taxonomy_by_feature
from src.governance.manuscript_contract import canonical_config_hash
from src.utils.config_loader import load_config


MANDATORY_WARNINGS = (
    "SHAP values are model attributions, not causal effects.",
    "Counterfactual model scenarios are not employee prescriptions.",
    "Removing sensitive or organisational fields does not prove fairness; proxy risk may remain.",
    "This is research-grade decision support, not an autonomous HR decision system.",
)


class ShapEvidenceError(RuntimeError):
    """Raised when canonical SHAP evidence violates its run or feature policy."""


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
        "run_id",
        "config_hash",
        "policy",
        "sample_index",
        "fold",
        "y_true",
        "y_pred",
        "confidence",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ShapEvidenceError(f"Representative-case predictions lack columns: {missing}")
    for identity in ("run_id", "config_hash", "policy"):
        observed = set(predictions[identity].dropna().astype(str).unique())
        if len(observed) != 1:
            raise ShapEvidenceError(
                f"Representative-case predictions require one canonical {identity}; observed={sorted(observed)}."
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
                    "run_id": str(row.run_id),
                    "config_hash": str(row.config_hash),
                    "policy": str(row.policy),
                    "case_type": case_type,
                    "sample_index": sample_index,
                    "sampling_reason": reason,
                    "true_class": int(row.y_true),
                    "predicted_class": int(row.y_pred),
                    "confidence": float(row.confidence),
                    "correct": bool(row.y_true == row.y_pred),
                    "fold": int(row.fold),
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
    run_id: str,
    config_hash: str,
    policy: str,
    taxonomy: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    importance = np.mean(np.abs(values), axis=tuple(range(values.ndim - 1)))
    rows = []
    for feature, value in zip(features, importance):
        rows.append(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "policy": policy,
                "feature": feature,
                "mean_abs_grouped_shap": float(value),
                **feature_governance(feature, taxonomy),
            }
        )
    frame = pd.DataFrame(rows).sort_values("mean_abs_grouped_shap", ascending=False).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame


def _fold_rankings(
    grouped_values: np.ndarray,
    folds: np.ndarray,
    features: Sequence[str],
    *,
    run_id: str,
    config_hash: str,
    policy: str,
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        importance = np.mean(np.abs(grouped_values[folds == fold]), axis=(0, 1))
        order = np.argsort(-importance)
        for rank, feature_index in enumerate(order, start=1):
            rows.append(
                {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "policy": policy,
                    "fold": int(fold),
                    "feature": features[int(feature_index)],
                    "rank": rank,
                    "mean_abs_grouped_shap": float(importance[int(feature_index)]),
                }
            )
    return pd.DataFrame(rows)


def shap_stability_pairwise(rankings: pd.DataFrame, top_k_values: Sequence[int]) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    fold_rankings = {
        int(fold): group.sort_values("rank")["feature"].tolist()
        for fold, group in rankings.groupby("fold")
    }
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
            rows.append(
                {
                    "run_id": rankings["run_id"].iloc[0],
                    "config_hash": rankings["config_hash"].iloc[0],
                    "policy": rankings["policy"].iloc[0],
                    "fold_a": fold_a,
                    "fold_b": fold_b,
                    "top_k": int(top_k),
                    "top_k_jaccard": float(len(set_a.intersection(set_b)) / len(union)) if union else 1.0,
                    "spearman_all_features": float(correlation),
                }
            )
    return pd.DataFrame(rows)


def summarize_stability(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for top_k, group in pairwise.groupby("top_k"):
        j_mean, j_std, j_low, j_high = _mean_ci(group["top_k_jaccard"])
        s_mean, s_std, s_low, s_high = _mean_ci(group["spearman_all_features"])
        rows.append(
            {
                "run_id": group["run_id"].iloc[0],
                "config_hash": group["config_hash"].iloc[0],
                "policy": group["policy"].iloc[0],
                "top_k": int(top_k),
                "n_fold_pairs": len(group),
                "jaccard_mean": j_mean,
                "jaccard_std": j_std,
                "jaccard_ci_low": j_low,
                "jaccard_ci_high": j_high,
                "spearman_mean": s_mean,
                "spearman_std": s_std,
                "spearman_ci_low": s_low,
                "spearman_ci_high": s_high,
            }
        )
    return pd.DataFrame(rows)


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
    run_id: str,
    config_hash: str,
    policy: str,
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
            "run_id": run_id,
            "config_hash": config_hash,
            "policy": policy,
            "model": "xgboost",
            "case_type": case.case_type,
            "sampling_reason": case.sampling_reason,
            "sample_index": sample_index,
            "fold": int(prediction["fold"]),
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
            f"Run ID: `{run_id}`  ",
            f"Config hash: `{config_hash}`  ",
            f"Sample index: `{sample_index}`; OOF fold: `{int(prediction['fold'])}`  ",
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
    forbidden_features: Sequence[str],
    run_id: str,
    config_hash: str,
) -> Dict[str, Any]:
    frames = [*global_tables, rankings, local_values]
    checked = 0
    for frame in frames:
        if "feature" in frame.columns:
            assert_feature_names_allowed(frame["feature"].astype(str), forbidden_features)
        for required, expected in (("run_id", run_id), ("config_hash", config_hash)):
            if required not in frame.columns or set(frame[required].astype(str)) != {str(expected)}:
                raise ShapEvidenceError(f"SHAP artifact frame has inconsistent {required}.")
        checked += 1
    return {
        "status": "passed",
        "run_id": run_id,
        "config_hash": config_hash,
        "frames_checked": checked,
        "forbidden_features_checked": list(forbidden_features),
    }


def _figure_6(global_importance: pd.DataFrame, output_dir: Path, *, run_id: str, config_hash: str, top_n: int) -> Dict[str, Path]:
    rows = global_importance.head(top_n).sort_values("mean_abs_grouped_shap")
    fig, axis = plt.subplots(figsize=(9.2, 6.6), constrained_layout=True)
    axis.barh(rows["feature"], rows["mean_abs_grouped_shap"], color="#176B87")
    axis.set_xlabel("Mean absolute grouped SHAP value (OOF)")
    axis.set_title("Figure 6. Global grouped SHAP attribution for the canonical primary model")
    axis.grid(axis="x", alpha=0.25)
    fig.text(0.01, 0.01, "Attribution, not causality. Proxy risk may remain after direct-field exclusion.", fontsize=8)
    description = f"run_id={run_id}; config_hash={config_hash}"
    png = output_dir / "figure_6_global_grouped_shap.png"
    svg = output_dir / "figure_6_global_grouped_shap.svg"
    fig.savefig(png, dpi=300, metadata={"Description": description})
    fig.savefig(svg, format="svg", metadata={"Title": "Figure 6 global grouped SHAP", "Description": description})
    plt.close(fig)
    return {"png": png, "svg": svg}


def _figure_7(
    representative_cases: pd.DataFrame,
    local_values: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
    *,
    run_id: str,
    config_hash: str,
) -> Dict[str, Path]:
    preferred = representative_cases[representative_cases["case_type"] == "incorrect_high_confidence"]
    case = (preferred if not preferred.empty else representative_cases).iloc[0]
    sample_index = int(case["sample_index"])
    prediction = predictions.set_index("sample_index").loc[sample_index]
    predicted_class = int(prediction["y_pred"])
    rows = local_values[
        (local_values["sample_index"] == sample_index)
        & (local_values["class_label"] == predicted_class)
    ].nlargest(12, "abs_grouped_shap_value").sort_values("grouped_shap_value")
    colors = ["#C44E52" if value < 0 else "#2A9D8F" for value in rows["grouped_shap_value"]]
    fig, axis = plt.subplots(figsize=(10.2, 7.0), constrained_layout=True)
    axis.barh(rows["feature"], rows["grouped_shap_value"], color=colors)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel(f"Grouped SHAP value for predicted class {predicted_class}")
    axis.set_title(
        f"Figure 7. OOF local reason code: {case['case_type']} (case {sample_index}, true {int(prediction['y_true'])}, predicted {predicted_class})"
    )
    axis.grid(axis="x", alpha=0.25)
    fig.text(
        0.01,
        0.01,
        "Model attribution only—not causality or employee advice. Human review required; proxy and calibration limitations remain.",
        fontsize=8,
    )
    description = f"run_id={run_id}; config_hash={config_hash}; sample_index={sample_index}"
    png = output_dir / "figure_7_local_reason_code.png"
    svg = output_dir / "figure_7_local_reason_code.svg"
    fig.savefig(png, dpi=300, metadata={"Description": description})
    fig.savefig(svg, format="svg", metadata={"Title": "Figure 7 local grouped SHAP reason code", "Description": description})
    plt.close(fig)
    return {"png": png, "svg": svg}


def run(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    run_id: str,
    config_hash: str | None = None,
) -> Dict[str, Path]:
    import shap

    raw_config = load_config(config_path)
    settings = raw_config.get("manuscript_final", raw_config)
    config_hash = config_hash or canonical_config_hash(raw_config)
    output = ensure_dir(Path(output_dir))
    local_dir = ensure_dir(output / "local_reason_codes")
    primary_policy = str(settings["feature_policies"]["primary_policy"])
    definition = settings["feature_policies"]["definitions"][primary_policy]
    forbidden = [str(value) for value in definition["excluded_features"]]
    target = settings.get("target", {})
    target_column = str(target.get("column", "PerformanceRating"))
    labels = [int(value) for value in target.get("labels", [2, 3, 4])]
    identifier_fields = settings.get("governance_fields", {}).get("identifier_fields", ["EmpNumber"])
    id_column = str(identifier_fields[0] if identifier_fields else "EmpNumber")
    data = load_validated_or_raw_data()
    X, excluded = exact_policy_frame(data, primary_policy, definition, target_column=target_column, id_column=id_column)
    assert_feature_names_allowed(X.columns, forbidden)
    y = data[target_column].astype(int)

    protocol = settings.get("shap", {})
    stability_protocol = protocol.get("stability", {}) if isinstance(protocol.get("stability", {}), Mapping) else {}
    local_protocol = protocol.get("local", {}) if isinstance(protocol.get("local", {}), Mapping) else {}
    n_splits = int(protocol.get("stability_folds", settings.get("evaluation", {}).get("cv", {}).get("n_splits", 10)))
    configured_top_k = stability_protocol.get("top_k")
    default_top_k = sorted({5, int(configured_top_k or 10), 15})
    top_k_values = [int(value) for value in protocol.get("top_k_values", default_top_k)]
    local_top_k = int(protocol.get("local_top_k", local_protocol.get("top_k_reason_codes", 5)))
    seed = resolve_seed(settings, "shap")
    parameters = _model_parameters(settings)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    feature_names = X.columns.tolist()
    grouped_oof = np.zeros((len(X), len(labels), len(feature_names)), dtype=float)
    probabilities_oof = np.zeros((len(X), len(labels)), dtype=float)
    predictions_oof = np.empty(len(X), dtype=int)
    fold_oof = np.empty(len(X), dtype=int)

    for fold, (train_positions, test_positions) in enumerate(splitter.split(X, y), start=1):
        X_train = X.iloc[train_positions]
        X_test = X.iloc[test_positions]
        pipeline = _fit_pipeline(X_train, y.iloc[train_positions], parameters, seed)
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["model"]
        probability = align_proba(pipeline.predict_proba(X_test), classifier.classes_, labels)
        predictions_oof[test_positions] = predict_labels_from_proba(probability, labels)
        probabilities_oof[test_positions] = probability
        fold_oof[test_positions] = fold
        numeric_columns, categorical_columns = infer_columns(X_train)
        group_names, mapping = get_group_mapping(preprocessor, numeric_columns, categorical_columns)
        transformed = preprocessor.transform(X_test)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        raw_values = shap.TreeExplainer(classifier.model_).shap_values(transformed)
        normalized = normalize_shap_values(
            raw_values,
            n_samples=len(test_positions),
            n_features=transformed.shape[1],
            n_classes=len(labels),
        )
        grouped = group_shap_values(normalized, group_names, mapping)
        grouped_oof[test_positions] = reorder_grouped_shap_to_feature_order(
            grouped,
            group_names,
            feature_names,
        )

    taxonomy = taxonomy_by_feature()
    global_importance = _importance_table(
        grouped_oof,
        feature_names,
        run_id=run_id,
        config_hash=config_hash,
        policy=primary_policy,
        taxonomy=taxonomy,
    )
    class_tables: Dict[int, pd.DataFrame] = {}
    for class_index, label in enumerate(labels):
        class_tables[label] = _importance_table(
            grouped_oof[:, class_index, :],
            feature_names,
            run_id=run_id,
            config_hash=config_hash,
            policy=primary_policy,
            taxonomy=taxonomy,
        )
        class_tables[label]["class_label"] = label

    rankings = _fold_rankings(
        grouped_oof,
        fold_oof,
        feature_names,
        run_id=run_id,
        config_hash=config_hash,
        policy=primary_policy,
    )
    pairwise = shap_stability_pairwise(rankings, top_k_values)
    stability = summarize_stability(pairwise)
    prediction_rows: list[Dict[str, Any]] = []
    local_rows: list[Dict[str, Any]] = []
    for position, sample_index in enumerate(X.index):
        prediction_row: Dict[str, Any] = {
            "run_id": run_id,
            "config_hash": config_hash,
            "policy": primary_policy,
            "sample_index": int(sample_index),
            "fold": int(fold_oof[position]),
            "y_true": int(y.iloc[position]),
            "y_pred": int(predictions_oof[position]),
            "confidence": float(np.max(probabilities_oof[position])),
            "correct": bool(y.iloc[position] == predictions_oof[position]),
        }
        for label_index, label in enumerate(labels):
            prediction_row[f"prob_class_{label}"] = float(probabilities_oof[position, label_index])
            for feature_index, feature in enumerate(feature_names):
                value = float(grouped_oof[position, label_index, feature_index])
                local_rows.append(
                    {
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "policy": primary_policy,
                        "sample_index": int(sample_index),
                        "fold": int(fold_oof[position]),
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
        forbidden_features=forbidden,
        run_id=run_id,
        config_hash=config_hash,
    )
    paths: Dict[str, Path] = {
        "global": output / "global_grouped_shap_importance.csv",
        "rankings": output / "fold_feature_rankings.csv",
        "pairwise": output / "shap_stability_pairwise.csv",
        "stability": output / "shap_stability_summary.csv",
        "representative": output / "representative_cases.csv",
        "predictions": output / "shap_oof_predictions.csv",
        "local_values": output / "local_grouped_shap_values.csv",
        "validation": output / "shap_artifact_validation.json",
        "metadata": output / "shap_metadata.json",
    }
    global_importance.to_csv(paths["global"], index=False)
    rankings.to_csv(paths["rankings"], index=False)
    pairwise.to_csv(paths["pairwise"], index=False)
    stability.to_csv(paths["stability"], index=False)
    representative.to_csv(paths["representative"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    local_values.to_csv(paths["local_values"], index=False)
    for label, table in class_tables.items():
        path = output / f"class_{label}_grouped_shap_importance.csv"
        table.to_csv(path, index=False)
        paths[f"class_{label}"] = path
    reason_code_paths = write_local_reason_codes(
        representative,
        local_values,
        predictions,
        X,
        labels,
        local_dir,
        top_k=local_top_k,
        run_id=run_id,
        config_hash=config_hash,
        policy=primary_policy,
        taxonomy=taxonomy,
    )
    write_json(paths["validation"], {**validation, "reason_code_files_checked": len(reason_code_paths)})
    figure_6 = _figure_6(
        global_importance,
        output,
        run_id=run_id,
        config_hash=config_hash,
        top_n=int(protocol.get("global_top_n", 15)),
    )
    figure_7 = _figure_7(
        representative,
        local_values,
        predictions,
        output,
        run_id=run_id,
        config_hash=config_hash,
    )
    paths.update(
        {
            "figure_6_png": figure_6["png"],
            "figure_6_svg": figure_6["svg"],
            "figure_7_png": figure_7["png"],
            "figure_7_svg": figure_7["svg"],
        }
    )
    write_json(
        paths["metadata"],
        {
            "stage": "shap_evidence",
            "run_id": run_id,
            "config_hash": config_hash,
            "policy": primary_policy,
            "excluded_features": excluded,
            "model": "xgboost",
            "model_parameters": parameters,
            "protocol": {
                "evaluation": "out_of_fold",
                "n_splits": n_splits,
                "grouping": "one-hot SHAP summed to raw feature families",
                "top_k_values": top_k_values,
                "local_top_k": local_top_k,
            },
            "seed": seed,
            "warnings": MANDATORY_WARNINGS,
            "outputs": {key: str(value) for key, value in paths.items() if key != "metadata"},
            "local_reason_code_files": [str(path) for path in reason_code_paths],
        },
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical OOF grouped SHAP and local reason-code evidence.")
    parser.add_argument("--config", default="configs/manuscript_final.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-hash", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            {
                key: str(value)
                for key, value in run(
                    arguments.config,
                    output_dir=arguments.output_dir,
                    run_id=arguments.run_id,
                    config_hash=arguments.config_hash,
                ).items()
            },
            indent=2,
            sort_keys=True,
        )
    )
