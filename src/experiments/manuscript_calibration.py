from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.core.io_utils import ensure_dir, write_json
from src.data.canonical_loader import load_canonical_dataset
from src.experiments.final_evidence_common import align_proba, calibrate_probabilities, predict_labels_from_proba
from src.experiments.leakage_safe_cv import LabelEncodedXGBClassifier, make_preprocessor
from src.experiments.manuscript_policy_ablation import _mean_ci, _model_parameters, exact_policy_frame, resolve_seed
from src.governance.manuscript_contract import canonical_config_hash
from src.models.evaluate import classification_metrics
from src.utils.config_loader import load_config


METHODS = ("raw", "sigmoid", "isotonic")
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "quadratic_weighted_kappa",
    "ordinal_mae",
    "severe_error_rate",
    "nll_log_loss",
    "multiclass_brier",
    "ece_confidence",
)


class CalibrationContractError(RuntimeError):
    """Raised when canonical nested calibration cannot be executed safely."""


def calibration_bin_rows(
    y_true: Sequence[int],
    probabilities: np.ndarray,
    labels: Sequence[int],
    *,
    run_id: str,
    config_hash: str,
    method: str,
    n_bins: int,
) -> list[Dict[str, Any]]:
    y_array = np.asarray(y_true, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[Dict[str, Any]] = []
    for label_index, label in enumerate(labels):
        scores = probabilities[:, label_index]
        outcomes = (y_array == int(label)).astype(float)
        assignments = np.digitize(scores, edges[1:-1], right=True) + 1
        for bin_index in range(1, n_bins + 1):
            mask = assignments == bin_index
            if not np.any(mask):
                continue
            predicted = float(scores[mask].mean())
            observed = float(outcomes[mask].mean())
            rows.append(
                {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "method": method,
                    "class_label": int(label),
                    "bin": bin_index,
                    "bin_low": float(edges[bin_index - 1]),
                    "bin_high": float(edges[bin_index]),
                    "n_samples": int(mask.sum()),
                    "mean_predicted_probability": predicted,
                    "observed_frequency": observed,
                    "absolute_gap": abs(predicted - observed),
                }
            )
    return rows


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for method, group in fold_metrics.groupby("method", sort=False):
        row: Dict[str, Any] = {
            "run_id": group["run_id"].iloc[0],
            "config_hash": group["config_hash"].iloc[0],
            "method": method,
            "n_folds": int(group["fold"].nunique()),
        }
        for metric in METRICS:
            mean, std, low, high = _mean_ci(pd.to_numeric(group[metric], errors="coerce"))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def select_calibration_method(summary: pd.DataFrame, selection_metrics: Sequence[str]) -> tuple[str, pd.DataFrame]:
    ranked = summary.copy()
    rank_columns: list[str] = []
    aliases = {"log_loss": "nll_log_loss", "ece": "ece_confidence", "brier": "multiclass_brier"}
    for metric in selection_metrics:
        metric = aliases.get(metric, metric)
        column = metric if metric.endswith("_mean") else f"{metric}_mean"
        if column not in ranked.columns:
            raise CalibrationContractError(f"Calibration selection metric is unavailable: {column}")
        rank_column = f"rank_{column}"
        ranked[rank_column] = ranked[column].rank(method="average", ascending=True)
        rank_columns.append(rank_column)
    ranked["selection_rank_sum"] = ranked[rank_columns].sum(axis=1)
    order = {method: index for index, method in enumerate(METHODS)}
    ranked["tie_break_order"] = ranked["method"].map(order)
    ranked = ranked.sort_values(["selection_rank_sum", "tie_break_order"]).reset_index(drop=True)
    ranked["selected"] = ranked.index == 0
    return str(ranked.iloc[0]["method"]), ranked


def _fit_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: Mapping[str, Any],
    seed: int,
) -> Pipeline:
    params = dict(parameters)
    params["random_state"] = seed
    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(X_train)),
            ("model", LabelEncodedXGBClassifier(**params)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def _plot_single_reliability(
    bins: pd.DataFrame,
    class_label: int,
    output_dir: Path,
    *,
    run_id: str,
    config_hash: str,
) -> Dict[str, Path]:
    subset = bins[bins["class_label"] == class_label]
    fig, axis = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    for method, method_rows in subset.groupby("method", sort=False):
        method_rows = method_rows.sort_values("mean_predicted_probability")
        axis.plot(
            method_rows["mean_predicted_probability"],
            method_rows["observed_frequency"],
            marker="o",
            linewidth=1.5,
            label=method,
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="ideal")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed frequency")
    axis.set_title(f"Reliability for performance class {class_label}")
    axis.grid(alpha=0.25)
    axis.legend()
    description = f"run_id={run_id}; config_hash={config_hash}; class={class_label}"
    png = output_dir / f"reliability_class_{class_label}.png"
    svg = output_dir / f"reliability_class_{class_label}.svg"
    fig.savefig(png, dpi=300, metadata={"Description": description})
    fig.savefig(svg, format="svg", metadata={"Title": f"Class {class_label} reliability", "Description": description})
    plt.close(fig)
    return {"png": png, "svg": svg}


def write_figure_5(
    bins: pd.DataFrame,
    summary: pd.DataFrame,
    labels: Sequence[int],
    output_dir: Path,
    *,
    selected_method: str,
    run_id: str,
    config_hash: str,
) -> Dict[str, Path]:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    selected_bins = bins[bins["method"] == selected_method]
    for axis, label in zip(axes[0], labels):
        rows = selected_bins[selected_bins["class_label"] == int(label)].sort_values("mean_predicted_probability")
        axis.plot(rows["mean_predicted_probability"], rows["observed_frequency"], marker="o", color="#176B87")
        axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed frequency")
        axis.set_title(f"A–C: class {label} reliability")
        axis.grid(alpha=0.25)

    methods = summary["method"].tolist()
    positions = np.arange(len(methods))
    width = 0.36
    axes[1, 0].bar(positions - width / 2, summary["ordinal_mae_mean"], width, label="Ordinal MAE", color="#176B87")
    axes[1, 0].bar(positions + width / 2, summary["severe_error_rate_mean"], width, label="Severe error rate", color="#DA7C30")
    axes[1, 0].set_xticks(positions, methods)
    axes[1, 0].set_title("D: ordinal error")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].bar(positions - width / 2, summary["nll_log_loss_mean"], width, label="Log loss", color="#176B87")
    axes[1, 1].bar(positions + width / 2, summary["multiclass_brier_mean"], width, label="Multiclass Brier", color="#7A5195")
    axes[1, 1].set_xticks(positions, methods)
    axes[1, 1].set_title("E: probability loss")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.25)

    axes[1, 2].bar(positions, summary["ece_confidence_mean"], color=["#176B87", "#2A9D8F", "#DA7C30"])
    axes[1, 2].set_xticks(positions, methods)
    axes[1, 2].set_title("F: expected calibration error")
    axes[1, 2].set_ylabel("ECE (lower is better)")
    axes[1, 2].grid(axis="y", alpha=0.25)
    fig.suptitle(
        f"Figure 5. Nested calibration and ordinal-error summary (selected: {selected_method})",
        fontsize=14,
    )
    description = f"run_id={run_id}; config_hash={config_hash}; selected_method={selected_method}"
    png = output_dir / "figure_5_calibration_ordinal_error.png"
    svg = output_dir / "figure_5_calibration_ordinal_error.svg"
    fig.savefig(png, dpi=300, metadata={"Description": description})
    fig.savefig(svg, format="svg", metadata={"Title": "Figure 5 calibration and ordinal error", "Description": description})
    plt.close(fig)
    return {"png": png, "svg": svg}


def write_rationale(
    path: Path,
    ranked_summary: pd.DataFrame,
    *,
    selected_method: str,
    run_id: str,
    config_hash: str,
) -> None:
    selected = ranked_summary[ranked_summary["selected"]].iloc[0]
    lines = [
        "# Canonical Calibration Method Rationale",
        "",
        f"Run ID: `{run_id}`  ",
        f"Config hash: `{config_hash}`",
        "",
        "Calibration uses outer stratified 10-fold evaluation. Within each outer training fold, the model is fitted on an inner training subset and raw, sigmoid, and isotonic probability outputs are evaluated only on the untouched outer test fold. Sigmoid/isotonic calibrators see only the inner calibration subset.",
        "",
        f"Selected method: `{selected_method}` by the predeclared lowest aggregate rank across log loss, multiclass Brier score, and ECE (rank sum {selected['selection_rank_sum']:.3f}).",
        "",
        "## Probability-Use Warning",
        "",
        "Probabilities are approximate model confidence estimates, not objective employee-performance probabilities. Use probability bands with calibration warnings and human review; do not use exact probabilities as autonomous HR decision thresholds.",
        "",
        "Isotonic calibration is flexible and may be unstable for the minority class because each inner calibration split is small. Method selection is descriptive for this declared protocol and dataset, not a guarantee of future calibration.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    run_id: str,
    config_hash: str | None = None,
) -> Dict[str, Path]:
    raw_config = load_config(config_path)
    settings = raw_config.get("manuscript_final", raw_config)
    config_hash = config_hash or canonical_config_hash(raw_config)
    output = ensure_dir(Path(output_dir))
    figures = ensure_dir(output / "figures")

    primary_policy = str(settings["feature_policies"]["primary_policy"])
    definition = settings["feature_policies"]["definitions"][primary_policy]
    target = settings.get("target", {})
    target_column = str(target.get("column", "PerformanceRating"))
    labels = [int(value) for value in target.get("labels", [2, 3, 4])]
    identifier_fields = settings.get("governance_fields", {}).get("identifier_fields", ["EmpNumber"])
    id_column = str(identifier_fields[0] if identifier_fields else "EmpNumber")
    data = load_canonical_dataset(config_path, "inx_primary").frame
    X, excluded = exact_policy_frame(data, primary_policy, definition, target_column=target_column, id_column=id_column)
    y = data[target_column].astype(int)

    protocol = settings.get("calibration", {})
    outer_splits = int(protocol.get("outer_folds", settings.get("evaluation", {}).get("cv", {}).get("n_splits", 10)))
    calibration_fraction = float(protocol.get("inner_calibration_fraction", 0.20))
    n_bins = int(protocol.get("n_bins", 10))
    methods = tuple(protocol.get("methods", METHODS))
    if methods != METHODS:
        raise CalibrationContractError(f"Canonical methods must be {METHODS}; received {methods}")
    if outer_splits != 10:
        raise CalibrationContractError("Canonical calibration requires outer stratified 10-fold CV.")
    if n_bins != 10:
        raise CalibrationContractError("Canonical calibration requires 10 bins unless the config is explicitly reviewed.")
    seed = resolve_seed(settings, "calibration")
    parameters = _model_parameters(settings)
    splitter = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)

    fold_rows: list[Dict[str, Any]] = []
    prediction_rows: list[Dict[str, Any]] = []
    for fold, (outer_train_positions, test_positions) in enumerate(splitter.split(X, y), start=1):
        X_outer_train = X.iloc[outer_train_positions]
        y_outer_train = y.iloc[outer_train_positions]
        X_inner, X_calibration, y_inner, y_calibration = train_test_split(
            X_outer_train,
            y_outer_train,
            test_size=calibration_fraction,
            random_state=seed + fold,
            stratify=y_outer_train,
        )
        X_test = X.iloc[test_positions]
        y_test = y.iloc[test_positions]
        pipeline = _fit_pipeline(X_inner, y_inner, parameters, seed)
        model_classes = pipeline.named_steps["model"].classes_
        calibration_raw = align_proba(pipeline.predict_proba(X_calibration), model_classes, labels)
        test_raw = align_proba(pipeline.predict_proba(X_test), model_classes, labels)
        for method in methods:
            probability = calibrate_probabilities(
                calibration_raw,
                y_calibration,
                test_raw,
                labels,
                method=method,
                seed=seed + fold,
            )
            prediction = predict_labels_from_proba(probability, labels)
            metrics = classification_metrics(y_test, prediction, probability, labels=labels)
            fold_rows.append(
                {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "policy": primary_policy,
                    "method": method,
                    "fold": fold,
                    "n_outer_train": len(outer_train_positions),
                    "n_inner_train": len(X_inner),
                    "n_calibration": len(X_calibration),
                    "n_test": len(test_positions),
                    **metrics,
                }
            )
            for row_position, sample_index in enumerate(X_test.index):
                row: Dict[str, Any] = {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "policy": primary_policy,
                    "method": method,
                    "fold": fold,
                    "sample_index": int(sample_index),
                    "y_true": int(y_test.loc[sample_index]),
                    "y_pred": int(prediction[row_position]),
                }
                for label_index, label in enumerate(labels):
                    row[f"prob_class_{label}"] = float(probability[row_position, label_index])
                prediction_rows.append(row)

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize_fold_metrics(fold_metrics)
    selection_metrics = protocol.get(
        "selection_metrics",
        ["nll_log_loss", "multiclass_brier", "ece_confidence"],
    )
    selected_method, ranked_summary = select_calibration_method(summary, selection_metrics)
    bin_rows: list[Dict[str, Any]] = []
    for method in methods:
        method_predictions = predictions[predictions["method"] == method].sort_values("sample_index")
        probabilities = method_predictions[[f"prob_class_{label}" for label in labels]].to_numpy(dtype=float)
        bin_rows.extend(
            calibration_bin_rows(
                method_predictions["y_true"].to_numpy(dtype=int),
                probabilities,
                labels,
                run_id=run_id,
                config_hash=config_hash,
                method=method,
                n_bins=n_bins,
            )
        )
    bins = pd.DataFrame(bin_rows)

    paths: Dict[str, Path] = {
        "fold_metrics": output / "calibration_fold_metrics.csv",
        "predictions": output / "calibration_predictions.csv",
        "bins": output / "calibration_bins.csv",
        "method_comparison": output / "calibration_method_comparison.csv",
        "uncertainty": output / "calibration_uncertainty.csv",
        "rationale": output / "selected_method_rationale.md",
        "figure_source_bins": output / "figure_5_reliability_source.csv",
        "figure_source_metrics": output / "figure_5_metric_source.csv",
        "metadata": output / "calibration_metadata.json",
    }
    fold_metrics.to_csv(paths["fold_metrics"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    bins.to_csv(paths["bins"], index=False)
    ranked_summary.to_csv(paths["method_comparison"], index=False)
    uncertainty_columns = [column for column in summary.columns if column.endswith(("_std", "_ci_low", "_ci_high"))]
    summary[["run_id", "config_hash", "method", "n_folds", *uncertainty_columns]].to_csv(paths["uncertainty"], index=False)
    bins.to_csv(paths["figure_source_bins"], index=False)
    summary.to_csv(paths["figure_source_metrics"], index=False)
    write_rationale(
        paths["rationale"],
        ranked_summary,
        selected_method=selected_method,
        run_id=run_id,
        config_hash=config_hash,
    )
    for label in labels:
        outputs = _plot_single_reliability(bins, label, figures, run_id=run_id, config_hash=config_hash)
        paths[f"reliability_class_{label}_png"] = outputs["png"]
        paths[f"reliability_class_{label}_svg"] = outputs["svg"]
    figure_5 = write_figure_5(
        bins,
        summary,
        labels,
        output,
        selected_method=selected_method,
        run_id=run_id,
        config_hash=config_hash,
    )
    paths["figure_5_png"] = figure_5["png"]
    paths["figure_5_svg"] = figure_5["svg"]
    write_json(
        paths["metadata"],
        {
            "stage": "calibration",
            "run_id": run_id,
            "config_hash": config_hash,
            "policy": primary_policy,
            "excluded_features": excluded,
            "protocol": {
                "outer_strategy": "StratifiedKFold",
                "outer_folds": outer_splits,
                "inner_calibration_fraction": calibration_fraction,
                "evaluation_scope": "outer_test_folds_only",
                "methods": methods,
                "n_bins": n_bins,
                "selection_metrics": selection_metrics,
            },
            "selected_method": selected_method,
            "seed": seed,
            "outputs": {key: str(value) for key, value in paths.items() if key != "metadata"},
            "probability_warning": "Approximate model confidence only; use bands and human review, not autonomous HR thresholds.",
        },
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical nested calibration evidence package.")
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
