from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.experiments.final_evidence_common import (
    CALIBRATION_DIR,
    FINAL_FEATURE_SETS,
    LABELS,
    MODEL_NAME,
    align_proba,
    append_task_registry,
    calibrate_probabilities,
    ensure_dir,
    fit_xgb_pipeline,
    get_current_xy,
    predict_labels_from_proba,
    save_json,
)
from src.models.evaluate import classification_metrics
from src.utils.experiment_registry import collect_package_versions, utc_now_iso


METHODS = ["raw", "sigmoid", "isotonic"]


def calibration_bin_rows(y_true: pd.Series, proba: np.ndarray, feature_set: str, method: str, n_bins: int = 10) -> List[Dict[str, Any]]:
    y_arr = y_true.astype(int).to_numpy()
    rows: List[Dict[str, Any]] = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for label_idx, label in enumerate(LABELS):
        scores = proba[:, label_idx]
        actual = (y_arr == label).astype(float)
        for bin_idx, (low, high) in enumerate(zip(bins[:-1], bins[1:]), start=1):
            mask = ((scores >= low) & (scores <= high)) if bin_idx == 1 else ((scores > low) & (scores <= high))
            if not np.any(mask):
                continue
            rows.append(
                {
                    "feature_set": feature_set,
                    "method": method,
                    "class_label": label,
                    "bin": bin_idx,
                    "bin_low": low,
                    "bin_high": high,
                    "n_samples": int(mask.sum()),
                    "mean_predicted_probability": float(scores[mask].mean()),
                    "observed_frequency": float(actual[mask].mean()),
                    "absolute_gap": float(abs(scores[mask].mean() - actual[mask].mean())),
                }
            )
    return rows


def summarize(fold_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "quadratic_weighted_kappa",
        "ordinal_mae",
        "severe_error_rate",
        "nll_log_loss",
        "multiclass_brier",
        "ece_confidence",
    ]
    rows = []
    for (feature_set, method), group in fold_df.groupby(["feature_set", "method"]):
        row = {"feature_set": feature_set, "method": method, "n_folds": int(len(group))}
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_reliability(bins_df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    for (feature_set, class_label), subset in bins_df.groupby(["feature_set", "class_label"]):
        fig, ax = plt.subplots(figsize=(6, 5))
        for method, method_df in subset.groupby("method"):
            method_df = method_df.sort_values("mean_predicted_probability")
            ax.plot(method_df["mean_predicted_probability"], method_df["observed_frequency"], marker="o", label=method)
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        ax.set_title(f"Reliability | {feature_set} | class {class_label}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"reliability_{feature_set}_class_{class_label}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def write_interpretation(summary_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Calibration Interpretation for Final Candidates",
        "",
        "Protocol: outer 10-fold CV. Inside each training fold, an inner train/calibration split is used. Sigmoid and isotonic calibrators are fitted only on the inner calibration split, then evaluated on the outer test fold.",
        "",
        "## Summary by Candidate",
        "",
    ]
    for feature_set, group in summary_df.groupby("feature_set"):
        best_ece = group.sort_values("ece_confidence_mean").iloc[0]
        best_log = group.sort_values("nll_log_loss_mean").iloc[0]
        best_brier = group.sort_values("multiclass_brier_mean").iloc[0]
        raw = group[group["method"] == "raw"].iloc[0]
        lines.extend(
            [
                f"### `{feature_set}`",
                f"- Best ECE: `{best_ece['method']}` ({best_ece['ece_confidence_mean']:.4f}).",
                f"- Best log loss: `{best_log['method']}` ({best_log['nll_log_loss_mean']:.4f}).",
                f"- Best Brier: `{best_brier['method']}` ({best_brier['multiclass_brier_mean']:.4f}).",
                f"- Raw nested macro-F1/QWK: {raw['macro_f1_mean']:.4f} / {raw['quadratic_weighted_kappa_mean']:.4f}.",
                "",
            ]
        )
    primary = summary_df[summary_df["feature_set"] == "no_salary_hike_no_attrition_no_department"].copy()
    ranked = primary.assign(
        rank_sum=primary["nll_log_loss_mean"].rank() + primary["multiclass_brier_mean"].rank() + primary["ece_confidence_mean"].rank()
    ).sort_values("rank_sum")
    best = ranked.iloc[0]
    raw = primary[primary["method"] == "raw"].iloc[0]
    lines.extend(
        [
            "## Required Answers",
            "",
            "### Which candidate has the best probability quality?",
            "The dashboard combines candidate-level metrics. For the primary candidate, the best average rank across log loss, Brier, and ECE is "
            f"`{best['method']}`.",
            "",
            "### Does calibration improve probability quality?",
            "Calibration is mixed unless it improves log loss, Brier, and ECE jointly. Do not select a method by ECE alone.",
            "",
            "### Does calibration introduce overfitting or instability risk?",
            "Yes, especially isotonic calibration because class 4 is small. Sigmoid is less flexible and usually safer with limited calibration data.",
            "",
            "### Should the final model output calibrated probabilities, raw probabilities, or probability bands with warnings?",
            f"Use probability bands with calibration warnings. If `{best['method']}` is used in tables, disclose the nested calibration protocol. Its macro-F1 difference versus raw in this protocol is {raw['macro_f1_mean'] - best['macro_f1_mean']:.4f}.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(n_splits: int = 10, seed: int = 42) -> Dict[str, Path]:
    ensure_dir(CALIBRATION_DIR)
    figures_dir = CALIBRATION_DIR / "figures"
    ensure_dir(figures_dir)
    fold_rows: List[Dict[str, Any]] = []
    bin_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []

    for feature_set in FINAL_FEATURE_SETS:
        X, y, _ = get_current_xy(feature_set, drop_sensitive=True)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
            X_outer_train = X.iloc[train_idx].copy()
            y_outer_train = y.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_test = y.iloc[test_idx].copy()
            X_inner, X_calib, y_inner, y_calib = train_test_split(
                X_outer_train,
                y_outer_train,
                test_size=0.20,
                random_state=seed + fold,
                stratify=y_outer_train,
            )
            pipeline = fit_xgb_pipeline(X_inner, y_inner, random_state=seed)
            classes = [int(c) for c in pipeline.named_steps["model"].classes_]
            calib_raw = align_proba(pipeline.predict_proba(X_calib), classes, LABELS)
            test_raw = align_proba(pipeline.predict_proba(X_test), classes, LABELS)
            for method in METHODS:
                proba = calibrate_probabilities(calib_raw, y_calib, test_raw, LABELS, method=method, seed=seed)
                pred = predict_labels_from_proba(proba, LABELS)
                metrics = classification_metrics(y_test, pred, proba, LABELS)
                fold_rows.append({"feature_set": feature_set, "method": method, "fold": fold, **metrics})
                bin_rows.extend(calibration_bin_rows(y_test, proba, feature_set, method, n_bins=10))
                for row_pos, sample_index in enumerate(X_test.index):
                    row = {
                        "feature_set": feature_set,
                        "method": method,
                        "fold": fold,
                        "sample_index": int(sample_index),
                        "y_true": int(y_test.loc[sample_index]),
                        "y_pred": int(pred[row_pos]),
                    }
                    for label_idx, label in enumerate(LABELS):
                        row[f"prob_class_{label}"] = float(proba[row_pos, label_idx])
                    pred_rows.append(row)
            print(f"[calibration] feature_set={feature_set} fold={fold}")

    fold_df = pd.DataFrame(fold_rows)
    summary_df = summarize(fold_df)
    bins_df = pd.DataFrame(bin_rows)
    pred_df = pd.DataFrame(pred_rows)

    summary_path = CALIBRATION_DIR / "calibration_summary.csv"
    bins_path = CALIBRATION_DIR / "calibration_bins.csv"
    fold_path = CALIBRATION_DIR / "calibration_fold_metrics.csv"
    pred_path = CALIBRATION_DIR / "calibration_predictions.csv"
    interp_path = CALIBRATION_DIR / "calibration_interpretation.md"
    meta_path = CALIBRATION_DIR / "metadata.json"
    summary_df.to_csv(summary_path, index=False)
    bins_df.to_csv(bins_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    plot_reliability(bins_df, figures_dir)
    write_interpretation(summary_df, interp_path)
    save_json(
        {
            "task": "final_candidate_calibration",
            "feature_sets": FINAL_FEATURE_SETS,
            "model": MODEL_NAME,
            "n_splits": n_splits,
            "seed": seed,
            "methods": METHODS,
            "package_versions": collect_package_versions(["numpy", "pandas", "scikit-learn", "xgboost", "matplotlib"]),
        },
        meta_path,
    )
    append_task_registry(
        run_id=f"final_calibration_{utc_now_iso()}",
        script="python -m src.experiments.final_calibration",
        feature_set="; ".join(FINAL_FEATURE_SETS),
        primary_metrics="log loss; multiclass Brier; ECE; class-wise calibration bins",
        output_dir=CALIBRATION_DIR,
        notes="Nested train/calibration split inside each outer fold for raw, sigmoid, and isotonic probability diagnostics.",
    )
    return {"summary": summary_path, "bins": bins_path, "interpretation": interp_path, "metadata": meta_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibration diagnostics for final XGBoost candidates.")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run(n_splits=args.n_splits, seed=args.seed))

