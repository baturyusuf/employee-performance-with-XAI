from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    cohen_kappa_score,
)

from src.data.preprocess import run_preprocessing
from src.utils.config import SETTINGS


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "ordinal_mae": float(mean_absolute_error(y_true, y_pred)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }
    return metrics


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_classification_report(y_true: pd.Series, y_pred: pd.Series, path: Path) -> pd.DataFrame:
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(path, index=True)
    return report_df


def save_confusion_matrix_outputs(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_dir: Path,
    split_name: str,
) -> pd.DataFrame:
    labels = sorted(list(SETTINGS.allowed_target_labels))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    cm_csv_path = output_dir / f"{split_name}_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path, index=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"{split_name.capitalize()} Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    cm_png_path = output_dir / f"{split_name}_confusion_matrix.png"
    fig.savefig(cm_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return cm_df


def extract_coefficient_table(
    model: LogisticRegression,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    For multiclass logistic regression:
    rows = classes
    cols = coefficients for each feature
    """
    coef_df = pd.DataFrame(model.coef_, columns=feature_names)
    coef_df.insert(0, "class_label", model.classes_)
    return coef_df


def save_top_features_by_class(
    coef_df: pd.DataFrame,
    output_dir: Path,
    top_k: int = 15,
) -> None:
    all_rows = []

    for _, row in coef_df.iterrows():
        class_label = row["class_label"]
        coeffs = row.drop(labels=["class_label"]).astype(float)

        top_positive = coeffs.sort_values(ascending=False).head(top_k)
        top_negative = coeffs.sort_values(ascending=True).head(top_k)

        for feature, value in top_positive.items():
            all_rows.append({
                "class_label": class_label,
                "direction": "positive",
                "feature": feature,
                "coefficient": float(value),
            })

        for feature, value in top_negative.items():
            all_rows.append({
                "class_label": class_label,
                "direction": "negative",
                "feature": feature,
                "coefficient": float(value),
            })

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(output_dir / "top_features_by_class.csv", index=False)


def train_baseline_model(
    test_size: float = 0.20,
    random_state: int = 42,
    drop_sensitive: bool = False,
    class_weight: str | None = None,
    max_iter: int = 3000,
    c_value: float = 1.0,
) -> Dict[str, Any]:
    """
    End-to-end baseline training:
    1) run preprocessing
    2) train multinomial logistic regression
    3) evaluate
    4) save artifacts
    """
    prep = run_preprocessing(
        test_size=test_size,
        random_state=random_state,
        drop_sensitive=drop_sensitive,
    )

    X_train = prep["X_train_processed"]
    X_test = prep["X_test_processed"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]
    metadata = prep["metadata"]

    output_dir = SETTINGS.artifacts_dir / "baseline"
    ensure_dir(output_dir)

    model = LogisticRegression(
        C=c_value,
        max_iter=max_iter,
        solver="lbfgs",
        class_weight=class_weight,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    y_train_pred = pd.Series(model.predict(X_train), index=y_train.index, name="prediction")
    y_test_pred = pd.Series(model.predict(X_test), index=y_test.index, name="prediction")

    if hasattr(model, "predict_proba"):
        y_train_proba = pd.DataFrame(
            model.predict_proba(X_train),
            columns=[f"class_{c}" for c in model.classes_],
            index=X_train.index,
        )
        y_test_proba = pd.DataFrame(
            model.predict_proba(X_test),
            columns=[f"class_{c}" for c in model.classes_],
            index=X_test.index,
        )
    else:
        y_train_proba = None
        y_test_proba = None

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    print("\n=== BASELINE TRAINING COMPLETE ===")
    print("Train metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save model
    model_path = output_dir / "baseline_logreg_model.joblib"
    joblib.dump(model, model_path)

    # Save predictions
    train_pred_df = pd.DataFrame({
        "y_true": y_train,
        "y_pred": y_train_pred,
    }, index=y_train.index)
    test_pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_test_pred,
    }, index=y_test.index)

    train_pred_df.to_csv(output_dir / "train_predictions.csv", index=True)
    test_pred_df.to_csv(output_dir / "test_predictions.csv", index=True)

    if y_train_proba is not None:
        y_train_proba.to_csv(output_dir / "train_pred_proba.csv", index=True)
    if y_test_proba is not None:
        y_test_proba.to_csv(output_dir / "test_pred_proba.csv", index=True)

    # Save reports
    train_report_df = save_classification_report(
        y_train,
        y_train_pred,
        output_dir / "train_classification_report.csv",
    )
    test_report_df = save_classification_report(
        y_test,
        y_test_pred,
        output_dir / "test_classification_report.csv",
    )

    save_confusion_matrix_outputs(y_train, y_train_pred, output_dir, split_name="train")
    save_confusion_matrix_outputs(y_test, y_test_pred, output_dir, split_name="test")

    # Save coefficients
    coef_df = extract_coefficient_table(model, X_train.columns.tolist())
    coef_df.to_csv(output_dir / "coefficients_by_class.csv", index=False)
    save_top_features_by_class(coef_df, output_dir, top_k=15)

    # Save full run summary
    run_summary = {
        "model_name": "multinomial_logistic_regression",
        "drop_sensitive": drop_sensitive,
        "class_weight": class_weight,
        "max_iter": max_iter,
        "C": c_value,
        "random_state": random_state,
        "test_size": test_size,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "model_artifact": str(model_path),
        "input_metadata": metadata,
        "output_files": {
            "train_predictions": str(output_dir / "train_predictions.csv"),
            "test_predictions": str(output_dir / "test_predictions.csv"),
            "train_classification_report": str(output_dir / "train_classification_report.csv"),
            "test_classification_report": str(output_dir / "test_classification_report.csv"),
            "train_confusion_matrix_csv": str(output_dir / "train_confusion_matrix.csv"),
            "test_confusion_matrix_csv": str(output_dir / "test_confusion_matrix.csv"),
            "train_confusion_matrix_png": str(output_dir / "train_confusion_matrix.png"),
            "test_confusion_matrix_png": str(output_dir / "test_confusion_matrix.png"),
            "coefficients_by_class": str(output_dir / "coefficients_by_class.csv"),
            "top_features_by_class": str(output_dir / "top_features_by_class.csv"),
        },
    }
    save_json(run_summary, output_dir / "run_summary.json")

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_report": train_report_df,
        "test_report": test_report_df,
        "coefficients": coef_df,
        "output_dir": output_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline logistic regression model.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test size ratio. Default: 0.20")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument(
        "--drop-sensitive",
        action="store_true",
        help="Drop fairness-sensitive columns during preprocessing.",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default=None,
        choices=[None, "balanced"],
        help="Use balanced class weights or leave as default.",
    )
    parser.add_argument("--max-iter", type=int, default=3000, help="Max iterations for logistic regression.")
    parser.add_argument("--c-value", type=float, default=1.0, help="Inverse regularization strength.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_baseline_model(
        test_size=args.test_size,
        random_state=args.random_state,
        drop_sensitive=args.drop_sensitive,
        class_weight=args.class_weight,
        max_iter=args.max_iter,
        c_value=args.c_value,
    )