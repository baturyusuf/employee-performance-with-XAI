from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

from src.data.preprocess import get_feature_groups, run_preprocessing
from src.utils.config import SETTINGS
import numpy as np

try:
    from catboost import CatBoostClassifier, Pool
except ImportError as exc:
    raise ImportError(
        "catboost is not installed. Please run: pip install catboost"
    ) from exc


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
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
    cm_df.to_csv(output_dir / f"{split_name}_confusion_matrix.csv", index=True)

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
    fig.savefig(output_dir / f"{split_name}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return cm_df


def prepare_catboost_inputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    drop_sensitive: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, List[str], List[int]]:
    """
    Make CatBoost-safe inputs:
    - categorical columns converted to string
    - missing categorical values filled
    - numeric/ordinal columns forced numeric where appropriate
    """
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()

    feature_groups = get_feature_groups(drop_sensitive=drop_sensitive)
    categorical_cols = feature_groups["categorical"]
    numeric_cols = feature_groups["numeric"]
    ordinal_cols = feature_groups["ordinal"]

    for col in categorical_cols:
        X_train_cb[col] = X_train_cb[col].astype("string").fillna("__MISSING__")
        X_test_cb[col] = X_test_cb[col].astype("string").fillna("__MISSING__")

    for col in numeric_cols + ordinal_cols:
        X_train_cb[col] = pd.to_numeric(X_train_cb[col], errors="coerce")
        X_test_cb[col] = pd.to_numeric(X_test_cb[col], errors="coerce")

    feature_names = X_train_cb.columns.tolist()
    cat_feature_indices = [feature_names.index(col) for col in categorical_cols if col in feature_names]

    return X_train_cb, X_test_cb, feature_names, cat_feature_indices


def maybe_compute_class_weights(
    y_train: pd.Series,
    use_balanced: bool,
) -> Optional[List[float]]:
    if not use_balanced:
        return None

    classes = np.array(sorted(y_train.unique().tolist()))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    return list(weights)

    classes = sorted(y_train.unique().tolist())
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    return list(weights)


def save_feature_importance(
    model: CatBoostClassifier,
    feature_names: List[str],
    output_dir: Path,
) -> pd.DataFrame:
    importances = model.get_feature_importance()
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False)

    fi_df.to_csv(output_dir / "feature_importance.csv", index=False)

    top_k = min(20, len(fi_df))
    top_df = fi_df.head(top_k).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_df["feature"], top_df["importance"])
    ax.set_title("Top Feature Importances (CatBoost)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance_top20.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return fi_df


def train_catboost_model(
    test_size: float = 0.20,
    random_state: int = 42,
    drop_sensitive: bool = False,
    use_balanced_class_weights: bool = False,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 100,
) -> Dict[str, Any]:
    prep = run_preprocessing(
        test_size=test_size,
        random_state=random_state,
        drop_sensitive=drop_sensitive,
    )

    X_train_raw = prep["X_train_raw"]
    X_test_raw = prep["X_test_raw"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]
    metadata = prep["metadata"]

    X_train_cb, X_test_cb, feature_names, cat_feature_indices = prepare_catboost_inputs(
        X_train=X_train_raw,
        X_test=X_test_raw,
        drop_sensitive=drop_sensitive,
    )

    class_weights = maybe_compute_class_weights(
        y_train=y_train,
        use_balanced=use_balanced_class_weights,
    )

    train_pool = Pool(
        data=X_train_cb,
        label=y_train,
        cat_features=cat_feature_indices,
        feature_names=feature_names,
    )
    test_pool = Pool(
        data=X_test_cb,
        label=y_test,
        cat_features=cat_feature_indices,
        feature_names=feature_names,
    )

    output_dir = SETTINGS.artifacts_dir / "catboost"
    ensure_dir(output_dir)

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_state,
        verbose=verbose_eval,
        early_stopping_rounds=early_stopping_rounds,
        class_weights=class_weights,
    )

    model.fit(
        train_pool,
        eval_set=test_pool,
        use_best_model=True,
    )

    y_train_pred = pd.Series(model.predict(train_pool).flatten().astype(int), index=y_train.index, name="prediction")
    y_test_pred = pd.Series(model.predict(test_pool).flatten().astype(int), index=y_test.index, name="prediction")

    y_train_proba = pd.DataFrame(
        model.predict_proba(train_pool),
        columns=[f"class_{c}" for c in model.classes_],
        index=y_train.index,
    )
    y_test_proba = pd.DataFrame(
        model.predict_proba(test_pool),
        columns=[f"class_{c}" for c in model.classes_],
        index=y_test.index,
    )

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    print("\n=== CATBOOST TRAINING COMPLETE ===")
    print("Train metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save model
    model_path = output_dir / "catboost_model.cbm"
    model.save_model(str(model_path))

    # Save joblib wrapper as convenience
    joblib.dump(model, output_dir / "catboost_model.joblib")

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
    y_train_proba.to_csv(output_dir / "train_pred_proba.csv", index=True)
    y_test_proba.to_csv(output_dir / "test_pred_proba.csv", index=True)

    # Save reports
    save_classification_report(y_train, y_train_pred, output_dir / "train_classification_report.csv")
    save_classification_report(y_test, y_test_pred, output_dir / "test_classification_report.csv")
    save_confusion_matrix_outputs(y_train, y_train_pred, output_dir, "train")
    save_confusion_matrix_outputs(y_test, y_test_pred, output_dir, "test")

    # Save feature importance
    fi_df = save_feature_importance(model, feature_names, output_dir)

    # Save feature metadata
    feature_meta = {
        "feature_names": feature_names,
        "categorical_feature_names": [feature_names[i] for i in cat_feature_indices],
        "categorical_feature_indices": cat_feature_indices,
    }
    save_json(feature_meta, output_dir / "feature_metadata.json")

    run_summary = {
        "model_name": "catboost_multiclass",
        "drop_sensitive": drop_sensitive,
        "use_balanced_class_weights": use_balanced_class_weights,
        "class_weights": class_weights,
        "random_state": random_state,
        "test_size": test_size,
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "l2_leaf_reg": l2_leaf_reg,
        "early_stopping_rounds": early_stopping_rounds,
        "best_iteration": int(model.get_best_iteration()) if model.get_best_iteration() is not None else None,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "input_metadata": metadata,
        "output_files": {
            "model_cbm": str(model_path),
            "model_joblib": str(output_dir / "catboost_model.joblib"),
            "train_predictions": str(output_dir / "train_predictions.csv"),
            "test_predictions": str(output_dir / "test_predictions.csv"),
            "train_pred_proba": str(output_dir / "train_pred_proba.csv"),
            "test_pred_proba": str(output_dir / "test_pred_proba.csv"),
            "train_classification_report": str(output_dir / "train_classification_report.csv"),
            "test_classification_report": str(output_dir / "test_classification_report.csv"),
            "feature_importance": str(output_dir / "feature_importance.csv"),
            "feature_importance_plot": str(output_dir / "feature_importance_top20.png"),
            "feature_metadata": str(output_dir / "feature_metadata.json"),
        },
    }
    save_json(run_summary, output_dir / "run_summary.json")

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "feature_importance": fi_df,
        "output_dir": output_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost multiclass model.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test size ratio. Default: 0.20")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument("--drop-sensitive", action="store_true", help="Drop fairness-sensitive columns.")
    parser.add_argument(
        "--balanced-class-weights",
        action="store_true",
        help="Use balanced class weights computed from y_train.",
    )
    parser.add_argument("--iterations", type=int, default=500, help="Max boosting iterations.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--depth", type=int, default=6, help="Tree depth.")
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0, help="L2 regularization.")
    parser.add_argument("--early-stopping-rounds", type=int, default=50, help="Early stopping patience.")
    parser.add_argument("--verbose-eval", type=int, default=100, help="Verbosity interval.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_catboost_model(
        test_size=args.test_size,
        random_state=args.random_state,
        drop_sensitive=args.drop_sensitive,
        use_balanced_class_weights=args.balanced_class_weights,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=args.verbose_eval,
    )