from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.data.preprocess import load_validated_or_raw_data, split_features_and_target
from src.utils.config import SETTINGS


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(text)).strip("_")


def prepare_lightgbm_frame(X: pd.DataFrame) -> pd.DataFrame:
    X_out = X.copy()

    for col in X_out.columns:
        if pd.api.types.is_numeric_dtype(X_out[col]):
            X_out[col] = pd.to_numeric(X_out[col], errors="coerce")
        else:
            X_out[col] = X_out[col].astype("category")

    return X_out


def get_split(drop_sensitive: bool, random_state: int, test_size: float):
    df = load_validated_or_raw_data()
    X, y = split_features_and_target(df, drop_sensitive=drop_sensitive)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y.astype(int),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return (
        prepare_lightgbm_frame(X_train),
        prepare_lightgbm_frame(X_test),
        y_train.astype(int),
        y_test.astype(int),
    )


def normalize_shap_values(raw_values: Any, n_samples: int, n_features: int, n_classes: int) -> np.ndarray:
    if isinstance(raw_values, list):
        arr = np.stack(raw_values, axis=1)
        return arr

    arr = np.asarray(raw_values)

    if arr.ndim == 2 and n_classes == 1:
        return arr[:, np.newaxis, :]

    if arr.ndim == 3:
        if arr.shape == (n_samples, n_classes, n_features):
            return arr
        if arr.shape == (n_classes, n_samples, n_features):
            return np.transpose(arr, (1, 0, 2))
        if arr.shape == (n_samples, n_features, n_classes):
            return np.transpose(arr, (0, 2, 1))

    raise ValueError(f"Unexpected LightGBM SHAP shape: {arr.shape}")


def display_matrix(X: pd.DataFrame, feature_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    plot_data = pd.DataFrame(index=X.index)
    display_data = pd.DataFrame(index=X.index)

    for feature in feature_names:
        series = X[feature]

        if pd.api.types.is_numeric_dtype(series):
            plot_data[feature] = pd.to_numeric(series, errors="coerce")
            display_data[feature] = series
        else:
            as_str = series.astype("string").fillna("__MISSING__")
            codes, _ = pd.factorize(as_str)
            plot_data[feature] = codes.astype(float)
            display_data[feature] = as_str.astype(str)

    plot_data = plot_data.fillna(plot_data.median(numeric_only=True))
    display_data = display_data.fillna("__MISSING__")

    return plot_data, display_data


def plot_summary(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    expected_values: Any,
    feature_names: List[str],
    class_labels: List[int],
    class_label: int,
    output_path: Path,
    top_k: int,
) -> None:
    class_idx = class_labels.index(class_label)
    plot_data, display_data = display_matrix(X, feature_names)

    if isinstance(expected_values, (list, np.ndarray)):
        base_value = np.asarray(expected_values)[class_idx]
    else:
        base_value = expected_values

    explanation = shap.Explanation(
        values=shap_values[:, class_idx, :],
        base_values=np.full(len(X), base_value),
        data=plot_data[feature_names].values,
        feature_names=feature_names,
    )
    explanation.display_data = display_data[feature_names].values

    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(
        explanation,
        max_display=top_k,
        show=False,
        plot_size=(10, 7),
        color_bar=True,
    )
    plt.title(f"LightGBM SHAP Beeswarm Summary | Class {class_label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_local_table(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: List[str],
    class_labels: List[int],
    sample_index: int,
    explained_class: int,
) -> pd.DataFrame:
    row_position = X.index.tolist().index(sample_index)
    class_idx = class_labels.index(explained_class)

    local_df = pd.DataFrame({
        "feature": feature_names,
        "feature_value": [X.loc[sample_index, f] for f in feature_names],
        "shap_value": shap_values[row_position, class_idx, :],
    })
    local_df["abs_shap_value"] = local_df["shap_value"].abs()
    local_df["direction"] = np.where(local_df["shap_value"] >= 0, "positive", "negative")
    return local_df.sort_values("abs_shap_value", ascending=False).reset_index(drop=True)


def plot_local(local_df: pd.DataFrame, output_path: Path, title: str, top_k: int = 12) -> None:
    data = local_df.head(top_k).iloc[::-1].copy()
    colors = np.where(data["shap_value"] >= 0, "#d62728", "#1f77b4")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(data["feature"], data["shap_value"], color=colors)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("SHAP contribution")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_cases(y_true: pd.Series, pred: pd.Series, proba: pd.DataFrame, class_labels: List[int]):
    cases = []

    for class_label in class_labels:
        subset = proba[pred == class_label]
        if len(subset):
            idx = int(subset[class_label].idxmax())
            cases.append((
                f"high_conf_class_{class_label}",
                idx,
                int(pred.loc[idx]),
                int(y_true.loc[idx]),
                float(proba.loc[idx, class_label]),
            ))

    uncertainty = proba.max(axis=1)
    idx = int(uncertainty.idxmin())
    cases.append((
        "most_uncertain",
        idx,
        int(pred.loc[idx]),
        int(y_true.loc[idx]),
        float(uncertainty.loc[idx]),
    ))

    wrong = pred[pred != y_true]
    if len(wrong):
        idx = int(wrong.index[0])
        cases.append((
            "misclassified_example",
            idx,
            int(pred.loc[idx]),
            int(y_true.loc[idx]),
            float(proba.loc[idx].max()),
        ))

    return cases


def run_lightgbm_shap(
    drop_sensitive: bool = True,
    random_state: int = 42,
    test_size: float = 0.20,
    top_k: int = 15,
) -> None:
    output_dir = SETTINGS.reports_dir / "xai" / "lightgbm"
    ensure_dir(output_dir)

    X_train, X_test, y_train, y_test = get_split(
        drop_sensitive=drop_sensitive,
        random_state=random_state,
        test_size=test_size,
    )

    cat_features = [
        col for col in X_train.columns
        if str(X_train[col].dtype) == "category"
    ]

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=random_state,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        categorical_feature=cat_features if cat_features else "auto",
    )

    pred = pd.Series(model.predict(X_test).astype(int), index=X_test.index)
    class_labels = [int(c) for c in model.classes_]
    proba = pd.DataFrame(model.predict_proba(X_test), columns=class_labels, index=X_test.index)

    feature_names = X_test.columns.tolist()
    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_test)
    shap_values = normalize_shap_values(
        raw_values=raw_shap,
        n_samples=len(X_test),
        n_features=len(feature_names),
        n_classes=len(class_labels),
    )

    expected_values = explainer.expected_value

    for class_label in class_labels:
        plot_summary(
            X=X_test,
            shap_values=shap_values,
            expected_values=expected_values,
            feature_names=feature_names,
            class_labels=class_labels,
            class_label=class_label,
            output_path=output_dir / f"summary_class_{class_label}.png",
            top_k=top_k,
        )

    cases = select_cases(y_test, pred, proba, class_labels)
    case_rows = []

    for case_name, sample_index, predicted_class, true_class, confidence in cases:
        local_df = build_local_table(
            X=X_test,
            shap_values=shap_values,
            feature_names=feature_names,
            class_labels=class_labels,
            sample_index=sample_index,
            explained_class=predicted_class,
        )
        local_df.to_csv(output_dir / f"{case_name}_local_shap_table.csv", index=False)
        plot_local(
            local_df,
            output_dir / f"{case_name}_local_explanation.png",
            f"{case_name} | pred={predicted_class} true={true_class} prob={confidence:.3f}",
        )

        case_rows.append({
            "case": case_name,
            "sample_index": sample_index,
            "predicted_class": predicted_class,
            "true_class": true_class,
            "confidence": confidence,
        })

    predictions = pd.DataFrame({
        "sample_index": X_test.index,
        "y_true": y_test.values,
        "y_pred": pred.values,
        "correct": y_test.values == pred.values,
    })
    for label in class_labels:
        predictions[f"prob_class_{label}"] = proba[label].values

    predictions.to_csv(output_dir / "lightgbm_test_predictions.csv", index=False)
    pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).T.to_csv(
        output_dir / "lightgbm_classification_report.csv"
    )
    pd.DataFrame(
        confusion_matrix(y_test, pred, labels=class_labels),
        index=[f"true_{x}" for x in class_labels],
        columns=[f"pred_{x}" for x in class_labels],
    ).to_csv(output_dir / "lightgbm_confusion_matrix.csv")

    pd.DataFrame(case_rows).to_csv(output_dir / "representative_cases.csv", index=False)

    save_json(
        {
            "drop_sensitive": drop_sensitive,
            "random_state": random_state,
            "test_size": test_size,
            "class_labels": class_labels,
            "categorical_features": cat_features,
            "output_dir": str(output_dir),
        },
        output_dir / "lightgbm_shap_metadata.json",
    )

    print("\n=== LIGHTGBM SHAP COMPLETE ===")
    print(f"Outputs: {output_dir}")
    for item in sorted(output_dir.glob("*")):
        print(f"  - {item.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightGBM SHAP analysis for employee performance project.")
    parser.add_argument("--drop-sensitive", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_lightgbm_shap(
        drop_sensitive=args.drop_sensitive,
        random_state=args.random_state,
        test_size=args.test_size,
        top_k=args.top_k,
    )
