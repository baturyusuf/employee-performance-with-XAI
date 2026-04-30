from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.preprocess import load_validated_or_raw_data, split_features_and_target
from src.utils.config import SETTINGS


FEATURE_SET_DEFINITIONS = {
    "no_salary_hike_no_attrition": [
        "EmpLastSalaryHikePercent",
        "Attrition",
    ],
    "no_salary_hike_no_attrition_no_department": [
        "EmpLastSalaryHikePercent",
        "Attrition",
        "EmpDepartment",
    ],
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return {str(k): to_jsonable(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(text)).strip("_")


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = []
    categorical_cols = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols, categorical_cols = infer_columns(X)
    transformers = []

    if numeric_cols:
        transformers.append(("num", "passthrough", numeric_cols))

    if categorical_cols:
        transformers.append(("cat", one_hot_encoder(), categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


class LabelEncodedXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        objective: str = "multi:softprob",
        eval_metric: str = "mlogloss",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier

        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective=self.objective,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            num_class=len(self.classes_),
        )
        self.model_.fit(X, y_encoded)
        return self

    def predict(self, X):
        pred_encoded = self.model_.predict(X).astype(int)
        return self.label_encoder_.inverse_transform(pred_encoded)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def apply_feature_set(X: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set not in FEATURE_SET_DEFINITIONS:
        raise ValueError(
            f"Unknown feature set '{feature_set}'. "
            f"Allowed values: {sorted(FEATURE_SET_DEFINITIONS)}"
        )

    removed = set(FEATURE_SET_DEFINITIONS[feature_set])
    keep_cols = [col for col in X.columns if col not in removed]
    return X[keep_cols].copy()


def get_split(
    feature_set: str,
    drop_sensitive: bool,
    test_size: float,
    random_state: int,
):
    df = load_validated_or_raw_data()
    X, y = split_features_and_target(df, drop_sensitive=drop_sensitive)
    X = apply_feature_set(X, feature_set=feature_set)
    y = y.astype(int)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def normalize_shap_values(raw_values: Any, n_samples: int, n_features: int, n_classes: int) -> np.ndarray:
    if isinstance(raw_values, list):
        return np.stack(raw_values, axis=1)

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

    raise ValueError(f"Unexpected XGBoost SHAP shape: {arr.shape}")


def get_group_mapping(
    preprocessor: ColumnTransformer,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[List[str], Dict[str, List[int]], List[str]]:
    group_names = numeric_cols + categorical_cols
    mapping: Dict[str, List[int]] = {name: [] for name in group_names}
    transformed_names = []

    col_idx = 0

    for col in numeric_cols:
        mapping[col].append(col_idx)
        transformed_names.append(col)
        col_idx += 1

    if categorical_cols:
        encoder = preprocessor.named_transformers_["cat"]

        for col, categories in zip(categorical_cols, encoder.categories_):
            for category in categories:
                mapping[col].append(col_idx)
                transformed_names.append(f"{col}={category}")
                col_idx += 1

    return group_names, mapping, transformed_names


def group_shap_values(
    shap_values: np.ndarray,
    group_names: List[str],
    mapping: Dict[str, List[int]],
) -> np.ndarray:
    n_samples, n_classes, _ = shap_values.shape
    grouped = np.zeros((n_samples, n_classes, len(group_names)))

    for group_idx, group_name in enumerate(group_names):
        cols = mapping[group_name]
        if cols:
            grouped[:, :, group_idx] = shap_values[:, :, cols].sum(axis=2)

    return grouped


def build_display_matrices(X: pd.DataFrame, group_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    plot_data = pd.DataFrame(index=X.index)
    display_data = pd.DataFrame(index=X.index)

    for feature in group_names:
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


def plot_grouped_beeswarm(
    X: pd.DataFrame,
    grouped_shap: np.ndarray,
    expected_values: Any,
    group_names: List[str],
    class_labels: List[int],
    class_label: int,
    output_path: Path,
    title: str,
    top_k: int,
) -> None:
    class_idx = class_labels.index(class_label)
    plot_data, display_data = build_display_matrices(X, group_names)

    if isinstance(expected_values, (list, np.ndarray)):
        base_arr = np.asarray(expected_values)
        if base_arr.ndim == 1 and len(base_arr) == len(class_labels):
            base_values = np.full(len(X), base_arr[class_idx])
        elif base_arr.ndim == 1 and len(base_arr) == len(X):
            base_values = base_arr
        else:
            base_values = np.full(len(X), float(base_arr.flatten()[0]))
    else:
        base_values = np.full(len(X), float(expected_values))

    explanation = shap.Explanation(
        values=grouped_shap[:, class_idx, :],
        base_values=base_values,
        data=plot_data[group_names].values,
        feature_names=group_names,
    )
    explanation.display_data = display_data[group_names].values

    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(
        explanation,
        max_display=top_k,
        show=False,
        plot_size=(10, 7),
        color_bar=True,
    )
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_local_table(
    X: pd.DataFrame,
    grouped_shap: np.ndarray,
    group_names: List[str],
    class_labels: List[int],
    sample_index: int,
    explained_class: int,
) -> pd.DataFrame:
    row_pos = X.index.tolist().index(sample_index)
    class_idx = class_labels.index(explained_class)

    local_df = pd.DataFrame({
        "feature": group_names,
        "feature_value": [X.loc[sample_index, f] for f in group_names],
        "grouped_shap_value": grouped_shap[row_pos, class_idx, :],
    })
    local_df["abs_grouped_shap_value"] = local_df["grouped_shap_value"].abs()
    local_df["direction"] = np.where(
        local_df["grouped_shap_value"] >= 0,
        "positive",
        "negative",
    )
    return local_df.sort_values("abs_grouped_shap_value", ascending=False).reset_index(drop=True)


def plot_local(local_df: pd.DataFrame, output_path: Path, title: str, top_k: int = 12) -> None:
    data = local_df.head(top_k).iloc[::-1].copy()
    colors = np.where(data["grouped_shap_value"] >= 0, "#d62728", "#1f77b4")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(data["feature"], data["grouped_shap_value"], color=colors)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Grouped SHAP contribution")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_representative_cases(
    y_true: pd.Series,
    pred: pd.Series,
    proba: pd.DataFrame,
    class_labels: List[int],
) -> List[Tuple[str, int, int, int, float]]:
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

    confidence = proba.max(axis=1)
    uncertain_idx = int(confidence.idxmin())
    cases.append((
        "most_uncertain",
        uncertain_idx,
        int(pred.loc[uncertain_idx]),
        int(y_true.loc[uncertain_idx]),
        float(confidence.loc[uncertain_idx]),
    ))

    wrong = pred[pred != y_true]
    if len(wrong):
        wrong_idx = int(wrong.index[0])
        cases.append((
            "misclassified_example",
            wrong_idx,
            int(pred.loc[wrong_idx]),
            int(y_true.loc[wrong_idx]),
            float(proba.loc[wrong_idx].max()),
        ))

    return cases


def run_xgboost_grouped_shap(
    feature_set: str,
    drop_sensitive: bool,
    test_size: float,
    random_state: int,
    top_k: int,
) -> None:
    output_dir = SETTINGS.reports_dir / "leakage_safe" / f"xgboost_{feature_set}" / "shap"
    ensure_dir(output_dir)

    X_train, X_test, y_train, y_test = get_split(
        feature_set=feature_set,
        drop_sensitive=drop_sensitive,
        test_size=test_size,
        random_state=random_state,
    )

    numeric_cols, categorical_cols = infer_columns(X_train)
    preprocessor = make_preprocessor(X_train)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    classifier = LabelEncodedXGBClassifier(random_state=random_state)
    classifier.fit(X_train_transformed, y_train)

    class_labels = [int(c) for c in classifier.classes_]
    pred = pd.Series(classifier.predict(X_test_transformed).astype(int), index=X_test.index)
    proba = pd.DataFrame(classifier.predict_proba(X_test_transformed), columns=class_labels, index=X_test.index)

    group_names, mapping, transformed_names = get_group_mapping(
        preprocessor=preprocessor,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    explainer = shap.TreeExplainer(classifier.model_)
    raw_shap = explainer.shap_values(X_test_transformed)
    transformed_shap = normalize_shap_values(
        raw_values=raw_shap,
        n_samples=X_test_transformed.shape[0],
        n_features=X_test_transformed.shape[1],
        n_classes=len(class_labels),
    )
    grouped_shap = group_shap_values(
        shap_values=transformed_shap,
        group_names=group_names,
        mapping=mapping,
    )

    expected_values = explainer.expected_value

    for class_label in class_labels:
        plot_grouped_beeswarm(
            X=X_test[group_names].copy(),
            grouped_shap=grouped_shap,
            expected_values=expected_values,
            group_names=group_names,
            class_labels=class_labels,
            class_label=class_label,
            output_path=output_dir / f"grouped_summary_class_{class_label}.png",
            title=f"XGBoost Grouped SHAP | {feature_set} | Class {class_label}",
            top_k=top_k,
        )

        class_idx = class_labels.index(class_label)
        class_importance = pd.DataFrame({
            "feature": group_names,
            "mean_abs_grouped_shap": np.mean(np.abs(grouped_shap[:, class_idx, :]), axis=0),
        }).sort_values("mean_abs_grouped_shap", ascending=False)
        class_importance.to_csv(output_dir / f"grouped_importance_class_{class_label}.csv", index=False)

    global_importance = pd.DataFrame({
        "feature": group_names,
        "mean_abs_grouped_shap": np.mean(np.abs(grouped_shap), axis=(0, 1)),
    }).sort_values("mean_abs_grouped_shap", ascending=False)
    global_importance.to_csv(output_dir / "global_grouped_shap_importance.csv", index=False)

    cases = select_representative_cases(
        y_true=y_test,
        pred=pred,
        proba=proba,
        class_labels=class_labels,
    )

    case_rows = []
    for case_name, sample_index, predicted_class, true_class, confidence in cases:
        local_df = build_local_table(
            X=X_test[group_names].copy(),
            grouped_shap=grouped_shap,
            group_names=group_names,
            class_labels=class_labels,
            sample_index=sample_index,
            explained_class=predicted_class,
        )
        local_df.to_csv(output_dir / f"{case_name}_local_grouped_shap_table.csv", index=False)

        plot_local(
            local_df=local_df,
            output_path=output_dir / f"{case_name}_local_explanation.png",
            title=f"{case_name} | pred={predicted_class} true={true_class} prob={confidence:.3f}",
        )

        case_rows.append({
            "case": case_name,
            "sample_index": sample_index,
            "predicted_class": predicted_class,
            "true_class": true_class,
            "confidence": confidence,
        })

    pd.DataFrame(case_rows).to_csv(output_dir / "representative_cases.csv", index=False)

    predictions = pd.DataFrame({
        "sample_index": X_test.index,
        "y_true": y_test.values,
        "y_pred": pred.values,
        "correct": y_test.values == pred.values,
    })
    for label in class_labels:
        predictions[f"prob_class_{label}"] = proba[label].values

    predictions.to_csv(output_dir / "xgboost_test_predictions.csv", index=False)

    pd.DataFrame(
        classification_report(y_test, pred, output_dict=True, zero_division=0)
    ).T.to_csv(output_dir / "xgboost_classification_report.csv")

    pd.DataFrame(
        confusion_matrix(y_test, pred, labels=class_labels),
        index=[f"true_{x}" for x in class_labels],
        columns=[f"pred_{x}" for x in class_labels],
    ).to_csv(output_dir / "xgboost_confusion_matrix.csv")

    save_json(
        {
            "model": "xgboost",
            "feature_set": feature_set,
            "drop_sensitive": drop_sensitive,
            "test_size": test_size,
            "random_state": random_state,
            "class_labels": class_labels,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "group_names": group_names,
            "transformed_feature_names": transformed_names,
            "group_mapping": mapping,
            "removed_features": FEATURE_SET_DEFINITIONS[feature_set],
            "output_dir": str(output_dir),
        },
        output_dir / "xgboost_grouped_shap_metadata.json",
    )

    print("\n=== XGBOOST GROUPED SHAP COMPLETE ===")
    print(f"Outputs: {output_dir}")
    for item in sorted(output_dir.glob("*")):
        print(f"  - {item.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped SHAP for leakage-safe XGBoost model.")
    parser.add_argument(
        "--feature-set",
        type=str,
        default="no_salary_hike_no_attrition",
        choices=sorted(FEATURE_SET_DEFINITIONS.keys()),
    )
    parser.add_argument("--drop-sensitive", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_xgboost_grouped_shap(
        feature_set=args.feature_set,
        drop_sensitive=args.drop_sensitive,
        test_size=args.test_size,
        random_state=args.random_state,
        top_k=args.top_k,
    )
