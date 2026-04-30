from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize

from src.data.preprocess import load_validated_or_raw_data, split_features_and_target
from src.utils.config import SETTINGS


FEATURE_SET_DEFINITIONS = {
    "full": [],
    "no_salary_hike": ["EmpLastSalaryHikePercent"],
    "no_salary_hike_no_attrition": ["EmpLastSalaryHikePercent", "Attrition"],
    "no_salary_hike_no_department": ["EmpLastSalaryHikePercent", "EmpDepartment"],
    "no_salary_hike_no_attrition_no_department": [
        "EmpLastSalaryHikePercent",
        "Attrition",
        "EmpDepartment",
    ],
}

MODEL_CHOICES = ["catboost", "lightgbm", "xgboost"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)


def safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols, categorical_cols = [], []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    numeric_cols, categorical_cols = infer_columns(X)
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler() if scale_numeric else "passthrough", numeric_cols))
    if categorical_cols:
        transformers.append(("cat", one_hot_encoder(), categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_lightgbm_frame(X: pd.DataFrame) -> pd.DataFrame:
    X_out = X.copy()
    for col in X_out.columns:
        if pd.api.types.is_numeric_dtype(X_out[col]):
            X_out[col] = pd.to_numeric(X_out[col], errors="coerce")
        else:
            X_out[col] = X_out[col].astype("category")
    return X_out


def prepare_catboost_frame(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[int]]:
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()
    categorical_cols = []
    for col in X_train_cb.columns:
        dtype_str = str(X_train_cb[col].dtype)
        if dtype_str == "object" or dtype_str.startswith("category") or dtype_str.startswith("string"):
            categorical_cols.append(col)
    for col in categorical_cols:
        X_train_cb[col] = X_train_cb[col].astype("string").fillna("__MISSING__")
        X_test_cb[col] = X_test_cb[col].astype("string").fillna("__MISSING__")
    for col in [c for c in X_train_cb.columns if c not in categorical_cols]:
        X_train_cb[col] = pd.to_numeric(X_train_cb[col], errors="coerce")
        X_test_cb[col] = pd.to_numeric(X_test_cb[col], errors="coerce")
    feature_names = X_train_cb.columns.tolist()
    cat_idx = [feature_names.index(c) for c in categorical_cols]
    return X_train_cb, X_test_cb, feature_names, cat_idx


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
        encoded_pred = self.model_.predict(X).astype(int)
        return self.label_encoder_.inverse_transform(encoded_pred)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def apply_feature_set(X: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set not in FEATURE_SET_DEFINITIONS:
        raise ValueError(f"Unknown feature set '{feature_set}'. Allowed: {sorted(FEATURE_SET_DEFINITIONS)}")
    remove_cols = set(FEATURE_SET_DEFINITIONS[feature_set])
    keep_cols = [c for c in X.columns if c not in remove_cols]
    return X[keep_cols].copy()


def load_feature_set_data(feature_set: str, drop_sensitive: bool, test_size: float, random_state: int) -> Dict[str, Any]:
    df = load_validated_or_raw_data()
    X_full, y = split_features_and_target(df, drop_sensitive=drop_sensitive)
    y = y.astype(int)
    X = apply_feature_set(X_full.copy(), feature_set=feature_set)
    labels = sorted(y.unique().tolist())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    removed_features = [c for c in FEATURE_SET_DEFINITIONS[feature_set] if c in X_full.columns]
    return {
        "X_train": X_train.copy(),
        "X_test": X_test.copy(),
        "y_train": y_train.copy(),
        "y_test": y_test.copy(),
        "labels": labels,
        "feature_names": X.columns.tolist(),
        "removed_features": removed_features,
    }


def multiclass_brier(y_true: Iterable[int], y_proba: np.ndarray, labels: List[int]) -> float:
    y_bin = label_binarize(list(y_true), classes=labels)
    return float(np.mean(np.sum((y_proba - y_bin) ** 2, axis=1)))


def expected_calibration_error(y_true: Iterable[int], y_pred: Iterable[int], y_proba: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    confidence = np.max(y_proba, axis=1)
    correct = (y_true == y_pred).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence > low) & (confidence <= high)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask])))
    return float(ece)


def calibration_bins(y_true: Iterable[int], y_pred: Iterable[int], y_proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    confidence = np.max(y_proba, axis=1)
    correct = (y_true == y_pred).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for idx, (low, high) in enumerate(zip(bins[:-1], bins[1:]), start=1):
        mask = (confidence > low) & (confidence <= high)
        rows.append({
            "bin": idx,
            "bin_low": low,
            "bin_high": high,
            "n_samples": int(mask.sum()),
            "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
            "accuracy": float(correct[mask].mean()) if mask.any() else np.nan,
        })
    return pd.DataFrame(rows)


def plot_reliability(bin_df: pd.DataFrame, output_path: Path, title: str) -> None:
    plot_df = bin_df.dropna(subset=["mean_confidence", "accuracy"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.plot(plot_df["mean_confidence"], plot_df["accuracy"], marker="o")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def metric_dict(y_true: Iterable[int], y_pred: Iterable[int], y_proba: Optional[np.ndarray], labels: List[int]) -> Dict[str, Any]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)
    out = {
        "accuracy": safe_float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": safe_float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": safe_float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_f1": safe_float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "macro_precision": safe_float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_precision": safe_float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "macro_recall": safe_float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_recall": safe_float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "ordinal_mae": safe_float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "quadratic_weighted_kappa": safe_float(cohen_kappa_score(y_true_arr, y_pred_arr, weights="quadratic")),
        "severe_error_rate": safe_float(np.mean(np.abs(y_true_arr - y_pred_arr) > 1)),
    }
    if y_proba is not None:
        out["nll_log_loss"] = safe_float(log_loss(y_true_arr, y_proba, labels=labels))
        out["multiclass_brier"] = safe_float(multiclass_brier(y_true_arr, y_proba, labels))
        out["ece_confidence"] = safe_float(expected_calibration_error(y_true_arr, y_pred_arr, y_proba, n_bins=10))
    return out


def artifact_dir(model_name: str, feature_set: str) -> Path:
    return SETTINGS.artifacts_dir / "leakage_safe" / f"{model_name}_{feature_set}"


def report_dir(model_name: str, feature_set: str) -> Path:
    return SETTINGS.reports_dir / "leakage_safe" / f"{model_name}_{feature_set}"


def fit_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, labels: List[int], random_state: int) -> Dict[str, Any]:
    if model_name == "catboost":
        X_train_cb, X_test_cb, feature_names, cat_idx = prepare_catboost_frame(X_train, X_test)
        train_pool = Pool(X_train_cb, y_train, cat_features=cat_idx, feature_names=feature_names)
        test_pool = Pool(X_test_cb, cat_features=cat_idx, feature_names=feature_names)
        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=random_state,
            verbose=False,
        )
        model.fit(train_pool)
        pred = model.predict(test_pool).flatten().astype(int)
        proba = model.predict_proba(test_pool)
        return {
            "model": model,
            "pred": pred,
            "proba": proba,
            "test_object": test_pool,
            "feature_names": feature_names,
            "cat_features": [feature_names[i] for i in cat_idx],
            "X_display": X_test_cb,
        }
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        X_train_lgbm = prepare_lightgbm_frame(X_train)
        X_test_lgbm = prepare_lightgbm_frame(X_test)
        categorical_features = [col for col in X_train_lgbm.columns if str(X_train_lgbm[col].dtype) == "category"]
        model = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=random_state, verbose=-1)
        model.fit(X_train_lgbm, y_train, categorical_feature=categorical_features if categorical_features else "auto")
        pred = model.predict(X_test_lgbm).astype(int)
        proba = model.predict_proba(X_test_lgbm)
        return {
            "model": model,
            "pred": pred,
            "proba": proba,
            "test_object": X_test_lgbm,
            "feature_names": X_train_lgbm.columns.tolist(),
            "cat_features": categorical_features,
            "X_display": X_test_lgbm,
        }
    if model_name == "xgboost":
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor(X_train, scale_numeric=False)),
            ("model", LabelEncodedXGBClassifier(random_state=random_state)),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test).astype(int)
        proba = pipeline.predict_proba(X_test)
        return {
            "model": pipeline,
            "pred": pred,
            "proba": proba,
            "test_object": X_test,
            "feature_names": X_train.columns.tolist(),
            "cat_features": infer_columns(X_train)[1],
            "X_display": X_test.copy(),
        }
    raise ValueError(f"Unsupported model name: {model_name}")


def save_model_artifact(model_name: str, feature_set: str, fit_output: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    out_dir = artifact_dir(model_name, feature_set)
    ensure_dir(out_dir)
    model = fit_output["model"]
    if model_name == "catboost":
        model.save_model(str(out_dir / "model.cbm"))
    else:
        joblib.dump(model, out_dir / "model.joblib")
    save_json({
        **metadata,
        "model_name": model_name,
        "feature_set": feature_set,
        "feature_names": fit_output["feature_names"],
        "categorical_features": fit_output["cat_features"],
        "artifact_dir": str(out_dir),
    }, out_dir / "metadata.json")


def export_diagnostics(model_name: str, feature_set: str, X_test: pd.DataFrame, y_test: pd.Series, pred: np.ndarray, proba: np.ndarray, labels: List[int], metadata: Dict[str, Any]) -> None:
    out_dir = report_dir(model_name, feature_set) / "diagnostics"
    ensure_dir(out_dir)
    pred_df = pd.DataFrame({
        "sample_index": X_test.index,
        "y_true": y_test.values,
        "y_pred": pred,
        "correct": y_test.values == pred,
    })
    for idx, label in enumerate(labels):
        pred_df[f"prob_class_{label}"] = proba[:, idx]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    cm = confusion_matrix(y_test, pred, labels=labels)
    pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels]).to_csv(out_dir / "confusion_matrix.csv")
    pd.DataFrame(classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)).T.to_csv(out_dir / "classification_report.csv")
    metrics = metric_dict(y_test, pred, proba, labels)
    save_json({**metadata, "metrics": metrics}, out_dir / "metrics.json")
    print(f"\n=== DIAGNOSTICS | {model_name} | {feature_set} ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")


def export_calibration(model_name: str, feature_set: str, y_test: pd.Series, pred: np.ndarray, proba: np.ndarray, metadata: Dict[str, Any]) -> None:
    out_dir = report_dir(model_name, feature_set) / "calibration"
    ensure_dir(out_dir)
    bins = calibration_bins(y_test, pred, proba, n_bins=10)
    bins.to_csv(out_dir / "calibration_bins.csv", index=False)
    plot_reliability(bins, out_dir / "reliability_diagram.png", f"Reliability Diagram | {model_name} | {feature_set}")
    save_json({**metadata, "calibration_bins": bins}, out_dir / "calibration_metadata.json")


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
    raise ValueError(f"Unexpected SHAP shape: {arr.shape}")


def normalize_catboost_shap(raw_values: Any, n_samples: int, n_features: int, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(raw_values)
    if arr.ndim == 2 and arr.shape == (n_samples, n_features + 1):
        return arr[:, :n_features][:, np.newaxis, :], arr[:, n_features][:, np.newaxis]
    if arr.ndim == 3:
        if arr.shape == (n_samples, n_classes, n_features + 1):
            return arr[:, :, :n_features], arr[:, :, n_features]
        if arr.shape == (n_classes, n_samples, n_features + 1):
            arr = np.transpose(arr, (1, 0, 2))
            return arr[:, :, :n_features], arr[:, :, n_features]
        if arr.shape == (n_samples, n_features + 1, n_classes):
            arr = np.transpose(arr, (0, 2, 1))
            return arr[:, :, :n_features], arr[:, :, n_features]
    raise ValueError(f"Unexpected CatBoost SHAP shape: {arr.shape}")


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


def plot_beeswarm(X: pd.DataFrame, shap_values: np.ndarray, base_values: Any, feature_names: List[str], labels: List[int], class_label: int, output_path: Path, title: str, top_k: int = 15) -> None:
    import shap
    class_idx = labels.index(class_label)
    plot_data, display_data = display_matrix(X, feature_names)
    base_arr = np.asarray(base_values)
    if base_arr.ndim == 2:
        class_base = base_arr[:, class_idx]
    elif base_arr.ndim == 1 and len(base_arr) == len(labels):
        class_base = np.full(len(X), base_arr[class_idx])
    elif base_arr.ndim == 1 and len(base_arr) == len(X):
        class_base = base_arr
    else:
        class_base = np.full(len(X), float(base_arr.flatten()[0]))
    explanation = shap.Explanation(
        values=shap_values[:, class_idx, :],
        base_values=class_base,
        data=plot_data[feature_names].values,
        feature_names=feature_names,
    )
    explanation.display_data = display_data[feature_names].values
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(explanation, max_display=top_k, show=False, plot_size=(10, 7), color_bar=True)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_local_table(X: pd.DataFrame, shap_values: np.ndarray, feature_names: List[str], labels: List[int], sample_index: int, explained_class: int) -> pd.DataFrame:
    row_position = X.index.tolist().index(sample_index)
    class_idx = labels.index(explained_class)
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


def select_representative_cases(y_test: pd.Series, pred: pd.Series, proba: pd.DataFrame, labels: List[int]) -> List[Tuple[str, int, int, int, float]]:
    cases = []
    for class_label in labels:
        subset = proba[pred == class_label]
        if len(subset):
            idx = int(subset[class_label].idxmax())
            cases.append((f"high_conf_class_{class_label}", idx, int(pred.loc[idx]), int(y_test.loc[idx]), float(proba.loc[idx, class_label])))
    confidence = proba.max(axis=1)
    uncertain_idx = int(confidence.idxmin())
    cases.append(("most_uncertain", uncertain_idx, int(pred.loc[uncertain_idx]), int(y_test.loc[uncertain_idx]), float(confidence.loc[uncertain_idx])))
    wrong = pred[pred != y_test]
    if len(wrong):
        wrong_idx = int(wrong.index[0])
        cases.append(("misclassified_example", wrong_idx, int(pred.loc[wrong_idx]), int(y_test.loc[wrong_idx]), float(proba.loc[wrong_idx].max())))
    return cases


def export_shap(model_name: str, feature_set: str, fit_output: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, labels: List[int], top_k: int, metadata: Dict[str, Any]) -> None:
    out_dir = report_dir(model_name, feature_set) / "shap"
    ensure_dir(out_dir)
    if model_name == "xgboost":
        print("SHAP export for XGBoost is not implemented in this script. Skipping.")
        return
    feature_names = fit_output["feature_names"]
    pred = pd.Series(fit_output["pred"], index=X_test.index)
    proba = pd.DataFrame(fit_output["proba"], columns=labels, index=X_test.index)
    if model_name == "catboost":
        raw_values = fit_output["model"].get_feature_importance(data=fit_output["test_object"], type="ShapValues")
        shap_values, base_values = normalize_catboost_shap(raw_values, len(X_test), len(feature_names), len(labels))
        X_display = fit_output["X_display"][feature_names].copy()
    elif model_name == "lightgbm":
        import shap
        X_display = fit_output["test_object"][feature_names].copy()
        explainer = shap.TreeExplainer(fit_output["model"])
        raw_values = explainer.shap_values(X_display)
        shap_values = normalize_shap_values(raw_values, len(X_display), len(feature_names), len(labels))
        base_values = explainer.expected_value
    else:
        raise ValueError(f"Unsupported model for SHAP: {model_name}")
    for class_label in labels:
        plot_beeswarm(
            X=X_display,
            shap_values=shap_values,
            base_values=base_values,
            feature_names=feature_names,
            labels=labels,
            class_label=class_label,
            output_path=out_dir / f"summary_class_{class_label}.png",
            title=f"{model_name.upper()} Leakage-Safe SHAP | {feature_set} | Class {class_label}",
            top_k=top_k,
        )
    cases = select_representative_cases(y_test, pred, proba, labels)
    case_rows = []
    for case_name, sample_index, predicted_class, true_class, confidence in cases:
        local_df = build_local_table(X_display, shap_values, feature_names, labels, sample_index, predicted_class)
        local_df.to_csv(out_dir / f"{case_name}_local_shap_table.csv", index=False)
        plot_local(local_df, out_dir / f"{case_name}_local_explanation.png", f"{case_name} | pred={predicted_class} true={true_class} prob={confidence:.3f}")
        case_rows.append({"case": case_name, "sample_index": sample_index, "predicted_class": predicted_class, "true_class": true_class, "confidence": confidence})
    pd.DataFrame(case_rows).to_csv(out_dir / "representative_cases.csv", index=False)
    global_importance = pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.mean(np.abs(shap_values), axis=(0, 1))}).sort_values("mean_abs_shap", ascending=False)
    global_importance.to_csv(out_dir / "global_shap_importance.csv", index=False)
    save_json({**metadata, "shap_output_dir": str(out_dir)}, out_dir / "shap_metadata.json")


def run_single(model_name: str, feature_set: str, task: str, drop_sensitive: bool, test_size: float, random_state: int, top_k: int) -> Dict[str, Any]:
    data = load_feature_set_data(feature_set, drop_sensitive, test_size, random_state)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    labels = data["labels"]
    metadata = {
        "model_name": model_name,
        "feature_set": feature_set,
        "drop_sensitive": drop_sensitive,
        "test_size": test_size,
        "random_state": random_state,
        "labels": labels,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "feature_names": X_train.columns.tolist(),
        "removed_features": data["removed_features"],
    }
    fit_output = fit_model(model_name, X_train, y_train, X_test, labels, random_state)
    if task in {"train", "all"}:
        save_model_artifact(model_name, feature_set, fit_output, metadata)
    if task in {"diagnostics", "all"}:
        export_diagnostics(model_name, feature_set, X_test, y_test, fit_output["pred"], fit_output["proba"], labels, metadata)
    if task in {"calibration", "all"}:
        export_calibration(model_name, feature_set, y_test, fit_output["pred"], fit_output["proba"], metadata)
    if task in {"shap", "all"}:
        export_shap(model_name, feature_set, fit_output, X_test, y_test, labels, top_k, metadata)
    return {**metadata, **metric_dict(y_test, fit_output["pred"], fit_output["proba"], labels)}


def resolve_models(model_arg: str) -> List[str]:
    if model_arg == "all":
        return MODEL_CHOICES
    if model_arg not in MODEL_CHOICES:
        raise ValueError(f"Unknown model '{model_arg}'. Allowed: all, {MODEL_CHOICES}")
    return [model_arg]


def run_pipeline(task: str, model_arg: str, feature_set: str, drop_sensitive: bool, test_size: float, random_state: int, top_k: int) -> None:
    rows = []
    for model_name in resolve_models(model_arg):
        print(f"\n=== RUNNING | task={task} | model={model_name} | feature_set={feature_set} ===")
        try:
            rows.append(run_single(model_name, feature_set, task, drop_sensitive, test_size, random_state, top_k))
        except Exception as exc:
            print(f"FAILED | model={model_name} | feature_set={feature_set} | error={exc}")
            rows.append({"model_name": model_name, "feature_set": feature_set, "error": str(exc)})
    summary_dir = SETTINGS.reports_dir / "leakage_safe" / "summary"
    ensure_dir(summary_dir)
    output_path = summary_dir / f"{task}_{feature_set}_{model_arg}_summary.csv"
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print("\n=== LEAKAGE-SAFE PIPELINE COMPLETE ===")
    print(f"Summary saved to: {output_path}")
    display_cols = ["model_name", "feature_set", "n_features", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "quadratic_weighted_kappa", "nll_log_loss", "multiclass_brier", "ece_confidence", "error"]
    existing_cols = [c for c in display_cols if c in df.columns]
    if existing_cols:
        print(df[existing_cols].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe model pipeline.")
    parser.add_argument("--task", type=str, default="all", choices=["train", "diagnostics", "calibration", "shap", "all"])
    parser.add_argument("--model", type=str, default="all", choices=["all", *MODEL_CHOICES])
    parser.add_argument("--feature-set", type=str, default="no_salary_hike_no_attrition", choices=sorted(FEATURE_SET_DEFINITIONS.keys()))
    parser.add_argument("--drop-sensitive", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        task=args.task,
        model_arg=args.model,
        feature_set=args.feature_set,
        drop_sensitive=args.drop_sensitive,
        test_size=args.test_size,
        random_state=args.random_state,
        top_k=args.top_k,
    )
