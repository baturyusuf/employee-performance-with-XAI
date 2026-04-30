from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize

from src.data.preprocess import load_validated_or_raw_data, run_preprocessing, split_features_and_target
from src.explainability.counterfactuals import (
    DEFAULT_ACTIONABLE_FEATURES,
    generate_counterfactual_candidates,
    predict_single_sample,
)
from src.models.train_catboost import prepare_catboost_inputs
from src.utils.config import SETTINGS


POST_OUTCOME_RISK_FEATURES = ["EmpLastSalaryHikePercent", "Attrition"]

ACTIONABILITY_SETS = {
    "employee_only": ["TrainingTimesLastYear", "EmpJobInvolvement"],
    "employee_manager": [
        "TrainingTimesLastYear",
        "EmpJobInvolvement",
        "EmpWorkLifeBalance",
        "EmpEnvironmentSatisfaction",
    ],
    "no_salary": [
        "TrainingTimesLastYear",
        "EmpJobInvolvement",
        "EmpWorkLifeBalance",
        "EmpEnvironmentSatisfaction",
    ],
    "full_default": list(DEFAULT_ACTIONABLE_FEATURES),
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
    categorical = []
    numeric = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric.append(col)
        else:
            categorical.append(col)

    return numeric, categorical


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    numeric, categorical = infer_columns(X)
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    transformers = []
    if numeric:
        transformers.append(("num", numeric_transformer, numeric))
    if categorical:
        transformers.append(("cat", one_hot_encoder(), categorical))

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
        pred = self.model_.predict(X).astype(int)
        return self.label_encoder_.inverse_transform(pred)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def metric_dict(y_true: Iterable[int], y_pred: Iterable[int], y_proba: Optional[np.ndarray], labels: List[int]) -> Dict[str, Any]:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)

    out = {
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": safe_float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": safe_float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": safe_float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": safe_float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_recall": safe_float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_recall": safe_float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "ordinal_mae": safe_float(mean_absolute_error(y_true, y_pred)),
        "quadratic_weighted_kappa": safe_float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "severe_error_rate": safe_float(np.mean(np.abs(y_true - y_pred) > 1)),
    }

    if y_proba is not None:
        out["nll_log_loss"] = safe_float(log_loss(y_true, y_proba, labels=labels))
        out["multiclass_brier"] = safe_float(multiclass_brier(y_true, y_proba, labels))

    return out


def summarize_folds(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    excluded = set(group_cols + ["fold"])
    metric_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        row["n_folds"] = int(len(group))
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def get_dataset(drop_sensitive: bool = True):
    df = load_validated_or_raw_data()
    X, y = split_features_and_target(df, drop_sensitive=drop_sensitive)
    return X.copy(), y.astype(int).copy()


def get_train_test(drop_sensitive: bool = True, random_state: int = 42, test_size: float = 0.20):
    df = load_validated_or_raw_data()
    X, y = split_features_and_target(df, drop_sensitive=drop_sensitive)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y.astype(int),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()


def fit_catboost(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, random_state: int, drop_sensitive: bool):
    X_train_cb, X_test_cb, feature_names, cat_idx = prepare_catboost_inputs(
        X_train=X_train,
        X_test=X_test,
        drop_sensitive=drop_sensitive,
    )

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
    return model, test_pool

def fit_catboost_for_current_columns(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    random_state: int,
):
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()

    categorical_cols = []
    for col in X_train_cb.columns:
        if (
            X_train_cb[col].dtype == "object"
            or str(X_train_cb[col].dtype).startswith("category")
            or str(X_train_cb[col].dtype).startswith("string")
        ):
            categorical_cols.append(col)

    for col in categorical_cols:
        X_train_cb[col] = X_train_cb[col].astype("string").fillna("__MISSING__")
        X_test_cb[col] = X_test_cb[col].astype("string").fillna("__MISSING__")

    for col in [c for c in X_train_cb.columns if c not in categorical_cols]:
        X_train_cb[col] = pd.to_numeric(X_train_cb[col], errors="coerce")
        X_test_cb[col] = pd.to_numeric(X_test_cb[col], errors="coerce")

    feature_names = X_train_cb.columns.tolist()
    cat_idx = [feature_names.index(c) for c in categorical_cols]

    train_pool = Pool(
        X_train_cb,
        y_train,
        cat_features=cat_idx,
        feature_names=feature_names,
    )

    test_pool = Pool(
        X_test_cb,
        cat_features=cat_idx,
        feature_names=feature_names,
    )

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
    return model, test_pool

def build_lightgbm_pipeline(X: pd.DataFrame, random_state: int):
    from lightgbm import LGBMClassifier

    return Pipeline([
        ("preprocessor", make_preprocessor(X, scale_numeric=False)),
        ("model", LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=random_state,
            verbose=-1,
        )),
    ])


def build_xgboost_pipeline(X: pd.DataFrame, random_state: int):
    return Pipeline([
        ("preprocessor", make_preprocessor(X, scale_numeric=False)),
        ("model", LabelEncodedXGBClassifier(random_state=random_state)),
    ])


def run_prediction_exports(random_state: int = 42, drop_sensitive: bool = True) -> None:
    out_dir = SETTINGS.reports_dir / "diagnostics" / "predictions"
    ensure_dir(out_dir)

    X_train, X_test, y_train, y_test = get_train_test(drop_sensitive=drop_sensitive, random_state=random_state)
    labels = sorted(y_train.unique().tolist())

    model_outputs = {}

    cat_model, cat_pool = fit_catboost(X_train, y_train, X_test, random_state=random_state, drop_sensitive=drop_sensitive)
    cat_pred = cat_model.predict(cat_pool).flatten().astype(int)
    cat_proba = cat_model.predict_proba(cat_pool)
    model_outputs["catboost"] = (cat_pred, cat_proba)

    for name, builder in [
        ("lightgbm", build_lightgbm_pipeline),
        ("xgboost", build_xgboost_pipeline),
    ]:
        try:
            pipe = builder(X_train, random_state)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test).astype(int)
            proba = pipe.predict_proba(X_test)
            model_outputs[name] = (pred, proba)
        except Exception as exc:
            print(f"{name} skipped: {exc}")

    for name, (pred, proba) in model_outputs.items():
        pred_df = pd.DataFrame({
            "sample_index": X_test.index,
            "y_true": y_test.values,
            "y_pred": pred,
            "correct": y_test.values == pred,
        })
        for i, label in enumerate(labels):
            pred_df[f"prob_class_{label}"] = proba[:, i]

        pred_df.to_csv(out_dir / f"{name}_test_predictions.csv", index=False)

        cm = confusion_matrix(y_test, pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
        cm_df.to_csv(out_dir / f"{name}_confusion_matrix.csv")

        report = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(out_dir / f"{name}_classification_report.csv")

        metrics = metric_dict(y_test, pred, proba, labels)
        save_json(metrics, out_dir / f"{name}_metrics.json")

    print(f"\nPrediction exports saved to: {out_dir}")


def run_leakage_ablation(n_splits: int = 10, random_state: int = 42, drop_sensitive: bool = True) -> None:
    out_dir = SETTINGS.reports_dir / "robustness" / "leakage_ablation"
    ensure_dir(out_dir)

    X, y = get_dataset(drop_sensitive=drop_sensitive)
    labels = sorted(y.unique().tolist())

    feature_sets = {
        "all_features": X.columns.tolist(),

        "no_salary_hike": [
            c for c in X.columns
            if c != "EmpLastSalaryHikePercent"
        ],

        "no_salary_hike_no_attrition": [
            c for c in X.columns
            if c not in ["EmpLastSalaryHikePercent", "Attrition"]
        ],

        "pre_evaluation_features_only": [
            c for c in X.columns
            if c not in ["EmpLastSalaryHikePercent", "Attrition"]
        ],

        "no_department": [
            c for c in X.columns
            if c != "EmpDepartment"
        ],

        "no_salary_hike_no_department": [
            c for c in X.columns
            if c not in ["EmpLastSalaryHikePercent", "EmpDepartment"]
        ],

        "no_salary_hike_no_attrition_no_department": [
            c for c in X.columns
            if c not in ["EmpLastSalaryHikePercent", "Attrition", "EmpDepartment"]
        ],
    }

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []

    for feature_set, features in feature_sets.items():
        features = [c for c in features if c in X.columns]

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X[features], y), start=1):
            X_train = X.iloc[train_idx][features].copy()
            X_test = X.iloc[test_idx][features].copy()
            y_train = y.iloc[train_idx].copy()
            y_test = y.iloc[test_idx].copy()

            for model_name in ["catboost", "lightgbm", "xgboost"]:
                try:
                    if model_name == "catboost":
                        model, pool = fit_catboost_for_current_columns(
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            random_state=random_state,
                        )
                        pred = model.predict(pool).flatten().astype(int)
                        proba = model.predict_proba(pool)
                    elif model_name == "lightgbm":
                        model = build_lightgbm_pipeline(X_train, random_state)
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test).astype(int)
                        proba = model.predict_proba(X_test)
                    else:
                        model = build_xgboost_pipeline(X_train, random_state)
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test).astype(int)
                        proba = model.predict_proba(X_test)

                    rows.append({
                        "feature_set": feature_set,
                        "model": model_name,
                        "fold": fold,
                        "n_features": len(features),
                        **metric_dict(y_test, pred, proba, labels),
                    })
                    print(f"[leakage] {feature_set}/{model_name}/fold={fold}")
                except Exception as exc:
                    rows.append({
                        "feature_set": feature_set,
                        "model": model_name,
                        "fold": fold,
                        "n_features": len(features),
                        "error": str(exc),
                    })
                    print(f"[leakage] {feature_set}/{model_name}/fold={fold} failed: {exc}")

    fold_df = pd.DataFrame(rows)
    summary_df = summarize_folds(fold_df, ["feature_set", "model", "n_features"])

    fold_df.to_csv(out_dir / "leakage_ablation_fold_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "leakage_ablation_summary.csv", index=False)

    print("\n=== LEAKAGE ABLATION COMPLETE ===")
    print(summary_df.sort_values("macro_f1_mean", ascending=False).to_string(index=False))


def multiclass_brier(y_true: Iterable[int], y_proba: np.ndarray, labels: List[int]) -> float:
    y_true_bin = label_binarize(list(y_true), classes=labels)
    return float(np.mean(np.sum((y_proba - y_true_bin) ** 2, axis=1)))


def expected_calibration_error(y_true: Iterable[int], y_pred: Iterable[int], y_proba: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    confidence = np.max(y_proba, axis=1)
    correctness = (y_true == y_pred).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if np.any(mask):
            acc = np.mean(correctness[mask])
            conf = np.mean(confidence[mask])
            ece += np.mean(mask) * abs(acc - conf)

    return float(ece)


def calibration_bins(y_true: Iterable[int], y_pred: Iterable[int], y_proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    confidence = np.max(y_proba, axis=1)
    correctness = (y_true == y_pred).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:]), start=1):
        mask = (confidence > lo) & (confidence <= hi)
        rows.append({
            "bin": idx,
            "bin_low": lo,
            "bin_high": hi,
            "n_samples": int(mask.sum()),
            "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
            "accuracy": float(correctness[mask].mean()) if mask.any() else np.nan,
        })

    return pd.DataFrame(rows)


def plot_reliability(bin_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_df = bin_df.dropna(subset=["mean_confidence", "accuracy"])
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.plot(plot_df["mean_confidence"], plot_df["accuracy"], marker="o")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_calibration(random_state: int = 42, drop_sensitive: bool = True, n_bins: int = 10) -> None:
    out_dir = SETTINGS.reports_dir / "robustness" / "calibration"
    ensure_dir(out_dir)

    X_train, X_test, y_train, y_test = get_train_test(drop_sensitive=drop_sensitive, random_state=random_state)
    labels = sorted(y_train.unique().tolist())

    models = {}

    cat_model, cat_pool = fit_catboost(X_train, y_train, X_test, random_state, drop_sensitive)
    cat_pred = cat_model.predict(cat_pool).flatten().astype(int)
    cat_proba = cat_model.predict_proba(cat_pool)
    models["catboost"] = (cat_pred, cat_proba)

    for name, builder in [("lightgbm", build_lightgbm_pipeline), ("xgboost", build_xgboost_pipeline)]:
        try:
            model = builder(X_train, random_state)
            model.fit(X_train, y_train)
            models[name] = (model.predict(X_test).astype(int), model.predict_proba(X_test))
        except Exception as exc:
            print(f"{name} skipped: {exc}")

    metric_rows = []
    for name, (pred, proba) in models.items():
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
            "nll_log_loss": log_loss(y_test, proba, labels=labels),
            "multiclass_brier": multiclass_brier(y_test, proba, labels),
            "ece_confidence": expected_calibration_error(y_test, pred, proba, n_bins=n_bins),
        }
        metric_rows.append(metrics)

        bins = calibration_bins(y_test, pred, proba, n_bins=n_bins)
        bins.to_csv(out_dir / f"{name}_calibration_bins.csv", index=False)
        plot_reliability(bins, out_dir / f"{name}_reliability.png", f"Reliability Diagram | {name}")

    pd.DataFrame(metric_rows).to_csv(out_dir / "calibration_metrics.csv", index=False)

    print("\n=== CALIBRATION ANALYSIS COMPLETE ===")
    print(pd.DataFrame(metric_rows).to_string(index=False))


def normalize_catboost_shap(raw_shap: Any, n_samples: int, n_features: int, n_classes: int):
    arr = np.asarray(raw_shap)

    if arr.ndim == 2 and arr.shape == (n_samples, n_features + 1):
        return arr[:, :n_features][:, np.newaxis, :]

    if arr.ndim == 3:
        if arr.shape == (n_samples, n_classes, n_features + 1):
            return arr[:, :, :n_features]
        if arr.shape == (n_classes, n_samples, n_features + 1):
            arr = np.transpose(arr, (1, 0, 2))
            return arr[:, :, :n_features]
        if arr.shape == (n_samples, n_features + 1, n_classes):
            arr = np.transpose(arr, (0, 2, 1))
            return arr[:, :, :n_features]

    raise ValueError(f"Unexpected SHAP shape: {arr.shape}")


def mean_abs_shap_ranking(model: CatBoostClassifier, pool: Pool, feature_names: List[str], class_count: int) -> pd.DataFrame:
    raw = model.get_feature_importance(data=pool, type="ShapValues")
    shap_values = normalize_catboost_shap(
        raw_shap=raw,
        n_samples=pool.num_row(),
        n_features=len(feature_names),
        n_classes=class_count,
    )

    overall = np.mean(np.abs(shap_values), axis=(0, 1))
    out = pd.DataFrame({"feature": feature_names, "mean_abs_shap": overall})
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a = set(a)
    b = set(b)
    return float(len(a & b) / len(a | b)) if len(a | b) else np.nan


def run_shap_stability(n_splits: int = 5, random_state: int = 42, drop_sensitive: bool = True) -> None:
    out_dir = SETTINGS.reports_dir / "robustness" / "shap_stability"
    ensure_dir(out_dir)

    X, y = get_dataset(drop_sensitive=drop_sensitive)
    labels = sorted(y.unique().tolist())

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    ranking_frames = []
    topk_rows = []
    corr_rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        X_train_cb, X_test_cb, feature_names, cat_idx = prepare_catboost_inputs(
            X_train=X_train,
            X_test=X_test,
            drop_sensitive=drop_sensitive,
        )

        train_pool = Pool(X_train_cb, y_train, cat_features=cat_idx, feature_names=feature_names)
        test_pool = Pool(X_test_cb, y_test, cat_features=cat_idx, feature_names=feature_names)

        model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=random_state,
            verbose=False,
        )
        model.fit(train_pool)

        ranking = mean_abs_shap_ranking(model, test_pool, feature_names, class_count=len(labels))
        ranking["fold"] = fold
        ranking_frames.append(ranking)

    all_rankings = pd.concat(ranking_frames, ignore_index=True)
    all_rankings.to_csv(out_dir / "fold_feature_rankings.csv", index=False)

    folds = sorted(all_rankings["fold"].unique().tolist())

    for k in [5, 10, 15]:
        for i, left in enumerate(folds):
            for right in folds[i + 1:]:
                top_left = all_rankings[all_rankings["fold"] == left].sort_values("rank").head(k)["feature"]
                top_right = all_rankings[all_rankings["fold"] == right].sort_values("rank").head(k)["feature"]
                topk_rows.append({"top_k": k, "fold_a": left, "fold_b": right, "jaccard": jaccard(top_left, top_right)})

    pivot = all_rankings.pivot(index="feature", columns="fold", values="rank")

    for i, left in enumerate(folds):
        for right in folds[i + 1:]:
            corr = pivot[left].corr(pivot[right], method="spearman")
            corr_rows.append({"fold_a": left, "fold_b": right, "spearman_rank_correlation": float(corr)})

    topk_df = pd.DataFrame(topk_rows)
    corr_df = pd.DataFrame(corr_rows)

    topk_df.to_csv(out_dir / "jaccard_topk.csv", index=False)
    corr_df.to_csv(out_dir / "spearman_rank_correlation.csv", index=False)

    summary = {
        "mean_jaccard_top5": float(topk_df[topk_df["top_k"] == 5]["jaccard"].mean()),
        "mean_jaccard_top10": float(topk_df[topk_df["top_k"] == 10]["jaccard"].mean()),
        "mean_jaccard_top15": float(topk_df[topk_df["top_k"] == 15]["jaccard"].mean()),
        "mean_spearman": float(corr_df["spearman_rank_correlation"].mean()),
    }
    save_json(summary, out_dir / "shap_stability_summary.json")

    print("\n=== SHAP STABILITY COMPLETE ===")
    print(summary)


def candidate_probability(candidate: Dict[str, Any]) -> float:
    return float(candidate.get("desired_class_probability", candidate.get("prob", np.nan)))


def changed_features(candidate: Dict[str, Any]) -> List[str]:
    if "changes" in candidate:
        return [c.get("feature") for c in candidate["changes"] if isinstance(c, dict) and "feature" in c]
    if "changed_features" in candidate:
        return [x.strip() for x in str(candidate["changed_features"]).split(",") if x.strip()]
    return []


def run_counterfactual_actionability(max_samples: int = 100, max_features_changed: int = 3, top_k: int = 5) -> None:
    out_dir = SETTINGS.reports_dir / "robustness" / "counterfactual_actionability"
    ensure_dir(out_dir)

    model_path = SETTINGS.artifacts_dir / "catboost" / "catboost_model.cbm"
    summary_path = SETTINGS.artifacts_dir / "catboost" / "run_summary.json"

    if not model_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Train CatBoost before running counterfactual actionability.")

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    prep = run_preprocessing(
        test_size=summary.get("test_size", 0.20),
        random_state=summary.get("random_state", 42),
        drop_sensitive=summary.get("drop_sensitive", False),
    )

    X_train = prep["X_train_raw"].copy()
    X_test = prep["X_test_raw"].copy()
    y_train = prep["y_train"].astype(int).copy()
    y_test = prep["y_test"].astype(int).copy()
    drop_sensitive = summary.get("drop_sensitive", False)
    classes = sorted([int(c) for c in model.classes_])

    sample_rows = []
    candidate_rows = []

    eligible = []
    for idx in X_test.index:
        pred_info = predict_single_sample(model, X_test.loc[[idx]], drop_sensitive)
        if int(pred_info["predicted_class"]) < max(classes):
            eligible.append(idx)

    for mode, feature_list in ACTIONABILITY_SETS.items():
        features = [f for f in feature_list if f in X_test.columns]

        for idx in eligible[:max_samples]:
            sample_df = X_test.loc[[idx]].copy()
            pred_info = predict_single_sample(model, sample_df, drop_sensitive)
            predicted_class = int(pred_info["predicted_class"])
            desired_class = min([c for c in classes if c > predicted_class])
            base_prob = float(pred_info["probabilities"].get(desired_class, 0.0))

            candidates = generate_counterfactual_candidates(
                model=model,
                original_sample_df=sample_df,
                original_prediction=predicted_class,
                desired_class=desired_class,
                X_train_raw=X_train,
                y_train=y_train,
                drop_sensitive=drop_sensitive,
                actionable_features=features,
                max_features_changed=max_features_changed,
                top_k=top_k,
            )

            valid = len(candidates) > 0
            best = candidates[0] if valid else None

            sample_rows.append({
                "actionability_mode": mode,
                "sample_index": int(idx),
                "true_class": int(y_test.loc[idx]),
                "predicted_class": predicted_class,
                "desired_class": desired_class,
                "base_desired_probability": base_prob,
                "valid_counterfactual_found": bool(valid),
                "best_desired_probability": candidate_probability(best) if valid else np.nan,
                "probability_gain": candidate_probability(best) - base_prob if valid else np.nan,
                "best_cost": float(best.get("cost", np.nan)) if valid else np.nan,
                "best_num_changes": int(best.get("num_changes", np.nan)) if valid else np.nan,
                "best_changed_features": ", ".join(changed_features(best)) if valid else "",
            })

            for rank, candidate in enumerate(candidates, start=1):
                candidate_rows.append({
                    "actionability_mode": mode,
                    "sample_index": int(idx),
                    "rank": rank,
                    "desired_class": desired_class,
                    "probability": candidate_probability(candidate),
                    "cost": float(candidate.get("cost", np.nan)),
                    "num_changes": int(candidate.get("num_changes", np.nan)),
                    "changed_features": ", ".join(changed_features(candidate)),
                })

    sample_df = pd.DataFrame(sample_rows)
    candidates_df = pd.DataFrame(candidate_rows)

    summary_df = sample_df.groupby("actionability_mode").agg(
        n_samples=("sample_index", "count"),
        validity_rate=("valid_counterfactual_found", "mean"),
        mean_probability_gain=("probability_gain", "mean"),
        mean_best_cost=("best_cost", "mean"),
        mean_best_num_changes=("best_num_changes", "mean"),
    ).reset_index()

    sample_df.to_csv(out_dir / "actionability_summary_by_sample.csv", index=False)
    candidates_df.to_csv(out_dir / "actionability_candidates.csv", index=False)
    summary_df.to_csv(out_dir / "actionability_summary.csv", index=False)

    print("\n=== COUNTERFACTUAL ACTIONABILITY COMPLETE ===")
    print(summary_df.to_string(index=False))


def binary_rates(y_true_bin: Iterable[int], y_pred_bin: Iterable[int]) -> Dict[str, float | int]:
    yt = np.asarray(list(y_true_bin), dtype=int)
    yp = np.asarray(list(y_pred_bin), dtype=int)

    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    return {
        "positive_prediction_rate": (tp + fp) / max(len(yt), 1),
        "tpr_equal_opportunity": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
        "support_positive": int(np.sum(yt == 1)),
        "support_negative": int(np.sum(yt == 0)),
    }


def full_test_with_sensitive(random_state: int = 42, test_size: float = 0.20):
    df = load_validated_or_raw_data()
    X, y = split_features_and_target(df, drop_sensitive=False)

    _, X_test, _, y_test = train_test_split(
        X,
        y.astype(int),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_test.copy(), y_test.copy()


def run_fairness_support(attributes: List[str], random_state: int = 42, min_supports: List[int] = [10, 20, 30]) -> None:
    out_dir = SETTINGS.reports_dir / "robustness" / "fairness_support"
    ensure_dir(out_dir)

    model_path = SETTINGS.artifacts_dir / "catboost" / "catboost_model.cbm"
    summary_path = SETTINGS.artifacts_dir / "catboost" / "run_summary.json"

    if not model_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Train CatBoost first.")

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    X_full_test, y_full_test = full_test_with_sensitive(
        random_state=summary.get("random_state", random_state),
        test_size=summary.get("test_size", 0.20),
    )

    prep = run_preprocessing(
        test_size=summary.get("test_size", 0.20),
        random_state=summary.get("random_state", random_state),
        drop_sensitive=summary.get("drop_sensitive", False),
    )
    X_model_test = prep["X_test_raw"].copy()
    y_model_test = prep["y_test"].astype(int).copy()

    if not X_model_test.index.equals(X_full_test.index):
        raise ValueError("Test split mismatch between full and model feature sets.")

    X_cb, _, feature_names, cat_idx = prepare_catboost_inputs(
        X_train=X_model_test,
        X_test=X_model_test.copy(),
        drop_sensitive=summary.get("drop_sensitive", False),
    )
    pool = Pool(X_cb, y_model_test, cat_features=cat_idx, feature_names=feature_names)
    y_pred = pd.Series(model.predict(pool).flatten().astype(int), index=X_model_test.index)
    proba = pd.DataFrame(model.predict_proba(pool), columns=[int(c) for c in model.classes_], index=X_model_test.index)

    labels = sorted([int(c) for c in model.classes_])
    group_rows = []
    disparity_rows = []
    warning_rows = []

    for attr in attributes:
        if attr not in X_full_test.columns:
            continue

        groups = X_full_test[attr].astype("string").fillna("__MISSING__")

        counts = groups.value_counts()
        for group_value, count in counts.items():
            if count < max(min_supports):
                warning_rows.append({
                    "attribute": attr,
                    "group_value": str(group_value),
                    "n_samples": int(count),
                    "warning": f"n_samples < {max(min_supports)}",
                })

        for cls in labels:
            for group_value in sorted(groups.unique().tolist()):
                mask = groups == group_value
                rates = binary_rates(
                    (y_model_test.loc[mask] == cls).astype(int),
                    (y_pred.loc[mask] == cls).astype(int),
                )
                group_rows.append({
                    "attribute": attr,
                    "group_value": str(group_value),
                    "class_label": cls,
                    "n_samples": int(mask.sum()),
                    "mean_predicted_probability": float(proba.loc[mask, cls].mean()),
                    **rates,
                })

    group_df = pd.DataFrame(group_rows)
    warning_df = pd.DataFrame(warning_rows)

    for min_support in min_supports:
        filtered = group_df[group_df["n_samples"] >= min_support].copy()
        for attr in sorted(filtered["attribute"].unique()):
            for cls in labels:
                subset = filtered[(filtered["attribute"] == attr) & (filtered["class_label"] == cls)]
                if len(subset) < 2:
                    continue

                for metric in [
                    "positive_prediction_rate",
                    "tpr_equal_opportunity",
                    "fpr",
                    "precision",
                    "mean_predicted_probability",
                ]:
                    vals = pd.to_numeric(subset[metric], errors="coerce").dropna()
                    if len(vals):
                        disparity_rows.append({
                            "min_support": min_support,
                            "attribute": attr,
                            "class_label": cls,
                            "metric": metric,
                            "n_groups_included": int(len(vals)),
                            "max_gap": float(vals.max() - vals.min()),
                            "min": float(vals.min()),
                            "max": float(vals.max()),
                        })

    disparity_df = pd.DataFrame(disparity_rows)

    group_df.to_csv(out_dir / "fairness_group_metrics.csv", index=False)
    warning_df.to_csv(out_dir / "small_group_warnings.csv", index=False)
    disparity_df.to_csv(out_dir / "fairness_support_summary.csv", index=False)

    print("\n=== SUPPORT-FILTERED FAIRNESS COMPLETE ===")
    print(disparity_df.head(30).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness checks for employee performance XAI project.")
    parser.add_argument("--task", type=str, default="leakage", choices=[
        "predictions",
        "leakage",
        "calibration",
        "shap-stability",
        "cf-actionability",
        "fairness-support",
        "all",
    ])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--drop-sensitive", action="store_true")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-features-changed", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--attributes",
        type=str,
        default="Gender,MaritalStatus,EmpDepartment,EducationBackground,BusinessTravelFrequency",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    attrs = [x.strip() for x in args.attributes.split(",") if x.strip()]

    if args.task == "predictions":
        run_prediction_exports(random_state=args.random_state, drop_sensitive=args.drop_sensitive)
    elif args.task == "leakage":
        run_leakage_ablation(
            n_splits=args.n_splits,
            random_state=args.random_state,
            drop_sensitive=args.drop_sensitive,
        )
    elif args.task == "calibration":
        run_calibration(
            random_state=args.random_state,
            drop_sensitive=args.drop_sensitive,
            n_bins=args.n_bins,
        )
    elif args.task == "shap-stability":
        run_shap_stability(
            n_splits=min(args.n_splits, 5),
            random_state=args.random_state,
            drop_sensitive=args.drop_sensitive,
        )
    elif args.task == "cf-actionability":
        run_counterfactual_actionability(
            max_samples=args.max_samples,
            max_features_changed=args.max_features_changed,
            top_k=args.top_k,
        )
    elif args.task == "fairness-support":
        run_fairness_support(
            attributes=attrs,
            random_state=args.random_state,
        )
    elif args.task == "all":
        run_prediction_exports(random_state=args.random_state, drop_sensitive=args.drop_sensitive)
        run_leakage_ablation(
            n_splits=args.n_splits,
            random_state=args.random_state,
            drop_sensitive=args.drop_sensitive,
        )
        run_calibration(
            random_state=args.random_state,
            drop_sensitive=args.drop_sensitive,
            n_bins=args.n_bins,
        )
        run_shap_stability(
            n_splits=min(args.n_splits, 5),
            random_state=args.random_state,
            drop_sensitive=args.drop_sensitive,
        )
        run_counterfactual_actionability(
            max_samples=args.max_samples,
            max_features_changed=args.max_features_changed,
            top_k=args.top_k,
        )
        run_fairness_support(
            attributes=attrs,
            random_state=args.random_state,
        )
