from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, label_binarize

from src.data.preprocess import load_validated_or_raw_data, split_features_and_target
from src.features.feature_sets import (
    apply_feature_set as apply_config_feature_set,
    build_feature_columns,
    get_feature_sets,
    leakage_safe_feature_sets,
)
from src.models.evaluate import classification_metrics
from src.utils.config import SETTINGS
from src.utils.experiment_registry import (
    append_registry_row,
    collect_package_versions,
    get_git_commit,
    utc_now_iso,
)

warnings.filterwarnings("ignore")


FEATURE_SET_CONFIG = get_feature_sets()
FEATURE_SET_DEFINITIONS = {
    name: definition.get("drop", [])
    for name, definition in FEATURE_SET_CONFIG.items()
}
DEFAULT_FEATURE_SETS = leakage_safe_feature_sets()
DEFAULT_MODELS = ["catboost", "lightgbm", "xgboost"]


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
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

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


def prepare_lightgbm_frame(X: pd.DataFrame) -> pd.DataFrame:
    X_out = X.copy()

    for col in X_out.columns:
        if pd.api.types.is_numeric_dtype(X_out[col]):
            X_out[col] = pd.to_numeric(X_out[col], errors="coerce")
        else:
            X_out[col] = X_out[col].astype("category")

    return X_out


def prepare_catboost_frame(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[int]]:
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()

    categorical_cols = []
    for col in X_train_cb.columns:
        dtype_str = str(X_train_cb[col].dtype)
        if (
            dtype_str == "object"
            or dtype_str.startswith("category")
            or dtype_str.startswith("string")
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
        pred_encoded = self.model_.predict(X).astype(int)
        return self.label_encoder_.inverse_transform(pred_encoded)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def apply_feature_set(X: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set not in FEATURE_SET_DEFINITIONS:
        raise ValueError(
            f"Unknown feature set: {feature_set}. "
            f"Allowed values: {sorted(FEATURE_SET_DEFINITIONS)}"
        )

    return apply_config_feature_set(X, feature_set_name=feature_set)


def removed_features_for(all_columns: Iterable[str], feature_set: str) -> List[str]:
    selected = set(build_feature_columns(all_columns, feature_set))
    return [col for col in all_columns if col not in selected]


def parse_csv_arg(value: str, allowed: List[str]) -> List[str]:
    if value == "all":
        return allowed

    parts = [x.strip() for x in value.split(",") if x.strip()]
    invalid = [x for x in parts if x not in allowed]

    if invalid:
        raise ValueError(f"Invalid values: {invalid}. Allowed: {allowed} or 'all'.")

    return parts


def multiclass_brier(y_true: Iterable[int], y_proba: np.ndarray, labels: List[int]) -> float:
    y_bin = label_binarize(list(y_true), classes=labels)
    return float(np.mean(np.sum((y_proba - y_bin) ** 2, axis=1)))


def expected_calibration_error(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    confidence = np.max(y_proba, axis=1)
    correct = (y_true == y_pred).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence > low) & (confidence <= high)
        if np.any(mask):
            bin_acc = float(np.mean(correct[mask]))
            bin_conf = float(np.mean(confidence[mask]))
            ece += float(np.mean(mask)) * abs(bin_acc - bin_conf)

    return float(ece)


def severe_error_rate(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)
    return float(np.mean(np.abs(yt - yp) > 1))


def adjacent_accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)
    return float(np.mean(np.abs(yt - yp) <= 1))


def metric_dict(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_proba: Optional[np.ndarray],
    labels: List[int],
) -> Dict[str, Any]:
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
        "adjacent_accuracy": safe_float(adjacent_accuracy(y_true_arr, y_pred_arr)),
        "severe_error_rate": safe_float(severe_error_rate(y_true_arr, y_pred_arr)),
    }

    if y_proba is not None:
        out["nll_log_loss"] = safe_float(log_loss(y_true_arr, y_proba, labels=labels))
        out["multiclass_brier"] = safe_float(multiclass_brier(y_true_arr, y_proba, labels))
        out["ece_confidence"] = safe_float(
            expected_calibration_error(y_true_arr, y_pred_arr, y_proba, n_bins=10)
        )

    return out


def fit_predict_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X_train_cb, X_test_cb, feature_names, cat_idx = prepare_catboost_frame(X_train, X_test)

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
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(train_pool)

    pred = model.predict(test_pool).flatten().astype(int)
    proba = model.predict_proba(test_pool)
    return pred, proba


def fit_predict_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    from lightgbm import LGBMClassifier

    X_train_lgbm = prepare_lightgbm_frame(X_train)
    X_test_lgbm = prepare_lightgbm_frame(X_test)
    categorical_features = [
        col for col in X_train_lgbm.columns
        if str(X_train_lgbm[col].dtype) == "category"
    ]

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(
        X_train_lgbm,
        y_train,
        categorical_feature=categorical_features if categorical_features else "auto",
    )

    pred = model.predict(X_test_lgbm).astype(int)
    proba = model.predict_proba(X_test_lgbm)
    return pred, proba


def fit_predict_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pipeline = Pipeline([
        ("preprocessor", make_preprocessor(X_train)),
        ("model", LabelEncodedXGBClassifier(random_state=random_state)),
    ])
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test).astype(int)
    proba = pipeline.predict_proba(X_test)
    return pred, proba


def fit_predict_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if model_name == "catboost":
        return fit_predict_catboost(X_train, y_train, X_test, random_state)
    if model_name == "lightgbm":
        return fit_predict_lightgbm(X_train, y_train, X_test, random_state)
    if model_name == "xgboost":
        return fit_predict_xgboost(X_train, y_train, X_test, random_state)

    raise ValueError(f"Unsupported model: {model_name}")


def summarize_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["feature_set", "model"]
    excluded = set(group_cols + ["fold"])

    metric_cols = [
        col for col in fold_df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(fold_df[col])
    ]

    rows = []
    for keys, group in fold_df.groupby(group_cols, dropna=False):
        row = {
            "feature_set": keys[0],
            "model": keys[1],
            "n_folds": int(len(group)),
        }

        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float)
    order = p.sort_values().index.tolist()
    m = len(order)
    adjusted = pd.Series(index=p.index, dtype=float)
    running = 0.0

    for rank, idx in enumerate(order, start=1):
        value = min((m - rank + 1) * p.loc[idx], 1.0)
        running = max(running, value)
        adjusted.loc[idx] = running

    return adjusted


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan

    means = [
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ]
    alpha = (1 - ci) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def bootstrap_ci_table(fold_df: pd.DataFrame, metric: str, seed: int = 42) -> pd.DataFrame:
    rows = []

    for (feature_set, model), group in fold_df.groupby(["feature_set", "model"]):
        values = pd.to_numeric(group[metric], errors="coerce").dropna().values
        low, high = bootstrap_ci(values, seed=seed)

        rows.append({
            "feature_set": feature_set,
            "model": model,
            "metric": metric,
            "mean": float(np.mean(values)) if len(values) else np.nan,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "ci_low": low,
            "ci_high": high,
        })

    return pd.DataFrame(rows).sort_values(["feature_set", "mean"], ascending=[True, False])


def pairwise_tests(fold_df: pd.DataFrame, metric: str, alpha: float = 0.05) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    try:
        from scipy.stats import friedmanchisquare, wilcoxon
    except Exception as exc:
        return pd.DataFrame([{"error": f"scipy unavailable: {exc}"}]), {
            "test": "unavailable",
            "error": str(exc),
        }

    rows = []
    omnibus_rows = []

    for feature_set, subset in fold_df.groupby("feature_set"):
        pivot = subset.pivot_table(
            index="fold",
            columns="model",
            values=metric,
            aggfunc="mean",
        ).dropna(axis=1)

        models = pivot.columns.tolist()

        if len(models) >= 3:
            try:
                stat, p_value = friedmanchisquare(*[pivot[m].values for m in models])
                omnibus_rows.append({
                    "feature_set": feature_set,
                    "test": "friedman",
                    "statistic": float(stat),
                    "p_value": float(p_value),
                })
            except Exception as exc:
                omnibus_rows.append({
                    "feature_set": feature_set,
                    "test": "friedman",
                    "error": str(exc),
                })

        for i, model_a in enumerate(models):
            for model_b in models[i + 1:]:
                try:
                    stat, p_value = wilcoxon(
                        pivot[model_a].values,
                        pivot[model_b].values,
                        zero_method="wilcox",
                        alternative="two-sided",
                    )
                    rows.append({
                        "feature_set": feature_set,
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric": metric,
                        "mean_a": float(pivot[model_a].mean()),
                        "mean_b": float(pivot[model_b].mean()),
                        "mean_diff_a_minus_b": float(pivot[model_a].mean() - pivot[model_b].mean()),
                        "wilcoxon_stat": float(stat),
                        "p_value": float(p_value),
                    })
                except Exception as exc:
                    rows.append({
                        "feature_set": feature_set,
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric": metric,
                        "error": str(exc),
                    })

    pair_df = pd.DataFrame(rows)

    if "p_value" in pair_df.columns:
        for feature_set, idx in pair_df.groupby("feature_set").groups.items():
            idx = list(idx)
            valid_idx = pair_df.loc[idx][pair_df.loc[idx, "p_value"].notna()].index
            if len(valid_idx):
                pair_df.loc[valid_idx, "p_holm"] = holm_adjust(pair_df.loc[valid_idx, "p_value"])
                pair_df.loc[valid_idx, "significant_holm"] = pair_df.loc[valid_idx, "p_holm"] < alpha

    omnibus = {
        "metric": metric,
        "alpha": alpha,
        "omnibus": omnibus_rows,
    }

    return pair_df, omnibus


def run_cv(
    feature_sets: List[str],
    models: List[str],
    n_splits: int,
    random_state: int,
    drop_sensitive: bool,
    alpha: float,
    output_dir: Optional[Path] = None,
    write_registry: bool = True,
) -> None:
    output_dir = output_dir or (SETTINGS.reports_dir / "leakage_safe_cv")
    if not output_dir.is_absolute():
        output_dir = SETTINGS.project_root / output_dir
    stats_dir = output_dir / "statistical_tests"
    ensure_dir(output_dir)
    ensure_dir(stats_dir)

    df = load_validated_or_raw_data()
    X_raw, y = split_features_and_target(df, drop_sensitive=drop_sensitive)
    y = y.astype(int)
    labels = sorted(y.unique().tolist())

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    rows = []

    for feature_set in feature_sets:
        X = apply_feature_set(X_raw.copy(), feature_set=feature_set)
        removed_features = removed_features_for(X_raw.columns, feature_set)

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_test = y.iloc[test_idx].copy()

            for model_name in models:
                try:
                    pred, proba = fit_predict_model(
                        model_name=model_name,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        random_state=random_state,
                    )
                    metrics = classification_metrics(y_test, pred, proba, labels)

                    row = {
                        "feature_set": feature_set,
                        "model": model_name,
                        "fold": fold,
                        "n_features": int(X.shape[1]),
                        "removed_features": ", ".join(removed_features),
                        **metrics,
                    }
                    rows.append(row)

                    print(
                        f"[cv] feature_set={feature_set} | "
                        f"model={model_name} | fold={fold} | "
                        f"macro_f1={metrics.get('macro_f1'):.4f} | "
                        f"qwk={metrics.get('quadratic_weighted_kappa'):.4f}"
                    )

                except Exception as exc:
                    rows.append({
                        "feature_set": feature_set,
                        "model": model_name,
                        "fold": fold,
                        "n_features": int(X.shape[1]),
                        "removed_features": ", ".join(removed_features),
                        "error": str(exc),
                    })
                    print(
                        f"[cv] FAILED | feature_set={feature_set} | "
                        f"model={model_name} | fold={fold} | error={exc}"
                    )

    fold_df = pd.DataFrame(rows)
    summary_df = summarize_metrics(fold_df)

    fold_path = output_dir / "fold_metrics.csv"
    summary_path = output_dir / "summary_metrics.csv"

    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    metrics_for_tests = [
        "macro_f1",
        "quadratic_weighted_kappa",
        "balanced_accuracy",
        "ordinal_mae",
        "nll_log_loss",
        "multiclass_brier",
        "ece_confidence",
    ]

    stat_outputs = {}

    for metric in metrics_for_tests:
        if metric not in fold_df.columns:
            continue

        ci_df = bootstrap_ci_table(fold_df, metric=metric, seed=random_state)
        pair_df, omnibus = pairwise_tests(fold_df, metric=metric, alpha=alpha)

        ci_path = stats_dir / f"{metric}_bootstrap_ci.csv"
        pair_path = stats_dir / f"{metric}_pairwise_tests.csv"
        meta_path = stats_dir / f"{metric}_metadata.json"

        ci_df.to_csv(ci_path, index=False)
        pair_df.to_csv(pair_path, index=False)
        save_json(omnibus, meta_path)

        stat_outputs[metric] = {
            "bootstrap_ci": str(ci_path),
            "pairwise_tests": str(pair_path),
            "metadata": str(meta_path),
        }

    metadata = {
        "n_splits": n_splits,
        "random_state": random_state,
        "drop_sensitive": drop_sensitive,
        "feature_sets": feature_sets,
        "feature_set_definitions": {
            name: FEATURE_SET_CONFIG.get(name, {})
            for name in feature_sets
        },
        "models": models,
        "labels": labels,
        "git_commit_if_available": get_git_commit(),
        "package_versions": collect_package_versions(
            ["numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "xgboost"]
        ),
        "outputs": {
            "fold_metrics": str(fold_path),
            "summary_metrics": str(summary_path),
            "statistical_tests": stat_outputs,
        },
    }
    save_json(metadata, output_dir / "metadata.json")

    if write_registry:
        append_registry_row(
            {
                "run_id": f"leakage_safe_cv_config_{utc_now_iso()}",
                "date_time": utc_now_iso(),
                "git_commit_if_available": metadata["git_commit_if_available"],
                "script": "python -m src.experiments.leakage_safe_cv",
                "config": "configs/feature_sets.yaml; configs/evaluation.yaml; configs/model_grid.yaml",
                "feature_set": "; ".join(feature_sets),
                "model": "; ".join(models),
                "seed": random_state,
                "cv_strategy": f"StratifiedKFold(n_splits={n_splits}, shuffle=True)",
                "primary_metrics": "macro_f1; quadratic_weighted_kappa; ordinal_mae; severe_error_rate; nll_log_loss; multiclass_brier; ece_confidence",
                "output_dir": str(output_dir.relative_to(SETTINGS.project_root)) if output_dir.is_relative_to(SETTINGS.project_root) else str(output_dir),
                "notes": "Config-backed leakage-safe CV. Final candidate feature sets exclude Age per researcher approval.",
                "decision_status": "candidate",
            }
        )

    print("\n=== LEAKAGE-SAFE CV COMPLETE ===")
    print(f"Fold metrics: {fold_path}")
    print(f"Summary metrics: {summary_path}")
    print("\nTop rows by macro_f1_mean:")
    if "macro_f1_mean" in summary_df.columns:
        print(summary_df.sort_values("macro_f1_mean", ascending=False).to_string(index=False))
    else:
        print(summary_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe cross-validation benchmark.")
    parser.add_argument(
        "--feature-sets",
        type=str,
        default="all",
        help="Comma-separated feature sets or 'all'.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated models or 'all'.",
    )
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--drop-sensitive", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-registry", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    allowed_feature_sets = DEFAULT_FEATURE_SETS if args.feature_sets == "all" else sorted(FEATURE_SET_DEFINITIONS)
    selected_feature_sets = parse_csv_arg(
        args.feature_sets,
        allowed=allowed_feature_sets,
    )
    selected_models = parse_csv_arg(
        args.models,
        allowed=DEFAULT_MODELS,
    )

    run_cv(
        feature_sets=selected_feature_sets,
        models=selected_models,
        n_splits=args.n_splits,
        random_state=args.random_state,
        drop_sensitive=args.drop_sensitive,
        alpha=args.alpha,
        output_dir=args.output_dir,
        write_registry=not args.no_registry,
    )
