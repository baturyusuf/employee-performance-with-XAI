from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.preprocess import load_validated_or_raw_data, split_features_and_target
from src.experiments.leakage_safe_cv import LabelEncodedXGBClassifier, make_preprocessor
from src.features.feature_sets import apply_feature_set
from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


FINAL_FEATURE_SETS = [
    "no_salary_hike_no_attrition",
    "no_salary_hike_no_attrition_no_department",
    "no_salary_hike_no_attrition_no_department_no_job_role",
]
PRIMARY_FEATURE_SET = "no_salary_hike_no_attrition_no_department"
MODEL_NAME = "xgboost"
LABELS = [2, 3, 4]

FAIRNESS_DIR = SETTINGS.reports_dir / "fairness" / "feature_set_sensitivity"
CALIBRATION_DIR = SETTINGS.reports_dir / "calibration" / "final_candidates"
XAI_DIR = SETTINGS.reports_dir / "xai" / "final_candidates"
COUNTERFACTUAL_DIR = SETTINGS.reports_dir / "counterfactuals" / "final_candidates"
MODEL_SELECTION_DIR = SETTINGS.reports_dir / "model_selection"
MODEL_CARD_DIR = SETTINGS.reports_dir / "model_card"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(SETTINGS.project_root))
    except ValueError:
        return str(path)


def registry_row(
    run_id: str,
    script: str,
    feature_set: str,
    primary_metrics: str,
    output_dir: Path,
    notes: str,
    model: str = MODEL_NAME,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "date_time": utc_now_iso(),
        "git_commit_if_available": get_git_commit(),
        "script": script,
        "config": "configs/feature_sets.yaml; configs/fairness.yaml; configs/feature_taxonomy.yaml; configs/counterfactuals.yaml",
        "feature_set": feature_set,
        "model": model,
        "seed": 42,
        "cv_strategy": "task_specific; see output metadata",
        "primary_metrics": primary_metrics,
        "output_dir": rel_path(output_dir),
        "notes": notes,
        "decision_status": "candidate",
    }


def append_task_registry(*args: Any, **kwargs: Any) -> None:
    append_registry_row(registry_row(*args, **kwargs))


def get_current_xy(feature_set: str, drop_sensitive: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    full_df = load_validated_or_raw_data()
    X_raw, y = split_features_and_target(full_df, drop_sensitive=drop_sensitive)
    X = apply_feature_set(X_raw.copy(), feature_set)
    return X, y.astype(int), full_df


def fit_xgb_pipeline(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> Pipeline:
    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(X_train)),
            ("model", LabelEncodedXGBClassifier(random_state=random_state)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def align_proba(proba: np.ndarray, classes: Iterable[int], labels: List[int] = LABELS) -> np.ndarray:
    classes_list = [int(c) for c in classes]
    out = np.zeros((len(proba), len(labels)), dtype=float)
    for source_idx, cls in enumerate(classes_list):
        if cls in labels:
            out[:, labels.index(cls)] = proba[:, source_idx]
    row_sums = out.sum(axis=1, keepdims=True)
    zero_mask = row_sums.flatten() <= 0
    if np.any(zero_mask):
        out[zero_mask, :] = 1.0 / len(labels)
        row_sums = out.sum(axis=1, keepdims=True)
    return out / row_sums


def predict_labels_from_proba(proba: np.ndarray, labels: List[int] = LABELS) -> np.ndarray:
    return np.asarray(labels, dtype=int)[np.argmax(proba, axis=1)]


def normalize_rows(proba: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(proba, dtype=float), 1e-8, 1.0)
    sums = arr.sum(axis=1, keepdims=True)
    sums[sums <= 0] = 1.0
    return arr / sums


def calibrate_probabilities(
    calib_proba: np.ndarray,
    calib_y: Iterable[int],
    test_proba: np.ndarray,
    labels: List[int] = LABELS,
    method: str = "raw",
    seed: int = 42,
) -> np.ndarray:
    calib_proba = normalize_rows(calib_proba)
    test_proba = normalize_rows(test_proba)
    if method == "raw":
        return test_proba

    y_arr = np.asarray(list(calib_y), dtype=int)
    calibrated = np.zeros_like(test_proba, dtype=float)

    for idx, label in enumerate(labels):
        binary_y = (y_arr == label).astype(int)
        if len(np.unique(binary_y)) < 2:
            calibrated[:, idx] = float(binary_y.mean())
            continue

        if method == "sigmoid":
            p_cal = np.clip(calib_proba[:, idx], 1e-6, 1 - 1e-6)
            p_test = np.clip(test_proba[:, idx], 1e-6, 1 - 1e-6)
            logits = np.log(p_cal / (1 - p_cal))
            test_logits = np.log(p_test / (1 - p_test))
            model = LogisticRegression(solver="lbfgs", random_state=seed)
            model.fit(logits.reshape(-1, 1), binary_y)
            calibrated[:, idx] = model.predict_proba(test_logits.reshape(-1, 1))[:, 1]
        elif method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(calib_proba[:, idx], binary_y)
            calibrated[:, idx] = iso.predict(test_proba[:, idx])
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    return normalize_rows(calibrated)


def bootstrap_mean_ci(values: Iterable[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(arr, size=len(arr), replace=True).mean()
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def dashboard_required_columns() -> List[str]:
    return [
        "feature_set",
        "model",
        "role",
        "macro_f1",
        "balanced_accuracy",
        "qwk",
        "ordinal_mae",
        "severe_error_rate",
        "adjacent_accuracy",
        "calibration_method",
        "log_loss",
        "multiclass_brier",
        "ece",
        "empdepartment_macro_f1_gap",
        "empdepartment_macro_f1_gap_ci_low",
        "empdepartment_macro_f1_gap_ci_high",
        "department_proxy_macro_f1",
        "emp_job_role_present",
        "top10_shap_jaccard",
        "shap_spearman",
        "employee_only_validity",
        "employee_manager_validity",
        "organization_allowed_validity",
        "full_default_validity",
        "no_salary_validity",
        "leakage_safe",
        "required_warnings",
        "recommendation_category",
    ]

