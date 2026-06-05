from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.preprocess import split_features_and_target
from src.data.preprocess import load_validated_or_raw_data
from src.features.feature_sets import apply_feature_set
from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


DEFAULT_OUTPUT_DIR = SETTINGS.reports_dir / "fairness"
DEFAULT_FEATURE_SET = "no_salary_hike_no_attrition_no_department"
PROXY_TARGET = "EmpDepartment"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols


def make_proxy_pipeline(X: pd.DataFrame, random_state: int) -> Pipeline:
    numeric_cols, categorical_cols = infer_columns(X)
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", one_hot_encoder(), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    classifier = LogisticRegression(
        max_iter=10000,
        class_weight="balanced",
        random_state=random_state,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", classifier)])


def cramers_v(feature: Iterable[Any], target: Iterable[Any]) -> float:
    table = pd.crosstab(pd.Series(feature), pd.Series(target))
    if table.empty or min(table.shape) <= 1:
        return 0.0

    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    if n == 0:
        return 0.0

    r, k = table.shape
    denominator = n * (min(k - 1, r - 1))
    if denominator == 0:
        return 0.0
    return float(np.sqrt(chi2 / denominator))


def feature_proxy_associations(X: pd.DataFrame, y: pd.Series, random_state: int) -> pd.DataFrame:
    rows = []
    y_encoded, _ = pd.factorize(y.astype(str))

    for feature in X.columns:
        series = X[feature]
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").fillna(series.median()).to_numpy().reshape(-1, 1)
            mi = float(mutual_info_classif(values, y_encoded, random_state=random_state, discrete_features=False)[0])
            association_type = "mutual_info_numeric"
            association_value = mi
        else:
            as_str = series.astype("string").fillna("__MISSING__").astype(str)
            encoded, _ = pd.factorize(as_str)
            mi = float(mutual_info_classif(encoded.reshape(-1, 1), y_encoded, random_state=random_state, discrete_features=True)[0])
            cv = cramers_v(as_str, y.astype(str))
            association_type = "categorical_mi_and_cramers_v"
            association_value = cv

        rows.append(
            {
                "feature": feature,
                "association_type": association_type,
                "mutual_info": mi,
                "cramers_v": association_value if association_type == "categorical_mi_and_cramers_v" else np.nan,
                "proxy_watchlist": feature in {"EmpJobRole", "EducationBackground", "BusinessTravelFrequency"},
            }
        )

    return pd.DataFrame(rows).sort_values(["mutual_info", "cramers_v"], ascending=[False, False])


def run_proxy_classifier_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    min_class_count = int(y.value_counts().min())
    splits = min(n_splits, min_class_count)
    if splits < 2:
        raise ValueError(f"Not enough support for proxy CV. Minimum department class count={min_class_count}.")

    pipeline = make_proxy_pipeline(X, random_state)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
    }
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)

    rows = []
    for fold in range(splits):
        rows.append(
            {
                "fold": fold + 1,
                "accuracy": float(results["test_accuracy"][fold]),
                "balanced_accuracy": float(results["test_balanced_accuracy"][fold]),
                "macro_f1": float(results["test_macro_f1"][fold]),
            }
        )
    return pd.DataFrame(rows)


def summarize_cv(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
        rows.append(
            {
                "metric": metric,
                "mean": float(fold_df[metric].mean()),
                "std": float(fold_df[metric].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def interpretation_text(
    feature_set: str,
    n_features: int,
    cv_summary: pd.DataFrame,
    associations: pd.DataFrame,
) -> str:
    top_features = associations.head(10)
    macro_f1 = cv_summary[cv_summary["metric"] == "macro_f1"]["mean"].iloc[0]
    balanced_accuracy = cv_summary[cv_summary["metric"] == "balanced_accuracy"]["mean"].iloc[0]

    lines = [
        "# Department Proxy Analysis",
        "",
        f"Feature set tested: `{feature_set}`",
        f"Number of remaining features: {n_features}",
        "",
        "## Proxy classifier result",
        "",
        f"A balanced logistic-regression proxy classifier predicted `EmpDepartment` from the remaining model features with macro-F1={macro_f1:.4f} and balanced accuracy={balanced_accuracy:.4f}.",
        "",
        "This does not prove discriminatory behavior. It estimates how much direct department information can still be reconstructed after removing `EmpDepartment`.",
        "",
        "## Top proxy-associated features",
        "",
    ]

    for row in top_features.itertuples(index=False):
        if pd.isna(row.cramers_v):
            assoc = f"mutual_info={row.mutual_info:.4f}"
        else:
            assoc = f"mutual_info={row.mutual_info:.4f}, cramers_v={row.cramers_v:.4f}"
        lines.append(f"- {row.feature}: {assoc}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If department is highly reconstructable from remaining features, department-free modeling reduces direct department use but does not eliminate organisational proxy risk. `EmpJobRole` is especially important to inspect because it may encode department structure.",
            "",
            "## Required follow-up",
            "",
            "- Compare fairness gaps for department-including and department-free candidate models.",
            "- Add proxy warnings to local reason codes when top features are organisational proxies.",
            "- Do not claim department removal proves fairness.",
        ]
    )

    return "\n".join(lines) + "\n"


def run_department_proxy_analysis(
    feature_set: str = DEFAULT_FEATURE_SET,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n_splits: int = 5,
    random_state: int = 42,
    write_registry: bool = True,
) -> Dict[str, Path]:
    ensure_dir(output_dir)

    df = load_validated_or_raw_data()
    X_full, _ = split_features_and_target(df, drop_sensitive=True)
    X = apply_feature_set(X_full, feature_set)
    y_department = df.loc[X.index, PROXY_TARGET].astype(str)

    fold_df = run_proxy_classifier_cv(X, y_department, n_splits=n_splits, random_state=random_state)
    summary_df = summarize_cv(fold_df)
    associations = feature_proxy_associations(X, y_department, random_state=random_state)

    outputs = {
        "feature_associations": output_dir / "proxy_analysis_department.csv",
        "cv_fold_metrics": output_dir / "proxy_analysis_department_cv_fold_metrics.csv",
        "cv_summary": output_dir / "proxy_analysis_department_cv_summary.csv",
        "interpretation": output_dir / "proxy_analysis_department_interpretation.md",
    }

    associations.to_csv(outputs["feature_associations"], index=False)
    fold_df.to_csv(outputs["cv_fold_metrics"], index=False)
    summary_df.to_csv(outputs["cv_summary"], index=False)
    outputs["interpretation"].write_text(
        interpretation_text(feature_set, X.shape[1], summary_df, associations),
        encoding="utf-8",
    )

    if write_registry:
        run_timestamp = utc_now_iso()
        safe_feature_set = feature_set.replace(" ", "_").replace("/", "_")
        append_registry_row(
            {
                "run_id": f"department_proxy_analysis_{safe_feature_set}_{run_timestamp}",
                "date_time": run_timestamp,
                "git_commit_if_available": get_git_commit(),
                "script": "python -m src.experiments.proxy_analysis",
                "config": "configs/feature_sets.yaml; configs/fairness.yaml; configs/feature_taxonomy.yaml",
                "feature_set": feature_set,
                "model": "balanced_logistic_regression_proxy_classifier",
                "seed": random_state,
                "cv_strategy": f"StratifiedKFold(n_splits={min(n_splits, int(y_department.value_counts().min()))}, shuffle=True)",
                "primary_metrics": summary_df.to_dict(orient="records"),
                "output_dir": str(output_dir.relative_to(SETTINGS.project_root)) if output_dir.is_relative_to(SETTINGS.project_root) else str(output_dir),
                "notes": "Proxy analysis for reconstructing EmpDepartment from department-free final-candidate features. No employee performance model was trained.",
                "decision_status": "candidate",
            }
        )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run department proxy analysis.")
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-registry", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = run_department_proxy_analysis(
        feature_set=args.feature_set,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        random_state=args.random_state,
        write_registry=not args.no_registry,
    )
    print("Department proxy analysis outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
