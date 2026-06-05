from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.experiments.final_evidence_common import (
    FINAL_FEATURE_SETS,
    MODEL_NAME,
    XAI_DIR,
    append_task_registry,
    ensure_dir,
    get_current_xy,
    save_json,
)
from src.experiments.leakage_safe_cv import LabelEncodedXGBClassifier, infer_columns, make_preprocessor
from src.utils.experiment_registry import collect_package_versions, utc_now_iso


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


def get_group_mapping(preprocessor: Any, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
    group_names = numeric_cols + categorical_cols
    mapping: Dict[str, List[int]] = {name: [] for name in group_names}
    col_idx = 0
    for col in numeric_cols:
        mapping[col].append(col_idx)
        col_idx += 1
    if categorical_cols:
        encoder = preprocessor.named_transformers_["cat"]
        for col, categories in zip(categorical_cols, encoder.categories_):
            for _ in categories:
                mapping[col].append(col_idx)
                col_idx += 1
    return group_names, mapping


def group_shap_values(shap_values: np.ndarray, group_names: List[str], mapping: Dict[str, List[int]]) -> np.ndarray:
    n_samples, n_classes, _ = shap_values.shape
    grouped = np.zeros((n_samples, n_classes, len(group_names)), dtype=float)
    for group_idx, group_name in enumerate(group_names):
        cols = mapping[group_name]
        if cols:
            grouped[:, :, group_idx] = shap_values[:, :, cols].sum(axis=2)
    return grouped


def group_transformed_importance(transformed_importance: np.ndarray, group_names: List[str], mapping: Dict[str, List[int]]) -> Dict[str, float]:
    return {group: float(np.asarray(transformed_importance)[cols].sum()) if cols else 0.0 for group, cols in mapping.items()}


def jaccard_at_k(rank_a: List[str], rank_b: List[str], k: int) -> float:
    set_a = set(rank_a[:k])
    set_b = set(rank_b[:k])
    if not set_a and not set_b:
        return 1.0
    return float(len(set_a.intersection(set_b)) / len(set_a.union(set_b)))


def spearman_for_rankings(rank_a: List[str], rank_b: List[str]) -> float:
    features = sorted(set(rank_a).union(rank_b))
    missing = len(features) + 1
    a = pd.Series({feature: (rank_a.index(feature) + 1 if feature in rank_a else missing) for feature in features})
    b = pd.Series({feature: (rank_b.index(feature) + 1 if feature in rank_b else missing) for feature in features})
    value = a.corr(b, method="spearman")
    return float(value) if pd.notna(value) else float("nan")


def write_interpretation(summary_df: pd.DataFrame, rankings_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# SHAP Stability Interpretation for Final Candidates",
        "",
        "SHAP values are model attributions, not causal effects. Stability measures whether global grouped feature-importance rankings are consistent across CV folds.",
        "",
        "## Stability Summary",
        "",
    ]
    for feature_set, group in summary_df.groupby("feature_set"):
        top10 = group[group["top_k"] == 10].iloc[0]
        top_features = (
            rankings_df[rankings_df["feature_set"] == feature_set]
            .sort_values(["fold", "rank"])
            .groupby("fold")
            .head(5)["feature"]
            .value_counts()
            .head(8)
            .index.tolist()
        )
        lines.extend(
            [
                f"### `{feature_set}`",
                f"- Mean top-10 Jaccard: {top10['mean_jaccard']:.4f}.",
                f"- Mean Spearman rank correlation: {top10['mean_spearman']:.4f}.",
                f"- Frequently recurring top features: {', '.join(top_features)}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Required Answers",
            "",
            "### Are the top explanatory features stable across folds?",
            "Use top-k Jaccard and Spearman jointly. High top-k Jaccard supports stable global discussion; low Spearman means lower-ranked features should not be overinterpreted.",
            "",
            "### Does removing EmpDepartment change the explanation structure?",
            "Removing EmpDepartment should be interpreted as an explanation-structure shift. If EmpJobRole or other organisational variables become prominent, describe this as proxy reliance rather than fairness mitigation.",
            "",
            "### Does removing EmpJobRole make explanations more or less stable?",
            "The strict job-role-free model must be judged jointly with utility. Higher stability alone is not sufficient if performance, calibration, or actionability are worse.",
            "",
            "### Which explanations are safe to discuss in the paper?",
            "Discuss only recurring grouped features from stable top-k rankings. Avoid local causal language and prescriptive HR claims.",
            "",
            "### Which explanations require proxy/fairness warnings?",
            "EmpJobRole, BusinessTravelFrequency, EducationBackground, DistanceFromHome, tenure, job level, and organisational-history features require proxy or governance warnings when important.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(n_splits: int = 5, seed: int = 42) -> Dict[str, Path]:
    import shap

    ensure_dir(XAI_DIR)
    ranking_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    for feature_set in FINAL_FEATURE_SETS:
        X, y, _ = get_current_xy(feature_set, drop_sensitive=True)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_rankings: Dict[int, List[str]] = {}
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
            X_train = X.iloc[train_idx].copy()
            y_train = y.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            numeric_cols, categorical_cols = infer_columns(X_train)
            preprocessor = make_preprocessor(X_train)
            X_train_t = preprocessor.fit_transform(X_train)
            X_test_t = preprocessor.transform(X_test)
            if hasattr(X_train_t, "toarray"):
                X_train_t = X_train_t.toarray()
            if hasattr(X_test_t, "toarray"):
                X_test_t = X_test_t.toarray()
            classifier = LabelEncodedXGBClassifier(random_state=seed)
            classifier.fit(X_train_t, y_train)
            group_names, mapping = get_group_mapping(preprocessor, numeric_cols, categorical_cols)
            explainer = shap.TreeExplainer(classifier.model_)
            raw_shap = explainer.shap_values(X_test_t)
            shap_arr = normalize_shap_values(raw_shap, X_test_t.shape[0], X_test_t.shape[1], len(classifier.classes_))
            grouped = group_shap_values(shap_arr, group_names, mapping)
            importance = np.mean(np.abs(grouped), axis=(0, 1))
            ranking_df = pd.DataFrame({"feature": group_names, "mean_abs_grouped_shap": importance}).sort_values("mean_abs_grouped_shap", ascending=False).reset_index(drop=True)
            ranking_df["rank"] = np.arange(1, len(ranking_df) + 1)
            fold_rankings[fold] = ranking_df["feature"].tolist()
            for row in ranking_df.itertuples(index=False):
                ranking_rows.append(
                    {
                        "feature_set": feature_set,
                        "fold": fold,
                        "feature": row.feature,
                        "rank": int(row.rank),
                        "mean_abs_grouped_shap": float(row.mean_abs_grouped_shap),
                    }
                )
            print(f"[shap] feature_set={feature_set} fold={fold}")
        for fold_a, fold_b in itertools.combinations(sorted(fold_rankings), 2):
            rank_a = fold_rankings[fold_a]
            rank_b = fold_rankings[fold_b]
            spearman = spearman_for_rankings(rank_a, rank_b)
            for k in [5, 10, 15]:
                pair_rows.append(
                    {
                        "feature_set": feature_set,
                        "fold_a": fold_a,
                        "fold_b": fold_b,
                        "top_k": k,
                        "jaccard": jaccard_at_k(rank_a, rank_b, k),
                        "spearman": spearman,
                    }
                )

    rankings_df = pd.DataFrame(ranking_rows)
    pairs_df = pd.DataFrame(pair_rows)
    summary_rows = []
    for (feature_set, top_k), group in pairs_df.groupby(["feature_set", "top_k"]):
        summary_rows.append(
            {
                "feature_set": feature_set,
                "top_k": int(top_k),
                "mean_jaccard": float(group["jaccard"].mean()),
                "std_jaccard": float(group["jaccard"].std(ddof=1)) if len(group) > 1 else 0.0,
                "mean_spearman": float(group["spearman"].mean()),
                "std_spearman": float(group["spearman"].std(ddof=1)) if len(group) > 1 else 0.0,
                "n_pairs": int(len(group)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = XAI_DIR / "shap_stability_summary.csv"
    rankings_path = XAI_DIR / "fold_feature_rankings.csv"
    pairwise_path = XAI_DIR / "shap_stability_pairwise.csv"
    interp_path = XAI_DIR / "shap_stability_interpretation.md"
    meta_path = XAI_DIR / "shap_stability_metadata.json"
    summary_df.to_csv(summary_path, index=False)
    rankings_df.to_csv(rankings_path, index=False)
    pairs_df.to_csv(pairwise_path, index=False)
    write_interpretation(summary_df, rankings_df, interp_path)
    save_json(
        {
            "task": "final_candidate_shap_stability",
            "feature_sets": FINAL_FEATURE_SETS,
            "model": MODEL_NAME,
            "n_splits": n_splits,
            "seed": seed,
            "package_versions": collect_package_versions(["numpy", "pandas", "scikit-learn", "xgboost", "shap"]),
        },
        meta_path,
    )
    append_task_registry(
        run_id=f"final_shap_stability_{utc_now_iso()}",
        script="python -m src.experiments.final_shap_stability",
        feature_set="; ".join(FINAL_FEATURE_SETS),
        primary_metrics="top-k Jaccard and Spearman rank stability of grouped SHAP importance",
        output_dir=XAI_DIR,
        notes="Grouped one-hot XGBoost SHAP values back to raw feature families across CV folds.",
    )
    return {"summary": summary_path, "rankings": rankings_path, "interpretation": interp_path, "metadata": meta_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped SHAP stability for final XGBoost candidates.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run(n_splits=args.n_splits, seed=args.seed))

