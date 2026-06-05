from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd

from src.utils.config_loader import load_feature_sets_config, load_feature_taxonomy_config


FeatureSetConfig = Dict[str, Dict[str, Any]]


def get_feature_sets(config: Dict[str, Any] | None = None) -> FeatureSetConfig:
    data = config if config is not None else load_feature_sets_config()
    feature_sets = data.get("feature_sets")
    if not isinstance(feature_sets, dict):
        raise ValueError("Feature set config must contain a 'feature_sets' object.")
    return feature_sets


def get_taxonomy_rows(config: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    data = config if config is not None else load_feature_taxonomy_config()
    rows = data.get("feature_taxonomy")
    if not isinstance(rows, list):
        raise ValueError("Feature taxonomy config must contain a 'feature_taxonomy' list.")
    return rows


def taxonomy_by_feature(config: Dict[str, Any] | None = None) -> Dict[str, Dict[str, Any]]:
    return {row["feature_name"]: row for row in get_taxonomy_rows(config)}


def features_by_control_type(control_types: Iterable[str], taxonomy_config: Dict[str, Any] | None = None) -> List[str]:
    allowed = set(control_types)
    return [
        row["feature_name"]
        for row in get_taxonomy_rows(taxonomy_config)
        if row.get("control_type") in allowed and row.get("allowed_for_final_model") is True
    ]


def build_feature_columns(
    all_columns: Iterable[str],
    feature_set_name: str,
    feature_sets_config: Dict[str, Any] | None = None,
    taxonomy_config: Dict[str, Any] | None = None,
) -> List[str]:
    """Return feature columns for a named feature set.

    The target and ID columns should be excluded by config. This function also
    defensively removes them if the caller passes a malformed config.
    """
    all_columns_list = list(all_columns)
    feature_sets = get_feature_sets(feature_sets_config)

    if feature_set_name not in feature_sets:
        raise ValueError(f"Unknown feature set: {feature_set_name}")

    definition = feature_sets[feature_set_name]

    if "include_by_control_type" in definition:
        included = set(features_by_control_type(definition["include_by_control_type"], taxonomy_config))
        columns = [col for col in all_columns_list if col in included]
    else:
        drop = set(definition.get("drop", []))
        columns = [col for col in all_columns_list if col not in drop]

    # Defense in depth: never return identifier or target as predictors.
    forbidden = {"EmpNumber", "PerformanceRating"}
    return [col for col in columns if col not in forbidden]


def apply_feature_set(
    df: pd.DataFrame,
    feature_set_name: str,
    feature_sets_config: Dict[str, Any] | None = None,
    taxonomy_config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    columns = build_feature_columns(
        df.columns,
        feature_set_name=feature_set_name,
        feature_sets_config=feature_sets_config,
        taxonomy_config=taxonomy_config,
    )
    return df.loc[:, columns].copy()


def assert_excludes(feature_columns: Iterable[str], excluded_columns: Iterable[str]) -> None:
    present = sorted(set(feature_columns).intersection(excluded_columns))
    if present:
        raise AssertionError(f"Excluded columns are present in feature set: {present}")


def feature_sets_by_deployment_status(status: str) -> List[str]:
    feature_sets = get_feature_sets()
    return [
        name
        for name, definition in feature_sets.items()
        if definition.get("deployment_status") == status
    ]


def leakage_safe_feature_sets() -> List[str]:
    return feature_sets_by_deployment_status("candidate")
