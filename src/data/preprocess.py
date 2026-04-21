from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import load_and_validate_data
from src.utils.config import SETTINGS


def load_validated_or_raw_data() -> pd.DataFrame:
    """
    Load validated dataset if it already exists.
    Otherwise load raw data, validate it, and save interim output.
    """
    if SETTINGS.interim_data_path.exists():
        df = pd.read_csv(SETTINGS.interim_data_path)
        return df

    return load_and_validate_data(save_interim=True)


def get_model_feature_columns(drop_sensitive: bool = False) -> List[str]:
    """
    Returns feature columns excluding ID and target.
    Optionally removes fairness-sensitive columns.
    """
    feature_cols = [
        col for col in SETTINGS.expected_columns
        if col not in {SETTINGS.id_col, SETTINGS.target_col}
    ]

    if drop_sensitive:
        feature_cols = [
            col for col in feature_cols
            if col not in SETTINGS.fairness_sensitive_columns
        ]

    return feature_cols


def get_feature_groups(drop_sensitive: bool = False) -> Dict[str, List[str]]:
    """
    Returns grouped feature lists to be used in preprocessing.
    """
    numeric_cols = SETTINGS.numeric_columns.copy()
    ordinal_cols = SETTINGS.ordinal_columns.copy()
    categorical_cols = SETTINGS.categorical_columns.copy()

    if drop_sensitive:
        numeric_cols = [c for c in numeric_cols if c not in SETTINGS.fairness_sensitive_columns]
        ordinal_cols = [c for c in ordinal_cols if c not in SETTINGS.fairness_sensitive_columns]
        categorical_cols = [c for c in categorical_cols if c not in SETTINGS.fairness_sensitive_columns]

    return {
        "numeric": numeric_cols,
        "ordinal": ordinal_cols,
        "categorical": categorical_cols,
    }


def create_one_hot_encoder() -> OneHotEncoder:
    """
    sklearn version compatibility helper.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_sklearn_preprocessor(drop_sensitive: bool = False) -> ColumnTransformer:
    """
    Build preprocessing pipeline for sklearn-compatible models.

    Numeric:
        median imputation + standard scaling

    Ordinal:
        median imputation + standard scaling
        (since values are ordered and usable as numeric ranks)

    Categorical:
        most_frequent imputation + one-hot encoding
    """
    feature_groups = get_feature_groups(drop_sensitive=drop_sensitive)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", create_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, feature_groups["numeric"]),
            ("ord", ordinal_pipeline, feature_groups["ordinal"]),
            ("cat", categorical_pipeline, feature_groups["categorical"]),
        ],
        remainder="drop",
    )

    return preprocessor


def split_features_and_target(
    df: pd.DataFrame,
    drop_sensitive: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into X and y.
    """
    feature_cols = get_model_feature_columns(drop_sensitive=drop_sensitive)

    X = df[feature_cols].copy()
    y = df[SETTINGS.target_col].astype(int).copy()

    return X, y


def get_preprocessed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extract transformed feature names from fitted ColumnTransformer.
    """
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
        return feature_names
    except Exception:
        # Fallback
        names: List[str] = []

        for transformer_name, transformer, cols in preprocessor.transformers_:
            if transformer_name == "remainder":
                continue

            if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
                onehot = transformer.named_steps["onehot"]
                cat_names = onehot.get_feature_names_out(cols).tolist()
                names.extend(cat_names)
            else:
                names.extend(cols)

        return names


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_series(s: pd.Series, path: Path, name: str = "target") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame(name=name).to_csv(path, index=False)


def run_preprocessing(
    test_size: float = 0.20,
    random_state: int = 42,
    drop_sensitive: bool = False,
) -> Dict[str, Any]:
    """
    Main preprocessing function.

    Saves:
    - raw train/test splits
    - fitted sklearn preprocessor
    - transformed train/test matrices as CSV
    - metadata JSON
    """
    df = load_validated_or_raw_data()

    X, y = split_features_and_target(df, drop_sensitive=drop_sensitive)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_sklearn_preprocessor(drop_sensitive=drop_sensitive)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    transformed_feature_names = get_preprocessed_feature_names(preprocessor)

    X_train_processed_df = pd.DataFrame(
        X_train_transformed,
        columns=transformed_feature_names,
        index=X_train.index,
    )
    X_test_processed_df = pd.DataFrame(
        X_test_transformed,
        columns=transformed_feature_names,
        index=X_test.index,
    )

    processed_dir = SETTINGS.processed_data_path.parent
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save raw splits
    save_dataframe(X_train, processed_dir / "X_train_raw.csv")
    save_dataframe(X_test, processed_dir / "X_test_raw.csv")
    save_series(y_train, processed_dir / "y_train.csv", name=SETTINGS.target_col)
    save_series(y_test, processed_dir / "y_test.csv", name=SETTINGS.target_col)

    # Save processed splits
    save_dataframe(X_train_processed_df, processed_dir / "X_train_preprocessed.csv")
    save_dataframe(X_test_processed_df, processed_dir / "X_test_preprocessed.csv")

    # Save fitted preprocessor
    preprocessor_path = processed_dir / "sklearn_preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    metadata = {
        "input_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "input_feature_count": int(X.shape[1]),
        "processed_feature_count": int(X_train_processed_df.shape[1]),
        "target_column": SETTINGS.target_col,
        "id_column_excluded": SETTINGS.id_col,
        "drop_sensitive": drop_sensitive,
        "sensitive_columns_removed": SETTINGS.fairness_sensitive_columns if drop_sensitive else [],
        "numeric_columns": get_feature_groups(drop_sensitive=drop_sensitive)["numeric"],
        "ordinal_columns": get_feature_groups(drop_sensitive=drop_sensitive)["ordinal"],
        "categorical_columns": get_feature_groups(drop_sensitive=drop_sensitive)["categorical"],
        "class_distribution_full": y.value_counts().sort_index().to_dict(),
        "class_distribution_train": y_train.value_counts().sort_index().to_dict(),
        "class_distribution_test": y_test.value_counts().sort_index().to_dict(),
        "test_size": test_size,
        "random_state": random_state,
        "artifacts": {
            "x_train_raw": str(processed_dir / "X_train_raw.csv"),
            "x_test_raw": str(processed_dir / "X_test_raw.csv"),
            "y_train": str(processed_dir / "y_train.csv"),
            "y_test": str(processed_dir / "y_test.csv"),
            "x_train_preprocessed": str(processed_dir / "X_train_preprocessed.csv"),
            "x_test_preprocessed": str(processed_dir / "X_test_preprocessed.csv"),
            "preprocessor": str(preprocessor_path),
        },
    }

    metadata_path = processed_dir / "preprocessing_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n=== PREPROCESSING COMPLETE ===")
    print(f"Input rows: {len(df)}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Input feature count: {X.shape[1]}")
    print(f"Processed feature count: {X_train_processed_df.shape[1]}")
    print(f"Drop sensitive columns: {drop_sensitive}")
    print(f"Saved artifacts to: {processed_dir}")

    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_processed_df,
        "X_test_processed": X_test_processed_df,
        "preprocessor": preprocessor,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess INX employee performance dataset.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test size ratio. Default: 0.20")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument(
        "--drop-sensitive",
        action="store_true",
        help="Drop fairness-sensitive columns such as Gender and MaritalStatus.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocessing(
        test_size=args.test_size,
        random_state=args.random_state,
        drop_sensitive=args.drop_sensitive,
    )