from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.explainability.canonical_shap_axis import (
    CanonicalShapAxisError,
    build_canonical_shap_axis,
    group_canonical_shap_values,
    normalize_multiclass_shap_values,
    validate_canonical_transformed_matrix,
)
from src.models.canonical_models import build_common_preprocessor


def _features() -> pd.DataFrame:
    # Raw order is intentionally interleaved. ColumnTransformer output is the
    # numeric block followed by the categorical block, while grouped SHAP must
    # return to this raw order.
    return pd.DataFrame(
        {
            "Role": ["engineering", "sales", "engineering", "sales"],
            "Tenure": [1.0, 2.0, 3.0, 4.0],
            "Travel": ["none", "rare", "frequent", "none"],
            "Training": [2, 3, 4, 5],
        }
    )


def _fitted():
    features = _features()
    return build_common_preprocessor(features).fit(features)


def _axis():
    features = _features()
    return build_canonical_shap_axis(
        _fitted(),
        raw_feature_order=features.columns,
        forbidden_features=["Age", "Gender", "MaritalStatus"],
    )


def test_fitted_axis_uses_output_slices_categories_and_preserves_raw_order() -> None:
    preprocessor = _fitted()
    contract = build_canonical_shap_axis(
        preprocessor,
        raw_feature_order=_features().columns,
    )

    assert contract.raw_feature_names == ("Role", "Tenure", "Travel", "Training")
    assert contract.numeric_feature_names == ("Tenure", "Training")
    assert contract.categorical_feature_names == ("Role", "Travel")
    assert contract.transformed_feature_names == (
        "numeric__Tenure",
        "numeric__Training",
        "categorical__Role_engineering",
        "categorical__Role_sales",
        "categorical__Travel_frequent",
        "categorical__Travel_none",
        "categorical__Travel_rare",
    )
    assert dict(contract.transformed_indices_by_raw) == {
        "Role": (2, 3),
        "Tenure": (0,),
        "Travel": (4, 5, 6),
        "Training": (1,),
    }
    assert contract.transformed_to_raw == (
        "Tenure",
        "Training",
        "Role",
        "Role",
        "Travel",
        "Travel",
        "Travel",
    )
    with pytest.raises(TypeError):
        contract.transformed_indices_by_raw["Role"] = (0,)  # type: ignore[index]


@pytest.mark.parametrize(
    "features",
    [
        pd.DataFrame({"A": [1.0, 2.0], "B": [3, 4]}),
        pd.DataFrame({"A": ["x", "y"], "B": ["m", "n"]}),
    ],
)
def test_axis_supports_canonical_numeric_only_and_categorical_only_blocks(
    features: pd.DataFrame,
) -> None:
    preprocessor = build_common_preprocessor(features).fit(features)
    contract = build_canonical_shap_axis(
        preprocessor,
        raw_feature_order=features.columns,
    )
    assert contract.raw_feature_names == tuple(features.columns)
    assert sorted(
        index
        for indices in contract.transformed_indices_by_raw.values()
        for index in indices
    ) == list(range(contract.n_transformed_features))


def test_realized_transformed_matrix_must_match_width_names_and_sample_count() -> None:
    features = _features()
    preprocessor = _fitted()
    contract = build_canonical_shap_axis(
        preprocessor,
        raw_feature_order=features.columns,
    )
    transformed = preprocessor.transform(features)
    validate_canonical_transformed_matrix(
        transformed,
        contract,
        n_samples=len(features),
    )
    contract.validate_transformed_matrix(transformed.to_numpy(), n_samples=len(features))

    with pytest.raises(CanonicalShapAxisError, match="feature-axis width drifted"):
        contract.validate_transformed_matrix(transformed.iloc[:, :-1])
    renamed = transformed.rename(columns={transformed.columns[0]: "numeric__Wrong"})
    with pytest.raises(CanonicalShapAxisError, match="columns do not equal"):
        contract.validate_transformed_matrix(renamed)
    with pytest.raises(CanonicalShapAxisError, match="sample-axis length drifted"):
        contract.validate_transformed_matrix(transformed, n_samples=len(features) + 1)


def test_raw_feature_order_and_forbidden_raw_names_fail_closed() -> None:
    preprocessor = _fitted()
    with pytest.raises(CanonicalShapAxisError, match="raw feature order"):
        build_canonical_shap_axis(
            preprocessor,
            raw_feature_order=list(reversed(_features().columns)),
        )
    with pytest.raises(CanonicalShapAxisError, match="Forbidden raw features"):
        build_canonical_shap_axis(
            preprocessor,
            raw_feature_order=_features().columns,
            forbidden_features=["role"],
        )


def test_forbidden_transformed_name_is_rejected_before_lineage_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocessor = _fitted()
    names = preprocessor.get_feature_names_out().copy()
    names[0] = "numeric__Age"
    monkeypatch.setattr(preprocessor, "get_feature_names_out", lambda: names)
    with pytest.raises(CanonicalShapAxisError, match="Forbidden transformed"):
        build_canonical_shap_axis(
            preprocessor,
            raw_feature_order=_features().columns,
            forbidden_features=["Age"],
        )


def test_get_feature_names_out_must_equal_reconstructed_fitted_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocessor = _fitted()
    names = preprocessor.get_feature_names_out().copy()
    names[-1] = "categorical__Travel_unrecorded"
    monkeypatch.setattr(preprocessor, "get_feature_names_out", lambda: names)
    with pytest.raises(CanonicalShapAxisError, match="does not equal lineage"):
        build_canonical_shap_axis(
            preprocessor,
            raw_feature_order=_features().columns,
        )


def test_transformed_output_index_overlap_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocessor = _fitted()
    monkeypatch.setattr(
        preprocessor,
        "output_indices_",
        {
            "numeric": slice(0, 2),
            "categorical": slice(1, 6),
            "remainder": slice(0, 0),
        },
    )
    with pytest.raises(CanonicalShapAxisError, match="overlap"):
        build_canonical_shap_axis(
            preprocessor,
            raw_feature_order=_features().columns,
        )


def test_transformed_output_index_gap_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocessor = _fitted()
    names = np.append(preprocessor.get_feature_names_out(), "numeric__Unowned")
    monkeypatch.setattr(preprocessor, "get_feature_names_out", lambda: names)
    with pytest.raises(CanonicalShapAxisError, match="gaps"):
        build_canonical_shap_axis(
            preprocessor,
            raw_feature_order=_features().columns,
        )


def test_one_hot_categories_must_match_fitted_output_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocessor = _fitted()
    encoder = preprocessor.named_transformers_["categorical"].named_steps["one_hot"]
    changed = list(encoder.categories_)
    changed[0] = np.append(changed[0], "support")
    monkeypatch.setattr(encoder, "categories_", changed)
    with pytest.raises(CanonicalShapAxisError, match="width"):
        build_canonical_shap_axis(
            preprocessor,
            raw_feature_order=_features().columns,
        )


def test_unfitted_or_noncanonical_transformers_are_rejected() -> None:
    features = _features()
    with pytest.raises(CanonicalShapAxisError, match="not fitted"):
        build_canonical_shap_axis(
            build_common_preprocessor(features),
            raw_feature_order=features.columns,
        )

    wrong_name = ColumnTransformer(
        [("num", Pipeline([("scale", StandardScaler())]), ["Tenure", "Training"])],
        remainder="drop",
    ).fit(features)
    with pytest.raises(CanonicalShapAxisError, match="Unexpected"):
        build_canonical_shap_axis(
            wrong_name,
            raw_feature_order=features.columns,
        )

    categorical_direct = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["Role", "Travel"],
            )
        ],
        remainder="drop",
    ).fit(features[["Role", "Travel"]])
    with pytest.raises(CanonicalShapAxisError, match="must be a fitted Pipeline"):
        build_canonical_shap_axis(
            categorical_direct,
            raw_feature_order=["Role", "Travel"],
        )


def test_normalize_accepts_all_declared_multiclass_layouts() -> None:
    transformed_count = _axis().n_transformed_features
    expected = np.arange(2 * 3 * transformed_count, dtype=float).reshape(
        2,
        3,
        transformed_count,
    )
    layouts = [
        expected,
        np.transpose(expected, (1, 0, 2)),
        np.transpose(expected, (0, 2, 1)),
        [expected[:, class_index, :] for class_index in range(3)],
        tuple(expected[:, class_index, :] for class_index in range(3)),
    ]
    for raw_values in layouts:
        observed = normalize_multiclass_shap_values(
            raw_values,
            n_samples=2,
            n_classes=3,
            n_transformed_features=transformed_count,
        )
        assert observed.dtype == np.float64
        assert observed.flags.c_contiguous
        np.testing.assert_array_equal(observed, expected)


def test_normalize_rejects_ambiguous_shape_drift_and_nonfinite_values() -> None:
    transformed_count = _axis().n_transformed_features
    with pytest.raises(CanonicalShapAxisError, match="Ambiguous"):
        normalize_multiclass_shap_values(
            np.zeros((3, 3, transformed_count)),
            n_samples=3,
            n_classes=3,
            n_transformed_features=transformed_count,
        )
    with pytest.raises(CanonicalShapAxisError, match="Unexpected"):
        normalize_multiclass_shap_values(
            np.zeros((2, 3, transformed_count - 1)),
            n_samples=2,
            n_classes=3,
            n_transformed_features=transformed_count,
        )
    values = np.zeros((2, 3, transformed_count))
    values[0, 0, 0] = np.nan
    with pytest.raises(CanonicalShapAxisError, match="non-finite"):
        normalize_multiclass_shap_values(
            values,
            n_samples=2,
            n_classes=3,
            n_transformed_features=transformed_count,
        )
    with pytest.raises(CanonicalShapAxisError, match="length differs"):
        normalize_multiclass_shap_values(
            [np.zeros((2, transformed_count))] * 2,
            n_samples=2,
            n_classes=3,
            n_transformed_features=transformed_count,
        )


def test_grouping_follows_raw_order_and_preserves_every_sample_class_sum() -> None:
    contract = _axis()
    normalized = np.arange(
        2 * 3 * contract.n_transformed_features,
        dtype=float,
    ).reshape(2, 3, contract.n_transformed_features)
    grouped = group_canonical_shap_values(normalized, contract)

    assert grouped.shape == (2, 3, 4)
    np.testing.assert_array_equal(grouped[:, :, 0], normalized[:, :, [2, 3]].sum(axis=2))
    np.testing.assert_array_equal(grouped[:, :, 1], normalized[:, :, 0])
    np.testing.assert_array_equal(grouped[:, :, 2], normalized[:, :, [4, 5, 6]].sum(axis=2))
    np.testing.assert_array_equal(grouped[:, :, 3], normalized[:, :, 1])
    np.testing.assert_allclose(
        grouped.sum(axis=2),
        normalized.sum(axis=2),
        rtol=1e-12,
        atol=1e-12,
    )


def test_grouping_rejects_transformed_shape_drift_and_nonfinite_values() -> None:
    contract = _axis()
    with pytest.raises(CanonicalShapAxisError, match="width drifted"):
        group_canonical_shap_values(
            np.zeros((2, 3, contract.n_transformed_features - 1)),
            contract,
        )
    values = np.zeros((2, 3, contract.n_transformed_features))
    values[1, 2, 0] = np.inf
    with pytest.raises(CanonicalShapAxisError, match="non-finite"):
        group_canonical_shap_values(values, contract)
