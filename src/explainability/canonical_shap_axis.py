"""Fail-closed transformed-axis and grouped-SHAP contracts.

The canonical benchmark fits a :class:`~sklearn.compose.ColumnTransformer`
with ``numeric`` and ``categorical`` transformer names.  Numeric columns emit
one transformed value each; categorical columns pass through a nested
``one_hot`` step and emit one value per fitted category.  This module derives
the SHAP grouping axis only from the *fitted* transformer metadata.  It does
not infer lineage by parsing category strings, which can be ambiguous when
feature names or category values contain underscores.

The contract deliberately fails closed when fitted selectors, output slices,
one-hot categories, feature-name lineage, transformed matrices, or SHAP axes
disagree.  Grouped values retain the declared raw-column order and are
accepted only when every transformed index is owned exactly once and the
per-sample/per-class SHAP sums are preserved within a tight float64 tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


GROUP_SUM_ATOL = 1e-12
GROUP_SUM_RTOL = 1e-12
_CANONICAL_TRANSFORMERS = frozenset({"numeric", "categorical"})


class CanonicalShapAxisError(ValueError):
    """Raised when fitted preprocessing and SHAP axes are incompatible."""


@dataclass(frozen=True)
class CanonicalShapAxisContract:
    """Immutable fitted lineage used to group transformed SHAP values.

    ``transformed_indices_by_raw`` is ordered by ``raw_feature_names`` when
    constructed by :func:`build_canonical_shap_axis`.  Mapping values contain
    zero-based transformed-column positions, never names reconstructed from
    category text.
    """

    raw_feature_names: tuple[str, ...]
    transformed_feature_names: tuple[str, ...]
    transformed_indices_by_raw: Mapping[str, tuple[int, ...]]
    transformed_to_raw: tuple[str, ...]
    numeric_feature_names: tuple[str, ...]
    categorical_feature_names: tuple[str, ...]

    @property
    def n_raw_features(self) -> int:
        return len(self.raw_feature_names)

    @property
    def n_transformed_features(self) -> int:
        return len(self.transformed_feature_names)

    def validate_transformed_matrix(
        self,
        transformed: Any,
        *,
        n_samples: int | None = None,
    ) -> None:
        """Validate the realized transformed matrix against fitted lineage.

        Named pandas output is checked column-for-column.  A numeric array is
        also accepted because SHAP implementations commonly consume NumPy;
        in that case the already-validated fitted lineage still governs the
        axis and the exact width is mandatory.
        """

        _validate_contract_integrity(self)
        shape = getattr(transformed, "shape", None)
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise CanonicalShapAxisError(
                "Transformed model input must be a two-dimensional matrix."
            )
        if shape[1] != self.n_transformed_features:
            raise CanonicalShapAxisError(
                "Transformed feature-axis width drifted from fitted lineage: "
                f"observed={shape[1]}, expected={self.n_transformed_features}."
            )
        if n_samples is not None:
            expected_samples = _positive_count(n_samples, "n_samples")
            if shape[0] != expected_samples:
                raise CanonicalShapAxisError(
                    "Transformed sample-axis length drifted: "
                    f"observed={shape[0]}, expected={expected_samples}."
                )
        if isinstance(transformed, pd.DataFrame):
            observed_names = _strict_names(
                transformed.columns,
                label="transformed matrix columns",
                allow_empty=False,
            )
            if observed_names != self.transformed_feature_names:
                raise CanonicalShapAxisError(
                    "Transformed matrix columns do not equal fitted "
                    "get_feature_names_out lineage."
                )


def _positive_count(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise CanonicalShapAxisError(f"{label} must be a positive integer.")
    count = int(value)
    if count <= 0:
        raise CanonicalShapAxisError(f"{label} must be a positive integer.")
    return count


def _strict_names(
    values: Sequence[Any],
    *,
    label: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    try:
        materialized = list(values)
    except TypeError as exc:
        raise CanonicalShapAxisError(f"{label} must be an ordered sequence.") from exc
    if not allow_empty and not materialized:
        raise CanonicalShapAxisError(f"{label} must not be empty.")
    non_strings = [repr(value) for value in materialized if not isinstance(value, str)]
    if non_strings:
        raise CanonicalShapAxisError(f"{label} must contain only strings: {non_strings}.")
    names = tuple(str(value) for value in materialized)
    blanks = [repr(value) for value in names if not value.strip()]
    if blanks:
        raise CanonicalShapAxisError(f"{label} contains blank names: {blanks}.")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise CanonicalShapAxisError(f"{label} contains duplicate names: {duplicates}.")
    return names


def _forbidden_name_set(forbidden_features: Sequence[str]) -> frozenset[str]:
    names = _strict_names(
        forbidden_features,
        label="forbidden feature names",
        allow_empty=True,
    )
    casefolded = [name.casefold() for name in names]
    duplicates = sorted({name for name in casefolded if casefolded.count(name) > 1})
    if duplicates:
        raise CanonicalShapAxisError(
            "Forbidden feature names must be unique under case folding: "
            f"{duplicates}."
        )
    return frozenset(casefolded)


def _transformed_forbidden_names(
    transformed_names: Sequence[str],
    forbidden: frozenset[str],
) -> list[str]:
    """Find transformed columns whose raw-family prefix is forbidden.

    For canonical names, text after ``<transformer>__`` is either the numeric
    raw name or ``<raw_name>_<category>``.  This check is defense in depth; the
    authoritative ownership check still comes from fitted selectors and
    ``categories_`` rather than parsing these strings.
    """

    matches: list[str] = []
    for transformed_name in transformed_names:
        local_name = transformed_name.split("__", 1)[-1].casefold()
        if any(
            local_name == forbidden_name
            or local_name.startswith(f"{forbidden_name}_")
            for forbidden_name in forbidden
        ):
            matches.append(transformed_name)
    return sorted(matches)


def _selector_names(selector: Any, *, transformer_name: str) -> tuple[str, ...]:
    if isinstance(selector, (str, bytes, slice)) or callable(selector):
        raise CanonicalShapAxisError(
            f"Transformer {transformer_name!r} must store an explicit fitted "
            "sequence of raw feature names."
        )
    if isinstance(selector, np.ndarray) and selector.dtype == bool:
        raise CanonicalShapAxisError(
            f"Transformer {transformer_name!r} cannot use a boolean selector."
        )
    return _strict_names(
        selector,
        label=f"{transformer_name} fitted selector",
        allow_empty=False,
    )


def _slice_positions(value: Any, *, n_transformed: int, label: str) -> tuple[int, ...]:
    if not isinstance(value, slice):
        raise CanonicalShapAxisError(f"{label} must be a fitted output slice.")
    if value.step not in (None, 1):
        raise CanonicalShapAxisError(f"{label} must have unit stride.")
    start = 0 if value.start is None else value.start
    stop = n_transformed if value.stop is None else value.stop
    if (
        isinstance(start, (bool, np.bool_))
        or isinstance(stop, (bool, np.bool_))
        or not isinstance(start, (int, np.integer))
        or not isinstance(stop, (int, np.integer))
    ):
        raise CanonicalShapAxisError(f"{label} bounds must be integers.")
    start_int, stop_int = int(start), int(stop)
    if start_int < 0 or stop_int < start_int or stop_int > n_transformed:
        raise CanonicalShapAxisError(
            f"{label} bounds [{start_int}, {stop_int}) fall outside transformed "
            f"width {n_transformed}."
        )
    return tuple(range(start_int, stop_int))


def _local_feature_names(
    transformer: Pipeline,
    columns: tuple[str, ...],
    *,
    transformer_name: str,
) -> tuple[str, ...]:
    try:
        values = transformer.get_feature_names_out(list(columns))
    except Exception as exc:  # sklearn raises several fitted/schema error types
        raise CanonicalShapAxisError(
            f"Transformer {transformer_name!r} cannot provide fitted feature-name lineage."
        ) from exc
    return _strict_names(
        values,
        label=f"{transformer_name} get_feature_names_out",
        allow_empty=False,
    )


def _categorical_lineage(
    transformer: Pipeline,
    columns: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if "one_hot" not in transformer.named_steps:
        raise CanonicalShapAxisError(
            "The canonical categorical pipeline requires a nested 'one_hot' step."
        )
    if tuple(transformer.named_steps)[-1] != "one_hot":
        raise CanonicalShapAxisError(
            "The canonical categorical 'one_hot' step must be the final pipeline step."
        )
    encoder = transformer.named_steps["one_hot"]
    if not isinstance(encoder, OneHotEncoder):
        raise CanonicalShapAxisError(
            "The canonical categorical 'one_hot' step must be sklearn OneHotEncoder."
        )
    if not hasattr(encoder, "categories_"):
        raise CanonicalShapAxisError("The nested one_hot encoder is not fitted.")
    categories = tuple(encoder.categories_)
    if len(categories) != len(columns):
        raise CanonicalShapAxisError(
            "one_hot categories_ does not align one-to-one with categorical selectors."
        )
    category_widths: list[int] = []
    for column, values in zip(columns, categories):
        category_values = np.asarray(values)
        if category_values.ndim != 1 or len(category_values) == 0:
            raise CanonicalShapAxisError(
                f"one_hot categories_ for {column!r} must be a non-empty vector."
            )
        category_widths.append(int(len(category_values)))
    if getattr(encoder, "drop", None) is not None or getattr(encoder, "drop_idx_", None) is not None:
        raise CanonicalShapAxisError(
            "Canonical grouped SHAP requires one_hot to retain every fitted category."
        )
    encoder_inputs = _strict_names(
        getattr(encoder, "feature_names_in_", ()),
        label="one_hot feature_names_in_",
        allow_empty=False,
    )
    if encoder_inputs != columns:
        raise CanonicalShapAxisError(
            "one_hot feature_names_in_ differs from the fitted categorical selector."
        )
    try:
        encoder_names = _strict_names(
            encoder.get_feature_names_out(list(columns)),
            label="one_hot get_feature_names_out",
            allow_empty=False,
        )
    except CanonicalShapAxisError:
        raise
    except Exception as exc:
        raise CanonicalShapAxisError(
            "The fitted one_hot encoder cannot provide feature-name lineage."
        ) from exc
    pipeline_names = _local_feature_names(
        transformer,
        columns,
        transformer_name="categorical",
    )
    if pipeline_names != encoder_names:
        raise CanonicalShapAxisError(
            "Categorical pipeline lineage differs from nested one_hot lineage."
        )
    expected_width = sum(category_widths)
    if len(encoder_names) != expected_width:
        raise CanonicalShapAxisError(
            "one_hot categories_ width differs from get_feature_names_out width: "
            f"categories={expected_width}, names={len(encoder_names)}."
        )
    return encoder_names, tuple(category_widths)


def build_canonical_shap_axis(
    preprocessor: ColumnTransformer,
    *,
    raw_feature_order: Sequence[str],
    forbidden_features: Sequence[str] = (),
) -> CanonicalShapAxisContract:
    """Build and validate raw-family ownership for a fitted preprocessor."""

    if not isinstance(preprocessor, ColumnTransformer):
        raise CanonicalShapAxisError(
            "Canonical SHAP lineage requires a fitted sklearn ColumnTransformer."
        )
    if not hasattr(preprocessor, "transformers_") or not hasattr(preprocessor, "output_indices_"):
        raise CanonicalShapAxisError("Canonical SHAP preprocessor is not fitted.")

    raw_names = _strict_names(
        raw_feature_order,
        label="canonical raw feature order",
        allow_empty=False,
    )
    fitted_inputs = _strict_names(
        getattr(preprocessor, "feature_names_in_", ()),
        label="ColumnTransformer feature_names_in_",
        allow_empty=False,
    )
    if fitted_inputs != raw_names:
        raise CanonicalShapAxisError(
            "Canonical raw feature order must equal fitted ColumnTransformer feature_names_in_."
        )

    forbidden = _forbidden_name_set(forbidden_features)
    forbidden_raw = sorted(name for name in raw_names if name.casefold() in forbidden)
    if forbidden_raw:
        raise CanonicalShapAxisError(
            f"Forbidden raw features entered the canonical SHAP axis: {forbidden_raw}."
        )

    try:
        transformed_names = _strict_names(
            preprocessor.get_feature_names_out(),
            label="ColumnTransformer get_feature_names_out",
            allow_empty=False,
        )
    except CanonicalShapAxisError:
        raise
    except Exception as exc:
        raise CanonicalShapAxisError(
            "Fitted ColumnTransformer cannot provide feature-name lineage."
        ) from exc
    forbidden_transformed = _transformed_forbidden_names(transformed_names, forbidden)
    if forbidden_transformed:
        raise CanonicalShapAxisError(
            "Forbidden transformed feature names entered the canonical SHAP axis: "
            f"{forbidden_transformed}."
        )

    n_transformed = len(transformed_names)
    output_indices = preprocessor.output_indices_
    if not isinstance(output_indices, Mapping):
        raise CanonicalShapAxisError("ColumnTransformer output_indices_ must be a mapping.")
    unknown_output_keys = sorted(set(map(str, output_indices)).difference(_CANONICAL_TRANSFORMERS | {"remainder"}))
    if unknown_output_keys:
        raise CanonicalShapAxisError(
            f"Unexpected ColumnTransformer output_indices_ keys: {unknown_output_keys}."
        )
    if "remainder" in output_indices:
        remainder_positions = _slice_positions(
            output_indices["remainder"],
            n_transformed=n_transformed,
            label="remainder output_indices_",
        )
        if remainder_positions:
            raise CanonicalShapAxisError(
                "Canonical SHAP does not permit transformed remainder columns."
            )

    selector_owners: dict[str, str] = {}
    indices_by_raw: dict[str, list[int]] = {name: [] for name in raw_names}
    expected_transformed_names: list[str | None] = [None] * n_transformed
    transformed_owner: list[str | None] = [None] * n_transformed
    numeric_names: tuple[str, ...] = ()
    categorical_names: tuple[str, ...] = ()
    seen_transformers: set[str] = set()

    for record in preprocessor.transformers_:
        if not isinstance(record, tuple) or len(record) != 3:
            raise CanonicalShapAxisError("Malformed fitted ColumnTransformer record.")
        transformer_name, transformer, selector = record
        if transformer_name == "remainder":
            remainder_columns = list(selector) if not isinstance(selector, slice) else [selector]
            if remainder_columns:
                raise CanonicalShapAxisError(
                    "Canonical selectors must cover every raw feature without remainder."
                )
            continue
        if transformer_name not in _CANONICAL_TRANSFORMERS:
            raise CanonicalShapAxisError(
                f"Unexpected fitted transformer {transformer_name!r}; expected numeric/categorical."
            )
        if transformer_name in seen_transformers:
            raise CanonicalShapAxisError(
                f"Duplicate fitted transformer name {transformer_name!r}."
            )
        seen_transformers.add(transformer_name)
        if not isinstance(transformer, Pipeline):
            raise CanonicalShapAxisError(
                f"Canonical transformer {transformer_name!r} must be a fitted Pipeline."
            )
        columns = _selector_names(selector, transformer_name=transformer_name)
        for column in columns:
            if column not in indices_by_raw:
                raise CanonicalShapAxisError(
                    f"Transformer {transformer_name!r} selected unknown raw feature {column!r}."
                )
            if column in selector_owners:
                raise CanonicalShapAxisError(
                    f"Raw feature {column!r} appears in multiple fitted transformers."
                )
            selector_owners[column] = transformer_name
        if transformer_name not in output_indices:
            raise CanonicalShapAxisError(
                f"Missing output_indices_ for transformer {transformer_name!r}."
            )
        positions = _slice_positions(
            output_indices[transformer_name],
            n_transformed=n_transformed,
            label=f"{transformer_name} output_indices_",
        )

        if transformer_name == "numeric":
            numeric_names = columns
            local_names = _local_feature_names(
                transformer,
                columns,
                transformer_name="numeric",
            )
            if local_names != columns:
                raise CanonicalShapAxisError(
                    "Numeric pipeline must emit exactly one column with unchanged lineage "
                    "for every numeric raw feature."
                )
            widths = (1,) * len(columns)
        else:
            categorical_names = columns
            local_names, widths = _categorical_lineage(transformer, columns)

        if len(positions) != len(local_names):
            raise CanonicalShapAxisError(
                f"{transformer_name} output_indices_ width differs from fitted "
                f"feature-name width: slice={len(positions)}, names={len(local_names)}."
            )
        offset = 0
        for column, width in zip(columns, widths):
            column_positions = positions[offset : offset + width]
            if len(column_positions) != width:
                raise CanonicalShapAxisError(
                    f"Transformed axis ended inside raw feature family {column!r}."
                )
            indices_by_raw[column].extend(column_positions)
            for position in column_positions:
                if transformed_owner[position] is not None:
                    raise CanonicalShapAxisError(
                        "Transformed output index overlap between raw feature families: "
                        f"index={position}, owners={transformed_owner[position]!r}/{column!r}."
                    )
                transformed_owner[position] = column
            offset += width
        if offset != len(positions):
            raise CanonicalShapAxisError(
                f"{transformer_name} transformed positions were not completely assigned."
            )
        for position, local_name in zip(positions, local_names):
            expected_name = f"{transformer_name}__{local_name}"
            if expected_transformed_names[position] is not None:
                raise CanonicalShapAxisError(
                    f"Transformed output index {position} received overlapping lineage."
                )
            expected_transformed_names[position] = expected_name

    missing_raw = [name for name in raw_names if name not in selector_owners]
    if missing_raw:
        raise CanonicalShapAxisError(
            f"Raw feature families are missing from fitted transformers: {missing_raw}."
        )
    missing_positions = [index for index, owner in enumerate(transformed_owner) if owner is None]
    if missing_positions:
        raise CanonicalShapAxisError(
            f"Transformed indices are not owned exactly once; gaps={missing_positions}."
        )
    if tuple(expected_transformed_names) != transformed_names:
        raise CanonicalShapAxisError(
            "ColumnTransformer get_feature_names_out does not equal lineage reconstructed "
            "from transformers_, output_indices_, and one_hot categories_."
        )

    ordered_mapping = {
        name: tuple(indices_by_raw[name])
        for name in raw_names
    }
    contract = CanonicalShapAxisContract(
        raw_feature_names=raw_names,
        transformed_feature_names=transformed_names,
        transformed_indices_by_raw=MappingProxyType(ordered_mapping),
        transformed_to_raw=tuple(str(owner) for owner in transformed_owner),
        numeric_feature_names=numeric_names,
        categorical_feature_names=categorical_names,
    )
    _validate_contract_integrity(contract)
    return contract


def _validate_contract_integrity(contract: CanonicalShapAxisContract) -> None:
    if not isinstance(contract, CanonicalShapAxisContract):
        raise CanonicalShapAxisError(
            "Grouped SHAP requires a CanonicalShapAxisContract instance."
        )
    raw = contract.raw_feature_names
    transformed = contract.transformed_feature_names
    if not raw or len(raw) != len(set(raw)):
        raise CanonicalShapAxisError("Canonical SHAP contract has invalid raw feature names.")
    if not transformed or len(transformed) != len(set(transformed)):
        raise CanonicalShapAxisError(
            "Canonical SHAP contract has invalid transformed feature names."
        )
    if tuple(contract.transformed_indices_by_raw) != raw:
        raise CanonicalShapAxisError(
            "Canonical SHAP grouping map does not preserve raw feature order."
        )
    if len(contract.transformed_to_raw) != len(transformed):
        raise CanonicalShapAxisError("Canonical SHAP transformed ownership width drifted.")
    observed_indices: list[int] = []
    expected_owners: list[str | None] = [None] * len(transformed)
    for feature in raw:
        indices = contract.transformed_indices_by_raw[feature]
        if not isinstance(indices, tuple) or not indices:
            raise CanonicalShapAxisError(
                f"Raw feature {feature!r} has no immutable transformed indices."
            )
        if tuple(sorted(indices)) != indices or len(indices) != len(set(indices)):
            raise CanonicalShapAxisError(
                f"Raw feature {feature!r} has invalid transformed-index ordering."
            )
        for index in indices:
            if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
                raise CanonicalShapAxisError("Transformed group indices must be integers.")
            position = int(index)
            if position < 0 or position >= len(transformed):
                raise CanonicalShapAxisError(
                    f"Transformed group index {position} is outside the fitted axis."
                )
            if expected_owners[position] is not None:
                raise CanonicalShapAxisError(
                    f"Transformed group index {position} appears in more than one raw family."
                )
            expected_owners[position] = feature
            observed_indices.append(position)
    if sorted(observed_indices) != list(range(len(transformed))):
        raise CanonicalShapAxisError(
            "Canonical SHAP contract does not map every transformed index exactly once."
        )
    if tuple(expected_owners) != contract.transformed_to_raw:
        raise CanonicalShapAxisError(
            "Canonical SHAP transformed-to-raw ownership is internally inconsistent."
        )
    numeric = contract.numeric_feature_names
    categorical = contract.categorical_feature_names
    if set(numeric).intersection(categorical) or set(numeric).union(categorical) != set(raw):
        raise CanonicalShapAxisError(
            "Numeric and categorical contract families must partition the raw features."
        )
    if tuple(name for name in raw if name in set(numeric)) != numeric:
        raise CanonicalShapAxisError("Numeric feature lineage does not preserve raw order.")
    if tuple(name for name in raw if name in set(categorical)) != categorical:
        raise CanonicalShapAxisError("Categorical feature lineage does not preserve raw order.")


def validate_canonical_transformed_matrix(
    transformed: Any,
    axis_contract: CanonicalShapAxisContract,
    *,
    n_samples: int | None = None,
) -> None:
    """Functional wrapper for transformed-matrix contract validation."""

    axis_contract.validate_transformed_matrix(transformed, n_samples=n_samples)


def _numeric_array(values: Any, *, label: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        raise CanonicalShapAxisError(f"{label} must contain numeric values.")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise CanonicalShapAxisError(f"{label} contains non-finite values.")
    return result


def normalize_multiclass_shap_values(
    raw_values: Any,
    *,
    n_samples: int,
    n_classes: int,
    n_transformed_features: int,
) -> np.ndarray:
    """Normalize accepted multiclass SHAP layouts to sample/class/feature.

    Accepted layouts are a class-separated list/tuple of ``(samples,
    transformed)`` matrices or one of these three arrays:

    - ``(samples, classes, transformed)``;
    - ``(classes, samples, transformed)``;
    - ``(samples, transformed, classes)``.

    If dimensions make multiple array layouts possible, the array is rejected
    rather than silently choosing an axis interpretation.
    """

    sample_count = _positive_count(n_samples, "n_samples")
    class_count = _positive_count(n_classes, "n_classes")
    transformed_count = _positive_count(
        n_transformed_features,
        "n_transformed_features",
    )
    if class_count < 2:
        raise CanonicalShapAxisError(
            "Multiclass SHAP normalization requires at least two classes."
        )

    if isinstance(raw_values, (list, tuple)):
        if len(raw_values) != class_count:
            raise CanonicalShapAxisError(
                "Class-separated SHAP output length differs from n_classes: "
                f"observed={len(raw_values)}, expected={class_count}."
            )
        class_arrays: list[np.ndarray] = []
        expected_shape = (sample_count, transformed_count)
        for class_index, values in enumerate(raw_values):
            class_array = _numeric_array(
                values,
                label=f"SHAP values for class index {class_index}",
            )
            if class_array.shape != expected_shape:
                raise CanonicalShapAxisError(
                    "Class-separated SHAP matrix shape drifted: "
                    f"class={class_index}, observed={class_array.shape}, "
                    f"expected={expected_shape}."
                )
            class_arrays.append(class_array)
        normalized = np.stack(class_arrays, axis=1)
    else:
        array = _numeric_array(raw_values, label="SHAP values")
        if array.ndim != 3:
            raise CanonicalShapAxisError(
                f"Multiclass SHAP array must be three-dimensional; observed={array.shape}."
            )
        target_shape = (sample_count, class_count, transformed_count)
        layouts: list[tuple[str, tuple[int, int, int]]] = []
        if array.shape == target_shape:
            layouts.append(("samples_classes_transformed", (0, 1, 2)))
        if array.shape == (class_count, sample_count, transformed_count):
            layouts.append(("classes_samples_transformed", (1, 0, 2)))
        if array.shape == (sample_count, transformed_count, class_count):
            layouts.append(("samples_transformed_classes", (0, 2, 1)))
        if not layouts:
            raise CanonicalShapAxisError(
                "Unexpected multiclass SHAP shape: "
                f"observed={array.shape}, expected dimensions "
                f"samples={sample_count}, classes={class_count}, "
                f"transformed={transformed_count}."
            )
        unique_permutations = {permutation for _, permutation in layouts}
        if len(unique_permutations) != 1:
            names = [name for name, _ in layouts]
            raise CanonicalShapAxisError(
                f"Ambiguous multiclass SHAP axis layout {array.shape}: {names}."
            )
        permutation = layouts[0][1]
        normalized = np.transpose(array, permutation)

    normalized = np.ascontiguousarray(normalized, dtype=np.float64)
    expected_normalized_shape = (sample_count, class_count, transformed_count)
    if normalized.shape != expected_normalized_shape:
        raise CanonicalShapAxisError(
            "Internal SHAP normalization shape mismatch: "
            f"observed={normalized.shape}, expected={expected_normalized_shape}."
        )
    if not np.all(np.isfinite(normalized)):
        raise CanonicalShapAxisError("Normalized SHAP values contain non-finite values.")
    return normalized


def group_canonical_shap_values(
    shap_values: Any,
    axis_contract: CanonicalShapAxisContract,
    *,
    atol: float = GROUP_SUM_ATOL,
    rtol: float = GROUP_SUM_RTOL,
) -> np.ndarray:
    """Sum transformed SHAP values into canonical raw feature families."""

    _validate_contract_integrity(axis_contract)
    if not isinstance(atol, (int, float, np.integer, np.floating)) or not np.isfinite(atol) or atol < 0:
        raise CanonicalShapAxisError("Grouped-SHAP atol must be a finite non-negative number.")
    if not isinstance(rtol, (int, float, np.integer, np.floating)) or not np.isfinite(rtol) or rtol < 0:
        raise CanonicalShapAxisError("Grouped-SHAP rtol must be a finite non-negative number.")
    values = _numeric_array(shap_values, label="Normalized SHAP values")
    if values.ndim != 3:
        raise CanonicalShapAxisError(
            "Normalized SHAP values must have shape (samples, classes, transformed)."
        )
    if values.shape[0] == 0 or values.shape[1] < 2:
        raise CanonicalShapAxisError(
            "Grouped multiclass SHAP requires samples and at least two classes."
        )
    if values.shape[2] != axis_contract.n_transformed_features:
        raise CanonicalShapAxisError(
            "SHAP transformed-feature width drifted from fitted lineage: "
            f"observed={values.shape[2]}, "
            f"expected={axis_contract.n_transformed_features}."
        )

    grouped = np.empty(
        (values.shape[0], values.shape[1], axis_contract.n_raw_features),
        dtype=np.float64,
    )
    for raw_index, feature in enumerate(axis_contract.raw_feature_names):
        transformed_indices = axis_contract.transformed_indices_by_raw[feature]
        grouped[:, :, raw_index] = np.sum(
            values[:, :, transformed_indices],
            axis=2,
            dtype=np.float64,
        )

    transformed_totals = np.sum(values, axis=2, dtype=np.float64)
    grouped_totals = np.sum(grouped, axis=2, dtype=np.float64)
    if not np.allclose(
        transformed_totals,
        grouped_totals,
        rtol=float(rtol),
        atol=float(atol),
        equal_nan=False,
    ):
        maximum_error = float(np.max(np.abs(transformed_totals - grouped_totals)))
        raise CanonicalShapAxisError(
            "Grouped SHAP failed per-sample/per-class sum preservation: "
            f"maximum_absolute_error={maximum_error:.17g}, "
            f"atol={float(atol):.3g}, rtol={float(rtol):.3g}."
        )
    return grouped


__all__ = [
    "GROUP_SUM_ATOL",
    "GROUP_SUM_RTOL",
    "CanonicalShapAxisContract",
    "CanonicalShapAxisError",
    "build_canonical_shap_axis",
    "group_canonical_shap_values",
    "normalize_multiclass_shap_values",
    "validate_canonical_transformed_matrix",
]
