from __future__ import annotations

import io
import zlib

import numpy as np

from src.experiments.hrdataset_replication_core import (
    CONDITIONAL_INFERENCE_NOTE,
    FOLD_DESCRIPTIVE_NOTE,
    REPLICATION_METRICS,
)
from src.models.oof_bootstrap import metric_definition


def test_replication_uses_one_paired_sample_level_resample_plan(
    hrdataset_replication_evidence,
) -> None:
    result = hrdataset_replication_evidence["result"]
    frames = (
        result.raw_metric_intervals,
        result.raw_policy_differences,
        result.calibration_metric_intervals,
        result.calibration_differences,
    )
    hashes = set()
    for frame in frames:
        assert not frame.empty
        assert set(frame["n_resamples"]) == {20}
        assert set(frame["n_valid"]) == {20}
        assert set(frame["method"]) == {"paired_stratified_percentile"}
        assert set(frame["strata"]) == {"outer_fold;y_true"}
        assert set(frame["conditional_inference_note"]) == {CONDITIONAL_INFERENCE_NOTE}
        hashes.update(frame["resample_hash"].astype(str))
    assert hashes == {result.protocol_metadata["bootstrap_resample_hash"]}
    assert result.protocol_metadata["bootstrap_strata"] == ["outer_fold", "y_true"]
    assert result.protocol_metadata["conditional_inference_note"] == CONDITIONAL_INFERENCE_NOTE

    plan = result.bootstrap_resample_plan
    receipt = plan.receipt
    assert receipt["resample_hash"] == result.protocol_metadata["bootstrap_resample_hash"]
    assert receipt["sample_order_sha256"] == result.protocol_metadata[
        "bootstrap_sample_order_sha256"
    ]
    assert receipt["compressed_indices_sha256"] == result.protocol_metadata[
        "bootstrap_compressed_indices_sha256"
    ]
    assert receipt["format"] == result.protocol_metadata[
        "bootstrap_resample_plan_format"
    ]
    assert receipt["format"] == "zlib_compressed_numpy_npy_v1"
    assert receipt["shape"] == [20, len(hrdataset_replication_evidence["target"])]
    restored = np.load(
        io.BytesIO(zlib.decompress(plan.compressed_indices_bytes)),
        allow_pickle=False,
    )
    assert restored.shape == (20, len(plan.sample_order))
    assert restored.dtype == np.dtype("<i8")
    assert restored.min() >= 0
    assert restored.max() < len(plan.sample_order)
    assert plan.sample_order["sample_position"].tolist() == list(
        range(len(plan.sample_order))
    )
    assert plan.sample_order["sample_index"].tolist() == sorted(
        hrdataset_replication_evidence["target"].index.tolist()
    )


def test_bootstrap_intervals_obey_metric_domains_and_policy_pairing(
    hrdataset_replication_evidence,
) -> None:
    result = hrdataset_replication_evidence["result"]
    intervals = result.raw_metric_intervals
    assert set(intervals["metric"]) == set(REPLICATION_METRICS)
    assert set(intervals["system_id"]) == {
        "department_free",
        "department_job_role_free",
    }
    for row in intervals.itertuples(index=False):
        definition = metric_definition(row.metric)
        assert row.ci_low <= row.point_estimate <= row.ci_high
        assert row.ci_low >= definition.lower_bound - 1e-12
        if np.isfinite(definition.upper_bound):
            assert row.ci_high <= definition.upper_bound + 1e-12

    differences = result.raw_policy_differences
    assert set(differences["comparison_id"]) == {
        "department_job_role_free_minus_department_free"
    }
    assert set(differences["metric"]) == set(REPLICATION_METRICS)
    assert not differences["primary_gate_comparison"].any()
    assert not differences["gate_eligible"].any()


def test_fold_results_are_descriptive_and_have_no_population_ci(
    hrdataset_replication_evidence,
) -> None:
    result = hrdataset_replication_evidence["result"]
    forbidden_ci_columns = {
        column
        for column in (*result.fold_metrics.columns, *result.fold_descriptive_summary.columns)
        if column.casefold() in {"ci_low", "ci_high"}
        or column.casefold().endswith(("_ci_low", "_ci_high", "_ci95_low", "_ci95_high"))
    }
    assert forbidden_ci_columns == set()
    assert not result.fold_metrics["population_confidence_interval_applicable"].any()
    assert set(result.fold_metrics["interpretation"]) == {FOLD_DESCRIPTIVE_NOTE}
    assert not result.fold_descriptive_summary[
        "population_confidence_interval_applicable"
    ].any()
    assert set(result.fold_descriptive_summary["interpretation"]) == {
        FOLD_DESCRIPTIVE_NOTE
    }
