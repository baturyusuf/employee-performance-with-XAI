from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.canonical_loader import load_canonical_dataset
from src.experiments import manuscript_supplementary_external as supplementary
from src.experiments import build_manuscript_evidence as builder
from src.governance.manuscript_contract import canonical_config_hash, sha256_file
from src.utils.config_loader import load_config


CONFIG_PATH = Path("configs/manuscript_final.yaml")
IBM_RAW = Path(__file__).resolve().parents[1] / "data/external/ibm_hr_analytics/raw.csv"


class _DeterministicBinaryPipeline:
    """Small serializable test double; scientific production never imports it."""

    def __init__(self, labels: tuple[int, int] = (3, 4)) -> None:
        self.classes_ = np.asarray(labels, dtype=int)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        numeric = features.select_dtypes(include=[np.number])
        values = numeric.iloc[:, 0].to_numpy(dtype=float)
        midpoint = float(np.median(values)) if len(values) else 0.0
        positive = np.where(values >= midpoint, 0.7, 0.3)
        return np.column_stack([1.0 - positive, positive])


def _side_input_record(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def test_protocol_freezes_three_noncomparable_task_strata_and_task_specific_metrics() -> None:
    config = load_config(CONFIG_PATH)
    protocol = supplementary.validate_protocol(config)

    assert tuple(protocol["tasks"]) == (
        "ibm_performance",
        "ibm_attrition",
        "employee_turnover",
    )
    assert protocol["model_protocol"]["selection_primary_metric"] == "macro_f1"
    assert protocol["model_protocol"]["selection_tie_break_metric"] == "balanced_accuracy"
    assert protocol["model_protocol"]["ordinal_tie_break_metrics_allowed"] is False
    assert protocol["cv"]["outer_splits"] == 10
    assert protocol["cv"]["inner_splits"] == 5
    assert protocol["uncertainty"]["n_resamples"] == 5000
    assert protocol["outputs"]["combined_cross_task_score_table_allowed"] is False
    assert protocol["direct_external_validation_of_primary_allowed"] is False
    assert protocol["locked_model_transport_performed"] is False
    assert protocol["transport_claim_allowed"] is False
    restricted = protocol["metrics"]["restricted_target_performance_robustness"]
    assert {"binary_brier", "roc_auc", "average_precision"}.issubset(restricted)
    assert not {
        "quadratic_weighted_kappa",
        "ordinal_mae",
        "adjacent_accuracy",
        "severe_error_rate",
    }.intersection(restricted)


def test_builder_uses_only_the_v2_task_bounded_runner() -> None:
    source = inspect.getsource(builder._run_external_robustness)
    assert "manuscript_supplementary_external" in source
    assert "manuscript_external_evidence" not in source
    assert "scientific_input_hash" in source
    assert "source_tree_hash" in source
    assert "expected_actual_input_receipts" in source
    assert "expected_side_input_hashes" in source


@pytest.mark.skipif(not IBM_RAW.is_file(), reason="requires the ignored local IBM dataset")
def test_stage_atomically_emits_exact_oof_model_lineage_and_no_combined_task_table(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_config(CONFIG_PATH)
    config_hash = canonical_config_hash(config)
    loaded = load_canonical_dataset(CONFIG_PATH, "ibm_hr_analytics", allow_download=False)
    model_grid = Path("configs/model_grid.yaml")
    acquisition = Path("configs/data_acquisition.yaml")
    mapping = Path("data/external/ibm_hr_analytics/schema_mapping.json")
    expected_side_inputs = {
        "model_search_space": _side_input_record(model_grid),
        "data_acquisition_contract": _side_input_record(acquisition),
        "external_ibm_hr_analytics_schema_mapping": _side_input_record(mapping),
    }

    def fake_fit(*args, **kwargs):
        return _DeterministicBinaryPipeline()

    monkeypatch.setattr(supplementary, "_fit_pipeline", fake_fit)
    output = tmp_path / "external_robustness"
    overrides = supplementary.SupplementaryExternalTestOnlyOverrides(
        candidate_indices=(0,),
        bootstrap_resamples=20,
        task_keys=("ibm_performance",),
    )
    paths = supplementary.run(
        CONFIG_PATH,
        output_dir=output,
        run_id="supplementary-contract-test",
        config_hash=config_hash,
        scientific_input_hash="a" * 64,
        source_tree_hash="b" * 64,
        git_commit="c" * 40,
        scope_contract_hash="d" * 64,
        expected_actual_input_receipts={"ibm_hr_analytics": loaded.receipt},
        expected_side_input_hashes=expected_side_inputs,
        expected_git_worktree_dirty=False,
        test_only_overrides=overrides,
    )

    assert output.is_dir()
    assert not list(tmp_path.glob("external_robustness.__staging__.*"))
    assert paths["restricted_target_source"].is_file()
    assert not (output / "related_binary_task_transfer.csv").exists()
    assert not (output / "ibm_attrition_task_transfer.csv").exists()
    assert not (output / "employee_turnover_task_transfer.csv").exists()

    predictions = pd.read_csv(output / "ibm_performance" / "oof_predictions.csv")
    receipts = pd.read_csv(output / "ibm_performance" / "outer_model_receipts.csv")
    intervals = pd.read_csv(output / "ibm_performance" / "metric_intervals.csv")
    applicability = pd.read_csv(output / "metric_applicability.csv", keep_default_na=False)
    metadata = json.loads((output / "stage_metadata.json").read_text(encoding="utf-8"))

    assert len(predictions) == 1470 * 3
    assert predictions.groupby("policy")["sample_index"].nunique().eq(1470).all()
    assert not predictions.duplicated(["policy", "sample_index"]).any()
    assert len(receipts) == 30
    assert receipts.groupby("policy")["outer_fold"].nunique().eq(10).all()
    assert set(predictions["model_artifact_path"]).issubset(
        set(receipts["model_artifact_path"])
    )
    for row in receipts.itertuples(index=False):
        model_path = output / "ibm_performance" / row.model_artifact_path
        assert model_path.is_file()
        assert sha256_file(model_path) == row.model_sha256
    assert set(intervals["n_resamples"]) == {20}
    assert set(intervals["metric"]) == {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "average_precision",
        "nll_log_loss",
        "binary_brier",
        "ece_confidence",
    }
    ordinal = applicability[
        applicability["metric"].isin(
            ["quadratic_weighted_kappa", "ordinal_mae", "adjacent_accuracy", "severe_error_rate"]
        )
    ]
    assert not ordinal["applicable"].any()
    assert set(ordinal["inapplicable_representation"]) == {"N/A"}
    assert metadata["canonical_eligible"] is False
    assert metadata["direct_external_validation_of_primary"] is False
    assert metadata["locked_model_transport"] is False

    manifest = pd.read_csv(output / "stage_artifact_manifest.csv")
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "stage_artifact_manifest.csv"
    }
    assert set(manifest["path"]) == actual
    assert all(
        hashlib.sha256((output / relative).read_bytes()).hexdigest() == digest
        for relative, digest in zip(manifest["path"], manifest["sha256"])
    )
