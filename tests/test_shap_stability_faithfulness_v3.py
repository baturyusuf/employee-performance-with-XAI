from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.shap_stability_faithfulness_v3 import (
    _mask_reference,
    _masked_frame,
    _prepare_inputs,
    _resolved_fold_model_path,
    explain_stability_run_v3,
    preflight_shap_stability_faithfulness_v3,
    stability_pairwise_v3,
    stability_summary_v3,
    summarize_faithfulness_v3,
)
from src.governance.shap_stability_faithfulness_contract_v3 import (
    DEFAULT_SHAP_STABILITY_FAITHFULNESS_CONTRACT,
)


def test_mask_reference_uses_training_median_and_deterministic_mode() -> None:
    training = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4],
            "category": ["b", "a", "b", "a"],
        }
    )
    reference = _mask_reference(training)
    assert reference == {"numeric": 2.5, "category": "a"}
    source = pd.DataFrame({"numeric": [10, 20], "category": ["x", "y"]})
    masked = _masked_frame(source, [["numeric"], ["category"]], reference)
    assert masked.to_dict("records") == [
        {"numeric": 2.5, "category": "x"},
        {"numeric": 20, "category": "a"},
    ]


def test_stability_pairwise_and_summary_have_exact_grids() -> None:
    rows = []
    features = [f"f{i:02d}" for i in range(20)]
    for stability_type, runs in (("model_seed", 6), ("outer_train_resample", 5)):
        for run in range(runs):
            order = features[run:] + features[:run]
            for rank, feature in enumerate(order, start=1):
                rows.append(
                    {
                        "stability_type": stability_type,
                        "run_label": f"{stability_type}_{run}",
                        "feature": feature,
                        "rank": rank,
                    }
                )
    pairwise = stability_pairwise_v3(pd.DataFrame(rows))
    assert len(pairwise) == (15 + 10) * 3
    assert pairwise.groupby(["stability_type", "left_run", "right_run"])["all_feature_spearman"].nunique().eq(1).all()
    canonical = pd.DataFrame(
        [
            {
                "top_k": value,
                "n_fold_pairs": 45,
                "jaccard_mean": 0.8,
                "jaccard_std": 0.1,
                "jaccard_median": 0.8,
                "jaccard_min": 0.5,
                "jaccard_max": 1.0,
                "spearman_mean": 0.9,
                "spearman_std": 0.05,
                "spearman_median": 0.9,
                "spearman_min": 0.7,
                "spearman_max": 1.0,
            }
            for value in (5, 10, 15)
        ]
    )
    summary = stability_summary_v3(pairwise, canonical)
    assert len(summary) == 9
    assert set(summary["stability_type"]) == {
        "canonical_outer_fold_pair",
        "model_seed",
        "outer_train_resample",
    }
    assert not summary["pair_independence_assumed"].any()
    assert not summary["confidence_interval_applicable"].any()


def test_faithfulness_summary_separates_guided_and_random_deletion_auc() -> None:
    rows = []
    guided = {1: 0.1, 3: 0.3, 5: 0.5}
    random = {1: 0.02, 3: 0.06, 5: 0.1}
    for sample_index in range(4):
        for count in (1, 3, 5):
            rows.append(
                {
                    "method": "shap_guided",
                    "random_repetition": 0,
                    "sample_index": sample_index,
                    "deleted_feature_count": count,
                    "probability_drop": guided[count],
                    "raw_margin_drop": 2 * guided[count],
                }
            )
            for repetition in (1, 2):
                rows.append(
                    {
                        "method": "random",
                        "random_repetition": repetition,
                        "sample_index": sample_index,
                        "deleted_feature_count": count,
                        "probability_drop": random[count],
                        "raw_margin_drop": 2 * random[count],
                    }
                )
    summary, contrasts, auc_sample, auc_summary, auc_contrast = summarize_faithfulness_v3(pd.DataFrame(rows))
    assert len(summary) == 9
    assert len(contrasts) == 6
    assert len(auc_sample) == 12
    assert len(auc_summary) == 3
    assert auc_contrast.loc[0, "guided_minus_random_mean"] > 0
    assert contrasts["guided_minus_random_mean"].gt(0).all()


@pytest.mark.skipif(
    not DEFAULT_SHAP_STABILITY_FAITHFULNESS_CONTRACT.is_file(),
    reason="Phase 2A contract is unavailable",
)
def test_real_phase2a_preflight_is_fit_free(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.experiments.shap_stability_faithfulness_v3._fit_pipeline",
        lambda *args, **kwargs: pytest.fail("Preflight attempted a fit"),
    )
    receipt = preflight_shap_stability_faithfulness_v3()
    assert receipt["status"] == "passed"
    assert receipt["sample_count"] == 1200
    assert receipt["feature_count"] == 20
    assert receipt["outer_model_count"] == 10
    assert receipt["planned_new_estimator_fit_calls"] == 100
    assert receipt["model_fit_count"] == 0
    assert receipt["network_calls"] == receipt["paid_api_calls"] == 0


def test_canonical_fold_model_paths_resolve_against_benchmark_directory() -> None:
    _, _, _, _, _, _, artifacts, _ = _prepare_inputs(
        DEFAULT_SHAP_STABILITY_FAITHFULNESS_CONTRACT
    )
    for fold_model in artifacts.fold_models.values():
        resolved = _resolved_fold_model_path(artifacts, fold_model)
        assert resolved.is_file()
        assert resolved.parent.name == "xgboost"


def test_exact_reference_grouped_shap_replays_canonical_global_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.experiments.shap_stability_faithfulness_v3._fit_pipeline",
        lambda *args, **kwargs: pytest.fail("Reference replay attempted a fit"),
    )
    contract, _, _, features, excluded, target, artifacts, model_definition = _prepare_inputs(
        DEFAULT_SHAP_STABILITY_FAITHFULNESS_CONTRACT
    )
    result = explain_stability_run_v3(
        stability_type="canonical_reference",
        run_label="canonical_seed_42",
        model_seed=42,
        subsample_seed=None,
        features=features,
        target=target,
        artifacts=artifacts,
        model_definition=model_definition,
        forbidden_features=excluded,
        maximum_additivity_error=1e-5,
        retain_grouped_oof=True,
    )
    canonical = pd.read_csv(contract["source_contracts"]["canonical_v2_shap_global"]["path"])
    observed = result.rankings.set_index("feature")
    expected = canonical.set_index("feature")
    assert len(result.fold_importance) == 200
    assert len(result.additivity_checks) == 10
    assert result.grouped_oof is not None and result.grouped_oof.shape == (1200, 3, 20)
    assert result.additivity_checks["maximum_raw_margin_additivity_error"].max() <= 1e-5
    assert result.additivity_checks["maximum_grouped_sum_error"].max() == 0
    assert np.allclose(
        observed.loc[expected.index, "mean_abs_grouped_shap"],
        expected["mean_abs_grouped_shap"],
        rtol=0.0,
        atol=1e-15,
    )
    assert observed.loc[expected.index, "rank"].astype(int).tolist() == expected["rank"].astype(int).tolist()
