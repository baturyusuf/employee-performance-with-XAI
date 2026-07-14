from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.data.external_adapters import load_external_dataset
from src.experiments import manuscript_external_evidence as external
from src.governance.manuscript_contract import canonical_config_hash
from src.utils.config_loader import load_config


CONFIG_PATH = Path("configs/manuscript_final.yaml")
ROOT = Path(__file__).resolve().parents[1]
INX_RAW = ROOT / "data/raw/inx_employee_performance.csv"
HR_RAW = ROOT / "data/external/hrdataset_v14/raw.csv"
REAL_INPUTS_AVAILABLE = INX_RAW.is_file() and HR_RAW.is_file()


def test_canonical_external_roles_are_distinct_and_config_backed() -> None:
    config = load_config(CONFIG_PATH)
    specs = (
        *external.configured_run_specs(config, scope="core"),
        *external.configured_run_specs(config, scope="supplementary"),
    )
    by_key = {spec.key: spec for spec in specs}

    assert by_key["hrdataset_v14"].role == "independent external performance-target replication"
    assert by_key["ibm_performance"].task_type == "restricted_target_performance_robustness"
    assert by_key["ibm_attrition"].task_type == "binary_attrition_transfer"
    assert by_key["employee_turnover"].task_type == "binary_turnover_transfer"
    assert all(spec.role != "employee-performance external validation" for spec in specs)


@pytest.mark.skipif(not HR_RAW.is_file(), reason="requires the ignored local HRDataset_v14 dataset")
def test_target_mapping_support_and_transport_infeasibility_are_computed() -> None:
    config = load_config(CONFIG_PATH)
    config_hash = canonical_config_hash(config)
    hr_spec = next(spec for spec in external.RUN_SPECS if spec.key == "hrdataset_v14")
    dataset = load_external_dataset("hrdataset_v14")
    support = external.target_support_table(
        dataset,
        run_id="test-run",
        config_hash=config_hash,
        spec=hr_spec,
        requested_splits=10,
    )
    mapping = external.target_mapping_table(
        dataset,
        run_id="test-run",
        config_hash=config_hash,
        spec=hr_spec,
    )
    _, assessment = external.compute_transport_assessment(
        config,
        run_id="test-run",
        config_hash=config_hash,
    )

    assert support.set_index("label")["n_rows"].to_dict() == {2: 31, 3: 243, 4: 37}
    assert mapping["n_rows"].sum() == len(dataset.canonical)
    assert mapping["mapping_complete"].all()
    assert assessment["status"] == "infeasible_or_too_limited"
    assert assessment["locked_inx_model_transported"] is False
    assert assessment["common_safe_features"] == [
        "EmpJobRole",
        "EmpJobSatisfaction",
        "ExperienceYearsAtThisCompany",
    ]


def test_inapplicable_binary_and_restricted_metrics_cannot_be_numeric() -> None:
    valid = pd.DataFrame(
        [
            {
                "task_type": "binary_attrition_transfer",
                "role": "related HR attrition task transfer",
                "macro_f1": 0.5,
                "severe_error_rate": None,
                "ordinal_mae": None,
                "quadratic_weighted_kappa": None,
                "multiclass_brier": None,
            },
            {
                "task_type": "restricted_target_performance_robustness",
                "role": "restricted-target performance robustness",
                "macro_f1": 0.5,
                "severe_error_rate": None,
                "ordinal_mae": None,
                "quadratic_weighted_kappa": None,
                "multiclass_brier": None,
            },
        ]
    )
    external.validate_task_metric_rows(valid)

    invalid = valid.iloc[[0]].copy()
    invalid.loc[:, "severe_error_rate"] = 0.0
    with pytest.raises(external.ExternalEvidenceError, match="Inapplicable metric"):
        external.validate_task_metric_rows(invalid)


@pytest.mark.skipif(not HR_RAW.is_file(), reason="requires the ignored local HRDataset_v14 dataset")
def test_hr_local_shap_is_fold_matched_to_the_oof_prediction(tmp_path: Path) -> None:
    config = copy.deepcopy(load_config(CONFIG_PATH)["manuscript_final"])
    config["evaluation"]["cv"]["n_splits"] = 2
    config["model"]["xgboost"]["n_estimators"] = 10
    config["model"]["xgboost"]["n_jobs"] = 1
    spec = replace(external.RUN_SPECS[0], policies=("conservative_primary",))
    paths = external._run_dataset_task(
        spec,
        output_dir=tmp_path / "external" / "hrdataset_v14",
        settings=config,
        run_id="oof-shap-test",
        config_hash="a" * 64,
    )

    predictions = pd.read_csv(paths["predictions"])
    local = pd.read_csv(paths["oof_local_shap"])
    assert set(local["evaluation_scope"]) == {"case_specific_oof_fold_model"}
    assert local["prediction_identity_verified"].all()
    observed = (
        local[["sample_index", "fold", "predicted_class"]]
        .drop_duplicates()
        .sort_values("sample_index")
        .reset_index(drop=True)
    )
    expected = (
        predictions[["sample_index", "fold", "y_pred"]]
        .rename(columns={"y_pred": "predicted_class"})
        .sort_values("sample_index")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(observed, expected)
    assert local.groupby(["sample_index", "class_label"])["feature"].nunique().min() > 0
    assert set(local["run_id"]) == {"oof-shap-test"}
    assert set(local["config_hash"]) == {"a" * 64}
    assert "actionability_summary" not in paths
    assert not (tmp_path / "external" / "hrdataset_v14" / "actionability_summary.csv").exists()


@pytest.mark.skipif(not REAL_INPUTS_AVAILABLE, reason="requires ignored local INX and HRDataset inputs")
def test_stage_writes_expected_versioned_layout_and_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(CONFIG_PATH)
    config_hash = canonical_config_hash(config)

    def fake_task(spec, *, output_dir, settings, run_id, config_hash):
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "performance_metrics.csv"
        pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "dataset_key": spec.key,
                    "task_type": spec.task_type,
                    "role": spec.role,
                    "policy": spec.policies[0],
                }
            ]
        ).to_csv(path, index=False)
        return {"policy_summary": path}

    def fake_transport(config, *, run_id, config_hash):
        return (
            pd.DataFrame(
                [
                    {
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "feature": "safe_feature",
                        "in_inx_canonical_primary": True,
                        "in_hrdataset_department_free": True,
                        "common_safe_feature": True,
                    }
                ]
            ),
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "status": "infeasible_or_too_limited",
                "locked_inx_model_transported": False,
                "n_common_safe_features": 1,
                "common_safe_features": ["safe_feature"],
                "minimum_feature_gate": 5,
            },
        )

    monkeypatch.setattr(external, "_run_dataset_task", fake_task)
    monkeypatch.setattr(external, "compute_transport_assessment", fake_transport)
    output = tmp_path / "manuscript-run" / "external"
    paths = external.run(
        CONFIG_PATH,
        scope="core",
        output_dir=output,
        run_id="versioned-test",
        config_hash=config_hash,
    )

    assert paths["external_dataset_roles"] == output / "external_dataset_roles.csv"
    assert all(path.resolve().is_relative_to(output.resolve()) for path in paths.values())
    roles = pd.read_csv(paths["external_dataset_roles"])
    assert set(roles["run_id"]) == {"versioned-test"}
    assert set(roles["config_hash"]) == {config_hash}
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["package_scope"] == "core"
    assert metadata["task_keys"] == ["hrdataset_v14"]
    assert metadata["canonical_dataset_keys_consumed"] == ["inx_primary", "hrdataset_v14"]
    assert metadata["paid_api_calls"] == 0
    assert metadata["locked_inx_model_transported"] is False
