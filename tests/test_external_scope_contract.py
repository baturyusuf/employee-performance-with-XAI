from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.experiments import manuscript_external_evidence as external
from src.governance.manuscript_contract import canonical_config_hash
from src.utils.config_loader import load_config


CONFIG_PATH = Path("configs/manuscript_final.yaml")


def test_external_scopes_are_immutable_exact_disjoint_and_exhaustive() -> None:
    assert external.EXTERNAL_SCOPE_TASK_KEYS["core"] == ("hrdataset_v14",)
    assert external.EXTERNAL_SCOPE_TASK_KEYS["supplementary"] == (
        "ibm_performance",
        "ibm_attrition",
        "employee_turnover",
    )
    core = {spec.key for spec in external.specs_for_scope("core")}
    supplementary = {spec.key for spec in external.specs_for_scope("supplementary")}
    assert core.isdisjoint(supplementary)
    assert core | supplementary == {spec.key for spec in external.RUN_SPECS}
    with pytest.raises(TypeError):
        external.EXTERNAL_SCOPE_TASK_KEYS["core"] = ("ibm_performance",)  # type: ignore[index]
    for prohibited in ("all", "hrdataset_v14", "", "CORE"):
        with pytest.raises(external.ExternalEvidenceError, match="scope"):
            external.specs_for_scope(prohibited)


def test_canonical_external_run_requires_an_explicit_scope(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="scope"):
        external.run(  # type: ignore[call-arg]
            CONFIG_PATH,
            output_dir=tmp_path / "external",
            run_id="missing-scope",
        )


@pytest.mark.parametrize("scope", ["core", "supplementary"])
def test_scoped_run_emits_no_cross_scope_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    config = load_config(CONFIG_PATH)
    config_hash = canonical_config_hash(config)
    binding_calls: list[tuple[list[str], bool]] = []
    transport_calls: list[str] = []

    def fake_bind(config_path, settings, *, preflight_dir, specs, include_inx_primary):
        binding_calls.append(([spec.key for spec in specs], include_inx_primary))
        runtime = dict(settings)
        runtime[external._CANONICAL_EXTERNAL_INPUTS_KEY] = {}
        return runtime

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
        transport_calls.append(run_id)
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

    monkeypatch.setattr(external, "_bind_canonical_external_inputs", fake_bind)
    monkeypatch.setattr(external, "_run_dataset_task", fake_task)
    monkeypatch.setattr(external, "compute_transport_assessment", fake_transport)

    output = tmp_path / scope
    paths = external.run(
        CONFIG_PATH,
        scope=scope,
        output_dir=output,
        run_id=f"{scope}-scope-test",
        config_hash=config_hash,
    )
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    roles = pd.read_csv(paths["external_dataset_roles"])
    all_relative_paths = {
        path.relative_to(output).as_posix().lower()
        for path in output.rglob("*")
        if path.is_file()
    }
    all_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore").lower()
        for path in output.rglob("*")
        if path.is_file()
    )

    assert metadata["package_scope"] == scope
    assert "actionability_summary.csv" not in all_relative_paths
    assert "actionability_status" not in all_text
    if scope == "core":
        assert binding_calls == [(["hrdataset_v14"], True)]
        assert transport_calls == ["core-scope-test"]
        assert metadata["task_keys"] == ["hrdataset_v14"]
        assert metadata["canonical_dataset_keys_consumed"] == ["inx_primary", "hrdataset_v14"]
        assert set(roles["dataset_key"]) == {"hrdataset_v14"}
        assert "performance_target_replication.csv" in all_relative_paths
        assert "cross_dataset_transport/transport_feasibility.json" in all_relative_paths
        assert not any("ibm" in path or "turnover" in path for path in all_relative_paths)
        assert "restricted_target_robustness.csv" not in all_relative_paths
        assert "related_binary_task_transfer.csv" not in all_relative_paths
        assert "ibm performancerating" not in all_text
        assert "employee turnover" not in all_text
    else:
        assert binding_calls == [(["ibm_performance", "ibm_attrition", "employee_turnover"], False)]
        assert transport_calls == []
        assert metadata["task_keys"] == ["ibm_performance", "ibm_attrition", "employee_turnover"]
        assert metadata["canonical_dataset_keys_consumed"] == [
            "ibm_hr_analytics",
            "ibm_hr_analytics_attrition",
            "employee_turnover",
        ]
        assert set(roles["dataset_key"]) == {"ibm_performance", "ibm_attrition", "employee_turnover"}
        assert "restricted_target_robustness.csv" in all_relative_paths
        assert "related_binary_task_transfer.csv" in all_relative_paths
        assert "performance_target_replication.csv" not in all_relative_paths
        assert not any(path.startswith("hrdataset_v14/") for path in all_relative_paths)
        assert not any(path.startswith("cross_dataset_transport/") for path in all_relative_paths)
        assert "hrdataset_v14" not in all_text
