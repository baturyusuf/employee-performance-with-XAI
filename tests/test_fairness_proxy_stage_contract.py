from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from src.experiments import manuscript_fairness_proxy as fairness_proxy
from src.governance.manuscript_contract import canonical_config_hash
from src.models import oof_bootstrap


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "manuscript_final.yaml"


def test_run_requires_all_current_run_scientific_upstreams_and_identity() -> None:
    parameters = inspect.signature(fairness_proxy.run).parameters

    assert tuple(parameters) == (
        "config_path",
        "shared_folds_dir",
        "model_benchmarks_dir",
        "policy_ablation_dir",
        "output_dir",
        "run_id",
        "config_hash",
        "scientific_input_hash",
    )
    for name in tuple(parameters)[1:]:
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_stage_has_no_independent_performance_fit_splitter_or_fold_t_inference() -> None:
    source = inspect.getsource(fairness_proxy)
    prohibited = (
        "LabelEncodedXGBClassifier",
        "make_preprocessor",
        "StratifiedKFold",
        "student_t",
        "_mean_t_interval",
        "generate_common_fold_oof_predictions",
        "n_bootstrap_override",
        "bootstrap_iterations",
    )

    for token in prohibited:
        assert token not in source


def test_stage_uses_the_central_paired_resample_implementation() -> None:
    assert (
        fairness_proxy.generate_stratified_resample_indices
        is oof_bootstrap.generate_stratified_resample_indices
    )
    source = inspect.getsource(fairness_proxy)
    assert "def _stratified_bootstrap_indices" not in source
    assert "BootstrapProtocol" in source
    assert "resample_hash" in source


def test_stage_binds_exact_policy_oof_shared_folds_and_benchmark_provenance() -> None:
    source = inspect.getsource(fairness_proxy)
    required_contract_tokens = (
        "read_shared_folds",
        "read_xgboost_oof_artifacts",
        "validate_consumer_fold_assignments",
        "policy_ablation_dir",
        "oof_predictions.csv",
        "bootstrap_metadata.json",
        "fold_contract_hash",
        "model_set_sha256",
        "scientific_input_hash",
    )

    for token in required_contract_tokens:
        assert token in source


def test_proxy_evidence_declares_exactly_once_shared_fold_and_alias_receipts() -> None:
    source = inspect.getsource(fairness_proxy)

    assert "proxy_oof_predictions" in source
    assert "proxy_equivalence" in source
    assert "proxy_target_absent_from_predictors" in source
    assert "outer_fold" in source
    assert "exactly_once" in source
    assert "paired" in source
    assert "proxy_target" in source


def test_populated_output_is_rejected_before_scientific_input_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "subgroup_proxy"
    output.mkdir()
    historical = output / "historical.txt"
    historical.write_text("preserve", encoding="utf-8")

    def unexpected_loader(*args, **kwargs):  # pragma: no cover - failure sentinel
        raise AssertionError("scientific inputs must not be accessed")

    monkeypatch.setattr(fairness_proxy, "load_canonical_dataset", unexpected_loader)

    with pytest.raises(
        fairness_proxy.FairnessProxyError,
        match="absent or an empty builder-owned directory",
    ):
        fairness_proxy.run(
            CONFIG_PATH,
            shared_folds_dir=tmp_path / "shared_folds",
            model_benchmarks_dir=tmp_path / "model_benchmarks",
            policy_ablation_dir=tmp_path / "policy_ablation",
            output_dir=output,
            run_id="unit-test",
            config_hash=canonical_config_hash(CONFIG_PATH),
            scientific_input_hash="b" * 64,
        )

    assert historical.read_text(encoding="utf-8") == "preserve"


def test_stage_source_declares_atomic_publication_and_relative_metadata() -> None:
    source = inspect.getsource(fairness_proxy)

    assert "TemporaryDirectory" in source
    assert ".replace(output)" in source or "os.replace" in source
    assert "relative_to(staging)" in source
