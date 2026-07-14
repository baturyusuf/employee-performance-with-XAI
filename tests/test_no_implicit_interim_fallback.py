from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import src.data.preprocess as preprocess


def test_legacy_loader_never_reads_existing_interim(monkeypatch, tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("raw", encoding="utf-8")
    interim = tmp_path / "interim.csv"
    interim.write_text("stale interim", encoding="utf-8")
    observed = {}
    expected = pd.DataFrame({"PerformanceRating": [2, 3]})

    def fake_load_and_validate_data(*, path, save_interim):
        observed["path"] = Path(path)
        observed["save_interim"] = save_interim
        return expected

    monkeypatch.setattr(
        preprocess,
        "SETTINGS",
        SimpleNamespace(raw_data_path=raw, interim_data_path=interim),
    )
    monkeypatch.setattr(preprocess, "load_and_validate_data", fake_load_and_validate_data)

    result = preprocess.load_validated_or_raw_data()

    assert result is expected
    assert observed == {"path": raw, "save_interim": False}


def test_legacy_loader_source_contains_no_interim_exists_branch() -> None:
    source = inspect.getsource(preprocess.load_validated_or_raw_data)
    assert "interim_data_path.exists" not in source
    assert "save_interim=False" in source


def test_manuscript_stages_use_explicit_canonical_loader() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "experiments"
    stage_files = [
        "manuscript_policy_ablation.py",
        "manuscript_calibration.py",
        "manuscript_shap_evidence.py",
        "manuscript_counterfactual_search.py",
        "manuscript_fairness_proxy.py",
    ]
    for filename in stage_files:
        source = (root / filename).read_text(encoding="utf-8")
        assert "load_validated_or_raw_data" not in source, filename
        assert 'load_canonical_dataset(config_path, "inx_primary")' in source, filename
