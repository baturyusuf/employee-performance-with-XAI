import json
from pathlib import Path
import pandas as pd
import pytest
from src.experiments.eswa_repeated_selection_objective_v1 import build_schedule, validate_contract
from src.experiments.repeated_nested_cv_v3 import TUNED_MODEL_NAMES
from src.governance.repeated_nested_cv_run_validator_v3 import validate_repeated_nested_cv_run_v3

SOURCE=Path("reports/major_revision_v3_runs/phase1c_v3_20260903T215015Z_78649c4/repeated_nested_cv")
pytestmark=pytest.mark.skipif(not SOURCE.is_dir(),reason="local row-level Phase 1C run is intentionally ignored")

def test_contract_reuses_exact_phase1c_sources_and_registries():
    contract,hashes=validate_contract()
    assert contract["models"]==list(TUNED_MODEL_NAMES)
    assert contract["selection_regimes"]["macro_f1"]["practical_tie_tolerance"]==0.001
    assert contract["selection_regimes"]["qwk"]["practical_tie_tolerance"]==0.001
    assert len(hashes)==5

def test_schedule_has_both_objectives_and_no_outer_test_selection():
    schedule,changes=build_schedule(pd.read_csv(SOURCE/"candidate_search_results.csv"))
    assert len(schedule)==300 and len(changes)==150
    assert set(schedule["selection_objective"])=={"macro_f1","qwk"}
    assert not schedule["outer_test_used_for_selection"].any()
    assert schedule.groupby(["repetition","outer_fold","model","selection_objective"]).size().eq(1).all()

def test_candidate_registry_is_unchanged_and_macro_schedule_replays_phase1c():
    source=pd.read_csv(SOURCE/"candidate_search_results.csv")
    schedule,_=build_schedule(source)
    assert set(source["model"])==set(TUNED_MODEL_NAMES)
    assert source.groupby(["repetition","outer_fold","model"])["n_inner_folds"].first().eq(5).all()
    macro=schedule[schedule["selection_objective"]=="macro_f1"]
    historical=source[source["selected_by_protocol"].astype(str).str.lower().eq("true")]
    merged=macro.merge(historical,on=["repetition","outer_fold","model"],validate="one_to_one")
    assert (merged["selected_candidate_index"]==merged["candidate_index"]).all()

def test_runner_reconstructs_persisted_folds_without_generating_new_splits():
    source=Path("src/experiments/eswa_repeated_selection_objective_v1.py").read_text(encoding="utf-8")
    assert "_rebuild_fold_artifacts" in source
    assert "generate_shared_folds" not in source

def test_complete_phase1c_source_run_passes_independent_validation():
    receipt=validate_repeated_nested_cv_run_v3(SOURCE)
    assert receipt["status"]=="passed"
    assert receipt["distinct_outer_assignment_count"]==5
    assert receipt["network_calls"]==receipt["paid_api_calls"]==0
