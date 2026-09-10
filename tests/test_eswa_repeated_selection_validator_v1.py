from pathlib import Path
import pytest
from src.governance.eswa_repeated_selection_validator_v1 import validate_run

RUN=Path("reports/major_revision_round2_runs/eswa_selection_v1_20260910T000000Z_914e1e0/repeated_selection_objective")
pytestmark=pytest.mark.skipif(not RUN.is_dir(),reason="local row-level ESWA run is intentionally ignored")

def test_complete_repeated_selection_run_independently_recomputes():
    receipt=validate_run(RUN)
    assert receipt["status"]=="PASS_INDEPENDENT_RECOMPUTATION"
    assert receipt["candidate_changes"]==92
    assert receipt["oof_rows"]==72000
    assert receipt["qwk_outer_fits"]==150
