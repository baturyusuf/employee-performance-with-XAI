from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.experiments.hrdataset_replication_core import (
    HRDatasetTestOnlyOverrides,
    evaluate_hrdataset_replication,
)
from src.utils.config_loader import load_config


@pytest.fixture(scope="session")
def hrdataset_replication_evidence() -> dict[str, Any]:
    """One real canonical-factory run shared by the focused contract tests."""

    rng = np.random.default_rng(731)
    n_rows = 60
    target = pd.Series(
        np.tile([2, 3, 4], n_rows // 3),
        index=pd.Index(range(n_rows), dtype=int),
        name="PerformanceRating",
    )
    identifiers = pd.Series(
        np.arange(10_000, 10_000 + n_rows),
        index=target.index,
        name="ExternalSampleId",
    )
    primary = pd.DataFrame(
        {
            "Signal": target.to_numpy(dtype=float) + rng.normal(0.0, 0.45, n_rows),
            "NumericContext": rng.normal(size=n_rows),
            "CategoryContext": np.where(
                target.to_numpy() == 2,
                "low",
                np.where(target.to_numpy() == 3, "middle", "high"),
            ),
        },
        index=target.index,
    )
    strict = primary.drop(columns=["CategoryContext"]).copy()
    policy_frames = {
        "department_free": primary,
        "department_job_role_free": strict,
    }
    policy_roles = {
        "department_free": "canonical_primary_external",
        "department_job_role_free": "strict_proxy_sensitivity_non_primary",
    }
    forbidden = {
        "department_free": (
            "PerformanceRating",
            "ExternalSampleId",
            "EmpNumber",
            "DeptID",
            "MarriedID",
            "EmpStatusID",
        ),
        "department_job_role_free": (
            "PerformanceRating",
            "ExternalSampleId",
            "EmpNumber",
            "DeptID",
            "PositionID",
            "MarriedID",
            "EmpStatusID",
            "CategoryContext",
        ),
    }
    test_overrides = HRDatasetTestOnlyOverrides(
        candidate_indices=(0,),
        bootstrap_resamples=20,
    )
    result = evaluate_hrdataset_replication(
        policy_frames,
        policy_roles,
        forbidden,
        target,
        identifiers,
        load_config("configs/model_grid.yaml"),
        primary_policy="department_free",
        run_id="hrdataset-core-contract-test",
        config_hash="a" * 64,
        scientific_input_hash="b" * 64,
        dataset_sha256="c" * 64,
        test_only_overrides=test_overrides,
    )
    return {
        "result": result,
        "target": target,
        "identifiers": identifiers,
        "policy_frames": policy_frames,
        "policy_roles": policy_roles,
        "forbidden": forbidden,
        "test_overrides": test_overrides,
    }
