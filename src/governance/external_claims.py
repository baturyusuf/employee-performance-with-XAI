from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


HRDATASET_REPLICATION_CLAIM = "independent external performance-target replication"


@dataclass(frozen=True)
class ExternalClaimBoundary:
    dataset_key: str
    allowed_claim: str
    transported_locked_inx_model: bool


EXTERNAL_CLAIM_BOUNDARIES: Mapping[str, ExternalClaimBoundary] = {
    "hrdataset_v14:primary": ExternalClaimBoundary(
        dataset_key="hrdataset_v14:primary",
        allowed_claim=HRDATASET_REPLICATION_CLAIM,
        transported_locked_inx_model=False,
    ),
    "ibm_hr_analytics:primary": ExternalClaimBoundary(
        dataset_key="ibm_hr_analytics:primary",
        allowed_claim="restricted-target performance robustness",
        transported_locked_inx_model=False,
    ),
    "ibm_hr_analytics:attrition": ExternalClaimBoundary(
        dataset_key="ibm_hr_analytics:attrition",
        allowed_claim="related HR attrition task transfer",
        transported_locked_inx_model=False,
    ),
    "employee_turnover:primary": ExternalClaimBoundary(
        dataset_key="employee_turnover:primary",
        allowed_claim="related HR turnover task transfer",
        transported_locked_inx_model=False,
    ),
}


def external_claim_boundary(dataset_name: str, target_kind: str = "primary") -> ExternalClaimBoundary:
    if dataset_name == "ibm_hr_analytics_attrition" and target_kind == "primary":
        dataset_name, target_kind = "ibm_hr_analytics", "attrition"
    key = f"{dataset_name}:{target_kind}"
    try:
        return EXTERNAL_CLAIM_BOUNDARIES[key]
    except KeyError as exc:
        raise ValueError(f"No external claim boundary is registered for '{key}'.") from exc


def external_allowed_claim(dataset_name: str, target_kind: str = "primary") -> str:
    return external_claim_boundary(dataset_name, target_kind).allowed_claim


def validate_configured_claim(dataset_name: str, target_kind: str, configured_claim: str) -> None:
    expected = external_allowed_claim(dataset_name, target_kind)
    if str(configured_claim).strip() != expected:
        raise ValueError(
            f"Claim boundary mismatch for {dataset_name}/{target_kind}: "
            f"expected '{expected}', observed '{configured_claim}'."
        )
