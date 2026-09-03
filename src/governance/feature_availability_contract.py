"""Validate and render the additive v3 feature-availability contract.

The contract is deliberately separate from the immutable v2 scientific
configuration.  It formalizes a prospective-sensitivity estimand without
claiming that the cross-sectional INX data contain feature or decision
timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CONTRACT_PATH = Path("configs/feature_availability_v3.json")
DEFAULT_ACQUISITION_PATH = Path("configs/data_acquisition.yaml")
DEFAULT_MARKDOWN_PATH = Path(
    "reports/research_log/major_revision_v3/FEATURE_AVAILABILITY_GOVERNANCE_CONTRACT.md"
)

REQUIRED_FEATURE_FIELDS = (
    "feature_name",
    "feature_family",
    "semantic_description",
    "likely_measurement_time",
    "relationship_to_target",
    "availability_at_prediction_time",
    "availability_confidence",
    "risk_type",
    "governance_type",
    "source_evidence",
    "primary_policy_status",
    "justification",
)

EXPECTED_RISK_TYPES = frozenset(
    {
        "identifier",
        "target_direct_leakage",
        "temporal_leakage",
        "outcome_proximal",
        "timing_uncertain",
        "sensitive_attribute",
        "organizational_proxy",
        "ordinary_predictor",
    }
)

EXPECTED_POLICIES = (
    ("P0", "INFORMATION_RICH_DIAGNOSTIC"),
    ("P1", "LEAKAGE_CONTROLLED"),
    ("P2", "GOVERNANCE_CONTROLLED"),
    ("P3", "PRIMARY_LEAKAGE_AWARE"),
    ("P4", "STRICT_PROSPECTIVE"),
    ("P5", "STRICT_PROXY"),
)

ALLOWED_AVAILABILITY_CONFIDENCE = frozenset({"low", "moderate", "not_applicable"})


class FeatureAvailabilityContractError(ValueError):
    """Raised when the v3 information contract is incomplete or overclaims."""


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FeatureAvailabilityContractError(
            f"Could not read JSON contract {path.as_posix()}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise FeatureAvailabilityContractError(
            f"JSON document {path.as_posix()} must contain an object."
        )
    return payload


def _require_nonempty_string(record: Mapping[str, Any], key: str, *, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FeatureAvailabilityContractError(
            f"{context} requires a non-empty string field {key!r}."
        )
    return value.strip()


def _unique_strings(values: Any, *, context: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise FeatureAvailabilityContractError(
            f"{context} must be a list of non-empty strings."
        )
    normalized = tuple(value.strip() for value in values)
    if len(normalized) != len(set(normalized)):
        raise FeatureAvailabilityContractError(f"{context} contains duplicates.")
    return normalized


def _canonical_semantic_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_prediction_scenario(contract: Mapping[str, Any]) -> None:
    scenario = contract.get("prediction_scenario")
    if not isinstance(scenario, Mapping):
        raise FeatureAvailabilityContractError("prediction_scenario must be an object.")
    for field in (
        "scenario_id",
        "statement",
        "evidence_status",
        "supported_interpretation",
        "prohibited_interpretation",
    ):
        _require_nonempty_string(scenario, field, context="prediction_scenario")
    evidence_status = str(scenario["evidence_status"])
    if evidence_status != "conceptual_estimand_only_no_observed_feature_or_decision_timestamps":
        raise FeatureAvailabilityContractError(
            "The prediction scenario must explicitly retain the no-observed-timestamps boundary."
        )
    prohibited = str(scenario["prohibited_interpretation"]).lower()
    for required_phrase in ("actually observed", "all leakage", "prospectively deployable"):
        if required_phrase not in prohibited:
            raise FeatureAvailabilityContractError(
                "prediction_scenario.prohibited_interpretation must mention "
                f"{required_phrase!r}."
            )
    _unique_strings(scenario.get("source_evidence"), context="prediction_scenario.source_evidence")


def _validate_features(
    contract: Mapping[str, Any],
    *,
    expected_columns: Sequence[str],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, int]]:
    features = contract.get("features")
    if not isinstance(features, list) or not features:
        raise FeatureAvailabilityContractError("features must be a non-empty list.")
    if not all(isinstance(feature, Mapping) for feature in features):
        raise FeatureAvailabilityContractError("Every features entry must be an object.")

    feature_names = []
    feature_by_name: dict[str, Mapping[str, Any]] = {}
    risk_counts = {risk: 0 for risk in EXPECTED_RISK_TYPES}
    availability_vocabulary = contract.get("availability_vocabulary")
    if not isinstance(availability_vocabulary, Mapping) or not availability_vocabulary:
        raise FeatureAvailabilityContractError("availability_vocabulary must be a non-empty object.")
    allowed_availability = set(map(str, availability_vocabulary))

    for index, feature in enumerate(features):
        context = f"features[{index}]"
        for field in REQUIRED_FEATURE_FIELDS:
            _require_nonempty_string(feature, field, context=context)
        name = str(feature["feature_name"])
        if name in feature_by_name:
            raise FeatureAvailabilityContractError(f"Duplicate feature contract for {name!r}.")
        feature_names.append(name)
        feature_by_name[name] = feature

        risk_type = str(feature["risk_type"])
        if risk_type not in EXPECTED_RISK_TYPES:
            raise FeatureAvailabilityContractError(
                f"Feature {name!r} uses unknown risk_type {risk_type!r}."
            )
        risk_counts[risk_type] += 1

        availability = str(feature["availability_at_prediction_time"])
        if availability not in allowed_availability:
            raise FeatureAvailabilityContractError(
                f"Feature {name!r} uses unknown availability state {availability!r}."
            )
        confidence = str(feature["availability_confidence"])
        if confidence not in ALLOWED_AVAILABILITY_CONFIDENCE:
            raise FeatureAvailabilityContractError(
                f"Feature {name!r} uses unsupported availability confidence {confidence!r}; "
                "the timestamp-free data cannot support high/confirmed availability."
            )
        combined_availability = f"{availability} {confidence}".lower()
        if "confirm" in combined_availability or "verified_available" in combined_availability:
            raise FeatureAvailabilityContractError(
                f"Feature {name!r} overclaims timestamp-verified availability."
            )

    if tuple(feature_names) != tuple(expected_columns):
        raise FeatureAvailabilityContractError(
            "Feature contract columns/order differ from the pinned acquisition schema: "
            f"contract={feature_names}, expected={list(expected_columns)}."
        )
    absent_risk_types = sorted(risk for risk, count in risk_counts.items() if count == 0)
    if absent_risk_types:
        raise FeatureAvailabilityContractError(
            f"The required risk taxonomy is not represented: {absent_risk_types}."
        )
    return feature_by_name, risk_counts


def _validate_policies(
    contract: Mapping[str, Any],
    *,
    feature_by_name: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    policies = contract.get("policies")
    if not isinstance(policies, list) or not all(
        isinstance(policy, Mapping) for policy in policies
    ):
        raise FeatureAvailabilityContractError("policies must be a list of objects.")
    observed_identity = tuple(
        (
            _require_nonempty_string(policy, "policy_id", context="policy"),
            _require_nonempty_string(policy, "name", context="policy"),
        )
        for policy in policies
    )
    if observed_identity != EXPECTED_POLICIES:
        raise FeatureAvailabilityContractError(
            f"Policy identity/order differs from P0-P5: {observed_identity}."
        )

    all_features = tuple(feature_by_name)
    all_feature_set = set(all_features)
    exclusions: dict[str, tuple[str, ...]] = {}
    retained: dict[str, tuple[str, ...]] = {}
    previous: set[str] = set()
    for policy in policies:
        policy_id = str(policy["policy_id"])
        _require_nonempty_string(policy, "role", context=f"policy {policy_id}")
        _require_nonempty_string(policy, "interpretation", context=f"policy {policy_id}")
        excluded = _unique_strings(
            policy.get("excluded_features"),
            context=f"policy {policy_id}.excluded_features",
        )
        unknown = sorted(set(excluded) - all_feature_set)
        if unknown:
            raise FeatureAvailabilityContractError(
                f"Policy {policy_id} excludes unknown features: {unknown}."
            )
        excluded_set = set(excluded)
        if not previous.issubset(excluded_set):
            raise FeatureAvailabilityContractError(
                f"Policy {policy_id} must include every exclusion from the preceding policy."
            )
        previous = excluded_set
        exclusions[policy_id] = excluded
        retained[policy_id] = tuple(name for name in all_features if name not in excluded_set)

    if set(exclusions["P0"]) != {"EmpNumber", "PerformanceRating"}:
        raise FeatureAvailabilityContractError("P0 must exclude only identifier and target.")
    required_p1 = {
        name
        for name, feature in feature_by_name.items()
        if feature["risk_type"]
        in {"identifier", "target_direct_leakage", "temporal_leakage", "outcome_proximal"}
    }
    if not required_p1.issubset(exclusions["P1"]):
        raise FeatureAvailabilityContractError(
            f"P1 fails to exclude all direct/temporal/outcome risks: {sorted(required_p1)}."
        )
    required_p2 = {
        name
        for name, feature in feature_by_name.items()
        if feature["risk_type"] == "sensitive_attribute"
    }
    if not required_p2.issubset(exclusions["P2"]):
        raise FeatureAvailabilityContractError(
            f"P2 fails to exclude all direct sensitive fields: {sorted(required_p2)}."
        )
    if "EmpDepartment" not in exclusions["P3"] or "EmpJobRole" in exclusions["P3"]:
        raise FeatureAvailabilityContractError(
            "P3 must exclude EmpDepartment and retain EmpJobRole for the declared continuity audit."
        )
    required_p4 = {
        name
        for name, feature in feature_by_name.items()
        if feature["risk_type"] == "timing_uncertain"
    }
    if not required_p4.issubset(exclusions["P4"]):
        raise FeatureAvailabilityContractError(
            f"P4 fails to exclude all timing-uncertain fields: {sorted(required_p4)}."
        )
    required_p5 = {
        name
        for name, feature in feature_by_name.items()
        if feature["risk_type"] == "organizational_proxy"
    }
    if not required_p5.issubset(exclusions["P5"]):
        raise FeatureAvailabilityContractError(
            f"P5 fails to exclude all declared organisational proxy fields: {sorted(required_p5)}."
        )

    exact_exclusions = {
        "P0": {
            name
            for name, feature in feature_by_name.items()
            if feature["risk_type"] in {"identifier", "target_direct_leakage"}
        },
        "P1": required_p1,
        "P2": required_p1 | required_p2,
        "P3": required_p1 | required_p2 | {"EmpDepartment"},
        "P4": required_p1 | required_p2 | {"EmpDepartment"} | required_p4,
        "P5": (
            required_p1
            | required_p2
            | {"EmpDepartment"}
            | required_p4
            | required_p5
        ),
    }
    for policy_id, expected in exact_exclusions.items():
        observed = set(exclusions[policy_id])
        if observed != expected:
            raise FeatureAvailabilityContractError(
                f"Policy {policy_id} must use its exact declared exclusion set; "
                f"unexpected={sorted(observed - expected)}, "
                f"missing={sorted(expected - observed)}."
            )

    p3_exclusions = set(exclusions["P3"])
    for name, feature in feature_by_name.items():
        status = str(feature["primary_policy_status"])
        if name in p3_exclusions and status.startswith("retained_"):
            raise FeatureAvailabilityContractError(
                f"Feature {name!r} is excluded by P3 but declares retained primary status."
            )
        if name not in p3_exclusions and not status.startswith("retained_"):
            raise FeatureAvailabilityContractError(
                f"Feature {name!r} is retained by P3 but its status is {status!r}."
            )
    return exclusions, retained


def validate_feature_availability_contract(
    contract_path: Path | str = DEFAULT_CONTRACT_PATH,
    acquisition_path: Path | str = DEFAULT_ACQUISITION_PATH,
) -> dict[str, Any]:
    """Validate the complete v3 INX feature/policy contract and return a receipt."""

    contract_file = Path(contract_path)
    acquisition_file = Path(acquisition_path)
    contract = _load_json_mapping(contract_file)
    acquisition = _load_json_mapping(acquisition_file)
    if contract.get("schema_version") != 1:
        raise FeatureAvailabilityContractError("schema_version must equal 1.")
    if contract.get("contract_id") != "inx_feature_availability_governance_v3":
        raise FeatureAvailabilityContractError(
            "contract_id must equal 'inx_feature_availability_governance_v3'."
        )
    if contract.get("dataset_key") != "inx_primary":
        raise FeatureAvailabilityContractError("dataset_key must equal 'inx_primary'.")
    if contract.get("physical_dataset_key") != "inx_employee_performance":
        raise FeatureAvailabilityContractError(
            "physical_dataset_key must equal 'inx_employee_performance'."
        )
    if contract.get("target") != "PerformanceRating":
        raise FeatureAvailabilityContractError("target must equal 'PerformanceRating'.")

    risk_vocabulary = _unique_strings(
        contract.get("risk_type_vocabulary"), context="risk_type_vocabulary"
    )
    if set(risk_vocabulary) != EXPECTED_RISK_TYPES or len(risk_vocabulary) != len(
        EXPECTED_RISK_TYPES
    ):
        raise FeatureAvailabilityContractError(
            "risk_type_vocabulary must contain the exact required eight-category taxonomy."
        )
    _validate_prediction_scenario(contract)

    try:
        physical = acquisition["data_acquisition"]["physical_datasets"][
            "inx_employee_performance"
        ]
        expected_columns = physical["expected_columns"]
        expected_count = int(physical["expected_column_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FeatureAvailabilityContractError(
            "Pinned acquisition contract lacks the INX ordered schema."
        ) from exc
    if not isinstance(expected_columns, list) or expected_count != len(expected_columns):
        raise FeatureAvailabilityContractError(
            "Pinned INX expected_column_count does not match expected_columns."
        )

    feature_by_name, risk_counts = _validate_features(
        contract, expected_columns=tuple(map(str, expected_columns))
    )
    exclusions, retained = _validate_policies(contract, feature_by_name=feature_by_name)
    limitations = _unique_strings(
        contract.get("global_limitations"), context="global_limitations"
    )
    limitations_text = " ".join(limitations).lower()
    for required_phrase in ("no feature-observation timestamps", "not verified prospective", "all leakage"):
        if required_phrase not in limitations_text:
            raise FeatureAvailabilityContractError(
                f"global_limitations must retain the boundary {required_phrase!r}."
            )

    return {
        "status": "passed",
        "contract_id": str(contract.get("contract_id")),
        "dataset_key": "inx_primary",
        "target": "PerformanceRating",
        "feature_count": len(feature_by_name),
        "policy_count": len(exclusions),
        "risk_type_counts": risk_counts,
        "policy_feature_counts": {
            policy_id: len(names) for policy_id, names in retained.items()
        },
        "contract_sha256": hashlib.sha256(contract_file.read_bytes()).hexdigest(),
        "contract_semantic_sha256": _canonical_semantic_sha256(contract),
        "acquisition_sha256": hashlib.sha256(acquisition_file.read_bytes()).hexdigest(),
        "timestamp_verified_feature_count": 0,
        "prediction_scenario_status": contract["prediction_scenario"]["evidence_status"],
    }


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def render_feature_availability_markdown(
    contract_path: Path | str = DEFAULT_CONTRACT_PATH,
    acquisition_path: Path | str = DEFAULT_ACQUISITION_PATH,
) -> str:
    """Return a manuscript-neutral Markdown rendering of the validated contract."""

    contract_file = Path(contract_path)
    contract = _load_json_mapping(contract_file)
    receipt = validate_feature_availability_contract(contract_file, acquisition_path)
    feature_names = tuple(str(feature["feature_name"]) for feature in contract["features"])

    lines = [
        "# Feature Availability and Governance Contract — v3",
        "",
        f"- Contract: `{receipt['contract_id']}`",
        f"- Dataset: `{receipt['dataset_key']}`",
        f"- Target: `{receipt['target']}`",
        f"- Contract SHA-256: `{receipt['contract_sha256']}`",
        "",
        "## Prediction scenario",
        "",
        str(contract["prediction_scenario"]["statement"]),
        "",
        f"**Evidence status:** `{contract['prediction_scenario']['evidence_status']}`",
        "",
        f"**Supported interpretation:** {contract['prediction_scenario']['supported_interpretation']}",
        "",
        f"**Prohibited interpretation:** {contract['prediction_scenario']['prohibited_interpretation']}",
        "",
        "## Policies",
        "",
        "| ID | Name | Role | Retained n | Excluded n | Retained features | Interpretation |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for policy in contract["policies"]:
        excluded = set(map(str, policy["excluded_features"]))
        retained = [name for name in feature_names if name not in excluded]
        lines.append(
            "| "
            + " | ".join(
                (
                    _escape_markdown(policy["policy_id"]),
                    _escape_markdown(policy["name"]),
                    _escape_markdown(policy["role"]),
                    str(len(retained)),
                    str(len(excluded)),
                    _escape_markdown("; ".join(retained)),
                    _escape_markdown(policy["interpretation"]),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Feature-level contract",
            "",
            "| Feature | Family | Description | Likely measurement time | Target relationship | Availability at prediction time | Confidence | Risk type | Governance type | Source evidence | P3 status | Justification |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for feature in contract["features"]:
        lines.append(
            "| "
            + " | ".join(
                _escape_markdown(feature[field]) for field in REQUIRED_FEATURE_FIELDS
            )
            + " |"
        )

    lines.extend(["", "## Mandatory limitations", ""])
    lines.extend(f"- {limitation}" for limitation in contract["global_limitations"])
    lines.extend(
        [
            "",
            "This contract is an additive v3 scientific side input. It does not modify or relabel canonical v2 evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FeatureAvailabilityContractError(
            f"Refusing to overwrite residual temporary file {temporary.as_posix()}."
        )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception as exc:
        try:
            temporary.unlink(missing_ok=True)
        except Exception as cleanup_exc:  # pragma: no cover - exceptional filesystem path
            if hasattr(exc, "add_note"):
                exc.add_note(f"Temporary-file cleanup also failed: {cleanup_exc}")
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--acquisition", type=Path, default=DEFAULT_ACQUISITION_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MARKDOWN_PATH)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    receipt = validate_feature_availability_contract(args.contract, args.acquisition)
    if not args.validate_only:
        rendered = render_feature_availability_markdown(args.contract, args.acquisition)
        _atomic_write_text(args.output, rendered)
        receipt["markdown_path"] = args.output.as_posix()
        receipt["markdown_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ACQUISITION_PATH",
    "DEFAULT_CONTRACT_PATH",
    "DEFAULT_MARKDOWN_PATH",
    "EXPECTED_POLICIES",
    "EXPECTED_RISK_TYPES",
    "FeatureAvailabilityContractError",
    "REQUIRED_FEATURE_FIELDS",
    "render_feature_availability_markdown",
    "validate_feature_availability_contract",
]
