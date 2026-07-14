from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.data.canonical_loader import load_canonical_dataset
from src.data.external_adapters import load_external_dataset
from src.governance.external_replication_contract import (
    ALWAYS_FORBIDDEN_FEATURE_ALIASES,
    CONSERVATIVE_PRIMARY_FEATURES,
    ExternalReplicationContractError,
    POLICY_ORDER,
    TARGET_MAPPING,
    expected_external_replication_contract,
    expected_schema_policy_variants,
    policy_exclusion_list,
    validate_external_replication_side_inputs,
)
from src.governance.manuscript_contract import (
    ManuscriptConfigError,
    load_manuscript_config,
    manuscript_settings,
    validate_manuscript_config,
)
from src.utils.config_loader import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "configs" / "manuscript_final.yaml"
SCHEMA_PATH = PROJECT_ROOT / "data" / "external" / "hrdataset_v14" / "schema_mapping.json"
PROVENANCE_PATH = PROJECT_ROOT / "configs" / "dataset_provenance.yaml"
DATASET_CARD_PATH = PROJECT_ROOT / "data" / "external" / "hrdataset_v14" / "dataset_card.md"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical() -> dict:
    return copy.deepcopy(load_manuscript_config(CONFIG_PATH))


def _set_nested(mapping: dict, path: tuple[str, ...], value: object) -> None:
    cursor = mapping
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def test_external_replication_section_is_the_exact_frozen_contract() -> None:
    settings = manuscript_settings(_canonical())
    assert settings["external_replication"] == expected_external_replication_contract()
    assert settings["external_replication"]["feature_policy_contract"]["reported_policy_order"] == list(
        POLICY_ORDER
    )


def test_exact_policy_exclusions_preserve_only_the_approved_differences() -> None:
    always_forbidden = {
        feature
        for values in ALWAYS_FORBIDDEN_FEATURE_ALIASES.values()
        for feature in values
    }
    exclusions = {name: set(policy_exclusion_list(name)) for name in POLICY_ORDER}

    assert always_forbidden <= exclusions["conservative_primary"]
    assert "EmpDepartment" in exclusions["conservative_primary"]
    assert "EmpJobRole" not in exclusions["conservative_primary"]
    assert {"Salary", "State", "Zip", "RecruitmentSource"} <= exclusions[
        "conservative_primary"
    ]

    assert "EmpDepartment" not in exclusions["department_including_audit"]
    assert "DeptID" in exclusions["department_including_audit"]
    assert "EmpJobRole" in exclusions["job_role_free_audit"]
    assert "PositionID" in exclusions["job_role_free_audit"]

    restored_proxy_fields = exclusions["conservative_primary"] - exclusions["proxy_rich_audit"]
    assert restored_proxy_fields == {"Salary", "State", "RecruitmentSource"}
    assert "Zip" in exclusions["proxy_rich_audit"]

    temporal_only = exclusions["temporality_restricted_audit"] - exclusions[
        "conservative_primary"
    ]
    assert temporal_only == {
        "EngagementSurvey",
        "EmpJobSatisfaction",
        "SpecialProjectsCount",
        "DaysLateLast30",
        "Absences",
    }
    for policy_name in POLICY_ORDER:
        assert always_forbidden <= exclusions[policy_name]
        assert "Zip" in exclusions[policy_name]


def test_subgroup_attributes_are_exact_and_absent_from_conservative_primary() -> None:
    external = manuscript_settings(_canonical())["external_replication"]
    subgroup = external["subgroup_diagnostics"]
    assert subgroup["attributes"] == {
        "protected_sensitive": {
            "type": "categorical",
            "features": ["Gender", "RaceEthnicity", "HispanicLatino", "MaritalStatus"],
        },
        "exploratory_operational": {
            "type": "categorical",
            "features": ["EmpDepartment"],
        },
    }
    specified = {
        feature
        for definition in subgroup["attributes"].values()
        for feature in definition["features"]
    }
    assert specified <= set(policy_exclusion_list("conservative_primary"))
    assert subgroup["all_specified_attributes_absent_from_conservative_primary"] is True


def test_proxy_diagnostic_predictor_folds_bootstrap_and_support_are_frozen() -> None:
    proxy = manuscript_settings(_canonical())["external_replication"]["proxy_diagnostics"]
    assert proxy["target"] == "EmpDepartment"
    assert proxy["target_aliases"] == ["Department", "DeptID", "EmpDepartment"]
    assert proxy["predictor_policy_sources"] == [
        "conservative_primary",
        "job_role_free_audit",
    ]
    assert proxy["same_exact_outer_folds_as_external_performance_models"] is True
    assert proxy["bootstrap"]["n_resamples"] == 5000
    assert proxy["bootstrap"]["stratify_by"] == ["outer_fold", "proxy_target"]
    assert proxy["class_support"] == {
        "required_outer_training_class_set": "complete_observed_proxy_target_class_set",
        "merge_classes_allowed": False,
        "drop_classes_allowed": False,
        "insufficient_support_status": (
            "not_estimated_insufficient_outer_training_class_support"
        ),
        "class_counts_and_zero_support_cells_reported": True,
    }
    for policy_name in proxy["predictor_policy_sources"]:
        # The adapter renames raw Department to canonical EmpDepartment before
        # policy projection; DeptID and the canonical target family must both be
        # excluded, while the proxy-stage flag removes every declared alias.
        assert {"DeptID", "EmpDepartment"} <= set(policy_exclusion_list(policy_name))
    assert proxy["target_aliases_absent_from_all_predictors"] is True


def test_primary_feature_governance_keys_and_temporality_are_exact() -> None:
    governance = manuscript_settings(_canonical())["external_replication"]["feature_governance"]
    assert governance["exact_primary_feature_families"] == list(CONSERVATIVE_PRIMARY_FEATURES)
    assert list(governance["features"]) == list(CONSERVATIVE_PRIMARY_FEATURES)
    assert governance["features"]["EmpJobRole"]["category"] == "operational_proxy_context"
    for feature in ("EngagementSurvey", "EmpJobSatisfaction"):
        assert governance["features"][feature]["temporality_status"] == (
            "timing_unverified_contemporaneous"
        )
    for feature in ("SpecialProjectsCount", "DaysLateLast30", "Absences"):
        assert governance["features"][feature]["temporality_status"] == (
            "timing_unverified_history_or_window"
        )
    assert governance["features"]["ExperienceYearsAtThisCompany"]["temporality_status"] == (
        "derived_at_last_review_timing_unverified_negative_durations_set_missing"
    )
    assert manuscript_settings(_canonical())["external_replication"]["shap"]["stability_top_k"] == 5
    assert manuscript_settings(_canonical())["external_replication"]["shap"]["attribution_unit"] == (
        "xgboost_raw_margin_score"
    )
    assert manuscript_settings(_canonical())["external_replication"]["shap"][
        "additivity_output_space"
    ] == "xgboost_raw_margin"
    assert manuscript_settings(_canonical())["external_replication"]["subgroup_diagnostics"][
        "probability_method"
    ] == "raw"
    warning = governance["model_scenario_only_warning"].casefold()
    assert "attribution only" in warning
    assert "causal" in warning and "actionable" in warning and "employee advice" in warning


def test_derived_tenure_negative_durations_are_explicitly_set_missing() -> None:
    loaded = load_canonical_dataset(CONFIG_PATH, "hrdataset_v14")
    dataset = load_external_dataset(
        "hrdataset_v14",
        raw_frame=loaded.frame,
        schema_mapping_path=SCHEMA_PATH,
    )
    tenure = dataset.canonical["ExperienceYearsAtThisCompany"]
    assert int(tenure.isna().sum()) == 2
    assert not (tenure.dropna() < 0).any()


def test_derived_tenure_has_no_missing_column_or_date_fallback() -> None:
    loaded = load_canonical_dataset(CONFIG_PATH, "hrdataset_v14")
    missing_reference = loaded.frame.drop(columns=["LastPerformanceReview_Date"])
    with pytest.raises(ValueError, match="reference column is missing"):
        load_external_dataset(
            "hrdataset_v14",
            raw_frame=missing_reference,
            schema_mapping_path=SCHEMA_PATH,
        )

    reference_drift = loaded.frame.copy()
    reference_drift.loc[reference_drift.index[0], "LastPerformanceReview_Date"] = None
    with pytest.raises(ValueError, match="reference-date support drifted"):
        load_external_dataset(
            "hrdataset_v14",
            raw_frame=reference_drift,
            schema_mapping_path=SCHEMA_PATH,
        )


def test_schema_mapping_has_exact_policy_roles_lists_and_leakage_aware_language() -> None:
    schema = _json(SCHEMA_PATH)
    assert schema["feature_policy_variants"] == expected_schema_policy_variants()
    assert "EmpStatusID" in schema["leakage_risk_columns"]
    assert {"MarriedID", "FromDiversityJobFairID"} <= set(
        schema["sensitive_audit_only_columns"]
    )
    assert {"DeptID", "PositionID"} <= set(schema["proxy_risk_columns"])
    assert "leakage-safe" not in json.dumps(schema, sort_keys=True).casefold()


def test_target_mapping_rationale_and_limitation_are_equal_across_all_contracts() -> None:
    settings = manuscript_settings(_canonical())
    external_target = settings["external_replication"]["target"]
    schema_target = _json(SCHEMA_PATH)["target"]
    provenance_binding = _json(PROVENANCE_PATH)["dataset_provenance"]["dataset_bindings"][
        "hrdataset_v14"
    ]

    assert external_target["mapping"] == dict(TARGET_MAPPING)
    assert schema_target["mapping"] == external_target["mapping"]
    assert provenance_binding["target_mapping"] == external_target["mapping"]
    assert schema_target["mapping_rationale"] == external_target["mapping_rationale"]
    assert provenance_binding["target_mapping_note"] == external_target["mapping_rationale"]
    assert schema_target["mapping_limitation"] == external_target["mapping_limitation"]
    assert provenance_binding["target_mapping_limitation"] == external_target["mapping_limitation"]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("external_replication", "cv", "inner_splits"), 3),
        (("external_replication", "cv", "outer_splits"), 5),
        (("external_replication", "model_protocol", "selection_primary_metric"), "accuracy"),
        (
            ("external_replication", "model_protocol", "selection_tie_break_metric"),
            "accuracy",
        ),
        (("external_replication", "calibration", "primary_method"), "isotonic"),
        (("external_replication", "uncertainty", "n_resamples"), 1000),
        (("external_replication", "shap", "model_refit_in_shap_stage"), True),
        (("external_replication", "shap", "attribution_unit"), "probability"),
        (("external_replication", "shap", "additivity_output_space"), "probability"),
        (("external_replication", "subgroup_diagnostics", "probability_method"), "sigmoid"),
        (
            (
                "external_replication",
                "subgroup_diagnostics",
                "legacy_fairness_helper_dependency_allowed",
            ),
            True,
        ),
        (
            (
                "external_replication",
                "subgroup_diagnostics",
                "attributes",
                "protected_sensitive",
                "features",
            ),
            ["Gender"],
        ),
        (("external_replication", "proxy_diagnostics", "bootstrap", "n_resamples"), 1000),
        (
            (
                "external_replication",
                "proxy_diagnostics",
                "class_support",
                "merge_classes_allowed",
            ),
            True,
        ),
        (
            (
                "external_replication",
                "feature_governance",
                "features",
                "EmpJobRole",
                "category",
            ),
            "ordinary_feature",
        ),
        (
            ("external_replication", "publication", "staged_atomic_publication_required"),
            False,
        ),
    ],
)
def test_protocol_drift_fails_closed(path: tuple[str, ...], replacement: object) -> None:
    malformed = _canonical()
    _set_nested(malformed["manuscript_final"], path, replacement)
    with pytest.raises(ManuscriptConfigError, match="frozen HRDataset_v14 10x5"):
        validate_manuscript_config(malformed)


def test_policy_role_or_restoration_drift_fails_closed() -> None:
    malformed = _canonical()
    policy = malformed["manuscript_final"]["external_replication"]["feature_policy_contract"][
        "policies"
    ]["proxy_rich_audit"]
    policy["restored_features"].append("Zip")
    with pytest.raises(ManuscriptConfigError, match="frozen HRDataset_v14 10x5"):
        validate_manuscript_config(malformed)


def test_boolean_external_seed_is_rejected() -> None:
    malformed = _canonical()
    malformed["manuscript_final"]["seeds"]["external_replication"] = True
    with pytest.raises(ManuscriptConfigError, match="protocol seeds"):
        validate_manuscript_config(malformed)


@pytest.mark.parametrize(
    "seed_name",
    ["external_replication", "inner_cv", "model", "calibration", "bootstrap", "fairness"],
)
def test_external_protocol_seed_values_are_exactly_bound(seed_name: str) -> None:
    malformed = _canonical()
    malformed["manuscript_final"]["seeds"][seed_name] = 99
    with pytest.raises(ManuscriptConfigError, match="protocol seeds"):
        validate_manuscript_config(malformed)


def test_copied_canonical_config_still_validates_external_side_inputs(tmp_path: Path) -> None:
    malformed = _canonical()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema["target"]["mapping"]["PIP"] = 3
    schema_path = tmp_path / "schema_mapping.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    malformed["manuscript_final"]["datasets"]["hrdataset_v14"][
        "schema_mapping_path"
    ] = str(schema_path)
    malformed["manuscript_final"]["provenance"]["scientific_side_inputs"][
        "external_hrdataset_v14_schema_mapping"
    ] = str(schema_path)
    copied_config = tmp_path / "copied_manuscript_final.yaml"
    copied_config.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ManuscriptConfigError, match="target mapping"):
        load_manuscript_config(copied_config)


def test_default_side_inputs_pass_semantic_equality_validation() -> None:
    settings = manuscript_settings(_canonical())
    validate_external_replication_side_inputs(settings, project_root=PROJECT_ROOT)


@pytest.mark.parametrize("drift_source", ["schema_policy", "schema_mapping", "provenance_mapping"])
def test_schema_or_provenance_semantic_drift_fails_closed(
    tmp_path: Path,
    drift_source: str,
) -> None:
    settings = copy.deepcopy(dict(manuscript_settings(_canonical())))
    schema = _json(SCHEMA_PATH)
    provenance = _json(PROVENANCE_PATH)

    if drift_source == "schema_policy":
        schema["feature_policy_variants"]["conservative_primary"]["exclude_columns"].remove(
            "DeptID"
        )
    elif drift_source == "schema_mapping":
        schema["target"]["mapping"]["PIP"] = 3
    else:
        provenance["dataset_provenance"]["dataset_bindings"]["hrdataset_v14"][
            "target_mapping"
        ]["PIP"] = 3

    schema_path = tmp_path / "schema_mapping.json"
    provenance_path = tmp_path / "dataset_provenance.yaml"
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    settings["datasets"]["hrdataset_v14"]["schema_mapping_path"] = schema_path.name
    settings["provenance"]["dataset_cards_config"] = provenance_path.name

    with pytest.raises(ExternalReplicationContractError):
        validate_external_replication_side_inputs(settings, project_root=tmp_path)


def test_dataset_card_uses_claim_bounded_leakage_aware_language() -> None:
    text = DATASET_CARD_PATH.read_text(encoding="utf-8")
    folded = text.casefold()
    normalized = " ".join(folded.split())
    assert "leakage-safe" not in folded
    assert "leakage-aware" in folded
    assert "not locked-model transport" in normalized
    assert "manual_review_required" in text
    assert "autonomous hr decision support" in normalized
