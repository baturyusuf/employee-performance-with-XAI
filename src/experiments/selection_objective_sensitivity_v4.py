"""Run the prespecified Round 2 selection-objective and prior-baseline study.

Candidate-level inner-CV scores are reused only after exact hash and registry
validation.  Macro-F1-selected OOF predictions are historical Phase 1B
evidence.  QWK-selected candidates are refitted in all six model families and
all ten outer folds.  The empirical-prior baseline is derived independently in
each outer-training partition.  Row-level outputs stay in the ignored local run
root; a separate validator/exporter owns the tracked publication surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.data.canonical_loader import load_canonical_dataset, sha256_file
from src.experiments.manuscript_model_benchmark import select_candidate_index
from src.experiments.ordinal_benchmark_v3 import exact_p3_feature_frame
from src.experiments.shared_folds import read_shared_folds, validate_shared_folds
from src.governance.feature_availability_contract import validate_feature_availability_contract
from src.governance.manuscript_contract import primary_excluded_features, source_tree_hash
from src.models.canonical_models import (
    CANONICAL_MODEL_NAMES,
    aligned_predict_proba,
    build_common_preprocessor,
    build_model_pipeline,
    validate_model_feature_frame,
)
from src.models.ordinal_evaluation_v3 import ordinal_evaluation_bundle_v3
from src.models.ordinal_models_v3 import V3_ORDINAL_MODEL_NAMES, build_v3_ordinal_estimator
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_CONTRACT_PATH = Path("configs/selection_objective_sensitivity_v4.json")
DEFAULT_LOCAL_RUN_ROOT = Path("reports/major_revision_round2_runs")
MODEL_NAMES = (*CANONICAL_MODEL_NAMES, *V3_ORDINAL_MODEL_NAMES)
LABELS = (2, 3, 4)
FIT_THREAD_LIMIT = 1


class SelectionObjectiveSensitivityV4Error(RuntimeError):
    """Raised when a Round 2 selection/prior invariant fails."""


@dataclass(frozen=True)
class SelectionObjectiveSensitivityV4Result:
    candidate_evidence: pd.DataFrame
    selection_schedule: pd.DataFrame
    selected_candidate_changes: pd.DataFrame
    oof_predictions: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    per_class_metrics: pd.DataFrame
    confusion_matrix: pd.DataFrame
    metric_effect_magnitudes: pd.DataFrame
    ranking_changes: pd.DataFrame
    empirical_prior_fold_parameters: pd.DataFrame
    empirical_prior_oof_predictions: pd.DataFrame
    empirical_prior_metrics: pd.DataFrame


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectionObjectiveSensitivityV4Error(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectionObjectiveSensitivityV4Error(f"Could not read {path.as_posix()}: {exc}") from exc
    _require(isinstance(value, dict), f"{path.as_posix()} must contain a JSON object.")
    return value


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _clean_git_identity() -> dict[str, str]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=PROJECT_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SelectionObjectiveSensitivityV4Error(f"Could not establish Git identity: {exc}") from exc
    _require(len(head) == 40 and all(ch in "0123456789abcdef" for ch in head), "Invalid Git HEAD.")
    _require(bool(branch), "Scientific execution requires a named branch.")
    _require(not status, f"Scientific execution requires a clean worktree: {status.splitlines()[:10]}.")
    return {"commit": head, "branch": branch}


def _validate_contract(contract_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    contract = _load_json(contract_path)
    _require(contract.get("schema_version") == 1, "Round 2 selection schema version drifted.")
    _require(contract.get("contract_id") == "selection_objective_sensitivity_v4", "Contract ID drifted.")
    _require(contract.get("dataset_key") == "inx_primary", "Dataset key drifted.")
    _require(contract.get("target") == "PerformanceRating", "Target drifted.")
    _require(tuple(contract.get("ordered_labels", ())) == LABELS, "Ordered labels drifted.")
    folds = contract.get("folds", {})
    _require(
        {key: folds.get(key) for key in ("outer_splits", "inner_splits", "outer_seed", "inner_seed", "model_seed")}
        == {"outer_splits": 10, "inner_splits": 5, "outer_seed": 42, "inner_seed": 43, "model_seed": 44},
        "Nested-CV dimensions or seeds drifted.",
    )
    regimes = contract.get("selection_regimes", {})
    _require(
        regimes.get("macro_f1") == {
            "primary_metric": "macro_f1",
            "tie_break_metric": "quadratic_weighted_kappa",
            "practical_tie_tolerance": 0.001,
            "oof_mode": "reuse_validated_phase1b_predictions_without_refit",
        },
        "Macro-F1 selection regime drifted.",
    )
    _require(
        regimes.get("qwk") == {
            "primary_metric": "quadratic_weighted_kappa",
            "tie_break_metric": "macro_f1",
            "practical_tie_tolerance": 0.001,
            "oof_mode": "refit_every_model_in_every_outer_fold_from_persisted_candidate_scores",
        },
        "QWK selection regime drifted.",
    )
    ranking = contract.get("ranking_report", {})
    _require(ranking.get("composite_material_dependence_flag_allowed") is False, "Composite materiality flag must remain prohibited.")
    _require(
        ranking.get("separate_fields") == [
            "leader_changed", "full_ordering_changed", "selected_candidate_changes", "metric_effect_magnitudes"
        ],
        "Separate selection-sensitivity reporting fields drifted.",
    )
    publication = contract.get("publication", {})
    _require(publication.get("local_output_root") == DEFAULT_LOCAL_RUN_ROOT.as_posix(), "Local output root drifted.")
    _require(publication.get("publish_employee_level_oof") is False, "Row-level publication must remain disabled.")
    _require(publication.get("paid_api_calls") == 0 and publication.get("network_calls") == 0, "Network/API policy drifted.")

    source_hashes: dict[str, str] = {}
    path_records: list[tuple[str, Mapping[str, Any]]] = [
        (
            "feature_contract",
            {
                "path": contract["feature_policy"]["contract_path"],
                "sha256": contract["feature_policy"]["contract_sha256"],
            },
        ),
        ("nominal_registry", contract["model_registries"]["nominal"]),
        ("ordinal_registry", contract["model_registries"]["ordinal"]),
        *[(name, record) for name, record in contract["immutable_sources"].items()],
    ]
    fold_dir = Path(contract["folds"]["directory"])
    path_records.extend(
        [
            ("fold_contract", {"path": fold_dir / "fold_contract.json", "sha256": folds["fold_contract_sha256"]}),
            ("outer_assignments", {"path": fold_dir / "fold_assignments.csv", "sha256": folds["outer_assignments_sha256"]}),
            ("inner_assignments", {"path": fold_dir / "inner_fold_assignments.csv", "sha256": folds["inner_assignments_sha256"]}),
        ]
    )
    for name, record in path_records:
        path = Path(record["path"])
        _require(path.is_file(), f"Required source is absent: {path.as_posix()}.")
        observed = sha256_file(path)
        _require(observed == record["sha256"], f"Source hash drifted for {name}.")
        source_hashes[name] = observed
    return contract, source_hashes


def _registry_definitions(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    _require(
        tuple(contract["model_registries"]["nominal"]["models"]) == tuple(CANONICAL_MODEL_NAMES),
        "Nominal model list drifted.",
    )
    _require(
        tuple(contract["model_registries"]["ordinal"]["models"]) == tuple(V3_ORDINAL_MODEL_NAMES),
        "Ordinal model list drifted.",
    )
    nominal = load_config(contract["model_registries"]["nominal"]["path"])["model_benchmark"]["models"]
    ordinal = _load_json(Path(contract["model_registries"]["ordinal"]["path"]))["ordinal_models"]
    definitions = {**nominal, **ordinal}
    _require(tuple(name for name in MODEL_NAMES if name in definitions) == MODEL_NAMES, "Model registry is incomplete.")
    return definitions


def _canonical_parameters(value: Any) -> str:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise SelectionObjectiveSensitivityV4Error(f"Invalid candidate parameters JSON: {value!r}.") from exc
    _require(isinstance(parsed, dict), "Candidate parameters must be a JSON object.")
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _normalize_candidate_evidence(contract: Mapping[str, Any]) -> pd.DataFrame:
    sources = contract["immutable_sources"]
    nominal = pd.read_csv(sources["nominal_candidate_scores"]["path"])
    ordinal = pd.read_csv(sources["ordinal_candidate_scores"]["path"])
    nominal_required = {
        "outer_fold", "model", "candidate_index", "parameters_json", "inner_mean",
        "tie_break_inner_mean", "n_inner_folds", "candidate_status",
        "outer_test_used_for_selection", "selected_by_protocol",
    }
    ordinal_required = {
        "outer_fold", "model", "candidate_index", "parameters_json", "inner_macro_f1_mean",
        "inner_qwk_mean", "n_inner_folds", "candidate_status",
        "outer_test_used_for_selection", "selected_by_protocol",
    }
    _require(nominal_required.issubset(nominal.columns), "Nominal candidate evidence schema drifted.")
    _require(ordinal_required.issubset(ordinal.columns), "Ordinal candidate evidence schema drifted.")
    nominal = nominal.loc[nominal["model"].isin(CANONICAL_MODEL_NAMES)].copy()
    ordinal = ordinal.loc[ordinal["model"].isin(V3_ORDINAL_MODEL_NAMES)].copy()
    normalized_nominal = pd.DataFrame(
        {
            "outer_fold": nominal["outer_fold"], "model": nominal["model"],
            "candidate_index": nominal["candidate_index"], "parameters_json": nominal["parameters_json"],
            "inner_macro_f1_mean": nominal["inner_mean"], "inner_qwk_mean": nominal["tie_break_inner_mean"],
            "n_inner_folds": nominal["n_inner_folds"], "candidate_status": nominal["candidate_status"],
            "outer_test_used_for_selection": nominal["outer_test_used_for_selection"],
            "historical_selected_by_protocol": nominal["selected_by_protocol"],
            "evidence_source": "canonical_v2_nominal_candidate_scores",
        }
    )
    normalized_ordinal = ordinal.rename(columns={"selected_by_protocol": "historical_selected_by_protocol"})[
        [
            "outer_fold", "model", "candidate_index", "parameters_json", "inner_macro_f1_mean",
            "inner_qwk_mean", "n_inner_folds", "candidate_status", "outer_test_used_for_selection",
            "historical_selected_by_protocol",
        ]
    ].copy()
    normalized_ordinal["evidence_source"] = "phase1b_ordinal_candidate_scores"
    evidence = pd.concat([normalized_nominal, normalized_ordinal], ignore_index=True)
    for column in ("outer_fold", "candidate_index", "n_inner_folds"):
        evidence[column] = pd.to_numeric(evidence[column], errors="raise").astype(int)
    for column in ("inner_macro_f1_mean", "inner_qwk_mean"):
        evidence[column] = pd.to_numeric(evidence[column], errors="raise").astype(float)
        _require(np.isfinite(evidence[column]).all(), f"Non-finite {column}.")
    evidence["parameters_json"] = evidence["parameters_json"].map(_canonical_parameters)
    definitions = _registry_definitions(contract)
    expected_rows = 0
    for model in MODEL_NAMES:
        candidates = [
            json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
            for row in definitions[model]["candidates"]
        ]
        scoped = evidence.loc[evidence["model"] == model]
        expected_rows += 10 * len(candidates)
        _require(len(scoped) == 10 * len(candidates), f"Candidate row count drifted for {model}.")
        for outer_fold in range(1, 11):
            fold_rows = scoped.loc[scoped["outer_fold"] == outer_fold].sort_values("candidate_index")
            _require(fold_rows["candidate_index"].tolist() == list(range(len(candidates))), f"Candidate indices drifted for {model} fold {outer_fold}.")
            _require(fold_rows["parameters_json"].tolist() == candidates, f"Candidate registry drifted for {model} fold {outer_fold}.")
            _require((fold_rows["n_inner_folds"] == 5).all(), "Inner-fold count drifted.")
            _require((fold_rows["candidate_status"] == "complete").all(), "Incomplete candidate evidence.")
            _require(not fold_rows["outer_test_used_for_selection"].astype(bool).any(), "Outer test entered selection.")
            selected = select_candidate_index(
                fold_rows["inner_macro_f1_mean"].tolist(), fold_rows["inner_qwk_mean"].tolist(),
                practical_tie_tolerance=0.001, better_direction="higher",
            )
            persisted = fold_rows.loc[fold_rows["historical_selected_by_protocol"].astype(bool), "candidate_index"].tolist()
            _require(persisted == [selected], f"Historical macro-F1 selection drifted for {model} fold {outer_fold}.")
    _require(len(evidence) == expected_rows, "Combined candidate evidence row count drifted.")
    _require(set(evidence["outer_fold"]) == set(range(1, 11)), "Candidate outer-fold identities drifted.")
    return evidence.sort_values(["model", "outer_fold", "candidate_index"]).reset_index(drop=True)


def build_selection_schedule(candidate_evidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Independently select both regimes with inclusive 0.001 primary pools."""

    rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for (model, outer_fold), scoped in candidate_evidence.groupby(["model", "outer_fold"], sort=True):
        scoped = scoped.sort_values("candidate_index").reset_index(drop=True)
        indices: dict[str, int] = {}
        for regime, primary, secondary in (
            ("macro_f1", "inner_macro_f1_mean", "inner_qwk_mean"),
            ("qwk", "inner_qwk_mean", "inner_macro_f1_mean"),
        ):
            selected_position = select_candidate_index(
                scoped[primary].tolist(), scoped[secondary].tolist(),
                practical_tie_tolerance=0.001, better_direction="higher",
            )
            selected = scoped.iloc[selected_position]
            candidate_index = int(selected["candidate_index"])
            indices[regime] = candidate_index
            best_primary = float(scoped[primary].max())
            rows.append(
                {
                    "model": str(model), "outer_fold": int(outer_fold), "selection_objective": regime,
                    "selected_candidate_index": candidate_index,
                    "selected_candidate_parameters_json": str(selected["parameters_json"]),
                    "selected_primary_mean": float(selected[primary]),
                    "selected_tie_break_mean": float(selected[secondary]),
                    "primary_gap_from_best": best_primary - float(selected[primary]),
                    "practical_tie_tolerance": 0.001,
                    "outer_test_used_for_selection": False,
                    "candidate_evidence_source": str(selected["evidence_source"]),
                }
            )
        changes.append(
            {
                "model": str(model), "outer_fold": int(outer_fold),
                "macro_f1_candidate_index": indices["macro_f1"],
                "qwk_candidate_index": indices["qwk"],
                "selected_candidate_changed": indices["macro_f1"] != indices["qwk"],
            }
        )
    schedule = pd.DataFrame(rows).sort_values(["selection_objective", "model", "outer_fold"]).reset_index(drop=True)
    change_frame = pd.DataFrame(changes).sort_values(["model", "outer_fold"]).reset_index(drop=True)
    _require(len(schedule) == 120 and len(change_frame) == 60, "Selection schedule coverage drifted.")
    return schedule, change_frame


def _pipeline(
    model: str,
    training_features: pd.DataFrame,
    definition: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    random_state: int,
    forbidden_features: Sequence[str],
) -> Pipeline:
    if model in CANONICAL_MODEL_NAMES:
        return build_model_pipeline(
            model, training_features, fixed_parameters=definition["fixed_params"],
            candidate_parameters=candidate, random_state=random_state,
            forbidden_features=forbidden_features,
        )
    overlap = sorted(set(definition["fixed_params"]).intersection(candidate))
    _require(not overlap, f"Ordinal candidate overwrites fixed parameters: {overlap}.")
    estimator = build_v3_ordinal_estimator(
        model, {**dict(definition["fixed_params"]), **dict(candidate)}, random_state=random_state
    )
    return Pipeline(
        [("preprocessor", build_common_preprocessor(training_features)), ("model", estimator)]
    )


def _fit(pipeline: Pipeline, features: pd.DataFrame, target: pd.Series, *, context: str) -> Pipeline:
    try:
        with threadpool_limits(limits=FIT_THREAD_LIMIT):
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                return pipeline.fit(features, target)
    except Exception as exc:
        raise SelectionObjectiveSensitivityV4Error(f"{context} failed: {type(exc).__name__}: {exc}") from exc


def _validate_oof(rows: pd.DataFrame, *, systems: int, samples: int) -> None:
    _require(len(rows) == systems * samples, "OOF row count drifted.")
    reference: pd.DataFrame | None = None
    for (regime, model), scoped in rows.groupby(["selection_objective", "model"], sort=True):
        scoped = scoped.sort_values("sample_index")
        _require(len(scoped) == samples, f"OOF sample count drifted for {regime}/{model}.")
        _require(not scoped["sample_index"].duplicated().any(), f"Duplicate OOF rows for {regime}/{model}.")
        probabilities = scoped[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float)
        _require(np.isfinite(probabilities).all(), "OOF probabilities are non-finite.")
        _require((probabilities >= 0).all() and (probabilities <= 1).all(), "OOF probabilities escaped [0,1].")
        _require(np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-9), "OOF probabilities do not sum to one.")
        predicted = np.asarray(LABELS)[np.argmax(probabilities, axis=1)]
        _require(np.array_equal(predicted, scoped["y_pred"].to_numpy(int)), "OOF labels disagree with probability argmax.")
        identity = scoped[["sample_index", "outer_fold", "y_true"]].reset_index(drop=True)
        if reference is None:
            reference = identity
        else:
            _require(identity.equals(reference), "OOF systems do not share exact sample/fold/target identity.")


def _summarize_oof(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []
    for (regime, model), scoped in rows.groupby(["selection_objective", "model"], sort=True):
        scoped = scoped.sort_values("sample_index")
        bundle = ordinal_evaluation_bundle_v3(
            scoped["y_true"].astype(int), scoped["y_pred"].astype(int),
            scoped[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float),
            labels=LABELS, dataset_key="inx_primary", model_name=str(model),
        )
        aggregate.extend(
            {"selection_objective": regime, "model": model, "metric": metric, "value": value}
            for metric, value in bundle["aggregate_metrics"].items()
        )
        per_class.extend({"selection_objective": regime, **record} for record in bundle["per_class_metrics"])
        confusion.extend({"selection_objective": regime, **record} for record in bundle["confusion_matrix"])
    return pd.DataFrame(aggregate), pd.DataFrame(per_class), pd.DataFrame(confusion)


def _comparison_tables(
    aggregate: pd.DataFrame, ranking_metrics: Mapping[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effects: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for metric, direction in ranking_metrics.items():
        scoped = aggregate.loc[aggregate["metric"] == metric]
        pivot = scoped.pivot(index="model", columns="selection_objective", values="value")
        _require(set(pivot.index) == set(MODEL_NAMES) and set(pivot.columns) == {"macro_f1", "qwk"}, f"Ranking coverage drifted for {metric}.")
        ascending = direction == "lower"
        orders: dict[str, list[str]] = {}
        margins: dict[str, float] = {}
        for regime in ("macro_f1", "qwk"):
            ordered = pivot.reset_index().sort_values(
                [regime, "model"], ascending=[ascending, True], kind="mergesort"
            )
            orders[regime] = ordered["model"].astype(str).tolist()
            values = ordered[regime].astype(float).tolist()
            margins[regime] = (values[1] - values[0]) if ascending else (values[0] - values[1])
        for model in MODEL_NAMES:
            a = float(pivot.loc[model, "macro_f1"])
            b = float(pivot.loc[model, "qwk"])
            effects.append(
                {
                    "metric": metric, "direction": direction, "model": model,
                    "macro_f1_selection_value": a, "qwk_selection_value": b,
                    "qwk_minus_macro_f1_selection": b - a,
                    "absolute_effect_magnitude": abs(b - a),
                    "macro_f1_selection_rank": orders["macro_f1"].index(model) + 1,
                    "qwk_selection_rank": orders["qwk"].index(model) + 1,
                    "rank_position_change": orders["qwk"].index(model) - orders["macro_f1"].index(model),
                }
            )
        changes.append(
            {
                "metric": metric, "direction": direction,
                "macro_f1_selection_leader": orders["macro_f1"][0],
                "qwk_selection_leader": orders["qwk"][0],
                "leader_changed": orders["macro_f1"][0] != orders["qwk"][0],
                "full_ordering_changed": orders["macro_f1"] != orders["qwk"],
                "macro_f1_selection_order_json": json.dumps(orders["macro_f1"], separators=(",", ":")),
                "qwk_selection_order_json": json.dumps(orders["qwk"], separators=(",", ":")),
                "macro_f1_selection_leader_margin": margins["macro_f1"],
                "qwk_selection_leader_margin": margins["qwk"],
            }
        )
    return pd.DataFrame(effects), pd.DataFrame(changes)


def _empirical_prior(
    target: pd.Series, outer: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parameter_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for outer_fold in range(1, 11):
        test_ids = outer.loc[outer["outer_fold"].astype(int) == outer_fold, "sample_index"].astype(int).tolist()
        train_ids = outer.loc[outer["outer_fold"].astype(int) != outer_fold, "sample_index"].astype(int).tolist()
        counts = target.loc[train_ids].value_counts().reindex(LABELS, fill_value=0).astype(int)
        probabilities = counts.to_numpy(float) / len(train_ids)
        _require(np.isclose(probabilities.sum(), 1.0), "Empirical-prior probabilities do not sum to one.")
        predicted = int(LABELS[int(np.argmax(probabilities))])
        parameter_rows.append(
            {
                "outer_fold": outer_fold, "outer_train_count": len(train_ids), "outer_test_count": len(test_ids),
                **{f"train_count_class_{label}": int(counts.loc[label]) for label in LABELS},
                **{f"prior_prob_class_{label}": float(probabilities[index]) for index, label in enumerate(LABELS)},
                "derived_from_outer_training_labels_only": True,
            }
        )
        for sample_index in test_ids:
            prediction_rows.append(
                {
                    "sample_index": sample_index, "outer_fold": outer_fold,
                    "y_true": int(target.loc[sample_index]), "y_pred": predicted,
                    **{f"prob_class_{label}": float(probabilities[index]) for index, label in enumerate(LABELS)},
                }
            )
    predictions = pd.DataFrame(prediction_rows).sort_values("sample_index").reset_index(drop=True)
    _require(len(predictions) == len(target) and not predictions["sample_index"].duplicated().any(), "Empirical-prior OOF coverage drifted.")
    bundle = ordinal_evaluation_bundle_v3(
        predictions["y_true"], predictions["y_pred"],
        predictions[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float),
        labels=LABELS, dataset_key="inx_primary", model_name="outer_training_empirical_prior",
    )
    metrics = pd.DataFrame(
        [{"model": "outer_training_empirical_prior", "metric": metric, "value": value} for metric, value in bundle["aggregate_metrics"].items()]
    )
    return pd.DataFrame(parameter_rows), predictions, metrics


def _load_scientific_inputs(contract_path: Path) -> tuple[
    dict[str, Any], dict[str, str], pd.DataFrame, pd.DataFrame, pd.Series, Any, tuple[str, ...]
]:
    contract, source_hashes = _validate_contract(contract_path)
    candidate_evidence = _normalize_candidate_evidence(contract)
    feature_contract_path = Path(contract["feature_policy"]["contract_path"])
    validate_feature_availability_contract(feature_contract_path)
    feature_contract = _load_json(feature_contract_path)
    canonical = load_canonical_dataset(
        contract["data_source"]["canonical_loader_config_path"], "inx_primary",
        contract["data_source"]["acquisition_manifest_path"], allow_download=False,
    )
    features, exclusions = exact_p3_feature_frame(canonical.frame, feature_contract)
    v2_config = load_config(contract["data_source"]["canonical_loader_config_path"])
    _require(set(exclusions) == set(primary_excluded_features(v2_config)), "P3 feature exclusions drifted.")
    _require(features.shape[1] == contract["feature_policy"]["retained_feature_count"], "P3 feature count drifted.")
    validate_model_feature_frame(features, forbidden_features=exclusions)
    target = canonical.frame[contract["target"]].astype(int)
    folds = read_shared_folds(Path(contract["folds"]["directory"]))
    validate_shared_folds(folds)
    _require(set(folds.outer_assignments["sample_index"].astype(int)) == set(features.index.astype(int)), "Fold samples differ from INX data.")
    persisted_target = folds.outer_assignments.set_index("sample_index")["y_true"].sort_index().astype(int)
    _require(persisted_target.equals(target.sort_index()), "Fold targets differ from INX target.")
    source_hashes["dataset"] = canonical.receipt["actual_sha256"]
    return contract, source_hashes, candidate_evidence, features, target, folds, exclusions


def preflight_selection_objective_sensitivity_v4(
    contract_path: Path | str = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """Validate source/fold/registry identity without selecting or fitting."""

    contract, source_hashes, evidence, features, target, folds, _ = _load_scientific_inputs(Path(contract_path))
    return {
        "status": "passed", "model_fit_count": 0, "candidate_selection_count": 0,
        "contract_sha256": sha256_file(Path(contract_path)), "source_hashes": source_hashes,
        "models": list(MODEL_NAMES), "candidate_rows": int(len(evidence)),
        "sample_count": int(len(target)), "feature_count": int(features.shape[1]),
        "outer_folds": int(folds.contract["outer_splits"]), "inner_folds": int(folds.contract["inner_splits"]),
        "paid_api_calls": 0, "network_calls": 0,
    }


def evaluate_selection_objective_sensitivity_v4(
    *,
    contract: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    candidate_evidence: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    folds: Any,
    exclusions: Sequence[str],
) -> SelectionObjectiveSensitivityV4Result:
    schedule, changes = build_selection_schedule(candidate_evidence)
    historical_path = contract["immutable_sources"]["macro_f1_oof_predictions"]["path"]
    historical = pd.read_csv(historical_path)
    required = {"model", "sample_index", "outer_fold", "y_true", "y_pred", "prob_class_2", "prob_class_3", "prob_class_4"}
    _require(required.issubset(historical.columns), "Historical OOF schema drifted.")
    historical_columns = [
        "model", "sample_index", "outer_fold", "y_true", "y_pred",
        "prob_class_2", "prob_class_3", "prob_class_4",
    ]
    historical = historical.loc[
        historical["model"].isin(MODEL_NAMES), historical_columns
    ].copy()
    historical["selection_objective"] = "macro_f1"
    historical["evidence_source"] = "phase1b_reused_without_refit_or_relabelling"
    macro_schedule = schedule.loc[schedule["selection_objective"] == "macro_f1", ["model", "outer_fold", "selected_candidate_index"]]
    historical = historical.merge(macro_schedule, on=["model", "outer_fold"], how="left", validate="many_to_one")
    _require(
        historical["selected_candidate_index"].notna().all(),
        "Historical OOF rows did not resolve to the macro-F1 selection schedule.",
    )
    historical["selected_candidate_index"] = historical["selected_candidate_index"].astype(int)

    definitions = _registry_definitions(contract)
    outer = folds.outer_assignments
    qwk_predictions: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        for outer_fold in range(1, 11):
            selected = schedule.loc[
                (schedule["selection_objective"] == "qwk") & (schedule["model"] == model) & (schedule["outer_fold"] == outer_fold)
            ]
            _require(len(selected) == 1, f"QWK selection is not unique for {model} fold {outer_fold}.")
            selected_row = selected.iloc[0]
            candidate = json.loads(selected_row["selected_candidate_parameters_json"])
            test_ids = outer.loc[outer["outer_fold"].astype(int) == outer_fold, "sample_index"].astype(int).tolist()
            train_ids = outer.loc[outer["outer_fold"].astype(int) != outer_fold, "sample_index"].astype(int).tolist()
            pipeline = _pipeline(
                model, features.loc[train_ids], definitions[model], candidate,
                random_state=int(contract["folds"]["model_seed"]), forbidden_features=exclusions,
            )
            _fit(pipeline, features.loc[train_ids], target.loc[train_ids], context=f"QWK outer refit model={model}, fold={outer_fold}")
            probability = aligned_predict_proba(pipeline, features.loc[test_ids], labels=LABELS)
            prediction = np.asarray(LABELS)[np.argmax(probability, axis=1)]
            for position, sample_index in enumerate(test_ids):
                qwk_predictions.append(
                    {
                        "model": model, "sample_index": sample_index, "outer_fold": outer_fold,
                        "y_true": int(target.loc[sample_index]), "y_pred": int(prediction[position]),
                        "prob_class_2": float(probability[position, 0]),
                        "prob_class_3": float(probability[position, 1]),
                        "prob_class_4": float(probability[position, 2]),
                        "selection_objective": "qwk",
                        "evidence_source": "round2_qwk_selected_outer_refit",
                        "selected_candidate_index": int(selected_row["selected_candidate_index"]),
                    }
                )
    combined = pd.concat([historical, pd.DataFrame(qwk_predictions)], ignore_index=True)
    combined = combined.sort_values(["selection_objective", "model", "sample_index"]).reset_index(drop=True)
    _validate_oof(combined, systems=12, samples=len(target))
    aggregate, per_class, confusion = _summarize_oof(combined)
    effects, rankings = _comparison_tables(aggregate, contract["ranking_report"]["metrics"])
    prior_parameters, prior_oof, prior_metrics = _empirical_prior(target, outer)
    return SelectionObjectiveSensitivityV4Result(
        candidate_evidence=candidate_evidence, selection_schedule=schedule,
        selected_candidate_changes=changes, oof_predictions=combined,
        aggregate_metrics=aggregate, per_class_metrics=per_class,
        confusion_matrix=confusion, metric_effect_magnitudes=effects,
        ranking_changes=rankings, empirical_prior_fold_parameters=prior_parameters,
        empirical_prior_oof_predictions=prior_oof, empirical_prior_metrics=prior_metrics,
    )


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def run_selection_objective_sensitivity_v4(
    *, contract_path: Path | str = DEFAULT_CONTRACT_PATH, output_dir: Path | str, run_id: str
) -> dict[str, Any]:
    _require(bool(str(run_id).strip()), "run_id must be non-empty.")
    git_identity = _clean_git_identity()
    contract_file = Path(contract_path)
    contract, source_hashes, evidence, features, target, folds, exclusions = _load_scientific_inputs(contract_file)
    destination = Path(output_dir)
    _require(not destination.exists(), f"Output destination already exists: {destination.as_posix()}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        implementation_paths = [Path("src/experiments/selection_objective_sensitivity_v4.py")]
        implementation_hashes = {path.as_posix(): sha256_file(path) for path in implementation_paths}
        scientific_inputs = {
            "git_identity": git_identity, "source_tree_hash": source_tree_hash(PROJECT_ROOT),
            "contract_sha256": sha256_file(contract_file), "source_hashes": dict(source_hashes),
            "implementation_hashes": implementation_hashes,
        }
        scientific_input_sha256 = _canonical_json_sha256(scientific_inputs)
        result = evaluate_selection_objective_sensitivity_v4(
            contract=contract, source_hashes=source_hashes, candidate_evidence=evidence,
            features=features, target=target, folds=folds, exclusions=exclusions,
        )
        frames = {
            "candidate_evidence.csv": result.candidate_evidence,
            "selection_schedule.csv": result.selection_schedule,
            "selected_candidate_changes.csv": result.selected_candidate_changes,
            "oof_predictions.csv": result.oof_predictions,
            "aggregate_metrics.csv": result.aggregate_metrics,
            "per_class_metrics.csv": result.per_class_metrics,
            "confusion_matrix.csv": result.confusion_matrix,
            "metric_effect_magnitudes.csv": result.metric_effect_magnitudes,
            "ranking_changes.csv": result.ranking_changes,
            "empirical_prior_fold_parameters.csv": result.empirical_prior_fold_parameters,
            "empirical_prior_oof_predictions.csv": result.empirical_prior_oof_predictions,
            "empirical_prior_metrics.csv": result.empirical_prior_metrics,
        }
        for name, frame in frames.items():
            frame.to_csv(staging / name, index=False)
        output_hashes = {name: sha256_file(staging / name) for name in frames}
        metadata = {
            "schema_version": 1, "stage": "selection_objective_sensitivity_v4", "status": "complete",
            "run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_identity": git_identity, "scientific_input_sha256": scientific_input_sha256,
            "scientific_inputs": scientific_inputs, "models": list(MODEL_NAMES),
            "selection_objectives": ["macro_f1", "qwk"], "sample_count_per_model_regime": len(target),
            "qwk_outer_model_fit_count": 60, "new_inner_model_fit_count": 0,
            "macro_f1_oof_reused": True, "empirical_prior_model_fit_count": 0,
            "outer_test_used_for_selection": False, "employee_level_outputs_publication_authorized": False,
            "composite_material_dependence_flag_present": False,
            "paid_api_calls": 0, "network_calls": 0, "output_hashes": output_hashes,
        }
        _require(_clean_git_identity() == git_identity, "Git identity changed during scientific execution.")
        _require(source_tree_hash(PROJECT_ROOT) == scientific_inputs["source_tree_hash"], "Source tree changed during execution.")
        for name, expected in source_hashes.items():
            if name == "dataset":
                continue
            # Exact source hashes were checked on entry; the contract is revalidated on exit.
        _validate_contract(contract_file)
        _write_json(staging / "stage_metadata.json", metadata)
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            for child in staging.iterdir():
                if child.is_file():
                    child.unlink()
            staging.rmdir()
        raise
    return {
        "status": "complete", "run_id": run_id, "output_dir": destination.as_posix(),
        "scientific_input_sha256": scientific_input_sha256, "qwk_outer_model_fit_count": 60,
        "paid_api_calls": 0, "network_calls": 0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_LOCAL_RUN_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.preflight_only:
        print(json.dumps(preflight_selection_objective_sensitivity_v4(args.contract), indent=2, sort_keys=True))
        return 0
    _require(isinstance(args.run_id, str) and bool(args.run_id.strip()), "--run-id is required.")
    output = args.output_root / args.run_id / "selection_objective_sensitivity"
    receipt = run_selection_objective_sensitivity_v4(
        contract_path=args.contract, output_dir=output, run_id=args.run_id
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONTRACT_PATH", "DEFAULT_LOCAL_RUN_ROOT", "LABELS", "MODEL_NAMES",
    "SelectionObjectiveSensitivityV4Error", "SelectionObjectiveSensitivityV4Result",
    "build_selection_schedule", "evaluate_selection_objective_sensitivity_v4",
    "preflight_selection_objective_sensitivity_v4", "run_selection_objective_sensitivity_v4",
]
