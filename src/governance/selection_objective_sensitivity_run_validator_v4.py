"""Independently validate a complete local Round 2 selection/prior run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.canonical_loader import load_canonical_dataset, sha256_file
from src.experiments.shared_folds import read_shared_folds, validate_shared_folds
from src.models.ordinal_evaluation_v3 import ordinal_evaluation_bundle_v3


DEFAULT_CONTRACT_PATH = Path("configs/selection_objective_sensitivity_v4.json")
LABELS = (2, 3, 4)
MODELS = (
    "logistic_regression",
    "random_forest",
    "lightgbm",
    "xgboost",
    "proportional_odds_logistic",
    "cumulative_threshold_xgboost",
)
FRAME_FILES = (
    "candidate_evidence.csv",
    "selection_schedule.csv",
    "selected_candidate_changes.csv",
    "oof_predictions.csv",
    "aggregate_metrics.csv",
    "per_class_metrics.csv",
    "confusion_matrix.csv",
    "metric_effect_magnitudes.csv",
    "ranking_changes.csv",
    "empirical_prior_fold_parameters.csv",
    "empirical_prior_oof_predictions.csv",
    "empirical_prior_metrics.csv",
)


class SelectionObjectiveRunValidationV4Error(RuntimeError):
    """Raised when stored Round 2 evidence fails independent recomputation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectionObjectiveRunValidationV4Error(message)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectionObjectiveRunValidationV4Error(f"Could not read {path.as_posix()}: {exc}") from exc
    _require(isinstance(value, dict), f"{path.as_posix()} must contain an object.")
    return value


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise SelectionObjectiveRunValidationV4Error(f"Could not read {path.as_posix()}: {exc}") from exc


def _assert_frame(actual: pd.DataFrame, expected: pd.DataFrame, *, name: str, atol: float = 1e-12) -> None:
    _require(list(actual.columns) == list(expected.columns), f"{name} columns drifted.")
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True), expected.reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=0.0, atol=atol,
        )
    except AssertionError as exc:
        raise SelectionObjectiveRunValidationV4Error(f"{name} recomputation mismatch: {exc}") from exc


def _git_blob_sha256(commit: str, path: str) -> str:
    try:
        value = subprocess.run(
            ["git", "show", f"{commit}:{path}"], check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SelectionObjectiveRunValidationV4Error(f"Could not read generation blob {commit}:{path}.") from exc
    return hashlib.sha256(value).hexdigest()


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    _require(mapped.notna().all(), "Boolean evidence contains invalid values.")
    return mapped.astype(bool)


def _candidate_evidence(contract: Mapping[str, Any]) -> pd.DataFrame:
    sources = contract["immutable_sources"]
    nominal = _frame(Path(sources["nominal_candidate_scores"]["path"]))
    ordinal = _frame(Path(sources["ordinal_candidate_scores"]["path"]))
    nominal = nominal.loc[nominal["model"].isin(MODELS[:4])]
    ordinal = ordinal.loc[ordinal["model"].isin(MODELS[4:])]
    a = pd.DataFrame(
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
    b = ordinal.rename(columns={"selected_by_protocol": "historical_selected_by_protocol"})[
        [
            "outer_fold", "model", "candidate_index", "parameters_json", "inner_macro_f1_mean",
            "inner_qwk_mean", "n_inner_folds", "candidate_status", "outer_test_used_for_selection",
            "historical_selected_by_protocol",
        ]
    ].copy()
    b["evidence_source"] = "phase1b_ordinal_candidate_scores"
    evidence = pd.concat([a, b], ignore_index=True)
    for column in ("outer_fold", "candidate_index", "n_inner_folds"):
        evidence[column] = pd.to_numeric(evidence[column], errors="raise").astype(int)
    for column in ("inner_macro_f1_mean", "inner_qwk_mean"):
        evidence[column] = pd.to_numeric(evidence[column], errors="raise").astype(float)
    evidence["parameters_json"] = evidence["parameters_json"].map(
        lambda value: json.dumps(json.loads(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    return evidence.sort_values(["model", "outer_fold", "candidate_index"]).reset_index(drop=True)


def _select(scoped: pd.DataFrame, primary: str, secondary: str) -> pd.Series:
    values = scoped[primary].to_numpy(float)
    best = float(np.max(values))
    eligible = scoped.loc[(best - scoped[primary].astype(float)) <= 0.001 + 1e-15]
    best_secondary = float(eligible[secondary].max())
    winners = eligible.loc[eligible[secondary].astype(float) == best_secondary]
    return winners.sort_values("candidate_index").iloc[0]


def _selection_tables(evidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for (model, outer_fold), scoped in evidence.groupby(["model", "outer_fold"], sort=True):
        scoped = scoped.sort_values("candidate_index")
        _require(len(scoped) in (6, 8), f"Candidate count drifted for {model}/{outer_fold}.")
        selected_indices: dict[str, int] = {}
        for regime, primary, secondary in (
            ("macro_f1", "inner_macro_f1_mean", "inner_qwk_mean"),
            ("qwk", "inner_qwk_mean", "inner_macro_f1_mean"),
        ):
            row = _select(scoped, primary, secondary)
            selected = int(row["candidate_index"])
            selected_indices[regime] = selected
            schedule.append(
                {
                    "model": model, "outer_fold": int(outer_fold), "selection_objective": regime,
                    "selected_candidate_index": selected,
                    "selected_candidate_parameters_json": row["parameters_json"],
                    "selected_primary_mean": float(row[primary]),
                    "selected_tie_break_mean": float(row[secondary]),
                    "primary_gap_from_best": float(scoped[primary].max() - row[primary]),
                    "practical_tie_tolerance": 0.001,
                    "outer_test_used_for_selection": False,
                    "candidate_evidence_source": row["evidence_source"],
                }
            )
        persisted = scoped.loc[_truthy(scoped["historical_selected_by_protocol"]), "candidate_index"].astype(int).tolist()
        _require(persisted == [selected_indices["macro_f1"]], f"Historical selection mismatch for {model}/{outer_fold}.")
        changes.append(
            {
                "model": model, "outer_fold": int(outer_fold),
                "macro_f1_candidate_index": selected_indices["macro_f1"],
                "qwk_candidate_index": selected_indices["qwk"],
                "selected_candidate_changed": selected_indices["macro_f1"] != selected_indices["qwk"],
            }
        )
    return (
        pd.DataFrame(schedule).sort_values(["selection_objective", "model", "outer_fold"]).reset_index(drop=True),
        pd.DataFrame(changes).sort_values(["model", "outer_fold"]).reset_index(drop=True),
    )


def _validate_oof(oof: pd.DataFrame, schedule: pd.DataFrame) -> None:
    required = {
        "selection_objective", "model", "sample_index", "outer_fold", "y_true", "y_pred",
        "prob_class_2", "prob_class_3", "prob_class_4", "selected_candidate_index", "evidence_source",
    }
    _require(required.issubset(oof.columns), "OOF schema drifted.")
    _require(len(oof) == 2 * len(MODELS) * 1200, "OOF row count drifted.")
    reference: pd.DataFrame | None = None
    for (regime, model), rows in oof.groupby(["selection_objective", "model"], sort=True):
        rows = rows.sort_values("sample_index")
        _require(len(rows) == 1200 and not rows["sample_index"].duplicated().any(), f"OOF coverage drifted for {regime}/{model}.")
        probability = rows[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float)
        _require(np.isfinite(probability).all(), "OOF probabilities are non-finite.")
        _require((probability >= 0).all() and (probability <= 1).all(), "OOF probabilities escaped [0,1].")
        _require(np.allclose(probability.sum(axis=1), 1.0, rtol=0, atol=1e-9), "OOF probability simplex failed.")
        _require(np.array_equal(np.asarray(LABELS)[np.argmax(probability, axis=1)], rows["y_pred"].astype(int)), "OOF argmax labels drifted.")
        identity = rows[["sample_index", "outer_fold", "y_true"]].astype(int).reset_index(drop=True)
        if reference is None:
            reference = identity
        else:
            _assert_frame(identity, reference, name="OOF identity", atol=0)
        expected_schedule = schedule.loc[
            (schedule["selection_objective"] == regime) & (schedule["model"] == model),
            ["outer_fold", "selected_candidate_index"],
        ]
        observed = rows[["outer_fold", "selected_candidate_index"]].drop_duplicates().sort_values("outer_fold").reset_index(drop=True)
        _assert_frame(observed, expected_schedule.sort_values("outer_fold").reset_index(drop=True), name=f"OOF schedule {regime}/{model}", atol=0)


def _summary(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []
    for (regime, model), rows in oof.groupby(["selection_objective", "model"], sort=True):
        rows = rows.sort_values("sample_index")
        bundle = ordinal_evaluation_bundle_v3(
            rows["y_true"].astype(int), rows["y_pred"].astype(int),
            rows[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float),
            labels=LABELS, dataset_key="inx_primary", model_name=model,
        )
        aggregate.extend(
            {"selection_objective": regime, "model": model, "metric": metric, "value": value}
            for metric, value in bundle["aggregate_metrics"].items()
        )
        classes.extend({"selection_objective": regime, **row} for row in bundle["per_class_metrics"])
        confusion.extend({"selection_objective": regime, **row} for row in bundle["confusion_matrix"])
    return (
        pd.DataFrame(aggregate).sort_values(
            ["selection_objective", "model", "metric"]
        ).reset_index(drop=True),
        pd.DataFrame(classes).sort_values(
            ["selection_objective", "model_name", "class_label"]
        ).reset_index(drop=True),
        pd.DataFrame(confusion).sort_values(
            ["selection_objective", "model_name", "true_label", "predicted_label"]
        ).reset_index(drop=True),
    )


def _comparison(aggregate: pd.DataFrame, metrics: Mapping[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    effects: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for metric, direction in metrics.items():
        pivot = aggregate.loc[aggregate["metric"] == metric].pivot(index="model", columns="selection_objective", values="value")
        ascending = direction == "lower"
        orders: dict[str, list[str]] = {}
        margins: dict[str, float] = {}
        for regime in ("macro_f1", "qwk"):
            ranked = pivot.reset_index().sort_values([regime, "model"], ascending=[ascending, True], kind="mergesort")
            orders[regime] = ranked["model"].tolist()
            values = ranked[regime].astype(float).tolist()
            margins[regime] = values[1] - values[0] if ascending else values[0] - values[1]
        for model in MODELS:
            a, b = float(pivot.loc[model, "macro_f1"]), float(pivot.loc[model, "qwk"])
            effects.append(
                {
                    "metric": metric, "direction": direction, "model": model,
                    "macro_f1_selection_value": a, "qwk_selection_value": b,
                    "qwk_minus_macro_f1_selection": b - a, "absolute_effect_magnitude": abs(b - a),
                    "macro_f1_selection_rank": orders["macro_f1"].index(model) + 1,
                    "qwk_selection_rank": orders["qwk"].index(model) + 1,
                    "rank_position_change": orders["qwk"].index(model) - orders["macro_f1"].index(model),
                }
            )
        changes.append(
            {
                "metric": metric, "direction": direction,
                "macro_f1_selection_leader": orders["macro_f1"][0], "qwk_selection_leader": orders["qwk"][0],
                "leader_changed": orders["macro_f1"][0] != orders["qwk"][0],
                "full_ordering_changed": orders["macro_f1"] != orders["qwk"],
                "macro_f1_selection_order_json": json.dumps(orders["macro_f1"], separators=(",", ":")),
                "qwk_selection_order_json": json.dumps(orders["qwk"], separators=(",", ":")),
                "macro_f1_selection_leader_margin": margins["macro_f1"],
                "qwk_selection_leader_margin": margins["qwk"],
            }
        )
    return pd.DataFrame(effects), pd.DataFrame(changes)


def _prior(target: pd.Series, outer: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parameters: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for fold in range(1, 11):
        test = outer.loc[outer["outer_fold"].astype(int) == fold, "sample_index"].astype(int).tolist()
        train = outer.loc[outer["outer_fold"].astype(int) != fold, "sample_index"].astype(int).tolist()
        counts = target.loc[train].value_counts().reindex(LABELS, fill_value=0).astype(int)
        probability = counts.to_numpy(float) / len(train)
        parameters.append(
            {
                "outer_fold": fold, "outer_train_count": len(train), "outer_test_count": len(test),
                **{f"train_count_class_{label}": int(counts.loc[label]) for label in LABELS},
                **{f"prior_prob_class_{label}": float(probability[index]) for index, label in enumerate(LABELS)},
                "derived_from_outer_training_labels_only": True,
            }
        )
        predicted = int(LABELS[int(np.argmax(probability))])
        for sample in test:
            predictions.append(
                {
                    "sample_index": sample, "outer_fold": fold, "y_true": int(target.loc[sample]), "y_pred": predicted,
                    **{f"prob_class_{label}": float(probability[index]) for index, label in enumerate(LABELS)},
                }
            )
    oof = pd.DataFrame(predictions).sort_values("sample_index").reset_index(drop=True)
    bundle = ordinal_evaluation_bundle_v3(
        oof["y_true"], oof["y_pred"], oof[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float),
        labels=LABELS, dataset_key="inx_primary", model_name="outer_training_empirical_prior",
    )
    metrics = pd.DataFrame(
        [{"model": "outer_training_empirical_prior", "metric": name, "value": value} for name, value in bundle["aggregate_metrics"].items()]
    ).sort_values(["model", "metric"]).reset_index(drop=True)
    return pd.DataFrame(parameters), oof, metrics


def validate_selection_objective_sensitivity_run_v4(
    run_dir: Path | str, contract_path: Path | str = DEFAULT_CONTRACT_PATH
) -> dict[str, Any]:
    root = Path(run_dir)
    contract_file = Path(contract_path)
    contract = _json(contract_file)
    metadata = _json(root / "stage_metadata.json")
    _require(metadata.get("stage") == "selection_objective_sensitivity_v4" and metadata.get("status") == "complete", "Run status drifted.")
    _require(metadata.get("qwk_outer_model_fit_count") == 60 and metadata.get("new_inner_model_fit_count") == 0, "Fit-count receipt drifted.")
    _require(metadata.get("macro_f1_oof_reused") is True and metadata.get("empirical_prior_model_fit_count") == 0, "Reuse/prior receipt drifted.")
    _require(metadata.get("composite_material_dependence_flag_present") is False, "Composite materiality flag is prohibited.")
    _require(metadata.get("outer_test_used_for_selection") is False, "Outer test entered selection.")
    _require(metadata.get("paid_api_calls") == 0 and metadata.get("network_calls") == 0, "Network/API receipt drifted.")
    _require(metadata["scientific_inputs"]["contract_sha256"] == sha256_file(contract_file), "Contract digest drifted.")
    _require(metadata["scientific_input_sha256"] == _canonical_digest(metadata["scientific_inputs"]), "Scientific-input digest drifted.")
    for name in FRAME_FILES:
        path = root / name
        _require(path.is_file(), f"Run output is missing: {name}.")
        _require(sha256_file(path) == metadata["output_hashes"].get(name), f"Output hash drifted: {name}.")
    _require(set(metadata["output_hashes"]) == set(FRAME_FILES), "Output hash inventory drifted.")
    commit = metadata["git_identity"]["commit"]
    for path, expected in metadata["scientific_inputs"]["implementation_hashes"].items():
        _require(_git_blob_sha256(commit, path) == expected, f"Generation implementation blob drifted: {path}.")
    for name, record in contract["immutable_sources"].items():
        _require(sha256_file(Path(record["path"])) == record["sha256"], f"Immutable source drifted: {name}.")

    evidence = _candidate_evidence(contract)
    _assert_frame(_frame(root / "candidate_evidence.csv"), evidence, name="candidate evidence")
    schedule, candidate_changes = _selection_tables(evidence)
    _assert_frame(_frame(root / "selection_schedule.csv"), schedule, name="selection schedule")
    _assert_frame(_frame(root / "selected_candidate_changes.csv"), candidate_changes, name="candidate changes")
    oof = _frame(root / "oof_predictions.csv")
    _validate_oof(oof, schedule)
    historical = _frame(Path(contract["immutable_sources"]["macro_f1_oof_predictions"]["path"]))
    columns = ["model", "sample_index", "outer_fold", "y_true", "y_pred", "prob_class_2", "prob_class_3", "prob_class_4"]
    expected_historical = historical.loc[historical["model"].isin(MODELS), columns].sort_values(["model", "sample_index"]).reset_index(drop=True)
    observed_historical = oof.loc[oof["selection_objective"] == "macro_f1", columns].sort_values(["model", "sample_index"]).reset_index(drop=True)
    _assert_frame(observed_historical, expected_historical, name="historical OOF replay")

    aggregate, per_class, confusion = _summary(oof)
    _assert_frame(_frame(root / "aggregate_metrics.csv"), aggregate, name="aggregate metrics")
    _assert_frame(_frame(root / "per_class_metrics.csv"), per_class, name="per-class metrics")
    _assert_frame(_frame(root / "confusion_matrix.csv"), confusion, name="confusion matrix", atol=0)
    effects, rankings = _comparison(aggregate, contract["ranking_report"]["metrics"])
    _assert_frame(_frame(root / "metric_effect_magnitudes.csv"), effects, name="metric effects")
    _assert_frame(_frame(root / "ranking_changes.csv"), rankings, name="ranking changes")
    _require(not any("material" in column.lower() for column in rankings.columns), "Ranking output contains a composite materiality field.")

    canonical = load_canonical_dataset(
        contract["data_source"]["canonical_loader_config_path"], "inx_primary",
        contract["data_source"]["acquisition_manifest_path"], allow_download=False,
    )
    target = canonical.frame[contract["target"]].astype(int)
    folds = read_shared_folds(Path(contract["folds"]["directory"]))
    validate_shared_folds(folds)
    prior_parameters, prior_oof, prior_metrics = _prior(target, folds.outer_assignments)
    _assert_frame(_frame(root / "empirical_prior_fold_parameters.csv"), prior_parameters, name="prior parameters")
    _assert_frame(_frame(root / "empirical_prior_oof_predictions.csv"), prior_oof, name="prior OOF")
    _assert_frame(_frame(root / "empirical_prior_metrics.csv"), prior_metrics, name="prior metrics")

    return {
        "status": "passed", "run_id": metadata["run_id"], "run_dir": root.as_posix(),
        "model_regime_oof_rows": int(len(oof)), "candidate_rows": int(len(evidence)),
        "selection_schedule_rows": int(len(schedule)), "qwk_outer_model_fit_count": 60,
        "empirical_prior_oof_rows": int(len(prior_oof)), "independent_metric_recomputation": True,
        "historical_oof_replay_exact": True, "composite_material_dependence_flag_present": False,
        "paid_api_calls": 0, "network_calls": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(validate_selection_objective_sensitivity_run_v4(args.run_dir, args.contract), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SelectionObjectiveRunValidationV4Error", "validate_selection_objective_sensitivity_run_v4"]
