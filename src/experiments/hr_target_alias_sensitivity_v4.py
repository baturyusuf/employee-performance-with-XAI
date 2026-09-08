"""Run the matched-sample Round 2 HR target-alias sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.data.canonical_loader import sha256_file
from src.data.external_adapters import build_feature_columns, load_external_dataset
from src.experiments.manuscript_model_benchmark import select_candidate_index
from src.experiments.shared_folds import generate_shared_folds, validate_shared_folds
from src.governance.manuscript_contract import source_tree_hash
from src.models.canonical_models import aligned_predict_proba, build_model_pipeline
from src.models.evaluate import classification_metrics
from src.models.ordinal_evaluation_v3 import ordinal_evaluation_bundle_v3
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_CONTRACT_PATH = Path("configs/hr_target_alias_sensitivity_v4.json")
DEFAULT_LOCAL_RUN_ROOT = Path("reports/major_revision_round2_runs")
LABELS = (2, 3, 4)
ARMS = ("historical_canonical_311", "restricted_canonical_309", "exclusion_refit_309")
FIT_THREAD_LIMIT = 1


class HRTargetAliasSensitivityV4Error(RuntimeError):
    """Raised when a target-alias sensitivity invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HRTargetAliasSensitivityV4Error(message)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HRTargetAliasSensitivityV4Error(f"Could not read {path.as_posix()}: {exc}") from exc


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _clean_git_identity() -> dict[str, str]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HRTargetAliasSensitivityV4Error(f"Could not establish Git identity: {exc}") from exc
    _require(len(head) == 40 and all(ch in "0123456789abcdef" for ch in head), "Invalid Git HEAD.")
    _require(bool(branch) and not status, f"Scientific execution requires a clean named branch: {status.splitlines()[:10]}.")
    return {"commit": head, "branch": branch}


def _validate_contract(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    contract = _load_json(path)
    _require(isinstance(contract, dict), "HR alias contract must be an object.")
    _require(contract.get("schema_version") == 1 and contract.get("contract_id") == "hr_target_alias_sensitivity_v4", "HR alias contract identity drifted.")
    _require(tuple(contract["target"]["ordered_labels"]) == LABELS, "Ordered labels drifted.")
    _require(contract["target"]["expected_disagreement_count"] == 2, "Expected disagreement count drifted.")
    _require(contract["target"]["canonical_rule_changes"] is False, "Canonical target rule must remain unchanged.")
    design = contract["phase3a_design"]
    expected_design = {
        "formulation_id": "primary_three_class", "repetition": 1, "outer_splits": 5,
        "inner_splits": 5, "outer_seed": 4201, "inner_seed": 4301,
        "model_seed": 4401, "selection_metric": "macro_f1",
        "tie_break_metric": "quadratic_weighted_kappa", "primary_tie_tolerance": 0.001,
        "candidate_count": 8,
    }
    _require(all(design.get(key) == value for key, value in expected_design.items()), "Phase 3A design drifted.")
    _require([row["id"] for row in contract["arms"]] == list(ARMS), "Sensitivity arms drifted.")
    comparisons = contract["comparisons"]
    _require(
        comparisons["sample_removal_effect"] == {
            "left": "historical_canonical_311", "right": "restricted_canonical_309",
            "interpretation": "fit_free_population_change_only",
        },
        "Sample-removal comparison drifted.",
    )
    _require(
        comparisons["matched_refit_effect"] == {
            "left": "restricted_canonical_309", "right": "exclusion_refit_309",
            "interpretation": "primary_training_and_data_rule_sensitivity_on_identical_population",
        },
        "Matched-refit comparison drifted.",
    )
    _require(contract["excluded_scope"] == ["sigmoid_calibration", "naive_baselines", "additional_repetitions"], "Excluded scope drifted.")
    publication = contract["publication"]
    _require(publication == {
        "local_output_root": DEFAULT_LOCAL_RUN_ROOT.as_posix(), "publish_row_identities": False,
        "publish_employee_level_oof": False, "paid_api_calls": 0, "network_calls": 0,
    }, "Publication controls drifted.")
    hashes: dict[str, str] = {}
    for name, record in contract["sources"].items():
        source = Path(record["path"])
        _require(source.is_file(), f"Required HR source is absent: {source.as_posix()}.")
        hashes[name] = sha256_file(source)
        _require(hashes[name] == record["sha256"], f"HR source hash drifted: {name}.")
    return contract, hashes


def identify_disagreement_indices(raw: pd.DataFrame, contract: Mapping[str, Any]) -> tuple[int, ...]:
    target = contract["target"]
    text = raw[target["canonical_column"]].astype(str).str.strip()
    expected_code = text.map(target["alias_audit_mapping"])
    observed_code = pd.to_numeric(raw[target["alias_column"]], errors="raise").astype(int)
    _require(expected_code.notna().all(), "Alias audit mapping is incomplete.")
    disagreements = tuple(int(value) for value in raw.index[expected_code.astype(int) != observed_code])
    _require(len(disagreements) == target["expected_disagreement_count"], "Target-alias disagreement count drifted.")
    return disagreements


def _prepare(contract_path: Path) -> tuple[dict[str, Any], dict[str, str], Any, pd.DataFrame, pd.Series, Mapping[str, Any], Any, tuple[int, ...]]:
    contract, hashes = _validate_contract(contract_path)
    dataset = load_external_dataset(
        "hrdataset_v14", raw_path=PROJECT_ROOT / contract["sources"]["raw_dataset"]["path"],
        schema_mapping_path=PROJECT_ROOT / contract["sources"]["schema_mapping"]["path"],
    )
    _require(dataset.raw.index.equals(pd.RangeIndex(311)), "HR raw sample index drifted.")
    features_order = tuple(build_feature_columns(dataset, "conservative_primary"))
    _require(features_order == tuple(contract["feature_policy"]["exact_features"]), "Conservative feature order drifted.")
    features = dataset.canonical.loc[:, list(features_order)].copy()
    raw_target = dataset.raw[contract["target"]["canonical_column"]].astype(str).str.strip()
    target = raw_target.map(contract["target"]["canonical_mapping"])
    _require(target.notna().all(), "Canonical target mapping is incomplete.")
    target = target.astype(int)
    target.index = features.index
    _require(tuple(sorted(target.unique())) == LABELS, "Canonical HR target support drifted.")
    disagreements = identify_disagreement_indices(dataset.raw, contract)
    model_grid = load_config(contract["sources"]["model_grid"]["path"])
    definition = model_grid["model_benchmark"]["models"]["xgboost"]
    _require(len(definition["candidates"]) == 8, "XGBoost candidate count drifted.")

    phase3a_oof = pd.read_csv(contract["sources"]["phase3a_oof_predictions"]["path"])
    canonical_oof = phase3a_oof.loc[
        (phase3a_oof["repetition"].astype(int) == 1)
        & (phase3a_oof["formulation_id"] == "primary_three_class")
        & (phase3a_oof["system"] == "xgboost_raw")
    ].copy()
    _require(len(canonical_oof) == 311 and not canonical_oof["sample_index"].duplicated().any(), "Phase 3A canonical OOF coverage drifted.")
    canonical_oof = canonical_oof.sort_values("sample_index").reset_index(drop=True)
    _require(canonical_oof["sample_index"].astype(int).tolist() == list(range(311)), "Phase 3A sample indices drifted.")
    _require(np.array_equal(canonical_oof["y_true"].astype(int), target.to_numpy()), "Phase 3A targets drifted.")

    source = dataset.canonical.copy()
    source["Phase3ATarget"] = target
    original = generate_shared_folds(
        source, target_column="Phase3ATarget", id_column="ExternalSampleId",
        run_id=f"{contract['phase3a_design']['source_run_id']}_primary_three_class_rep1",
        config_hash=contract["sources"]["phase3a_contract"]["sha256"],
        scientific_input_hash=contract["phase3a_design"]["source_scientific_input_sha256"],
        dataset_key="hrdataset_v14_primary_three_class",
        dataset_sha256=contract["sources"]["raw_dataset"]["sha256"],
        outer_splits=5, inner_splits=5, seed=4201, inner_seed=4301,
    )
    validate_shared_folds(original)
    _require(original.contract["fold_contract_hash"] == contract["phase3a_design"]["fold_contract_hash"], "Reconstructed Phase 3A fold identity drifted.")
    observed_outer = canonical_oof.set_index("sample_index")["outer_fold"].astype(int).sort_index()
    expected_outer = original.outer_assignments.set_index("sample_index")["outer_fold"].astype(int).sort_index()
    _require(observed_outer.equals(expected_outer), "Canonical OOF outer folds differ from reconstructed Phase 3A folds.")
    return contract, hashes, dataset, features, target, definition, original, disagreements


def _fit(pipeline: Any, features: pd.DataFrame, target: pd.Series, *, context: str) -> Any:
    try:
        with threadpool_limits(limits=FIT_THREAD_LIMIT):
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                return pipeline.fit(features, target)
    except Exception as exc:
        raise HRTargetAliasSensitivityV4Error(f"{context} failed: {type(exc).__name__}: {exc}") from exc


def _scores(target: pd.Series, prediction: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    values = classification_metrics(target, prediction, probability, list(LABELS))
    return float(values["macro_f1"]), float(values["quadratic_weighted_kappa"])


def _summaries(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []
    for arm in ARMS:
        rows = oof.loc[oof["arm"] == arm].sort_values("sample_index")
        bundle = ordinal_evaluation_bundle_v3(
            rows["y_true"].astype(int), rows["y_pred"].astype(int),
            rows[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float),
            labels=LABELS, dataset_key="hrdataset_v14", model_name="xgboost_raw",
        )
        aggregate.extend({"arm": arm, "metric": name, "value": value, "sample_count": len(rows)} for name, value in bundle["aggregate_metrics"].items())
        classes.extend({"arm": arm, **row} for row in bundle["per_class_metrics"])
        confusion.extend({"arm": arm, **row} for row in bundle["confusion_matrix"])
    return (
        pd.DataFrame(aggregate).sort_values(["arm", "metric"]).reset_index(drop=True),
        pd.DataFrame(classes).sort_values(["arm", "class_label"]).reset_index(drop=True),
        pd.DataFrame(confusion).sort_values(["arm", "true_label", "predicted_label"]).reset_index(drop=True),
    )


def build_comparison_deltas(aggregate: pd.DataFrame, contract: Mapping[str, Any]) -> pd.DataFrame:
    pivot = aggregate.pivot(index="metric", columns="arm", values="value")
    rows: list[dict[str, Any]] = []
    for comparison_id, comparison in contract["comparisons"].items():
        left, right = comparison["left"], comparison["right"]
        for metric in sorted(pivot.index):
            left_value, right_value = float(pivot.loc[metric, left]), float(pivot.loc[metric, right])
            rows.append(
                {
                    "comparison_id": comparison_id, "interpretation": comparison["interpretation"],
                    "left_arm": left, "right_arm": right, "metric": metric,
                    "left_value": left_value, "right_value": right_value,
                    "right_minus_left": right_value - left_value,
                    "absolute_difference": abs(right_value - left_value),
                    "evaluation_population_matched": comparison_id == "matched_refit_effect",
                }
            )
    return pd.DataFrame(rows).sort_values(["comparison_id", "metric"]).reset_index(drop=True)


def preflight_hr_target_alias_sensitivity_v4(contract_path: Path | str = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    contract, hashes, _, features, target, definition, folds, disagreements = _prepare(Path(contract_path))
    retained = target.drop(index=list(disagreements))
    return {
        "status": "passed", "model_fit_count": 0, "contract_sha256": sha256_file(Path(contract_path)),
        "source_hashes": hashes, "historical_sample_count": len(target), "matched_sample_count": len(retained),
        "disagreement_count": len(disagreements), "historical_support": {str(k): int(v) for k, v in target.value_counts().sort_index().items()},
        "matched_support": {str(k): int(v) for k, v in retained.value_counts().sort_index().items()},
        "feature_count": features.shape[1], "candidate_count": len(definition["candidates"]),
        "outer_folds": folds.contract["outer_splits"], "inner_folds": folds.contract["inner_splits"],
        "row_identities_published": False, "paid_api_calls": 0, "network_calls": 0,
    }


def run_hr_target_alias_sensitivity_v4(
    *, contract_path: Path | str = DEFAULT_CONTRACT_PATH, output_dir: Path | str, run_id: str
) -> dict[str, Any]:
    _require(bool(str(run_id).strip()), "run_id must be non-empty.")
    git_identity = _clean_git_identity()
    contract_file = Path(contract_path)
    contract, source_hashes, _, features, target, definition, folds, disagreements = _prepare(contract_file)
    destination = Path(output_dir)
    _require(not destination.exists(), f"Output destination already exists: {destination.as_posix()}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        implementation_path = Path("src/experiments/hr_target_alias_sensitivity_v4.py")
        scientific_inputs = {
            "git_identity": git_identity, "source_tree_hash": source_tree_hash(PROJECT_ROOT),
            "contract_sha256": sha256_file(contract_file), "source_hashes": source_hashes,
            "implementation_hashes": {implementation_path.as_posix(): sha256_file(implementation_path)},
            "disagreement_sample_set_sha256": _canonical_json_sha256(sorted(disagreements)),
        }
        scientific_input_sha256 = _canonical_json_sha256(scientific_inputs)
        original_oof = pd.read_csv(contract["sources"]["phase3a_oof_predictions"]["path"])
        original_oof = original_oof.loc[
            (original_oof["repetition"].astype(int) == 1)
            & (original_oof["formulation_id"] == "primary_three_class")
            & (original_oof["system"] == "xgboost_raw")
        ].sort_values("sample_index")
        base_columns = ["sample_index", "outer_fold", "y_true", "y_pred", "selected_candidate_index", "prob_class_2", "prob_class_3", "prob_class_4"]
        historical = original_oof[base_columns].copy()
        historical["arm"] = "historical_canonical_311"
        historical["evidence_source"] = "phase3a_repetition1_raw_oof_reused"
        restricted = historical.loc[~historical["sample_index"].astype(int).isin(disagreements)].copy()
        restricted["arm"] = "restricted_canonical_309"
        restricted["evidence_source"] = "fit_free_phase3a_oof_restricted_to_matched_population"

        outer = folds.outer_assignments
        inner = folds.inner_assignments
        retained_ids = set(int(value) for value in features.index) - set(disagreements)
        candidate_rows: list[dict[str, Any]] = []
        selected_rows: list[dict[str, Any]] = []
        refit_rows: list[dict[str, Any]] = []
        candidates = [dict(value) for value in definition["candidates"]]
        historical_selected = pd.read_csv(contract["sources"]["phase3a_selected_hyperparameters"]["path"])
        historical_selected = historical_selected.loc[
            (historical_selected["repetition"].astype(int) == 1)
            & (historical_selected["formulation_id"] == "primary_three_class")
        ]
        for outer_fold in range(1, 6):
            test_ids = [int(value) for value in outer.loc[outer["outer_fold"].astype(int) == outer_fold, "sample_index"] if int(value) in retained_ids]
            train_ids = [int(value) for value in outer.loc[outer["outer_fold"].astype(int) != outer_fold, "sample_index"] if int(value) in retained_ids]
            scoped_inner = inner.loc[(inner["outer_fold"].astype(int) == outer_fold) & (~inner["sample_index"].astype(int).isin(disagreements))]
            _require(set(scoped_inner["sample_index"].astype(int)) == set(train_ids), "Restricted inner membership differs from restricted outer training.")
            macro_means: list[float] = []
            qwk_means: list[float] = []
            fold_candidate_rows: list[dict[str, Any]] = []
            for candidate_index, candidate in enumerate(candidates):
                macro_scores: list[float] = []
                qwk_scores: list[float] = []
                for inner_fold in range(1, 6):
                    validation_ids = scoped_inner.loc[scoped_inner["inner_fold"].astype(int) == inner_fold, "sample_index"].astype(int).tolist()
                    development_ids = sorted(set(train_ids) - set(validation_ids))
                    pipeline = build_model_pipeline(
                        "xgboost", features.loc[development_ids], fixed_parameters=definition["fixed_params"],
                        candidate_parameters=candidate, random_state=4401,
                    )
                    _fit(pipeline, features.loc[development_ids], target.loc[development_ids], context=f"alias inner outer={outer_fold} candidate={candidate_index} inner={inner_fold}")
                    probability = aligned_predict_proba(pipeline, features.loc[validation_ids], labels=LABELS)
                    prediction = np.asarray(LABELS)[np.argmax(probability, axis=1)]
                    macro, qwk = _scores(target.loc[validation_ids], prediction, probability)
                    macro_scores.append(macro)
                    qwk_scores.append(qwk)
                macro_means.append(float(np.mean(macro_scores)))
                qwk_means.append(float(np.mean(qwk_scores)))
                fold_candidate_rows.append(
                    {
                        "outer_fold": outer_fold, "candidate_index": candidate_index,
                        "parameters_json": json.dumps(candidate, sort_keys=True, separators=(",", ":")),
                        "inner_macro_f1_scores_json": json.dumps(macro_scores, separators=(",", ":")),
                        "inner_macro_f1_mean": macro_means[-1], "inner_qwk_scores_json": json.dumps(qwk_scores, separators=(",", ":")),
                        "inner_qwk_mean": qwk_means[-1], "n_inner_folds": 5,
                        "outer_test_used_for_selection": False,
                    }
                )
            selected_index = select_candidate_index(
                macro_means, qwk_means, practical_tie_tolerance=0.001, better_direction="higher"
            )
            for row in fold_candidate_rows:
                row["selected_by_protocol"] = row["candidate_index"] == selected_index
            candidate_rows.extend(fold_candidate_rows)
            old = historical_selected.loc[historical_selected["outer_fold"].astype(int) == outer_fold, "selected_candidate_index"]
            _require(len(old) == 1, "Historical selected candidate is not unique.")
            selected_rows.append(
                {
                    "outer_fold": outer_fold, "historical_candidate_index": int(old.iloc[0]),
                    "exclusion_refit_candidate_index": selected_index,
                    "selected_candidate_changed": int(old.iloc[0]) != selected_index,
                    "exclusion_candidate_parameters_json": json.dumps(candidates[selected_index], sort_keys=True, separators=(",", ":")),
                    "selected_inner_macro_f1_mean": macro_means[selected_index],
                    "selected_inner_qwk_mean": qwk_means[selected_index],
                    "outer_test_used_for_selection": False,
                }
            )
            final = build_model_pipeline(
                "xgboost", features.loc[train_ids], fixed_parameters=definition["fixed_params"],
                candidate_parameters=candidates[selected_index], random_state=4401,
            )
            _fit(final, features.loc[train_ids], target.loc[train_ids], context=f"alias outer refit fold={outer_fold}")
            probability = aligned_predict_proba(final, features.loc[test_ids], labels=LABELS)
            prediction = np.asarray(LABELS)[np.argmax(probability, axis=1)]
            for position, sample_index in enumerate(test_ids):
                refit_rows.append(
                    {
                        "sample_index": sample_index, "outer_fold": outer_fold,
                        "y_true": int(target.loc[sample_index]), "y_pred": int(prediction[position]),
                        "selected_candidate_index": selected_index,
                        "prob_class_2": float(probability[position, 0]), "prob_class_3": float(probability[position, 1]),
                        "prob_class_4": float(probability[position, 2]), "arm": "exclusion_refit_309",
                        "evidence_source": "round2_exclusion_selection_and_outer_refit",
                    }
                )
        oof = pd.concat([historical, restricted, pd.DataFrame(refit_rows)], ignore_index=True)
        oof = oof.sort_values(["arm", "sample_index"]).reset_index(drop=True)
        expected_counts = {"historical_canonical_311": 311, "restricted_canonical_309": 309, "exclusion_refit_309": 309}
        _require(oof.groupby("arm").size().to_dict() == expected_counts, "Alias OOF arm counts drifted.")
        restricted_ids = set(oof.loc[oof["arm"] == "restricted_canonical_309", "sample_index"].astype(int))
        refit_ids = set(oof.loc[oof["arm"] == "exclusion_refit_309", "sample_index"].astype(int))
        _require(restricted_ids == refit_ids == retained_ids, "Matched 309-row evaluation populations differ.")
        aggregate, per_class, confusion = _summaries(oof)
        deltas = build_comparison_deltas(aggregate, contract)
        population_receipt = {
            "historical_sample_count": 311, "matched_sample_count": 309,
            "disagreement_count": 2,
            "disagreement_sample_set_sha256": _canonical_json_sha256(sorted(disagreements)),
            "historical_sample_set_sha256": _canonical_json_sha256(sorted(range(311))),
            "matched_sample_set_sha256": _canonical_json_sha256(sorted(retained_ids)),
            "restricted_and_refit_sample_sets_identical": True,
            "sample_removal_comparison_is_fit_free": True,
            "primary_refit_comparison_has_identical_evaluation_population": True,
            "row_identities_publication_authorized": False,
        }
        frames = {
            "oof_predictions.csv": oof,
            "candidate_search_results.csv": pd.DataFrame(candidate_rows).sort_values(["outer_fold", "candidate_index"]),
            "selected_hyperparameters.csv": pd.DataFrame(selected_rows).sort_values("outer_fold"),
            "aggregate_metrics.csv": aggregate, "per_class_metrics.csv": per_class,
            "confusion_matrix.csv": confusion, "comparison_deltas.csv": deltas,
        }
        for name, frame in frames.items():
            frame.to_csv(staging / name, index=False)
        (staging / "population_receipt.json").write_text(json.dumps(population_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output_paths = [*frames, "population_receipt.json"]
        metadata = {
            "schema_version": 1, "stage": "hr_target_alias_sensitivity_v4", "status": "complete",
            "run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_identity": git_identity, "scientific_input_sha256": scientific_input_sha256,
            "scientific_inputs": scientific_inputs, "historical_sample_count": 311,
            "matched_sample_count": 309, "disagreement_count": 2,
            "candidate_fit_count": 200, "outer_model_fit_count": 5, "total_model_fit_count": 205,
            "calibration_fit_count": 0, "baseline_fit_count": 0,
            "matched_refit_evaluation_population_identical": True,
            "sample_removal_effect_fit_free": True, "outer_test_used_for_selection": False,
            "employee_level_outputs_publication_authorized": False, "paid_api_calls": 0, "network_calls": 0,
            "output_hashes": {name: sha256_file(staging / name) for name in output_paths},
        }
        _require(_clean_git_identity() == git_identity, "Git identity changed during HR alias execution.")
        _require(source_tree_hash(PROJECT_ROOT) == scientific_inputs["source_tree_hash"], "Source tree changed during HR alias execution.")
        _validate_contract(contract_file)
        (staging / "stage_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        "scientific_input_sha256": scientific_input_sha256, "total_model_fit_count": 205,
        "historical_sample_count": 311, "matched_sample_count": 309,
        "paid_api_calls": 0, "network_calls": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_LOCAL_RUN_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.preflight_only:
        print(json.dumps(preflight_hr_target_alias_sensitivity_v4(args.contract), indent=2, sort_keys=True))
        return 0
    _require(isinstance(args.run_id, str) and bool(args.run_id.strip()), "--run-id is required.")
    output = args.output_root / args.run_id / "hr_target_alias_sensitivity"
    print(json.dumps(run_hr_target_alias_sensitivity_v4(contract_path=args.contract, output_dir=output, run_id=args.run_id), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS", "DEFAULT_CONTRACT_PATH", "HRTargetAliasSensitivityV4Error",
    "build_comparison_deltas", "identify_disagreement_indices",
    "preflight_hr_target_alias_sensitivity_v4", "run_hr_target_alias_sensitivity_v4",
]
