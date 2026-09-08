"""Independently validate the matched-sample Round 2 HR alias run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.canonical_loader import sha256_file
from src.models.ordinal_evaluation_v3 import ordinal_evaluation_bundle_v3


DEFAULT_CONTRACT_PATH = Path("configs/hr_target_alias_sensitivity_v4.json")
LABELS = (2, 3, 4)
ARMS = ("historical_canonical_311", "restricted_canonical_309", "exclusion_refit_309")
OUTPUTS = (
    "oof_predictions.csv",
    "candidate_search_results.csv",
    "selected_hyperparameters.csv",
    "aggregate_metrics.csv",
    "per_class_metrics.csv",
    "confusion_matrix.csv",
    "comparison_deltas.csv",
    "population_receipt.json",
)


class HRTargetAliasRunValidationV4Error(RuntimeError):
    """Raised when stored HR alias evidence fails recomputation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HRTargetAliasRunValidationV4Error(message)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HRTargetAliasRunValidationV4Error(f"Could not read {path.as_posix()}: {exc}") from exc


def _frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise HRTargetAliasRunValidationV4Error(f"Could not read {path.as_posix()}: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _git_blob_sha256(commit: str, path: str) -> str:
    try:
        blob = subprocess.run(["git", "show", f"{commit}:{path}"], check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HRTargetAliasRunValidationV4Error(f"Could not read generation blob {commit}:{path}.") from exc
    return hashlib.sha256(blob).hexdigest()


def _assert_frame(actual: pd.DataFrame, expected: pd.DataFrame, *, name: str, atol: float = 1e-12) -> None:
    _require(list(actual.columns) == list(expected.columns), f"{name} columns drifted.")
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False,
            check_exact=False, rtol=0.0, atol=atol,
        )
    except AssertionError as exc:
        raise HRTargetAliasRunValidationV4Error(f"{name} recomputation mismatch: {exc}") from exc


def _disagreements(raw: pd.DataFrame, contract: Mapping[str, Any]) -> tuple[int, ...]:
    target = contract["target"]
    expected = raw[target["canonical_column"]].astype(str).str.strip().map(target["alias_audit_mapping"])
    observed = pd.to_numeric(raw[target["alias_column"]], errors="raise").astype(int)
    _require(expected.notna().all(), "Alias audit mapping is incomplete.")
    values = tuple(int(value) for value in raw.index[expected.astype(int) != observed])
    _require(len(values) == 2, "Target-alias disagreement count drifted.")
    return values


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    result = series.astype(str).str.lower().map({"true": True, "false": False})
    _require(result.notna().all(), "Invalid Boolean evidence.")
    return result.astype(bool)


def _select(candidate_rows: pd.DataFrame) -> int:
    best = float(candidate_rows["inner_macro_f1_mean"].max())
    eligible = candidate_rows.loc[
        (best - candidate_rows["inner_macro_f1_mean"].astype(float)) <= 0.001 + 1e-15
    ]
    best_qwk = float(eligible["inner_qwk_mean"].max())
    winners = eligible.loc[eligible["inner_qwk_mean"].astype(float) == best_qwk]
    return int(winners.sort_values("candidate_index").iloc[0]["candidate_index"])


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
        aggregate.extend({"arm": arm, "metric": metric, "value": value, "sample_count": len(rows)} for metric, value in bundle["aggregate_metrics"].items())
        classes.extend({"arm": arm, **record} for record in bundle["per_class_metrics"])
        confusion.extend({"arm": arm, **record} for record in bundle["confusion_matrix"])
    return (
        pd.DataFrame(aggregate).sort_values(["arm", "metric"]).reset_index(drop=True),
        pd.DataFrame(classes).sort_values(["arm", "class_label"]).reset_index(drop=True),
        pd.DataFrame(confusion).sort_values(["arm", "true_label", "predicted_label"]).reset_index(drop=True),
    )


def _deltas(aggregate: pd.DataFrame, contract: Mapping[str, Any]) -> pd.DataFrame:
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


def validate_hr_target_alias_run_v4(
    run_dir: Path | str, contract_path: Path | str = DEFAULT_CONTRACT_PATH
) -> dict[str, Any]:
    root, contract_file = Path(run_dir), Path(contract_path)
    contract = _json(contract_file)
    metadata = _json(root / "stage_metadata.json")
    _require(isinstance(contract, dict) and isinstance(metadata, dict), "Contract/metadata must be objects.")
    _require(metadata.get("stage") == "hr_target_alias_sensitivity_v4" and metadata.get("status") == "complete", "Run status drifted.")
    expected_counts = {
        "historical_sample_count": 311, "matched_sample_count": 309, "disagreement_count": 2,
        "candidate_fit_count": 200, "outer_model_fit_count": 5, "total_model_fit_count": 205,
        "calibration_fit_count": 0, "baseline_fit_count": 0,
    }
    _require(all(metadata.get(key) == value for key, value in expected_counts.items()), "Run count receipt drifted.")
    for field in (
        "matched_refit_evaluation_population_identical", "sample_removal_effect_fit_free"
    ):
        _require(metadata.get(field) is True, f"Required matched-sample receipt drifted: {field}.")
    _require(metadata.get("outer_test_used_for_selection") is False, "Outer test entered selection.")
    _require(metadata.get("paid_api_calls") == 0 and metadata.get("network_calls") == 0, "Network/API receipt drifted.")
    _require(metadata["scientific_inputs"]["contract_sha256"] == sha256_file(contract_file), "Contract digest drifted.")
    _require(metadata["scientific_input_sha256"] == _digest(metadata["scientific_inputs"]), "Scientific-input digest drifted.")
    _require(set(metadata["output_hashes"]) == set(OUTPUTS), "Output inventory drifted.")
    for name in OUTPUTS:
        path = root / name
        _require(path.is_file() and sha256_file(path) == metadata["output_hashes"][name], f"Output hash drifted: {name}.")
    commit = metadata["git_identity"]["commit"]
    for path, expected in metadata["scientific_inputs"]["implementation_hashes"].items():
        _require(_git_blob_sha256(commit, path) == expected, f"Generation implementation blob drifted: {path}.")
    for name, record in contract["sources"].items():
        _require(sha256_file(Path(record["path"])) == record["sha256"], f"Source hash drifted: {name}.")

    raw = _frame(Path(contract["sources"]["raw_dataset"]["path"]))
    disagreements = _disagreements(raw, contract)
    _require(metadata["scientific_inputs"]["disagreement_sample_set_sha256"] == _digest(sorted(disagreements)), "Disagreement-set digest drifted.")
    oof = _frame(root / "oof_predictions.csv")
    required = {"sample_index", "outer_fold", "y_true", "y_pred", "selected_candidate_index", "prob_class_2", "prob_class_3", "prob_class_4", "arm", "evidence_source"}
    _require(required.issubset(oof.columns), "OOF schema drifted.")
    _require(oof.groupby("arm").size().to_dict() == {"exclusion_refit_309": 309, "historical_canonical_311": 311, "restricted_canonical_309": 309}, "OOF arm counts drifted.")
    for arm, rows in oof.groupby("arm", sort=True):
        _require(not rows["sample_index"].duplicated().any(), f"Duplicate OOF sample in {arm}.")
        probability = rows[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float)
        _require(np.isfinite(probability).all() and (probability >= 0).all() and (probability <= 1).all(), f"Invalid probabilities in {arm}.")
        _require(np.allclose(probability.sum(axis=1), 1.0, rtol=0, atol=1e-9), f"Probability simplex failed in {arm}.")
        _require(np.array_equal(np.asarray(LABELS)[np.argmax(probability, axis=1)], rows["y_pred"].astype(int)), f"Argmax labels drifted in {arm}.")

    source_oof = _frame(Path(contract["sources"]["phase3a_oof_predictions"]["path"]))
    source_oof = source_oof.loc[
        (source_oof["repetition"].astype(int) == 1)
        & (source_oof["formulation_id"] == "primary_three_class")
        & (source_oof["system"] == "xgboost_raw")
    ]
    compare_columns = ["sample_index", "outer_fold", "y_true", "y_pred", "selected_candidate_index", "prob_class_2", "prob_class_3", "prob_class_4"]
    historical = oof.loc[oof["arm"] == "historical_canonical_311", compare_columns].sort_values("sample_index").reset_index(drop=True)
    expected_historical = source_oof[compare_columns].sort_values("sample_index").reset_index(drop=True)
    _assert_frame(historical, expected_historical, name="historical canonical replay")
    restricted = oof.loc[oof["arm"] == "restricted_canonical_309", compare_columns].sort_values("sample_index").reset_index(drop=True)
    expected_restricted = expected_historical.loc[~expected_historical["sample_index"].astype(int).isin(disagreements)].reset_index(drop=True)
    _assert_frame(restricted, expected_restricted, name="fit-free restricted canonical replay")
    refit = oof.loc[oof["arm"] == "exclusion_refit_309", compare_columns].sort_values("sample_index").reset_index(drop=True)
    _require(restricted[["sample_index", "outer_fold", "y_true"]].astype(int).equals(refit[["sample_index", "outer_fold", "y_true"]].astype(int)), "Matched 309-row identities/folds/targets differ.")

    candidates = _frame(root / "candidate_search_results.csv")
    _require(len(candidates) == 40 and set(candidates["outer_fold"].astype(int)) == set(range(1, 6)), "Candidate coverage drifted.")
    _require(not _truthy(candidates["outer_test_used_for_selection"]).any(), "Outer test entered candidate selection.")
    selected = _frame(root / "selected_hyperparameters.csv")
    _require(len(selected) == 5 and not _truthy(selected["outer_test_used_for_selection"]).any(), "Selected schedule drifted.")
    historical_selected = _frame(Path(contract["sources"]["phase3a_selected_hyperparameters"]["path"]))
    historical_selected = historical_selected.loc[
        (historical_selected["repetition"].astype(int) == 1)
        & (historical_selected["formulation_id"] == "primary_three_class")
    ].set_index("outer_fold")["selected_candidate_index"].astype(int)
    for fold in range(1, 6):
        scoped = candidates.loc[candidates["outer_fold"].astype(int) == fold].sort_values("candidate_index")
        _require(scoped["candidate_index"].astype(int).tolist() == list(range(8)), f"Candidate indices drifted in fold {fold}.")
        independently_selected = _select(scoped)
        row = selected.loc[selected["outer_fold"].astype(int) == fold].iloc[0]
        _require(int(row["exclusion_refit_candidate_index"]) == independently_selected, f"Selection mismatch in fold {fold}.")
        _require(int(row["historical_candidate_index"]) == int(historical_selected.loc[fold]), f"Historical schedule mismatch in fold {fold}.")
        observed = refit.loc[refit["outer_fold"].astype(int) == fold, "selected_candidate_index"].astype(int).unique().tolist()
        _require(observed == [independently_selected], f"Refit OOF candidate mismatch in fold {fold}.")

    aggregate, per_class, confusion = _summaries(oof)
    _assert_frame(_frame(root / "aggregate_metrics.csv"), aggregate, name="aggregate metrics")
    _assert_frame(_frame(root / "per_class_metrics.csv"), per_class, name="per-class metrics")
    _assert_frame(_frame(root / "confusion_matrix.csv"), confusion, name="confusion matrix", atol=0)
    deltas = _deltas(aggregate, contract)
    _assert_frame(_frame(root / "comparison_deltas.csv"), deltas, name="comparison deltas")
    _require(_truthy(deltas.loc[deltas["comparison_id"] == "matched_refit_effect", "evaluation_population_matched"]).all(), "Matched refit flag drifted.")
    _require(not _truthy(deltas.loc[deltas["comparison_id"] == "sample_removal_effect", "evaluation_population_matched"]).any(), "Sample-removal comparison was mislabelled matched.")

    receipt = _json(root / "population_receipt.json")
    retained = sorted(set(range(311)) - set(disagreements))
    expected_receipt = {
        "disagreement_count": 2, "disagreement_sample_set_sha256": _digest(sorted(disagreements)),
        "historical_sample_count": 311, "historical_sample_set_sha256": _digest(list(range(311))),
        "matched_sample_count": 309, "matched_sample_set_sha256": _digest(retained),
        "primary_refit_comparison_has_identical_evaluation_population": True,
        "restricted_and_refit_sample_sets_identical": True,
        "row_identities_publication_authorized": False, "sample_removal_comparison_is_fit_free": True,
    }
    _require(receipt == expected_receipt, "Population receipt drifted.")
    return {
        "status": "passed", "run_id": metadata["run_id"], "run_dir": root.as_posix(),
        "historical_oof_replay_exact": True, "restricted_oof_fit_free_replay_exact": True,
        "matched_309_identity_fold_target_exact": True, "aggregate_recomputed": True,
        "candidate_rows": 40, "selected_rows": 5, "oof_rows": len(oof),
        "row_identities_published": False, "paid_api_calls": 0, "network_calls": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(validate_hr_target_alias_run_v4(args.run_dir, args.contract), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["HRTargetAliasRunValidationV4Error", "validate_hr_target_alias_run_v4"]
