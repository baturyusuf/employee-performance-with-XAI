"""Generate the Round 2 publication-ready method reproducibility tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_OUTPUT = Path("reports/research_log/major_revision_round2/METHOD_REPRODUCIBILITY_TABLES")
SOURCE_HASHES = {
    "configs/feature_availability_v3.json": "6dd5fdde534e379cceacfaa01e865d1551310fb632b691f5b937ef39394e93cf",
    "configs/model_grid.yaml": "d8fb0584d7106f8941ca8e1b0e0a9c58e13f39519c6c75b383ff333d92d41617",
    "configs/ordinal_benchmark_v3.json": "39bcc62580515888783120a00ed807c0ede0f4c46f587f50897aced4c7999b02",
    "configs/policy_retuning_v3.json": "d10c6f6c5e3a61e3895220f4d43a8d682e4d98c83b165f6694b20570ae950d22",
    "configs/repeated_nested_cv_v3.json": "5681e521cfbaff5963494212fcc047116056fd7d66393346df0463aef7553af9",
    "configs/hrdataset_sensitivity_v3.json": "9a94052e8fc96ff0894c89cfe99eca898e1e0da5ed697c7fabac99e4fb22799b",
    "configs/calibration_diagnostics_v3.json": "258a81b0a3a4038218ca9f185252ed9fcc12c8f7056314e9760f5285ffec831b",
    "configs/selection_objective_sensitivity_v4.json": "ed46b263644c1d1fc49969c3d7a2936e2452eab02240e66a4b53016d9eb3f719",
    "configs/hr_target_alias_sensitivity_v4.json": "87f447ac5ad9e6b33254c8f63d34cef0c0b4ebbd183610b6db009372ac5d6fb7",
}
POLICY_LABELS = {
    "P0": "Information-Rich Diagnostic",
    "P1": "Leakage-Risk-Controlled",
    "P2": "Direct-Sensitive-Attribute-Excluded",
    "P3": "Primary Leakage-Aware",
    "P4": "Prospective-Plausibility",
    "P5": "Strict Proxy-Reduced Prospective-Plausibility",
}
EXPECTED_RETAINED_COUNTS = {"P0": 26, "P1": 24, "P2": 21, "P3": 20, "P4": 13, "P5": 6}


class MethodReproducibilityTablesV4Error(RuntimeError):
    """Raised when a source or generated table violates its frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MethodReproducibilityTablesV4Error(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_contracts(project_root: Path) -> dict[str, Mapping[str, Any]]:
    contracts: dict[str, Mapping[str, Any]] = {}
    for relative, expected_hash in SOURCE_HASHES.items():
        path = project_root / relative
        _require(path.is_file(), f"Missing method source: {relative}.")
        _require(_sha256(path) == expected_hash, f"Method source hash drifted: {relative}.")
        contracts[relative] = json.loads(path.read_text(encoding="utf-8"))
    return contracts


def build_table_s1(contracts: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    contract = contracts["configs/feature_availability_v3.json"]
    policies = {row["policy_id"]: row for row in contract["policies"]}
    _require(tuple(policies) == tuple(POLICY_LABELS), "Feature-policy order drifted.")
    _require(len(contract["features"]) == 28, "Expected 28 declared fields including identifier and target.")
    rows: list[dict[str, Any]] = []
    for feature in contract["features"]:
        row: dict[str, Any] = {
            "feature_name": feature["feature_name"],
            "semantic_family": feature["feature_family"],
            "semantic_description": feature["semantic_description"],
            "timing_status": feature["availability_at_prediction_time"],
            "risk_type": feature["risk_type"],
            "governance_type": feature["governance_type"],
        }
        for policy_id in POLICY_LABELS:
            row[policy_id] = "Excluded" if feature["feature_name"] in policies[policy_id]["excluded_features"] else "Included"
        row["justification"] = feature["justification"]
        row["timestamp_caveat"] = "No row-level feature timestamp verifies prospective availability."
        rows.append(row)
    table = pd.DataFrame(rows)
    retained = {policy: int(table[policy].eq("Included").sum()) for policy in POLICY_LABELS}
    _require(retained == EXPECTED_RETAINED_COUNTS, f"Policy retained-feature counts drifted: {retained}.")
    _require(table["feature_name"].is_unique, "Feature names are not unique.")
    return table


def build_table_s2(contracts: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    ordinal = contracts["configs/ordinal_benchmark_v3.json"]
    nominal_grid = contracts["configs/model_grid.yaml"]["model_benchmark"]
    selection_v4 = contracts["configs/selection_objective_sensitivity_v4.json"]
    rows: list[dict[str, Any]] = []
    preprocessing = ordinal["preprocessing"]
    rows.extend(
        [
            {
                "section": "preprocessing",
                "item_id": "numeric",
                "display_name": "Numeric preprocessing",
                "estimator": preprocessing["factory"],
                "fixed_parameters": _json_cell(preprocessing["numeric"]),
                "candidate_grid": "not_applicable",
                "candidate_count": 0,
                "fit_scope": preprocessing["fit_scope"],
                "selection_primary": "not_applicable",
                "selection_tie_break": "not_applicable",
                "tie_tolerance": np.nan,
                "failure_policy": "fail_entire_stage",
                "outer_test_role": "never_fit",
                "unknown_category_handling": "not_applicable",
            },
            {
                "section": "preprocessing",
                "item_id": "categorical",
                "display_name": "Categorical preprocessing",
                "estimator": preprocessing["factory"],
                "fixed_parameters": _json_cell(preprocessing["categorical"]),
                "candidate_grid": "not_applicable",
                "candidate_count": 0,
                "fit_scope": preprocessing["fit_scope"],
                "selection_primary": "not_applicable",
                "selection_tie_break": "not_applicable",
                "tie_tolerance": np.nan,
                "failure_policy": "fail_entire_stage",
                "outer_test_role": "never_fit",
                "unknown_category_handling": "one_hot_handle_unknown_ignore",
            },
        ]
    )
    common = {
        "fit_scope": preprocessing["fit_scope"],
        "selection_primary": ordinal["selection"]["primary_metric"],
        "selection_tie_break": ordinal["selection"]["tie_break_metric"],
        "tie_tolerance": ordinal["selection"]["primary_tie_tolerance"],
        "failure_policy": nominal_grid["candidate_failure_policy"],
        "outer_test_role": "evaluation_only_never_selection_or_preprocessing",
        "unknown_category_handling": "one_hot_handle_unknown_ignore",
    }
    for model_id, definition in nominal_grid["models"].items():
        rows.append(
            {
                "section": "trained_model",
                "item_id": model_id,
                "display_name": definition["display_name"],
                "estimator": definition["estimator"],
                "fixed_parameters": _json_cell(definition["fixed_params"]),
                "candidate_grid": _json_cell(definition["candidates"]),
                "candidate_count": len(definition["candidates"]),
                **common,
            }
        )
    for model_id, definition in ordinal["ordinal_models"].items():
        extra = {
            key: value
            for key, value in definition.items()
            if key in {"cumulative_probability_rule", "crossing_probability_rule"}
        }
        fixed = {**definition["fixed_params"], **extra}
        rows.append(
            {
                "section": "trained_model",
                "item_id": model_id,
                "display_name": definition["display_name"],
                "estimator": definition["estimator"],
                "fixed_parameters": _json_cell(fixed),
                "candidate_grid": _json_cell(definition["candidates"]),
                "candidate_count": len(definition["candidates"]),
                **common,
            }
        )
    for regime_id, regime in selection_v4["selection_regimes"].items():
        rows.append(
            {
                "section": "selection_regime",
                "item_id": regime_id,
                "display_name": f"{regime_id} selection",
                "estimator": "all_six_trained_models",
                "fixed_parameters": "not_applicable",
                "candidate_grid": "prespecified_model_specific_registry",
                "candidate_count": np.nan,
                "fit_scope": "inner_validation_predictions_within_current_outer_training_partition",
                "selection_primary": regime["primary_metric"],
                "selection_tie_break": f"{regime['tie_break_metric']}; then lowest_candidate_index",
                "tie_tolerance": regime["practical_tie_tolerance"],
                "failure_policy": nominal_grid["candidate_failure_policy"],
                "outer_test_role": "evaluation_only_never_selection",
                "unknown_category_handling": "inherited_from_common_preprocessor",
            }
        )
    table = pd.DataFrame(rows)
    model_rows = table.loc[table["section"] == "trained_model"]
    _require(len(model_rows) == 6 and model_rows["item_id"].is_unique, "Expected all six trained models exactly once.")
    _require(model_rows["candidate_count"].astype(int).tolist() == [6, 8, 8, 8, 6, 8], "Candidate counts drifted.")
    return table


def _format_seed_schedule(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = ("outer_seed", "inner_seed", "model_seed", "calibration_seed", "baseline_seed")
    parts = []
    for row in rows:
        seeds = "/".join(str(row[field]) for field in fields if field in row)
        parts.append(f"r{row['repetition']}:{seeds}")
    return ";".join(parts)


def build_table_s3(contracts: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    ordinal = contracts["configs/ordinal_benchmark_v3.json"]
    repeated = contracts["configs/repeated_nested_cv_v3.json"]
    policy = contracts["configs/policy_retuning_v3.json"]
    hr = contracts["configs/hrdataset_sensitivity_v3.json"]
    calibration = contracts["configs/calibration_diagnostics_v3.json"]
    selection = contracts["configs/selection_objective_sensitivity_v4.json"]
    alias = contracts["configs/hr_target_alias_sensitivity_v4.json"]
    canonical = ordinal["shared_nested_cv"]
    canonical_seeds = f"outer={canonical['outer_seed']};inner={canonical['inner_seed']};model={canonical['model_seed']}"
    rows = [
        {
            "analysis": "Canonical INX six-model benchmark",
            "population_or_policy": "INX/P3",
            "repetitions": 1,
            "outer_design": f"{canonical['outer_strategy']}({canonical['outer_splits']},shuffle={str(canonical['outer_shuffle']).lower()})",
            "inner_design": f"{canonical['inner_strategy']}({canonical['inner_splits']},shuffle={str(canonical['inner_shuffle']).lower()})",
            "seed_schedule": canonical_seeds,
            "fold_identity": "canonical_v2 persisted outer+inner assignments; exact shared folds across models",
            "oof_coverage": canonical["oof_coverage"],
            "calibration_isolation": "raw benchmark; separate sigmoid calibration uses outer-training cross-fitted inner OOF only",
            "selection_score_source": "inner validation predictions within each outer-training partition",
            "reuse_or_refit_boundary": ordinal["canonical_v2_comparison_source"]["reuse_boundary"],
            "outer_test_role": canonical["outer_test_usage"],
        },
        {
            "analysis": "INX repeated nested-CV sensitivity",
            "population_or_policy": "INX/P3",
            "repetitions": repeated["design"]["repetitions"],
            "outer_design": f"{repeated['design']['outer_strategy']}({repeated['design']['outer_splits']},shuffle=true)",
            "inner_design": f"{repeated['design']['inner_strategy']}({repeated['design']['inner_splits']},shuffle=true)",
            "seed_schedule": _format_seed_schedule(repeated["design"]["seed_schedule"]),
            "fold_identity": "shared exactly across models within repetition; different across repetitions",
            "oof_coverage": "every sample exactly once per model per repetition",
            "calibration_isolation": "not part of this raw-model variability analysis",
            "selection_score_source": "new inner validation predictions within each repetition/outer partition",
            "reuse_or_refit_boundary": "all six models refitted in every repetition; canonical OOF reuse prohibited",
            "outer_test_role": repeated["design"]["outer_test_usage"],
        },
        {
            "analysis": "Policy fixed-schedule sensitivity",
            "population_or_policy": "INX/P0-P5",
            "repetitions": 1,
            "outer_design": f"persisted StratifiedKFold({policy['design']['outer_splits']})",
            "inner_design": f"persisted StratifiedKFold({policy['design']['inner_splits']})",
            "seed_schedule": canonical_seeds,
            "fold_identity": "same exact canonical outer+inner assignments for all policies and estimands",
            "oof_coverage": "every sample exactly once per policy",
            "calibration_isolation": "raw probabilities; no calibration selected on outer test",
            "selection_score_source": policy["fixed_hyperparameter_estimand"]["fold_specific_schedule_source"],
            "reuse_or_refit_boundary": "P0-P3 exact fixed-policy OOF reuse; P4-P5 outer refit using P3 fold schedule",
            "outer_test_role": policy["design"]["outer_test_usage"],
        },
        {
            "analysis": "Policy independently retuned sensitivity",
            "population_or_policy": "INX/P0-P5",
            "repetitions": 1,
            "outer_design": f"persisted StratifiedKFold({policy['design']['outer_splits']})",
            "inner_design": f"persisted StratifiedKFold({policy['design']['inner_splits']})",
            "seed_schedule": canonical_seeds,
            "fold_identity": "same exact canonical outer+inner assignments for all policies and estimands",
            "oof_coverage": "every sample exactly once per policy",
            "calibration_isolation": "raw probabilities; no calibration selected on outer test",
            "selection_score_source": policy["independently_retuned_estimand"]["selection_scope"],
            "reuse_or_refit_boundary": "each policy independently selected and refitted; P3 must exactly replay canonical benchmark",
            "outer_test_role": policy["design"]["outer_test_usage"],
        },
        {
            "analysis": "Selection-objective sensitivity",
            "population_or_policy": "INX/P3",
            "repetitions": 1,
            "outer_design": f"persisted StratifiedKFold({selection['folds']['outer_splits']})",
            "inner_design": f"persisted StratifiedKFold({selection['folds']['inner_splits']})",
            "seed_schedule": f"outer={selection['folds']['outer_seed']};inner={selection['folds']['inner_seed']};model={selection['folds']['model_seed']}",
            "fold_identity": "same exact canonical outer+inner assignments in both regimes",
            "oof_coverage": "every sample exactly once per model and regime",
            "calibration_isolation": "raw probabilities; empirical prior uses outer-training labels only",
            "selection_score_source": "persisted exhaustive inner-candidate macro-F1 and QWK scores",
            "reuse_or_refit_boundary": "macro-F1 OOF exact reuse; QWK-selected models refitted in every outer fold",
            "outer_test_role": "evaluation_only_never_selection",
        },
        {
            "analysis": "Canonical sigmoid calibration diagnostics",
            "population_or_policy": "INX/P3/XGBoost",
            "repetitions": 1,
            "outer_design": f"persisted StratifiedKFold({calibration['canonical_identity']['outer_folds']})",
            "inner_design": "five inner folds for cross-fitted calibration training predictions",
            "seed_schedule": canonical_seeds,
            "fold_identity": "exact canonical XGBoost outer folds and fold-specific selected candidates",
            "oof_coverage": "every outer-test sample exactly once per raw/sigmoid system",
            "calibration_isolation": calibration["source_calibration"]["calibrator_training"],
            "selection_score_source": calibration["source_calibration"]["method_selection"],
            "reuse_or_refit_boundary": "diagnostic stage reuses validated raw and predeclared sigmoid OOF probabilities",
            "outer_test_role": calibration["source_calibration"]["outer_test_role"],
        },
        {
            "analysis": "HRDataset target-formulation/CV sensitivity",
            "population_or_policy": "HRDataset_v14/conservative seven features",
            "repetitions": hr["design"]["repetitions"],
            "outer_design": f"{hr['design']['outer_strategy']}({hr['design']['outer_splits']},shuffle=true)",
            "inner_design": f"{hr['design']['inner_strategy']}({hr['design']['inner_splits']},shuffle=true)",
            "seed_schedule": _format_seed_schedule(hr["design"]["seed_schedule"]),
            "fold_identity": "shared across systems within formulation/repetition; independently stratified across formulations",
            "oof_coverage": "every sample exactly once per system/formulation/repetition",
            "calibration_isolation": hr["calibration"]["training_source"],
            "selection_score_source": "inner macro-F1 with QWK tie-break inside each outer-training partition",
            "reuse_or_refit_boundary": "all formulation/repetition models fitted locally; no locked INX model transport",
            "outer_test_role": hr["design"]["outer_test_usage"],
        },
        {
            "analysis": "HR target-alias matched-sample sensitivity",
            "population_or_policy": "HRDataset_v14/311 historical and matched 309",
            "repetitions": 1,
            "outer_design": f"persisted Phase3A repetition-1 StratifiedKFold({alias['phase3a_design']['outer_splits']})",
            "inner_design": f"persisted Phase3A repetition-1 StratifiedKFold({alias['phase3a_design']['inner_splits']})",
            "seed_schedule": f"outer={alias['phase3a_design']['outer_seed']};inner={alias['phase3a_design']['inner_seed']};model={alias['phase3a_design']['model_seed']}",
            "fold_identity": "original fold identities restricted consistently after two-row exclusion",
            "oof_coverage": "311 historical; identical 309 rows exactly once in restricted and refit arms",
            "calibration_isolation": "excluded from scope; raw XGBoost only",
            "selection_score_source": "restricted inner folds for exclusion/refit; historical schedule retained only as comparator",
            "reuse_or_refit_boundary": "311 OOF reuse; fit-free restriction to 309; separate selection+refit on the same 309 population",
            "outer_test_role": "evaluation_only_never_selection",
        },
    ]
    table = pd.DataFrame(rows)
    _require(len(table) == 8 and table["analysis"].is_unique, "Expected eight distinct design rows.")
    return table


def _markdown_table(frame: pd.DataFrame) -> str:
    def render(value: Any) -> str:
        if pd.isna(value):
            return "—"
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(render(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows]) + "\n"


def methods_notes() -> str:
    return """# Reproducibility Notes for the Ordinal Models

## Common preprocessing

Preprocessing is fitted anew inside the current inner-development or outer-training partition. Numeric variables receive median imputation followed by standard scaling. Categorical variables receive most-frequent imputation and dense one-hot encoding with unknown categories ignored. Outer-test rows are used only for evaluation and never to fit preprocessing, select a candidate, select a calibration method, or choose a seed or information policy.

## Proportional-odds model

The proportional-odds logistic model is a cumulative-link model with shared feature coefficients across ordered cut points and fitted thresholds constrained to increase. Its shared-slope proportional-odds assumption is a model restriction, not an empirical statement that effects are identical in the data. Candidate selection varies regularization strength and optional class balancing under the same nested-CV rule as the other trained models.

## Cumulative-threshold XGBoost

For ordered labels 2/3/4, cumulative-threshold XGBoost independently fits binary XGBoost tasks for each observed training threshold, estimating `P(Y > 2)` and `P(Y > 3)`. Independently fitted tasks can cross. For each row, a nonincreasing pool-adjacent-violators (PAVA) projection corrects the cumulative probabilities before class probabilities are reconstructed by differencing: `P(Y=2)=1-P(Y>2)`, `P(Y=3)=P(Y>2)-P(Y>3)`, and `P(Y=4)=P(Y>3)`. The resulting vector is clipped only for numerical safety and normalized to the probability simplex.

This construction supplies ordered probabilities but does not guarantee superior extreme-class recall. Candidate selection, class imbalance, cumulative decomposition, PAVA projection, and ordinal-distance metrics are plausible methodological explanations for observed trade-offs unless directly isolated by a prespecified experiment.

## Selection and probability evaluation

The canonical regime selects the highest inner macro-F1 candidate, admits candidates within an inclusive 0.001 primary-score tolerance, then uses QWK and finally the lowest candidate index as deterministic tie-breaks. Round 2 reverses macro-F1 and QWK only in the designated sensitivity regime; it does not add an MAE- or RPS-selected regime. Probability reporting includes log loss, multiclass Brier score, confidence ECE, and ranked probability score. The RPS is the mean squared cumulative-probability error across the `K-1` ordered thresholds.

All results remain conditional on the stated data, feature policy, folds, seeds, candidate registries, and calibration contract. No table establishes prospective validity, leakage freedom, causal feature effects, fairness, or deployment readiness.
"""


def readme_text() -> str:
    source_lines = "\n".join(f"- `{path}` — `{digest}`" for path, digest in SOURCE_HASHES.items())
    labels = "\n".join(f"- {policy}: {label}" for policy, label in POLICY_LABELS.items())
    return f"""# Round 2 Method Reproducibility Tables

These additive tables are generated from hash-pinned, previously approved scientific contracts. They do not modify historical Phase 1–5 evidence or the manuscript.

- `TABLE_S1_FEATURE_AVAILABILITY.csv/.md`: all 28 declared fields and P0–P5 inclusion decisions.
- `TABLE_S2_PREPROCESSING_HYPERPARAMETER_SEARCH.csv/.md`: common preprocessing, six complete trained-model registries, and both Round 2 selection regimes.
- `TABLE_S3_CV_SEED_CONTRACT.csv/.md`: exact fold, seed, OOF, calibration, score-source, and reuse/refit boundaries.
- `METHODS_NOTES.md`: self-contained ordinal-model and probability-evaluation prose for later manuscript integration after claim approval.
- `SOURCE_HASHES.json`: exact source identity.

## Manuscript-facing policy labels

{labels}

P4 and P5 are timestamp-unverified sensitivity policies, not prospective validation. No policy is labelled leakage-free.

## Source hashes

{source_lines}
"""


def build_outputs(project_root: Path | str = Path(".")) -> dict[str, str]:
    contracts = _load_contracts(Path(project_root))
    s1 = build_table_s1(contracts)
    s2 = build_table_s2(contracts)
    s3 = build_table_s3(contracts)
    outputs = {
        "TABLE_S1_FEATURE_AVAILABILITY.csv": s1.to_csv(index=False, lineterminator="\n"),
        "TABLE_S1_FEATURE_AVAILABILITY.md": "# Table S1. Complete Feature Availability and Governance Contract\n\n" + _markdown_table(s1),
        "TABLE_S2_PREPROCESSING_HYPERPARAMETER_SEARCH.csv": s2.to_csv(index=False, lineterminator="\n", float_format="%.17g"),
        "TABLE_S2_PREPROCESSING_HYPERPARAMETER_SEARCH.md": "# Table S2. Preprocessing and Hyperparameter Search Contract\n\n" + _markdown_table(s2),
        "TABLE_S3_CV_SEED_CONTRACT.csv": s3.to_csv(index=False, lineterminator="\n"),
        "TABLE_S3_CV_SEED_CONTRACT.md": "# Table S3. Exact Cross-Validation and Seed Contract\n\n" + _markdown_table(s3),
        "METHODS_NOTES.md": methods_notes(),
        "README.md": readme_text(),
        "SOURCE_HASHES.json": json.dumps(SOURCE_HASHES, indent=2, sort_keys=True) + "\n",
    }
    return outputs


def write_method_reproducibility_tables_v4(
    output_dir: Path | str = DEFAULT_OUTPUT,
    project_root: Path | str = Path("."),
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = build_outputs(project_root)
    existing = {path.name for path in destination.iterdir() if path.is_file()}
    unexpected = existing - set(outputs)
    _require(not unexpected, f"Unexpected method output files: {sorted(unexpected)}.")
    for name, content in outputs.items():
        (destination / name).write_text(content, encoding="utf-8", newline="\n")
    return destination


def validate_method_reproducibility_tables_v4(
    output_dir: Path | str = DEFAULT_OUTPUT,
    project_root: Path | str = Path("."),
) -> dict[str, Any]:
    destination = Path(output_dir)
    expected = build_outputs(project_root)
    observed_names = {path.name for path in destination.iterdir() if path.is_file()}
    _require(observed_names == set(expected), f"Method output inventory drifted: {sorted(observed_names)}.")
    for name, content in expected.items():
        _require((destination / name).read_text(encoding="utf-8") == content, f"Method output drifted: {name}.")
    return {
        "status": "passed",
        "file_count": len(expected),
        "feature_count": 28,
        "trained_model_count": 6,
        "design_row_count": 8,
        "payload_sha256": hashlib.sha256(
            "".join(f"{name}:{_sha256(destination / name)}\n" for name in sorted(expected)).encode("utf-8")
        ).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.validate_only:
        write_method_reproducibility_tables_v4(args.output_dir, args.project_root)
    print(json.dumps(validate_method_reproducibility_tables_v4(args.output_dir, args.project_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MethodReproducibilityTablesV4Error",
    "build_outputs",
    "build_table_s1",
    "build_table_s2",
    "build_table_s3",
    "validate_method_reproducibility_tables_v4",
    "write_method_reproducibility_tables_v4",
]
