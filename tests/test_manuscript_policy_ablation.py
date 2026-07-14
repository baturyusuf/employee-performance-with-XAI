from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.experiments.manuscript_policy_ablation as policy_module
from src.experiments.manuscript_policy_ablation import (
    IDENTITY_FIELDS,
    PRIMARY_METRIC_ORDER,
    PolicyAblationError,
    _comparison_specs,
    _policy_definitions,
    _settings,
    _validate_resample_binding,
    _validate_comparison_protocol,
    exact_policy_frame,
    leakage_sensitivity_indices,
    run,
)
from src.governance.manuscript_contract import canonical_config_hash
from src.models.oof_bootstrap import BootstrapResult, metric_definition


CONFIG_HASH: str
SCIENTIFIC_HASH = "a" * 64
RESAMPLE_HASH = "b" * 64
MODEL_SET_HASH = "c" * 64
DATASET_HASH = "d" * 64
FOLD_HASH = "e" * 64
PRIMARY = "no_salary_hike_no_attrition_no_department"


def _comparison_protocol() -> dict[str, object]:
    return {
        "evaluation_type": "matched_oof_feature_access_sensitivity",
        "outer_folds_source": "shared_folds.outer_fold_assignments",
        "primary_oof_source": "model_benchmarks.exact_xgboost_oof_predictions",
        "primary_oof_replay_probability_atol": 1e-12,
        "non_primary_hyperparameters_source": (
            "model_benchmarks.xgboost_selected_candidate_by_outer_fold"
        ),
        "independent_policy_tuning": False,
        "preprocessing_fit_scope": "outer_training_partition_only",
        "uncertainty_source": "evaluation.bootstrap",
        "fold_summary_scope": "descriptive_variability_only_no_population_ci",
        "pairwise_inference": (
            "pointwise_paired_bootstrap_intervals_no_multiplicity_adjusted_rejection_claim"
        ),
        "full_feature_comparator_boundary": (
            "diagnostic_information_rich_comparator_not_guaranteed_optimized_upper_bound"
        ),
    }


def _config() -> dict[str, object]:
    return {
        "manuscript_final": {
            "target": {
                "column": "PerformanceRating",
                "labels": [2, 3, 4],
                "problem_type": "ordinal_multiclass_performance",
            },
            "feature_policies": {
                "primary_policy": PRIMARY,
                "definitions": {
                    "full_feature_upper_bound": {
                        "excluded_features": ["EmpNumber", "PerformanceRating"],
                        "role": "diagnostic_upper_bound_never_deployable",
                        "audit_only": True,
                    },
                    "no_salary_hike": {
                        "excluded_features": [
                            "EmpNumber",
                            "PerformanceRating",
                            "EmpLastSalaryHikePercent",
                        ],
                        "role": "leakage_ablation_non_primary",
                        "audit_only": True,
                    },
                    "no_salary_hike_no_attrition": {
                        "excluded_features": [
                            "Age",
                            "Gender",
                            "MaritalStatus",
                            "EmpLastSalaryHikePercent",
                            "Attrition",
                            "EmpNumber",
                            "PerformanceRating",
                        ],
                        "role": "governed_leakage_ablation_non_primary",
                        "audit_only": False,
                    },
                    PRIMARY: {
                        "excluded_features": [
                            "Age",
                            "Gender",
                            "MaritalStatus",
                            "EmpDepartment",
                            "EmpLastSalaryHikePercent",
                            "Attrition",
                            "EmpNumber",
                            "PerformanceRating",
                        ],
                        "role": "canonical_primary",
                        "audit_only": False,
                    },
                    "no_salary_hike_no_attrition_no_department_no_job_role": {
                        "excluded_features": [
                            "Age",
                            "Gender",
                            "MaritalStatus",
                            "EmpDepartment",
                            "EmpJobRole",
                            "EmpLastSalaryHikePercent",
                            "Attrition",
                            "EmpNumber",
                            "PerformanceRating",
                        ],
                        "role": "strict_proxy_sensitivity_non_primary",
                        "audit_only": False,
                    },
                    "no_salary_hike_no_attrition_sensitive_retaining_audit": {
                        "excluded_features": [
                            "EmpLastSalaryHikePercent",
                            "Attrition",
                            "EmpNumber",
                            "PerformanceRating",
                        ],
                        "role": "sensitive_retaining_leakage_ablation_non_primary",
                        "audit_only": True,
                    },
                },
                "comparison_protocol": _comparison_protocol(),
            },
            "governance_fields": {"identifier_fields": ["EmpNumber"]},
            "evaluation": {
                "metric_applicability": {
                    "ordinal_multiclass_performance": {
                        "applicable": list(PRIMARY_METRIC_ORDER),
                    }
                },
                "bootstrap": {
                    "n_resamples": 5000,
                    "confidence_level": 0.95,
                    "seed": "bootstrap",
                    "method": "paired_stratified_percentile",
                    "stratify_by": ["outer_fold", "y_true"],
                    "quantile_method": "linear",
                },
            },
            "seeds": {"model": 42, "bootstrap": 43},
            "model": {"xgboost": {"n_estimators": 1, "n_jobs": 1}},
        }
    }


def _data() -> pd.DataFrame:
    rows = 30
    return pd.DataFrame(
        {
            "EmpNumber": [f"E{value:03d}" for value in range(rows)],
            "Age": [25 + value % 20 for value in range(rows)],
            "Gender": ["F" if value % 2 else "M" for value in range(rows)],
            "MaritalStatus": ["Single" if value % 2 else "Married" for value in range(rows)],
            "EmpDepartment": ["Sales" if value % 2 else "IT" for value in range(rows)],
            "EmpJobRole": ["A" if value % 3 else "B" for value in range(rows)],
            "EmpLastSalaryHikePercent": [10 + value % 4 for value in range(rows)],
            "Attrition": ["No" if value % 3 else "Yes" for value in range(rows)],
            "Signal": np.linspace(0.0, 1.0, rows),
            "PerformanceRating": [2, 3, 4] * 10,
        }
    )


def _bundle(data: pd.DataFrame, config_hash: str) -> SimpleNamespace:
    outer = pd.DataFrame(
        {
            "run_id": "run-1",
            "config_hash": config_hash,
            "scientific_input_hash": SCIENTIFIC_HASH,
            "dataset_key": "inx_primary",
            "dataset_sha256": DATASET_HASH,
            "fold_contract_hash": FOLD_HASH,
            "sample_index": data.index,
            "y_true": data["PerformanceRating"].astype(int),
            "outer_fold": np.repeat(np.arange(1, 11), 3),
        }
    )
    folds = SimpleNamespace(
        outer_assignments=outer,
        contract={
            "run_id": "run-1",
            "config_hash": config_hash,
            "scientific_input_hash": SCIENTIFIC_HASH,
            "dataset_sha256": DATASET_HASH,
            "fold_contract_hash": FOLD_HASH,
            "outer_splits": 10,
        },
    )
    primary_features, _ = exact_policy_frame(
        data,
        PRIMARY,
        _config()["manuscript_final"]["feature_policies"]["definitions"][PRIMARY],
        target_column="PerformanceRating",
        id_column="EmpNumber",
    )
    selected = pd.DataFrame(
        [
            {
                "outer_fold": fold,
                "selected_candidate_index": 0,
                "fixed_parameters_json": json.dumps(
                    {
                        "n_estimators": 1,
                        "n_jobs": 1,
                        "objective": "multi:softprob",
                        "eval_metric": "mlogloss",
                    }
                ),
                "selected_candidate_parameters_json": json.dumps({"max_depth": 2}),
            }
            for fold in range(1, 11)
        ]
    )
    probability = np.tile(np.asarray([0.2, 0.6, 0.2]), (len(data), 1))
    oof = outer[["sample_index", "outer_fold", "y_true"]].copy()
    oof["y_pred"] = 3
    oof["selected_candidate_index"] = 0
    for column, label in enumerate((2, 3, 4)):
        oof[f"prob_class_{label}"] = probability[:, column]
    models = {
        fold: SimpleNamespace(selected_candidate_index=0, sha256=f"{fold:064x}")
        for fold in range(1, 11)
    }
    return SimpleNamespace(
        identity=SimpleNamespace(
            run_id="run-1",
            config_hash=config_hash,
            scientific_input_hash=SCIENTIFIC_HASH,
            fold_contract_hash=FOLD_HASH,
        ),
        folds=folds,
        oof_predictions=oof,
        selected_hyperparameters=selected,
        fold_models=models,
        labels=(2, 3, 4),
        raw_feature_order=tuple(primary_features.columns),
        model_set_sha256=MODEL_SET_HASH,
        upstream_file_hashes={"selected_hyperparameters": "f" * 64},
        baseline_gate={"resample_hash": RESAMPLE_HASH},
    )


class _FakePipeline:
    fit_calls: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
    forbidden_calls: list[tuple[str, ...]] = []

    def __init__(self, parameters: dict[str, object]) -> None:
        self.classes_ = np.asarray([2, 3, 4])
        self._parameters = parameters
        self.named_steps = {
            "preprocessor": SimpleNamespace(feature_names_in_=np.asarray([], dtype=object)),
            "model": self,
        }

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "_FakePipeline":
        self.fit_calls.append((tuple(features.index), tuple(features.columns)))
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        self.named_steps["preprocessor"].feature_names_in_ = np.asarray(
            features.columns, dtype=object
        )
        return self

    def get_params(self, deep: bool = False) -> dict[str, object]:
        del deep
        return dict(self._parameters)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), 3, dtype=int)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.tile(np.asarray([0.2, 0.6, 0.2]), (len(features), 1))


def _fake_bootstrap(predictions: pd.DataFrame, **kwargs: object) -> BootstrapResult:
    systems = predictions["system_id"].drop_duplicates().tolist()
    metrics = list(kwargs["metrics"])
    comparisons = list(kwargs["comparisons"])
    protocol = kwargs["protocol"]
    assert protocol.n_resamples == 5000
    assert kwargs["primary_metric"] is None
    assert all(not comparison.primary_gate for comparison in comparisons)
    points: dict[tuple[str, str], float] = {}
    base = {
        "accuracy": 0.60,
        "balanced_accuracy": 0.55,
        "macro_f1": 0.50,
        "quadratic_weighted_kappa": 0.40,
        "ordinal_mae": 0.50,
        "severe_error_rate": 0.10,
        "nll_log_loss": 1.00,
        "multiclass_brier": 0.70,
        "ece_confidence": 0.10,
    }
    interval_rows = []
    for system_position, system in enumerate(systems):
        for metric in metrics:
            value = base[metric] + system_position * 0.001
            points[(system, metric)] = value
            definition = metric_definition(metric)
            low = max(definition.lower_bound, value - 0.01)
            high = value + 0.01
            if np.isfinite(definition.upper_bound):
                high = min(definition.upper_bound, high)
            interval_rows.append(
                {
                    "system_id": system,
                    "task_type": "ordinal_multiclass_performance",
                    "metric": metric,
                    "point_estimate": value,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_std": 0.005,
                    "n_samples": 30,
                    "n_resamples": 5000,
                    "n_valid": 5000,
                    "confidence_level": 0.95,
                    "method": "paired_stratified_percentile",
                    "strata": "outer_fold;y_true",
                    "seed": 43,
                    "resample_hash": RESAMPLE_HASH,
                    "better_direction": definition.better_direction,
                    "domain_low": definition.lower_bound,
                    "domain_high": definition.upper_bound,
                }
            )
    difference_rows = []
    for comparison in comparisons:
        for metric in metrics:
            definition = metric_definition(metric)
            value_a = points[(comparison.system_a, metric)]
            value_b = points[(comparison.system_b, metric)]
            raw = value_a - value_b
            oriented = raw if definition.better_direction == "higher" else -raw
            difference_rows.append(
                {
                    "comparison_id": comparison.comparison_id,
                    "system_a": comparison.system_a,
                    "system_b": comparison.system_b,
                    "task_type": "ordinal_multiclass_performance",
                    "metric": metric,
                    "estimate_a": value_a,
                    "estimate_b": value_b,
                    "raw_difference_a_minus_b": raw,
                    "raw_difference_ci_low": raw - 0.01,
                    "raw_difference_ci_high": raw + 0.01,
                    "improvement_oriented_difference": oriented,
                    "improvement_ci_low": oriented - 0.01,
                    "improvement_ci_high": oriented + 0.01,
                    "bootstrap_std": 0.005,
                    "n_samples": 30,
                    "n_resamples": 5000,
                    "n_valid": 5000,
                    "confidence_level": 0.95,
                    "method": "paired_stratified_percentile",
                    "strata": "outer_fold;y_true",
                    "seed": 43,
                    "resample_hash": RESAMPLE_HASH,
                    "better_direction": definition.better_direction,
                    "primary_metric": None,
                    "primary_gate_comparison": False,
                    "gate_eligible": False,
                    "gate_triggered": False,
                }
            )
    return BootstrapResult(
        metric_intervals=pd.DataFrame(interval_rows),
        paired_differences=pd.DataFrame(difference_rows),
        metadata={
            "task_type": "ordinal_multiclass_performance",
            "labels": [2, 3, 4],
            "systems": systems,
            "metrics": metrics,
            "comparison_ids": [value.comparison_id for value in comparisons],
            "primary_metric": None,
            "n_samples": 30,
            "n_resamples": 5000,
            "confidence_level": 0.95,
            "seed": 43,
            "strata_columns": ["outer_fold", "y_true"],
            "method": "paired_stratified_percentile",
            "quantile_method": "linear",
            "resample_hash": RESAMPLE_HASH,
        },
        resample_plan=SimpleNamespace(),
    )


class ManuscriptPolicyAblationTests(unittest.TestCase):
    def test_exact_policy_frame_requires_explicit_id_and_target(self) -> None:
        frame = pd.DataFrame({"EmpNumber": ["E1"], "Age": [30], "PerformanceRating": [3]})
        with self.assertRaises(PolicyAblationError):
            exact_policy_frame(
                frame,
                "bad",
                {"excluded_features": ["PerformanceRating"]},
                target_column="PerformanceRating",
                id_column="EmpNumber",
            )

    def test_exact_policy_frame_applies_only_declared_definition(self) -> None:
        frame = pd.DataFrame(
            {
                "EmpNumber": ["E1"],
                "Age": [30],
                "Gender": ["F"],
                "Signal": [1.0],
                "PerformanceRating": [3],
            }
        )
        result, excluded = exact_policy_frame(
            frame,
            "primary",
            {"excluded_features": ["EmpNumber", "PerformanceRating", "Age", "Gender"]},
            target_column="PerformanceRating",
            id_column="EmpNumber",
        )
        self.assertEqual(list(result.columns), ["Signal"])
        self.assertEqual(excluded, ["EmpNumber", "PerformanceRating", "Age", "Gender"])

    def test_comparison_protocol_is_exact_and_fail_closed(self) -> None:
        settings = _config()["manuscript_final"]
        self.assertEqual(dict(_validate_comparison_protocol(settings)), _comparison_protocol())
        settings["feature_policies"]["comparison_protocol"]["independent_policy_tuning"] = True
        with self.assertRaisesRegex(PolicyAblationError, "frozen matched-policy contract"):
            _validate_comparison_protocol(settings)

    def test_direct_stage_preflight_rejects_repository_policy_projection_drift(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            conflicting = {
                PRIMARY: {
                    "drop": [
                        "Age",
                        "EmpDepartment",
                        "EmpLastSalaryHikePercent",
                        "Attrition",
                        "EmpNumber",
                        "PerformanceRating",
                    ]
                }
            }
            with patch.object(
                policy_module,
                "repository_feature_policy_projection",
                return_value=conflicting,
            ):
                with self.assertRaisesRegex(PolicyAblationError, "projection is incompatible"):
                    _settings(path)

    def test_policy_scope_is_exactly_the_frozen_six_definitions(self) -> None:
        settings = _config()["manuscript_final"]
        definitions = settings["feature_policies"]["definitions"]
        self.assertEqual(len(_policy_definitions(settings)), 6)
        definitions["unplanned_audit"] = {
            "excluded_features": ["EmpNumber", "PerformanceRating"],
            "role": "audit",
            "audit_only": True,
        }
        with self.assertRaisesRegex(PolicyAblationError, "exactly the frozen six-policy"):
            _policy_definitions(settings)

    def test_pairwise_specs_are_all_pairs_and_never_model_gates(self) -> None:
        policies = list(_config()["manuscript_final"]["feature_policies"]["definitions"])
        specs, predeclared = _comparison_specs(policies)
        self.assertEqual(len(specs), 15)
        self.assertTrue(all(not spec.primary_gate for spec in specs))
        self.assertIn(("full_feature_upper_bound", "no_salary_hike"), predeclared)
        self.assertIn(
            ("no_salary_hike", "no_salary_hike_no_attrition_sensitive_retaining_audit"),
            predeclared,
        )
        self.assertIn(
            ("no_salary_hike_no_attrition_sensitive_retaining_audit", "no_salary_hike_no_attrition"),
            predeclared,
        )

    def test_policy_bootstrap_must_reuse_benchmark_resample_plan(self) -> None:
        self.assertEqual(
            _validate_resample_binding(
                {"resample_hash": RESAMPLE_HASH}, {"resample_hash": RESAMPLE_HASH}
            ),
            RESAMPLE_HASH,
        )
        with self.assertRaisesRegex(PolicyAblationError, "differs from the benchmark"):
            _validate_resample_binding(
                {"resample_hash": RESAMPLE_HASH}, {"resample_hash": "f" * 64}
            )

    def test_lsi_does_not_normalize_unbounded_log_loss(self) -> None:
        policies = ["full_feature_upper_bound", "reduced"]
        rows = []
        for metric in ("macro_f1", "nll_log_loss"):
            rows.append(
                {
                    "system_a": "full_feature_upper_bound",
                    "system_b": "reduced",
                    "metric": metric,
                    "improvement_oriented_difference": 0.1,
                    "improvement_ci_low": 0.02,
                    "improvement_ci_high": 0.18,
                    "n_valid": 5000,
                }
            )
        identity = {field: field for field in IDENTITY_FIELDS}
        result = leakage_sensitivity_indices(
            pd.DataFrame(rows),
            policies=policies,
            metrics=("macro_f1", "nll_log_loss"),
            identity=identity,
            resample_hash=RESAMPLE_HASH,
            n_samples=30,
            n_resamples=5000,
        )
        log_loss = result[
            (result["policy"] == "reduced") & (result["metric"] == "nll_log_loss")
        ].iloc[0]
        macro = result[
            (result["policy"] == "reduced") & (result["metric"] == "macro_f1")
        ].iloc[0]
        self.assertEqual(log_loss["normalization_status"], "not_applicable_unbounded_domain")
        self.assertTrue(np.isnan(log_loss["domain_normalized_degradation"]))
        self.assertAlmostEqual(macro["domain_normalized_degradation"], 0.1)

    def test_canonical_run_uses_no_splitter_legacy_model_or_fold_inference(self) -> None:
        source = inspect.getsource(run)
        for prohibited in (
            "StratifiedKFold",
            "LabelEncodedXGBClassifier",
            "make_preprocessor",
            "wilcoxon",
            "student_t",
            "_mean_ci",
            "_model_parameters",
        ):
            self.assertNotIn(prohibited, source)

    def test_run_reuses_primary_and_fits_nonprimary_on_exact_outer_train(self) -> None:
        data = _data()
        config = _config()
        config_hash = canonical_config_hash(config)
        bundle = _bundle(data, config_hash)
        _FakePipeline.fit_calls = []
        _FakePipeline.forbidden_calls = []
        threadpool_calls: list[int] = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "policy"

            def fake_reader(*args: object, **kwargs: object) -> SimpleNamespace:
                self.assertEqual(kwargs["expected_run_id"], "run-1")
                self.assertEqual(kwargs["expected_config_hash"], config_hash)
                self.assertEqual(kwargs["expected_scientific_input_hash"], SCIENTIFIC_HASH)
                self.assertEqual(tuple(kwargs["expected_feature_columns"]), bundle.raw_feature_order)
                self.assertEqual(tuple(kwargs["expected_labels"]), (2, 3, 4))
                return bundle

            def fake_pipeline(*args: object, **kwargs: object) -> _FakePipeline:
                self.assertEqual(args[0], "xgboost")
                self.assertEqual(kwargs["random_state"], 42)
                forbidden = tuple(kwargs["forbidden_features"])
                self.assertIn("PerformanceRating", forbidden)
                self.assertIn("EmpNumber", forbidden)
                training_features = args[1]
                self.assertTrue(set(forbidden).isdisjoint(training_features.columns))
                _FakePipeline.forbidden_calls.append(forbidden)
                return _FakePipeline(
                    {**dict(kwargs["fixed_parameters"]), **dict(kwargs["candidate_parameters"])}
                )

            @contextmanager
            def fake_threadpool_limits(*, limits: int):
                threadpool_calls.append(limits)
                yield

            with (
                patch.object(
                    policy_module,
                    "load_canonical_dataset",
                    return_value=SimpleNamespace(
                        frame=data.copy(), receipt={"actual_sha256": DATASET_HASH}
                    ),
                ),
                patch.object(policy_module, "read_xgboost_oof_artifacts", side_effect=fake_reader),
                patch.object(policy_module, "validate_xgboost_oof_replay") as replay,
                patch.object(policy_module, "build_model_pipeline", side_effect=fake_pipeline),
                patch.object(policy_module, "threadpool_limits", side_effect=fake_threadpool_limits),
                patch.object(policy_module, "compute_paired_oof_bootstrap", side_effect=_fake_bootstrap),
                patch.object(policy_module, "_mean_ci", side_effect=AssertionError("legacy helper called")),
                patch.object(
                    policy_module,
                    "_model_parameters",
                    side_effect=AssertionError("legacy helper called"),
                ),
            ):
                paths = run(
                    config_path,
                    shared_folds_dir=root / "shared_folds",
                    model_benchmarks_dir=root / "model_benchmarks",
                    output_dir=output,
                    run_id="run-1",
                    config_hash=config_hash,
                    scientific_input_hash=SCIENTIFIC_HASH,
                )

            policies = list(config["manuscript_final"]["feature_policies"]["definitions"])
            replay.assert_called_once()
            self.assertEqual(replay.call_args.kwargs["probability_atol"], 1e-12)
            self.assertEqual(len(_FakePipeline.fit_calls), (len(policies) - 1) * 10)
            self.assertEqual(threadpool_calls, [1] * ((len(policies) - 1) * 10))
            self.assertEqual(len(_FakePipeline.forbidden_calls), (len(policies) - 1) * 10)
            self.assertTrue(any("Age" in forbidden for forbidden in _FakePipeline.forbidden_calls))
            primary_ids = set(bundle.oof_predictions["sample_index"])
            oof = pd.read_csv(paths["oof_predictions"])
            primary_rows = oof[oof["policy"] == PRIMARY]
            self.assertEqual(set(primary_rows["sample_index"]), primary_ids)
            self.assertEqual(
                set(primary_rows["model_fit_mode"]), {"exact_benchmark_oof_reuse_no_refit"}
            )
            self.assertEqual(oof.groupby("policy")["sample_index"].nunique().to_dict(), {p: 30 for p in policies})
            self.assertEqual(oof.groupby("policy").size().to_dict(), {p: 30 for p in policies})
            self.assertTrue(
                all(
                    len(indices) == 27
                    for indices, _ in _FakePipeline.fit_calls
                )
            )

            summary = pd.read_csv(paths["policy_summary"])
            self.assertEqual(set(summary["policy"]), set(policies))
            self.assertNotIn("weighted_f1_mean", summary.columns)
            self.assertNotIn("macro_f1_mean", summary.columns)
            self.assertIn("macro_f1_oof", summary.columns)
            self.assertIn("macro_f1_fold_mean", summary.columns)
            self.assertTrue((summary["n_resamples"] == 5000).all())
            self.assertTrue(
                (summary["fold_variability_status"] == "descriptive_only_not_population_ci").all()
            )
            intervals = pd.read_csv(paths["policy_metric_intervals"])
            self.assertEqual(len(intervals), len(policies) * len(PRIMARY_METRIC_ORDER))
            self.assertEqual(set(IDENTITY_FIELDS).difference(intervals.columns), set())
            self.assertTrue((intervals["n_valid"] == 5000).all())
            pairwise = pd.read_csv(paths["policy_pairwise_tests"])
            self.assertEqual(len(pairwise), 15 * len(PRIMARY_METRIC_ORDER))
            self.assertFalse(pairwise["gate_eligible"].astype(bool).any())
            self.assertFalse(pairwise["gate_triggered"].astype(bool).any())
            self.assertTrue(pairwise["predeclared_comparison"].astype(bool).any())
            self.assertTrue(pairwise["adjacent_policy_step"].astype(bool).any())

            lsi = pd.read_csv(paths["leakage_sensitivity_index"])
            log_loss = lsi[lsi["metric"] == "nll_log_loss"]
            self.assertTrue(log_loss["domain_normalized_degradation"].isna().all())
            self.assertTrue(
                (log_loss["normalization_status"] == "not_applicable_unbounded_domain").all()
            )
            figure_source = pd.read_csv(paths["figure_source"])
            self.assertIn("full_feature_upper_bound", set(figure_source["policy"]))
            self.assertTrue(figure_source["audit_only"].astype(bool).any())
            self.assertGreater(paths["figure_png"].stat().st_size, 0)
            self.assertGreater(paths["figure_svg"].stat().st_size, 0)

            receipts = pd.read_csv(paths["policy_fit_receipts"])
            self.assertEqual(len(receipts), len(policies) * 10)
            self.assertTrue((receipts["execution_status"] == "complete").all())
            self.assertTrue((receipts["threadpool_limit"] == 1).all())
            self.assertTrue((receipts["n_train"] == 27).all())
            self.assertTrue((receipts["n_test"] == 3).all())
            primary_receipts = receipts[receipts["policy"] == PRIMARY]
            self.assertFalse(primary_receipts["stage_fit_performed"].astype(bool).any())
            self.assertTrue(primary_receipts["primary_benchmark_oof_reused"].astype(bool).all())
            schedule = pd.read_csv(paths["policy_hyperparameter_schedule"])
            self.assertTrue((schedule["planned_fit_threadpool_limit"] == 1).all())
            self.assertTrue(schedule["parameter_source"].str.contains("primary_policy").all())

            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            self.assertEqual(metadata["protocol"]["comparison_protocol"], _comparison_protocol())
            self.assertFalse(metadata["protocol"]["policy_independently_tuned"])
            self.assertTrue(all(not Path(value).is_absolute() for value in metadata["outputs"].values()))
            self.assertNotIn("fold_assignments", paths)

    def test_populated_output_is_rejected_before_scientific_input_access(self) -> None:
        config = _config()
        config_hash = canonical_config_hash(config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "policy"
            output.mkdir()
            (output / "stale.csv").write_text("stale\n", encoding="utf-8")
            with patch.object(
                policy_module,
                "load_canonical_dataset",
                side_effect=AssertionError("loader must not run"),
            ):
                with self.assertRaisesRegex(PolicyAblationError, "absent or an empty"):
                    run(
                        config_path,
                        shared_folds_dir=root / "shared",
                        model_benchmarks_dir=root / "benchmark",
                        output_dir=output,
                        run_id="run-1",
                        config_hash=config_hash,
                        scientific_input_hash=SCIENTIFIC_HASH,
                    )

    def test_late_failure_leaves_no_partial_policy_artifact_directory(self) -> None:
        data = _data()
        config = _config()
        config_hash = canonical_config_hash(config)
        bundle = _bundle(data, config_hash)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "policy"
            output.mkdir()

            def fake_pipeline(*args: object, **kwargs: object) -> _FakePipeline:
                return _FakePipeline(
                    {**dict(kwargs["fixed_parameters"]), **dict(kwargs["candidate_parameters"])}
                )

            with (
                patch.object(
                    policy_module,
                    "load_canonical_dataset",
                    return_value=SimpleNamespace(
                        frame=data.copy(), receipt={"actual_sha256": DATASET_HASH}
                    ),
                ),
                patch.object(policy_module, "read_xgboost_oof_artifacts", return_value=bundle),
                patch.object(policy_module, "validate_xgboost_oof_replay"),
                patch.object(policy_module, "build_model_pipeline", side_effect=fake_pipeline),
                patch.object(
                    policy_module,
                    "compute_paired_oof_bootstrap",
                    side_effect=_fake_bootstrap,
                ),
                patch.object(
                    policy_module,
                    "write_tradeoff_figure",
                    side_effect=RuntimeError("injected late figure failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected late figure failure"):
                    run(
                        config_path,
                        shared_folds_dir=root / "shared",
                        model_benchmarks_dir=root / "benchmark",
                        output_dir=output,
                        run_id="run-1",
                        config_hash=config_hash,
                        scientific_input_hash=SCIENTIFIC_HASH,
                    )

            self.assertTrue(output.is_dir())
            self.assertFalse(any(output.iterdir()))
            self.assertFalse(any(path.name.startswith(".policy.") for path in root.iterdir()))


if __name__ == "__main__":
    unittest.main()
