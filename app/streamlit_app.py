from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier, Pool

from src.data.preprocess import run_preprocessing
from src.explainability.counterfactuals import (
    DEFAULT_ACTIONABLE_FEATURES,
    generate_counterfactual_candidates,
)
from src.explainability.reason_codes import (
    build_reason_codes,
    build_executive_summary,
)
from src.chatbot.chat_engine import GuardrailedChatEngine
from src.models.train_catboost import prepare_catboost_inputs
from src.utils.config import SETTINGS


st.set_page_config(
    page_title="Employee Performance XAI Dashboard",
    layout="wide",
)


# =========================
# Utilities
# =========================
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_catboost_model(model_path: Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


@st.cache_resource(show_spinner=False)
def get_model_and_summary() -> Tuple[CatBoostClassifier, Dict[str, Any]]:
    catboost_dir = SETTINGS.artifacts_dir / "catboost"
    model_path = catboost_dir / "catboost_model.cbm"
    summary_path = catboost_dir / "run_summary.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            f"Please run: python -m src.models.train_catboost --drop-sensitive"
        )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Run summary not found: {summary_path}. "
            f"Please train the CatBoost model first."
        )

    model = load_catboost_model(model_path)
    summary = load_json(summary_path)
    return model, summary


@st.cache_data(show_spinner=False)
def get_rebuilt_data(drop_sensitive: bool, test_size: float, random_state: int) -> Dict[str, Any]:
    return run_preprocessing(
        test_size=test_size,
        random_state=random_state,
        drop_sensitive=drop_sensitive,
    )


def build_pool_from_df(
    df: pd.DataFrame,
    y_dummy: pd.Series | None,
    drop_sensitive: bool,
) -> Tuple[pd.DataFrame, Pool]:
    y_dummy = y_dummy if y_dummy is not None else pd.Series([0] * len(df), index=df.index)

    X_cb, _, feature_names, cat_feature_indices = prepare_catboost_inputs(
        X_train=df,
        X_test=df.copy(),
        drop_sensitive=drop_sensitive,
    )

    pool = Pool(
        data=X_cb,
        label=y_dummy,
        cat_features=cat_feature_indices,
        feature_names=feature_names,
    )
    return X_cb, pool


def predict_with_model(
    model: CatBoostClassifier,
    df: pd.DataFrame,
    drop_sensitive: bool,
) -> Tuple[pd.Series, pd.DataFrame]:
    X_cb, pool = build_pool_from_df(df, None, drop_sensitive)

    pred = pd.Series(
        model.predict(pool).flatten().astype(int),
        index=df.index,
        name="prediction",
    )
    proba = pd.DataFrame(
        model.predict_proba(pool),
        columns=[int(c) for c in model.classes_],
        index=df.index,
    )
    return pred, proba


def get_local_shap_for_df(
    model: CatBoostClassifier,
    df: pd.DataFrame,
    drop_sensitive: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    """
    Returns:
        shap_values: (n_samples, n_classes, n_features)
        base_values: (n_samples, n_classes)
        feature_names
        class_labels
    """
    X_cb, pool = build_pool_from_df(df, None, drop_sensitive)
    raw_shap = model.get_feature_importance(data=pool, type="ShapValues")

    arr = np.array(raw_shap)
    n_samples = len(X_cb)
    n_features = X_cb.shape[1]
    class_labels = [int(c) for c in model.classes_]
    n_classes = len(class_labels)

    if arr.ndim == 2 and arr.shape == (n_samples, n_features + 1):
        shap_values = arr[:, :n_features][:, np.newaxis, :]
        base_values = arr[:, n_features][:, np.newaxis]
        return shap_values, base_values, X_cb.columns.tolist(), class_labels

    if arr.ndim == 3:
        if arr.shape == (n_samples, n_classes, n_features + 1):
            shap_values = arr[:, :, :n_features]
            base_values = arr[:, :, n_features]
            return shap_values, base_values, X_cb.columns.tolist(), class_labels

        if arr.shape == (n_classes, n_samples, n_features + 1):
            arr = np.transpose(arr, (1, 0, 2))
            shap_values = arr[:, :, :n_features]
            base_values = arr[:, :, n_features]
            return shap_values, base_values, X_cb.columns.tolist(), class_labels

        if arr.shape == (n_samples, n_features + 1, n_classes):
            arr = np.transpose(arr, (0, 2, 1))
            shap_values = arr[:, :, :n_features]
            base_values = arr[:, :, n_features]
            return shap_values, base_values, X_cb.columns.tolist(), class_labels

    raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")


def build_local_contribution_df(
    raw_row: pd.Series,
    shap_row: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    local_df = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_value": [raw_row.get(f) for f in feature_names],
            "shap_value": shap_row,
        }
    )
    local_df["abs_shap_value"] = local_df["shap_value"].abs()
    local_df["direction"] = np.where(local_df["shap_value"] >= 0, "positive", "negative")
    local_df = local_df.sort_values(by="abs_shap_value", ascending=False).reset_index(drop=True)
    return local_df


def display_prediction_block(predicted_class: int, proba_row: pd.Series) -> None:
    c1, c2, c3 = st.columns([1, 2, 2])
    with c1:
        st.metric("Predicted Class", predicted_class)
    with c2:
        st.write("**Class Probabilities**")
        st.dataframe(
            pd.DataFrame(
                {
                    "class": proba_row.index.astype(str),
                    "probability": proba_row.values,
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    with c3:
        st.bar_chart(proba_row.rename("probability"))


def display_local_explanation(
    predicted_class: int,
    true_class: int | None,
    proba_for_pred: float,
    local_df: pd.DataFrame,
) -> None:
    st.subheader("Local Explanation")

    info = {
        "predicted_class": predicted_class,
        "true_class": true_class if true_class is not None else predicted_class,
        "predicted_probability_for_predicted_class": proba_for_pred,
        "explained_class": predicted_class,
        "sample_index": -1,
    }

    reason_codes = build_reason_codes(
        local_df=local_df,
        top_k_positive=4,
        top_k_negative=4,
        min_abs_shap=0.005,
    )
    summary_text = build_executive_summary(
        meta=info,
        reason_codes=reason_codes,
    )

    left, right = st.columns([1, 1])

    with left:
        st.write("**Top SHAP Contributions**")
        st.dataframe(local_df.head(12), use_container_width=True, hide_index=True)

    with right:
        plot_df = local_df.head(10).copy().iloc[::-1]
        st.write("**Top 10 Feature Contributions**")
        st.bar_chart(
            plot_df.set_index("feature")["shap_value"]
        )

    st.markdown("**Reason Codes**")
    pos_df = pd.DataFrame(reason_codes["positive_reason_codes"])
    neg_df = pd.DataFrame(reason_codes["negative_reason_codes"])

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Positive Drivers**")
        if len(pos_df) > 0:
            st.dataframe(pos_df[["feature_label_tr", "feature_value_description", "shap_value", "sentence_tr"]], use_container_width=True, hide_index=True)
        else:
            st.info("No positive drivers found.")

    with c2:
        st.write("**Negative Drivers**")
        if len(neg_df) > 0:
            st.dataframe(neg_df[["feature_label_tr", "feature_value_description", "shap_value", "sentence_tr"]], use_container_width=True, hide_index=True)
        else:
            st.info("No negative drivers found.")

    st.markdown("**Executive Summary**")
    st.text(summary_text)


def build_manual_input_df(drop_sensitive: bool) -> pd.DataFrame:
    st.subheader("Manual Input")

    cols = {}

    c1, c2, c3 = st.columns(3)

    with c1:
        cols["Age"] = st.number_input("Age", min_value=18, max_value=100, value=32)
        cols["DistanceFromHome"] = st.number_input("DistanceFromHome", min_value=0, max_value=100, value=10)
        cols["EmpEducationLevel"] = st.selectbox("EmpEducationLevel", [1, 2, 3, 4, 5], index=2)
        cols["EmpEnvironmentSatisfaction"] = st.selectbox("EmpEnvironmentSatisfaction", [1, 2, 3, 4], index=2)
        cols["EmpHourlyRate"] = st.number_input("EmpHourlyRate", min_value=0, max_value=500, value=65)
        cols["EmpJobInvolvement"] = st.selectbox("EmpJobInvolvement", [1, 2, 3, 4], index=2)
        cols["EmpJobLevel"] = st.selectbox("EmpJobLevel", [1, 2, 3, 4, 5], index=1)
        cols["EmpJobSatisfaction"] = st.selectbox("EmpJobSatisfaction", [1, 2, 3, 4], index=2)
        cols["NumCompaniesWorked"] = st.number_input("NumCompaniesWorked", min_value=0, max_value=20, value=2)

    with c2:
        cols["EmpLastSalaryHikePercent"] = st.number_input("EmpLastSalaryHikePercent", min_value=0, max_value=100, value=15)
        cols["EmpRelationshipSatisfaction"] = st.selectbox("EmpRelationshipSatisfaction", [1, 2, 3, 4], index=2)
        cols["TotalWorkExperienceInYears"] = st.number_input("TotalWorkExperienceInYears", min_value=0, max_value=50, value=5)
        cols["TrainingTimesLastYear"] = st.number_input("TrainingTimesLastYear", min_value=0, max_value=20, value=2)
        cols["EmpWorkLifeBalance"] = st.selectbox("EmpWorkLifeBalance", [1, 2, 3, 4], index=2)
        cols["ExperienceYearsAtThisCompany"] = st.number_input("ExperienceYearsAtThisCompany", min_value=0, max_value=50, value=3)
        cols["ExperienceYearsInCurrentRole"] = st.number_input("ExperienceYearsInCurrentRole", min_value=0, max_value=50, value=2)
        cols["YearsSinceLastPromotion"] = st.number_input("YearsSinceLastPromotion", min_value=0, max_value=30, value=1)
        cols["YearsWithCurrManager"] = st.number_input("YearsWithCurrManager", min_value=0, max_value=30, value=2)

    with c3:
        if not drop_sensitive:
            cols["Gender"] = st.selectbox("Gender", ["Male", "Female"])
            cols["MaritalStatus"] = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])

        cols["EducationBackground"] = st.selectbox(
            "EducationBackground",
            ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
        )
        cols["EmpDepartment"] = st.selectbox(
            "EmpDepartment",
            ["Sales", "Development", "Research & Development", "Human Resources", "Data Science", "Finance"],
        )
        cols["EmpJobRole"] = st.selectbox(
            "EmpJobRole",
            [
                "Sales Executive",
                "Developer",
                "Manager",
                "Research Scientist",
                "Human Resources",
                "Business Analyst",
                "Healthcare Representative",
                "Laboratory Technician",
                "Manufacturing Director",
                "Sales Representative",
            ],
        )
        cols["BusinessTravelFrequency"] = st.selectbox(
            "BusinessTravelFrequency",
            ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
        )
        cols["OverTime"] = st.selectbox("OverTime", ["Yes", "No"])
        cols["Attrition"] = st.selectbox("Attrition", ["Yes", "No"])

    row = {col: cols[col] for col in get_model_input_columns(drop_sensitive=drop_sensitive)}
    df = pd.DataFrame([row], index=[0])
    return df


def get_model_input_columns(drop_sensitive: bool) -> List[str]:
    cols = [c for c in SETTINGS.expected_columns if c not in {SETTINGS.id_col, SETTINGS.target_col}]
    if drop_sensitive:
        cols = [c for c in cols if c not in SETTINGS.fairness_sensitive_columns]
    return cols


def show_global_xai_outputs() -> None:
    st.subheader("Global XAI Outputs")

    xai_dir = SETTINGS.reports_dir / "xai"
    overall_csv = xai_dir / "global_shap_importance.csv"
    overall_png = xai_dir / "global_shap_top20.png"
    class_2_png = xai_dir / "class_2_top_shap.png"
    class_3_png = xai_dir / "class_3_top_shap.png"
    class_4_png = xai_dir / "class_4_top_shap.png"

    if overall_csv.exists():
        overall_df = pd.read_csv(overall_csv)
        st.write("**Global SHAP Importance**")
        st.dataframe(overall_df.head(20), use_container_width=True, hide_index=True)
    else:
        st.info("Global SHAP CSV not found. Run: python -m src.explainability.shap_global")

    cols = st.columns(2)
    with cols[0]:
        if overall_png.exists():
            st.image(str(overall_png), caption="Overall Global SHAP Top 20")
        else:
            st.info("Overall SHAP plot not found.")

    with cols[1]:
        if class_3_png.exists():
            st.image(str(class_3_png), caption="Class 3 Global SHAP")
        else:
            st.info("Class 3 SHAP plot not found.")

    cols2 = st.columns(2)
    with cols2[0]:
        if class_2_png.exists():
            st.image(str(class_2_png), caption="Class 2 Global SHAP")
    with cols2[1]:
        if class_4_png.exists():
            st.image(str(class_4_png), caption="Class 4 Global SHAP")


def show_fairness_outputs() -> None:
    st.subheader("Fairness Overview")

    fairness_dir = SETTINGS.reports_dir / "fairness"
    summary_path = fairness_dir / "fairness_report_summary.json"

    if not summary_path.exists():
        st.info("Fairness summary not found. Run: python -m src.explainability.fairness_report")
        return

    summary = load_json(summary_path)

    st.write("**Overall Test Metrics**")
    overall_df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in summary["overall_test_metrics"].items()]
    )
    st.dataframe(overall_df, use_container_width=True, hide_index=True)

    st.write("**Attribute Disparity Summary**")
    rows = []
    for item in summary.get("attribute_summaries", []):
        row = {"attribute": item["attribute"]}
        row.update(item["disparity"])
        rows.append(row)
    if rows:
        disp_df = pd.DataFrame(rows)
        st.dataframe(disp_df, use_container_width=True, hide_index=True)

    for attribute in summary.get("audited_attributes_available", []):
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in attribute).strip("_")
        csv_path = fairness_dir / f"{slug}_group_metrics.csv"
        acc_png = fairness_dir / f"{slug}_accuracy.png"
        f1_png = fairness_dir / f"{slug}_weighted_f1.png"

        st.markdown(f"### {attribute}")
        cols = st.columns(2)
        with cols[0]:
            if csv_path.exists():
                group_df = pd.read_csv(csv_path)
                st.dataframe(group_df, use_container_width=True, hide_index=True)
            else:
                st.info(f"{attribute} group metrics CSV not found.")
        with cols[1]:
            if acc_png.exists():
                st.image(str(acc_png), caption=f"{attribute} Accuracy by Group")
            if f1_png.exists():
                st.image(str(f1_png), caption=f"{attribute} Weighted F1 by Group")


def render_counterfactual_section(
    model: CatBoostClassifier,
    summary: Dict[str, Any],
    prep_data: Dict[str, Any],
) -> None:
    st.subheader("Counterfactual Suggestions")

    X_test_raw = prep_data["X_test_raw"]
    y_test = prep_data["y_test"].astype(int)
    X_train_raw = prep_data["X_train_raw"]
    y_train = prep_data["y_train"].astype(int)
    drop_sensitive = summary.get("drop_sensitive", False)

    sample_position = st.number_input(
        "Test sample position",
        min_value=0,
        max_value=len(X_test_raw) - 1,
        value=0,
        step=1,
    )
    selected_index = int(X_test_raw.index[sample_position])

    original_df = X_test_raw.loc[[selected_index]].copy()
    pred, proba = predict_with_model(model, original_df, drop_sensitive)
    predicted_class = int(pred.iloc[0])
    true_class = int(y_test.loc[selected_index])

    class_labels = sorted([int(c) for c in model.classes_])
    possible_target_classes = [c for c in class_labels if c > predicted_class]

    if not possible_target_classes:
        st.warning("This sample is already predicted as the highest class.")
        return

    desired_class = st.selectbox("Desired target class", possible_target_classes, index=0)
    max_features_changed = st.slider("Max features changed", 1, 5, 3)
    actionable_features = st.multiselect(
        "Actionable features",
        options=[c for c in DEFAULT_ACTIONABLE_FEATURES if c in original_df.columns],
        default=[c for c in DEFAULT_ACTIONABLE_FEATURES if c in original_df.columns],
    )

    if st.button("Generate Counterfactual", use_container_width=True):
        candidates = generate_counterfactual_candidates(
            model=model,
            original_sample_df=original_df,
            original_prediction=predicted_class,
            desired_class=desired_class,
            X_train_raw=X_train_raw,
            y_train=y_train,
            drop_sensitive=drop_sensitive,
            actionable_features=actionable_features,
            max_features_changed=max_features_changed,
            top_k=5,
        )

        st.write(f"**Selected sample index:** {selected_index}")
        st.write(f"**True class:** {true_class}")
        st.write(f"**Predicted class:** {predicted_class}")
        st.write("**Initial probabilities**")
        st.dataframe(
            pd.DataFrame(
                {"class": proba.columns.astype(str), "probability": proba.iloc[0].values}
            ),
            use_container_width=True,
            hide_index=True,
        )

        if not candidates:
            st.warning("No valid counterfactual candidate found under the current constraints.")
            return

        best = candidates[0]
        st.success(
            f"Best candidate found | target class probability: {best['desired_class_probability']:.4f} | cost: {best['cost']:.4f}"
        )

        st.write("**Actions**")
        actions_df = pd.DataFrame(best["actions_tr"])
        st.dataframe(actions_df, use_container_width=True, hide_index=True)

        st.write("**Top Candidate Table**")
        table_rows = []
        for i, cand in enumerate(candidates, start=1):
            table_rows.append(
                {
                    "rank": i,
                    "prototype_index": cand["prototype_index"],
                    "desired_class_probability": cand["desired_class_probability"],
                    "num_changes": cand["num_changes"],
                    "cost": cand["cost"],
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


def render_test_sample_explorer(
    model: CatBoostClassifier,
    summary: Dict[str, Any],
    prep_data: Dict[str, Any],
) -> None:
    st.subheader("Test Sample Explorer")

    X_test_raw = prep_data["X_test_raw"]
    y_test = prep_data["y_test"].astype(int)
    drop_sensitive = summary.get("drop_sensitive", False)

    sample_position = st.number_input(
        "Sample position in X_test",
        min_value=0,
        max_value=len(X_test_raw) - 1,
        value=0,
        step=1,
        key="sample_position_explorer",
    )
    selected_index = int(X_test_raw.index[sample_position])

    sample_df = X_test_raw.loc[[selected_index]].copy()
    pred, proba = predict_with_model(model, sample_df, drop_sensitive)

    predicted_class = int(pred.iloc[0])
    true_class = int(y_test.loc[selected_index])

    st.write(f"**Sample index:** {selected_index}")
    st.write(f"**True class:** {true_class}")

    display_prediction_block(predicted_class, proba.iloc[0])

    st.write("**Raw Input Row**")
    st.dataframe(sample_df.T.rename(columns={selected_index: "value"}), use_container_width=True)

    shap_values, base_values, feature_names, class_labels = get_local_shap_for_df(
        model=model,
        df=sample_df,
        drop_sensitive=drop_sensitive,
    )

    class_idx = class_labels.index(predicted_class)
    local_df = build_local_contribution_df(
        raw_row=sample_df.iloc[0],
        shap_row=shap_values[0, class_idx, :],
        feature_names=feature_names,
    )

    display_local_explanation(
        predicted_class=predicted_class,
        true_class=true_class,
        proba_for_pred=float(proba.loc[selected_index, predicted_class]),
        local_df=local_df,
    )


def render_manual_prediction(
    model: CatBoostClassifier,
    summary: Dict[str, Any],
) -> None:
    drop_sensitive = summary.get("drop_sensitive", False)

    manual_df = build_manual_input_df(drop_sensitive=drop_sensitive)

    if st.button("Run Prediction", use_container_width=True):
        pred, proba = predict_with_model(model, manual_df, drop_sensitive)
        predicted_class = int(pred.iloc[0])

        display_prediction_block(predicted_class, proba.iloc[0])

        shap_values, _, feature_names, class_labels = get_local_shap_for_df(
            model=model,
            df=manual_df,
            drop_sensitive=drop_sensitive,
        )
        class_idx = class_labels.index(predicted_class)
        local_df = build_local_contribution_df(
            raw_row=manual_df.iloc[0],
            shap_row=shap_values[0, class_idx, :],
            feature_names=feature_names,
        )

        display_local_explanation(
            predicted_class=predicted_class,
            true_class=None,
            proba_for_pred=float(proba.iloc[0][predicted_class]),
            local_df=local_df,
        )


def render_llm_governance_audit() -> None:
    st.subheader("LLM Governance & Audit")
    st.caption(
        "The LLM layer interprets structured XAI evidence only. It is not the predictive model and does not make HR decisions."
    )

    explanation_root = SETTINGS.reports_dir / "llm_explanations"
    case_dirs = sorted([p for p in explanation_root.glob("case_*") if p.is_dir()])
    case_options = [p.name.replace("case_", "") for p in case_dirs]

    st.markdown(
        """
        <style>
        .governance-card {
            border: 1px solid #d8e2dc;
            border-radius: 14px;
            padding: 1rem;
            background: linear-gradient(135deg, #f8fbf7 0%, #eef5f2 100%);
        }
        .warning-card {
            border-left: 5px solid #b45309;
            padding: 0.85rem 1rem;
            background: #fff7ed;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="warning-card">
        <strong>Governance boundary:</strong> explanations are for research review only.
        Do not use this system for hiring, firing, compensation, promotion, discipline, or autonomous employee evaluation.
        SHAP is attribution, not causality.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Evaluation Snapshot")
    snapshot_cols = st.columns(4)
    real_summary_path = explanation_root / "real_llm_eval_summary.csv"
    if real_summary_path.exists():
        real_summary = pd.read_csv(real_summary_path).tail(1).iloc[0].to_dict()
        metrics = [
            ("Faithfulness", real_summary.get("faithfulness_pass_rate")),
            ("Agent Success", real_summary.get("agent_success_rate")),
            ("Guardrail Refusal", real_summary.get("unsafe_prompt_refusal_rate")),
            ("Warning Consistency", real_summary.get("warning_consistency_rate")),
        ]
        for col, (label, value) in zip(snapshot_cols, metrics):
            with col:
                st.metric(label, value)
    else:
        st.info("Real LLM evaluation summary is unavailable.")

    evidence_tab, case_tab, audit_tab, chatbot_tab = st.tabs(
        ["Evidence Dashboard", "Case Explanation", "Agent Audit", "Guardrailed Chatbot"]
    )

    with evidence_tab:
        col1, col2 = st.columns(2)
        with col1:
            summary_path = explanation_root / "llm_governance_eval_summary.csv"
            if summary_path.exists():
                st.write("**Offline/Deterministic LLM Governance Evaluation**")
                st.dataframe(pd.read_csv(summary_path), use_container_width=True, hide_index=True)
            else:
                st.warning("LLM governance evaluation summary is unavailable. Run python -m src.llm.evaluate_llm_governance.")
            guardrail_path = SETTINGS.reports_dir / "chatbot_eval" / "guardrail_evaluation.csv"
            if guardrail_path.exists():
                guardrail_df = pd.read_csv(guardrail_path)
                st.write("**Chatbot Guardrail Evaluation**")
                st.dataframe(
                    guardrail_df.groupby("prompt_type")["pass"].mean().reset_index(name="pass_rate"),
                    use_container_width=True,
                    hide_index=True,
                )
        with col2:
            taxonomy_path = SETTINGS.reports_dir / "governance_reports" / "warning_taxonomy.csv"
            if taxonomy_path.exists():
                st.write("**Canonical Warning Taxonomy**")
                st.dataframe(pd.read_csv(taxonomy_path), use_container_width=True, hide_index=True)
            interpretation_path = explanation_root / "real_llm_eval_interpretation.md"
            if interpretation_path.exists():
                st.write("**Real OpenAI Evaluation Interpretation**")
                st.markdown(interpretation_path.read_text(encoding="utf-8")[:3000])

    with case_tab:
        selected_case = st.selectbox(
            "Select generated evidence case",
            options=case_options if case_options else ["unavailable"],
            disabled=not case_options,
        )
        if case_options:
            case_path = explanation_root / f"case_{selected_case}"
            evidence_path = case_path / "structured_evidence.json"
            explanation_path = case_path / "governed_explanation.json"
            left, right = st.columns(2)
            with left:
                if evidence_path.exists():
                    st.write("**Structured Prediction Evidence**")
                    st.json(load_json(evidence_path))
            with right:
                if explanation_path.exists():
                    st.write("**Governed Explanation**")
                    explanation = load_json(explanation_path)
                    st.markdown(explanation.get("short_explanation", ""))
                    st.json(explanation)

    with audit_tab:
        audit_root = SETTINGS.reports_dir / "agent_audits"
        audit_options = sorted(p.name for p in audit_root.glob("*governance_audit.md")) if audit_root.exists() else []
        selected_audit = st.selectbox(
            "Select agent audit report",
            options=audit_options if audit_options else ["unavailable"],
            disabled=not audit_options,
        )
        if audit_options:
            audit_path = audit_root / selected_audit
            st.markdown(audit_path.read_text(encoding="utf-8")[:6000])

    with chatbot_tab:
        st.write("**Guardrailed Chatbot**")
        selected_case_for_chat = st.selectbox(
            "Case context",
            options=case_options if case_options else ["unavailable"],
            disabled=not case_options,
            key="llm_governance_chat_case",
        )
        question = st.text_input(
            "Ask an audit question",
            value="Why are full-feature models not deployable?",
        )
        if st.button("Ask Guardrailed Chatbot", use_container_width=True):
            response = GuardrailedChatEngine().answer(
                question,
                case_id=selected_case_for_chat if case_options else None,
            )
            if response.allowed:
                st.info(response.answer)
            else:
                st.error(response.answer)
            if response.guardrail_reasons:
                st.write("Guardrail reasons:", response.guardrail_reasons)


# =========================
# App
# =========================
def main() -> None:
    st.title("Employee Performance XAI Dashboard")
    st.caption("Leakage-aware XAI, governance audit, and guardrailed LLM explanation dashboard")

    try:
        model, summary = get_model_and_summary()
    except Exception as exc:
        model = None
        summary = {}
        prep_data = None
        st.sidebar.warning("Legacy CatBoost dashboard unavailable.")
        st.sidebar.caption(str(exc))
    else:
        st.sidebar.header("Model Info")
        st.sidebar.write(f"**Model:** {summary.get('model_name')}")
        st.sidebar.write(f"**Drop sensitive:** {summary.get('drop_sensitive')}")
        st.sidebar.write(f"**Best iteration:** {summary.get('best_iteration')}")
        st.sidebar.write(f"**Test weighted F1:** {summary.get('test_metrics', {}).get('weighted_f1')}")
        st.sidebar.write(f"**Test macro F1:** {summary.get('test_metrics', {}).get('macro_f1')}")
        st.sidebar.write(f"**Test QWK:** {summary.get('test_metrics', {}).get('quadratic_weighted_kappa')}")
        prep_data = get_rebuilt_data(
            drop_sensitive=summary.get("drop_sensitive", False),
            test_size=summary.get("test_size", 0.20),
            random_state=summary.get("random_state", 42),
        )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Manual Prediction",
            "Test Sample Explorer",
            "Global XAI",
            "Fairness",
            "Counterfactuals",
            "LLM Governance & Audit",
        ]
    )

    with tab1:
        if model is None:
            st.warning("Legacy CatBoost model is unavailable. The LLM Governance tab remains available.")
        else:
            render_manual_prediction(model, summary)

    with tab2:
        if model is None or prep_data is None:
            st.warning("Legacy CatBoost model is unavailable. The LLM Governance tab remains available.")
        else:
            render_test_sample_explorer(model, summary, prep_data)

    with tab3:
        show_global_xai_outputs()

    with tab4:
        show_fairness_outputs()

    with tab5:
        if model is None or prep_data is None:
            st.warning("Legacy CatBoost model is unavailable. The LLM Governance tab remains available.")
        else:
            render_counterfactual_section(model, summary, prep_data)

    with tab6:
        render_llm_governance_audit()


if __name__ == "__main__":
    main()
