from __future__ import annotations

import io
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# -----------------------------------------------------------------------------
# Configuration and shared helpers
# -----------------------------------------------------------------------------

def get_api_base_url() -> str:
    explicit_url = os.getenv("API_BASE_URL", "").strip()
    if explicit_url:
        return explicit_url.rstrip("/")

    api_hostport = os.getenv("API_HOSTPORT", "").strip()
    if api_hostport:
        return f"http://{api_hostport}".rstrip("/")

    api_host = os.getenv("API_HOST", "").strip()
    api_port = os.getenv("API_PORT", "10000").strip()
    if api_host:
        return f"http://{api_host}:{api_port}".rstrip("/")

    return "http://127.0.0.1:8000"


API_BASE_URL = get_api_base_url()
REQUEST_TIMEOUT = (10, 180)
ID_COLUMNS = ("vehicle_id", "truck_id", "record_id", "id")

st.set_page_config(
    page_title="Scania APS | Predictive Maintenance",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.35rem; padding-bottom: 2.2rem; max-width: 1500px;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(120,120,120,.18);}
    .hero {padding: 1.1rem 1.35rem; border-radius: 15px; background: linear-gradient(110deg,#142d42,#1b5260); color: white; margin-bottom: 1rem;}
    .hero h1 {font-size: 1.9rem; margin: 0 0 .35rem 0; color: white;}
    .hero p {margin: 0; color: #e6f0f3; font-size: .98rem;}
    .small-muted {color: #78838c; font-size: .85rem;}
    .section-label {font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #5f7a88; margin-bottom: .25rem;}
    .decision-inspect {color:#b42318; font-weight:700;}
    .decision-normal {color:#067647; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str) -> Any:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=REQUEST_TIMEOUT)
    if not response.ok:
        raise RuntimeError(extract_error(response))
    return response.json()


def api_post_file(path: str, file_name: str, file_bytes: bytes) -> Any:
    response = requests.post(
        f"{API_BASE_URL}{path}",
        files={"file": (file_name, file_bytes, "text/csv")},
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(extract_error(response))
    return response.json()


def api_post_json(path: str, payload: dict[str, Any]) -> Any:
    response = requests.post(
        f"{API_BASE_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT
    )
    if not response.ok:
        raise RuntimeError(extract_error(response))
    return response.json()


def extract_error(response: requests.Response) -> str:
    try:
        body = response.json()
        detail = body.get("detail", body)
        return f"API error ({response.status_code}): {detail}"
    except Exception:
        return f"API error ({response.status_code}): {response.text[:600]}"


@st.cache_data(show_spinner=False)
def cached_model_info(api_url: str) -> dict[str, Any]:
    response = requests.get(f"{api_url}/model-info", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def api_status() -> tuple[bool, str]:
    try:
        result = api_get("/health")
        if result.get("model_loaded"):
            return True, "API and model online"
        return False, "API is running, but the model is not loaded"
    except Exception as exc:
        return False, str(exc)


def parse_csv_for_details(file_bytes: bytes, required_features: list[str]) -> pd.DataFrame:
    """Parse either a normal CSV or the original UCI APS file with 20 preamble rows."""
    candidates: list[pd.DataFrame] = []
    errors: list[str] = []
    for skiprows in (0, 20):
        try:
            frame = pd.read_csv(
                io.BytesIO(file_bytes),
                skiprows=skiprows,
                na_values=["na", "NA", "Na", "N/A", "n/a", ""],
                low_memory=False,
            )
            candidates.append(frame)
        except Exception as exc:
            errors.append(str(exc))
    if not candidates:
        raise ValueError("Could not parse the uploaded CSV. " + "; ".join(errors))

    def matches(frame: pd.DataFrame) -> int:
        return sum(column in frame.columns for column in required_features)

    frame = max(candidates, key=matches)
    if required_features and matches(frame) < len(required_features):
        missing = [c for c in required_features if c not in frame.columns]
        raise ValueError(
            f"Could not identify the expected sensor header. Missing {len(missing)} required columns: "
            f"{missing[:15]}. Upload the original APS sensor schema."
        )
    return frame


def build_record_ids(frame: pd.DataFrame) -> list[str]:
    id_column = next((column for column in ID_COLUMNS if column in frame.columns), None)
    if id_column:
        values = frame[id_column].astype("string").fillna("").tolist()
        return [value if str(value).strip() else f"Truck-{i + 1:06d}" for i, value in enumerate(values)]
    return [f"Truck-{i + 1:06d}" for i in range(len(frame))]


def json_safe_row(row: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            result[str(key)] = None
        elif isinstance(value, np.generic):
            result[str(key)] = value.item()
        else:
            result[str(key)] = value
    return result


def show_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def get_batch_state() -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    return st.session_state.get("batch_results_df"), st.session_state.get("batch_payload")


def format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


# -----------------------------------------------------------------------------
# Sidebar and API readiness
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🚛 Scania APS")
    st.caption("Predictive maintenance workspace")
    page = st.radio(
        "Workspace",
        ["Overview", "Batch Scoring", "Inspection Queue", "Truck Details", "Model Monitoring"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Backend connection")
    if st.button("Refresh API status", use_container_width=True):
        st.cache_data.clear()
    online, status_message = api_status()
    if online:
        st.success(status_message)
    else:
        st.error("Backend unavailable")
        st.caption(status_message)
    st.caption(f"API: {API_BASE_URL}")

if not online:
    st.title("Scania APS Failure Prediction")
    st.error("The FastAPI backend is not reachable. Start the backend or check the API_BASE_URL configuration.")
    st.code("uvicorn backend.main:app --host 0.0.0.0 --port 8000", language="bash")
    st.stop()

try:
    model_info = cached_model_info(API_BASE_URL)
except Exception as exc:
    st.error(f"Could not retrieve model information: {exc}")
    st.stop()

threshold = float(model_info.get("threshold", 0.01811))
required_features = model_info.get("required_features", [])


# -----------------------------------------------------------------------------
# Page 1: Overview
# -----------------------------------------------------------------------------

if page == "Overview":
    show_hero(
        "Scania APS Failure Prediction",
        "A cost-sensitive predictive maintenance workspace for prioritising truck inspections.",
    )

    try:
        metrics_payload = api_get("/model-metrics")
        metrics = metrics_payload.get("metrics", {})
    except Exception as exc:
        metrics_payload, metrics = {}, {}
        st.warning(f"Historical model metrics are unavailable: {exc}")

    batch_df, batch_payload = get_batch_state()
    latest_summary = (batch_payload or {}).get("summary", {})

    total_scored = latest_summary.get("total_vehicles")
    flagged = latest_summary.get("vehicles_recommended_for_inspection")
    inspection_rate = latest_summary.get("inspection_rate")

    st.markdown('<div class="section-label">Model reference</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected model", model_info.get("model_name", "Baseline XGBoost"))
    col2.metric("Decision threshold", f"{threshold:.5f}")
    col3.metric("Holdout PR-AUC", f"{metrics.get('pr_auc', 0):.5f}" if metrics else "—")
    col4.metric("Holdout recall", format_percent(metrics.get("recall")) if metrics else "—")

    st.markdown('<div class="section-label">Latest batch</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vehicles scored", f"{int(total_scored):,}" if total_scored is not None else "—")
    c2.metric("Flagged for inspection", f"{int(flagged):,}" if flagged is not None else "—")
    c3.metric("Batch inspection rate", format_percent(inspection_rate) if inspection_rate is not None else "—")
    c4.metric("Score type", "Uncalibrated")

    st.info(
        "The model produces uncalibrated scores. The saved threshold is applied to convert each score into an inspection decision; scores should not be interpreted as calibrated failure probabilities."
    )

    st.subheader("Model holdout performance")
    if metrics:
        left, right = st.columns([1.2, 1])
        with left:
            perf = pd.DataFrame(
                {"Metric": ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1-score", "Accuracy"],
                 "Value": [metrics.get("pr_auc"), metrics.get("roc_auc"), metrics.get("precision"),
                           metrics.get("recall"), metrics.get("f1_score"), metrics.get("accuracy")]}
            )
            perf["Value"] = pd.to_numeric(perf["Value"], errors="coerce")
            fig = px.bar(perf, x="Metric", y="Value", range_y=[0, 1], text=perf["Value"].map(lambda x: f"{x:.3f}"))
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=25, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            cost_data = pd.DataFrame({
                "Outcome": ["True positives", "False positives", "True negatives", "False negatives"],
                "Vehicles": [metrics.get("true_positives", 0), metrics.get("false_positives", 0),
                             metrics.get("true_negatives", 0), metrics.get("false_negatives", 0)]
            })
            fig = px.bar(cost_data, x="Outcome", y="Vehicles", color="Outcome", text="Vehicles")
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=25, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Official holdout: {metrics_payload.get('holdout_rows', '—'):,} rows; "
            f"historical cost = {metrics.get('total_cost', '—'):,}; "
            f"inspection rate = {format_percent(metrics.get('inspection_rate'))}."
        )
    else:
        st.warning("No historical holdout metrics are configured in the backend.")


# -----------------------------------------------------------------------------
# Page 2: Batch scoring
# -----------------------------------------------------------------------------

elif page == "Batch Scoring":
    show_hero("Batch Scoring", "Upload a raw APS sensor CSV and score every truck through the saved preprocessing and XGBoost pipeline.")
    st.caption(f"Expected input: {model_info.get('raw_input_feature_count', len(required_features))} sensor columns. Optional target column `class` is ignored.")

    uploaded = st.file_uploader("Upload sensor readings (.csv)", type=["csv"], key="batch_upload")
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        st.write(f"**File:** {uploaded.name} · **Size:** {len(file_bytes) / 1024:.1f} KB")
        col_validate, col_score = st.columns(2)
        with col_validate:
            validate_clicked = st.button("Validate CSV", use_container_width=True)
        with col_score:
            score_clicked = st.button("Score all trucks", type="primary", use_container_width=True)

        if validate_clicked:
            with st.spinner("Validating sensor schema and data quality..."):
                try:
                    validation = api_post_file("/validate", uploaded.name, file_bytes)
                    st.session_state["last_validation"] = validation
                    st.success("CSV validation passed.")
                    st.json(validation.get("data_quality", {}))
                except Exception as exc:
                    st.error(str(exc))

        if score_clicked:
            with st.spinner("Scoring vehicles. Large files may take a little longer..."):
                try:
                    result = api_post_file("/predict/batch", uploaded.name, file_bytes)
                    result_df = pd.DataFrame(result.get("results", []))
                    raw_df = parse_csv_for_details(file_bytes, required_features)
                    st.session_state["batch_payload"] = result
                    st.session_state["batch_results_df"] = result_df
                    st.session_state["batch_raw_df"] = raw_df
                    st.session_state["batch_file_name"] = uploaded.name
                    st.success(f"Scoring completed for {len(result_df):,} vehicles.")
                except Exception as exc:
                    st.error(str(exc))

    batch_df, batch_payload = get_batch_state()
    if batch_df is not None and batch_payload is not None:
        st.divider()
        summary = batch_payload.get("summary", {})
        a, b, c, d = st.columns(4)
        a.metric("Vehicles analysed", f"{summary.get('total_vehicles', 0):,}")
        b.metric("Flagged for inspection", f"{summary.get('vehicles_recommended_for_inspection', 0):,}")
        c.metric("Not flagged", f"{summary.get('vehicles_not_flagged', 0):,}")
        d.metric("Inspection rate", format_percent(summary.get("inspection_rate")))

        st.subheader("Prediction distribution")
        chart_col, quality_col = st.columns([1.1, 1])
        with chart_col:
            if not batch_df.empty:
                decision_counts = batch_df["decision"].value_counts().rename_axis("Decision").reset_index(name="Vehicles")
                fig = px.pie(decision_counts, names="Decision", values="Vehicles", hole=0.62)
                fig.update_layout(height=300, margin=dict(l=5, r=5, t=10, b=5))
                st.plotly_chart(fig, use_container_width=True)
        with quality_col:
            st.markdown("**Data quality checks**")
            quality = batch_payload.get("data_quality", {})
            st.metric("Missing sensor cells", f"{quality.get('missing_sensor_cells', 0):,}")
            st.metric("Rows with missing sensors", f"{quality.get('rows_with_missing_sensors', 0):,}")
            st.metric("Invalid numeric cells", f"{quality.get('invalid_numeric_cells_coerced_to_missing', 0):,}")
            if quality.get("ignored_extra_columns"):
                st.caption("Ignored extra columns: " + ", ".join(quality["ignored_extra_columns"][:10]))

        st.subheader("Highest-risk vehicles")
        st.dataframe(batch_df.head(15), use_container_width=True, hide_index=True)
        st.download_button(
            "Download complete prediction report (CSV)",
            data=batch_df.to_csv(index=False).encode("utf-8"),
            file_name="scania_aps_predictions.csv",
            mime="text/csv",
            type="primary",
        )
        st.caption("Scores are uncalibrated model outputs. The inspection decision uses the saved OOF-selected threshold.")


# -----------------------------------------------------------------------------
# Page 3: Inspection queue
# -----------------------------------------------------------------------------

elif page == "Inspection Queue":
    show_hero("Inspection Queue", "Review and prioritise vehicles flagged by the APS prediction model.")
    batch_df, batch_payload = get_batch_state()
    if batch_df is None or batch_df.empty:
        st.info("No scored batch is available. Go to **Batch Scoring** and upload a sensor CSV first.")
    else:
        flagged_only = st.toggle("Show vehicles flagged for inspection only", value=True)
        minimum_score = st.slider("Minimum model score", 0.0, 1.0, 0.0, 0.001, format="%.3f")
        queue = batch_df.copy()
        if flagged_only:
            queue = queue[queue["decision"] == "INSPECT"]
        queue = queue[queue["aps_failure_score"] >= minimum_score]
        queue = queue.sort_values("aps_failure_score", ascending=False)

        st.metric("Vehicles in current queue", f"{len(queue):,}")
        st.dataframe(
            queue[["record_id", "aps_failure_score", "decision", "priority", "missing_sensor_count"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "aps_failure_score": st.column_config.NumberColumn("Uncalibrated score", format="%.5f"),
                "missing_sensor_count": st.column_config.NumberColumn("Missing sensor values", format="%d"),
            },
        )
        st.download_button(
            "Download current queue",
            data=queue.to_csv(index=False).encode("utf-8"),
            file_name="scania_inspection_queue.csv",
            mime="text/csv",
        )
        st.caption("Rows are sorted by model score. The model provides an inspection flag, not a validated high/medium/low severity scale.")


# -----------------------------------------------------------------------------
# Page 4: Truck details and local SHAP explanation
# -----------------------------------------------------------------------------

elif page == "Truck Details":
    show_hero("Truck Details", "Inspect an individual prediction and request a local SHAP explanation.")
    batch_df, batch_payload = get_batch_state()
    raw_df = st.session_state.get("batch_raw_df")

    if batch_df is None or raw_df is None or batch_df.empty:
        st.info("Score a CSV on the **Batch Scoring** page before opening individual truck details.")
    else:
        selected_id = st.selectbox("Select a truck", batch_df["record_id"].astype(str).tolist())
        result_row = batch_df.loc[batch_df["record_id"].astype(str) == selected_id].iloc[0]

        raw_ids = build_record_ids(raw_df)
        try:
            raw_index = raw_ids.index(selected_id)
        except ValueError:
            raw_index = None

        c1, c2, c3 = st.columns(3)
        c1.metric("Uncalibrated model score", f"{float(result_row['aps_failure_score']):.5f}")
        c2.metric("Decision", str(result_row["decision"]))
        c3.metric("Missing sensor values", int(result_row["missing_sensor_count"]))

        if raw_index is None:
            st.error("Could not map this result back to the uploaded raw row. Re-run batch scoring with the original CSV.")
        else:
            raw_record = json_safe_row(raw_df.iloc[raw_index])
            with st.expander("View raw sensor record"):
                st.dataframe(pd.DataFrame([raw_record]), use_container_width=True, hide_index=True)

            if st.button("Generate SHAP explanation", type="primary"):
                with st.spinner("Calculating local SHAP contributions..."):
                    try:
                        explanation = api_post_json(
                            "/explain",
                            {"record_id": selected_id, "features": raw_record, "top_n": 12},
                        )
                        st.session_state["last_explanation"] = explanation
                        st.session_state["last_explanation_id"] = selected_id
                    except Exception as exc:
                        st.error(str(exc))

            explanation = st.session_state.get("last_explanation")
            explanation_id = st.session_state.get("last_explanation_id")
            if explanation and explanation_id == selected_id:
                st.subheader("Local feature contributions")
                st.caption(explanation.get("note", "SHAP explains model output, not causality."))
                contributions = pd.DataFrame(explanation.get("top_contributions", []))
                if not contributions.empty:
                    contributions["direction_label"] = contributions["direction"].map({
                        "pushes_towards_aps_failure": "Pushes towards APS failure",
                        "pushes_away_from_aps_failure": "Pushes away from APS failure",
                    })
                    contributions = contributions.sort_values("shap_value")
                    fig = px.bar(
                        contributions,
                        x="shap_value",
                        y="feature",
                        orientation="h",
                        color="direction_label",
                        color_discrete_map={
                            "Pushes towards APS failure": "#C2413B",
                            "Pushes away from APS failure": "#16856B",
                        },
                        labels={"shap_value": "SHAP contribution (raw model output)", "feature": "Feature"},
                    )
                    fig.update_layout(height=470, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="Direction")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(contributions[["feature", "shap_value", "direction_label"]], use_container_width=True, hide_index=True)
                else:
                    st.info("No SHAP contributions were returned.")
                st.caption(f"Explanation scale: {explanation.get('explanation_scale', 'raw model output')}. SHAP contributions are not probability-point changes or causal effects.")


# -----------------------------------------------------------------------------
# Page 5: Historical model monitoring
# -----------------------------------------------------------------------------

elif page == "Model Monitoring":
    show_hero("Model Monitoring", "Historical holdout diagnostics and data-quality checks. These are not live model-performance metrics.")
    try:
        payload = api_get("/model-metrics")
    except Exception as exc:
        st.error(f"Could not load model monitoring data: {exc}")
        st.stop()

    metrics = payload.get("metrics", {})
    st.subheader("Official holdout metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.5f}")
    k2.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.5f}")
    k3.metric("Recall", format_percent(metrics.get("recall")))
    k4.metric("Total cost", f"{metrics.get('total_cost', 0):,}")

    st.write(
        f"Evaluation source: **{payload.get('evaluation_source', 'historical evaluation')}** · "
        f"Rows: **{payload.get('holdout_rows', '—'):,}** · "
        f"Threshold: **{float(payload.get('threshold', threshold)):.5f}**"
    )
    st.caption("The selected XGBoost scores are uncalibrated. The calibration curve is shown only as a reliability diagnostic.")

    curves = payload.get("curves", {})
    if not curves:
        st.warning(payload.get("monitoring_curves_note", "Monitoring curves have not been exported from the notebook yet."))
    else:
        pr = curves.get("precision_recall", {})
        if pr.get("precision") and pr.get("recall"):
            st.subheader("Precision–Recall curve")
            pr_frame = pd.DataFrame({"Recall": pr["recall"], "Precision": pr["precision"]})
            pr_frame = pr_frame.replace([np.inf, -np.inf], np.nan).dropna()
            fig = px.line(pr_frame, x="Recall", y="Precision")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), yaxis_range=[0, 1], xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        calibration = curves.get("calibration", {})
        if calibration.get("mean_predicted_probability"):
            st.subheader("Calibration reliability curve")
            cal_frame = pd.DataFrame({
                "Mean predicted score": calibration["mean_predicted_probability"],
                "Observed positive fraction": calibration["observed_positive_fraction"],
            })
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                line=dict(color="gray", dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=cal_frame["Mean predicted score"], y=cal_frame["Observed positive fraction"],
                mode="lines+markers", name="XGBoost holdout diagnostic"
            ))
            fig.update_layout(height=360, xaxis_title="Mean predicted score", yaxis_title="Observed positive fraction", xaxis_range=[0, 1], yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The final XGBoost model was not calibrated. This chart diagnoses its score reliability; it does not apply a calibration transformation.")

        cost_curve = curves.get("cost_vs_threshold", [])
        if cost_curve:
            st.subheader("Cost versus threshold")
            cost_frame = pd.DataFrame(cost_curve)
            fig = px.line(cost_frame, x="threshold", y="cost", labels={"threshold": "Decision threshold", "cost": "Misclassification cost"})
            fig.add_vline(x=float(payload.get("threshold", threshold)), line_dash="dash", line_color="red", annotation_text="Selected OOF threshold")
            fig.update_layout(height=370, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("This is a historical holdout cost curve for reporting only. Do not use it to retune the threshold.")

    batch_df, batch_payload = get_batch_state()
    if batch_payload:
        st.subheader("Latest upload: data-quality checks")
        quality = batch_payload.get("data_quality", {})
        q1, q2, q3 = st.columns(3)
        q1.metric("Missing sensor cells", f"{quality.get('missing_sensor_cells', 0):,}")
        q2.metric("Rows with missing sensors", f"{quality.get('rows_with_missing_sensors', 0):,}")
        q3.metric("Invalid numeric cells", f"{quality.get('invalid_numeric_cells_coerced_to_missing', 0):,}")

st.sidebar.divider()
st.sidebar.caption("Scores are uncalibrated. Inspection decisions use the saved OOF-selected threshold.")
