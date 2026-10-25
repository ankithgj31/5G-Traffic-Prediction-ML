
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 5G Traffic Prediction - Streamlit Demonstration App
# Model: Extra Trees (selected from the supplied trained model)
# Target: next 5-second Required_Bandwidth
# ------------------------------------------------------------

st.set_page_config(
    page_title="5G Traffic Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = "https://zenodo.org/records/21625252/files/augmented_dataset_ready.csv?download=1"
DATA_FILE = Path("5G_dataset.csv")
MODEL_FILE = Path("5G_Traffic_Final_Lightweight_Model.joblib")

# ------------------------- Styling --------------------------
st.markdown("""
<style>
.main-title {
    font-size: 2.3rem;
    font-weight: 800;
    margin-bottom: 0.1rem;
}
.subtitle {
    color: #6b7280;
    font-size: 1.05rem;
    margin-bottom: 1.3rem;
}
.metric-card {
    padding: 0.8rem 1rem;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
}
.small-note {
    color: #6b7280;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ----------------------- Data functions ---------------------
@st.cache_data(show_spinner=False)
def load_raw_data():
    if not DATA_FILE.exists():
        import requests
        r = requests.get(DATA_URL, timeout=120)
        r.raise_for_status()
        DATA_FILE.write_bytes(r.content)

    df = pd.read_csv(DATA_FILE)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    numeric_cols = [
        "Signal_Strength",
        "Latency",
        "Required_Bandwidth",
        "Allocated_Bandwidth",
        "Resource_Allocation",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def engineer_features(raw_df):
    df = raw_df.copy()

    # Same feature engineering used in the supplied notebook.
    df["Target_Next_Bandwidth"] = df["Required_Bandwidth"].shift(-1)

    seconds_in_day = (
        df["Timestamp"].dt.hour * 3600
        + df["Timestamp"].dt.minute * 60
        + df["Timestamp"].dt.second
    )

    df["hour"] = df["Timestamp"].dt.hour
    df["minute"] = df["Timestamp"].dt.minute
    df["second"] = df["Timestamp"].dt.second
    df["day_of_week"] = df["Timestamp"].dt.dayofweek

    df["time_sin"] = np.sin(2 * np.pi * seconds_in_day / 86400)
    df["time_cos"] = np.cos(2 * np.pi * seconds_in_day / 86400)

    bw = df["Required_Bandwidth"]

    for lag in [1, 2, 3, 5, 10]:
        df[f"bw_lag_{lag}"] = bw.shift(lag)

    df["bw_roll_mean_3"] = bw.shift(1).rolling(3).mean()
    df["bw_roll_std_3"] = bw.shift(1).rolling(3).std()
    df["bw_roll_mean_5"] = bw.shift(1).rolling(5).mean()
    df["bw_roll_std_5"] = bw.shift(1).rolling(5).std()

    df["bw_change_1"] = bw - bw.shift(1)
    df["bw_change_2"] = bw - bw.shift(2)

    df = df.dropna().reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    package = joblib.load(MODEL_FILE)
    return package


@st.cache_data(show_spinner=False)
def calculate_test_predictions(df, _model):
    features = [
        "Application_Type",
        "Signal_Strength",
        "Latency",
        "Required_Bandwidth",
        "Allocated_Bandwidth",
        "Resource_Allocation",
        "hour",
        "minute",
        "second",
        "day_of_week",
        "time_sin",
        "time_cos",
        "bw_lag_1",
        "bw_lag_2",
        "bw_lag_3",
        "bw_lag_5",
        "bw_lag_10",
        "bw_roll_mean_3",
        "bw_roll_std_3",
        "bw_roll_mean_5",
        "bw_roll_std_5",
        "bw_change_1",
        "bw_change_2",
    ]

    split = int(len(df) * 0.80)
    X_test = df[features].iloc[split:].copy()
    y_test = df["Target_Next_Bandwidth"].iloc[split:].copy()
    pred = _model.predict(X_test)

    return y_test, pred, df.iloc[split:].copy()


# --------------------------- Load ----------------------------
st.markdown('<div class="main-title">📡 5G Traffic Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Machine Learning-Based 5G Traffic Prediction • '
    'Next 5-second bandwidth demand forecasting</div>',
    unsafe_allow_html=True,
)

try:
    raw_df = load_raw_data()
    df = engineer_features(raw_df)
    package = load_model()
    model = package["model"]
    model_name = package.get("model_name", "Extra Trees")
except Exception as e:
    st.error("The application could not load the trained model or dataset.")
    st.code(str(e))
    st.warning(
        "Use the exact Python dependencies in requirements.txt. "
        "The trained .joblib was created with scikit-learn 1.6.1."
    )
    st.stop()

features = package.get("features", [
    "Application_Type", "Signal_Strength", "Latency",
    "Required_Bandwidth", "Allocated_Bandwidth",
    "Resource_Allocation", "hour", "minute", "second",
    "day_of_week", "time_sin", "time_cos",
    "bw_lag_1", "bw_lag_2", "bw_lag_3", "bw_lag_5", "bw_lag_10",
    "bw_roll_mean_3", "bw_roll_std_3", "bw_roll_mean_5",
    "bw_roll_std_5", "bw_change_1", "bw_change_2"
])

# ------------------------- Sidebar ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Open section",
    ["Dashboard", "Live Prediction", "Traffic Analysis", "Model Performance", "About Project"],
)

st.sidebar.divider()
st.sidebar.caption("Trained model")
st.sidebar.write(f"**{model_name}**")
st.sidebar.caption("Target: next 5-second Required_Bandwidth")

# ------------------------- Dashboard -------------------------
if page == "Dashboard":
    st.subheader("Project Dashboard")

    split = int(len(df) * 0.80)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Records", f"{len(raw_df):,}")
    c2.metric("Usable Records", f"{len(df):,}")
    c3.metric("Model", model_name)
    c4.metric("R² Score", "0.8366")

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Traffic Trend")
        n = min(500, len(df))
        chart_df = df[["Timestamp", "Required_Bandwidth"]].tail(n).set_index("Timestamp")
        st.line_chart(chart_df)

    with right:
        st.subheader("Project Results")
        result_table = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "R²"],
            "Value": [322.32, 454.79, 0.8366]
        })
        st.dataframe(result_table, hide_index=True, use_container_width=True)

        st.info(
            "The project predicts the Required Bandwidth at the next "
            "5-second observation using current and historical network features."
        )

    st.subheader("System Workflow")
    st.markdown("""
    **Dataset → Preprocessing → Feature Engineering → Train/Test Split → 
    ML Regression → Next-Interval Prediction → Evaluation → Visualization**
    """)

# ---------------------- Live Prediction ----------------------
elif page == "Live Prediction":
    st.subheader("🔮 Live Next-Interval Prediction")
    st.write(
        "Select a real observation from the dataset. The application automatically "
        "constructs the same lag, rolling, and time features used during training."
    )

    max_idx = len(df) - 1
    row_idx = st.slider(
        "Select current network observation",
        min_value=10,
        max_value=max_idx - 1,
        value=max(10, max_idx - 100),
        step=1,
    )

    row = df.iloc[row_idx]

    c1, c2, c3 = st.columns(3)
    with c1:
        app_type = st.number_input(
            "Application Type",
            value=int(row["Application_Type"]),
            step=1,
        )
        signal = st.number_input(
            "Signal Strength",
            value=float(row["Signal_Strength"]),
            format="%.3f",
        )
    with c2:
        latency = st.number_input(
            "Latency",
            value=float(row["Latency"]),
            format="%.3f",
        )
        required = st.number_input(
            "Current Required Bandwidth",
            value=float(row["Required_Bandwidth"]),
            format="%.3f",
        )
    with c3:
        allocated = st.number_input(
            "Allocated Bandwidth",
            value=float(row["Allocated_Bandwidth"]),
            format="%.3f",
        )
        resource = st.number_input(
            "Resource Allocation",
            value=float(row["Resource_Allocation"]),
            format="%.3f",
        )

    st.caption(
        f"Observation time: {row['Timestamp']}  |  "
        f"Actual next bandwidth (for demonstration only): "
        f"{row['Target_Next_Bandwidth']:,.2f}"
    )

    if st.button("🚀 Predict Next Bandwidth", type="primary", use_container_width=True):
        input_row = row[features].copy()

        # Replace the directly editable current-observation values.
        input_row["Application_Type"] = app_type
        input_row["Signal_Strength"] = signal
        input_row["Latency"] = latency
        input_row["Required_Bandwidth"] = required
        input_row["Allocated_Bandwidth"] = allocated
        input_row["Resource_Allocation"] = resource

        input_df = pd.DataFrame([input_row], columns=features)

        try:
            prediction = float(model.predict(input_df)[0])

            st.success("Prediction completed successfully.")

            a, b, c = st.columns(3)
            a.metric("Predicted Next Bandwidth", f"{prediction:,.2f}")
            b.metric("Current Bandwidth", f"{required:,.2f}")
            b.metric(
                "Change",
                f"{prediction - required:,.2f}",
            )

            actual = float(row["Target_Next_Bandwidth"])
            error = actual - prediction

            st.write("### Prediction vs Actual")
            comparison = pd.DataFrame({
                "Value": [prediction, actual],
            }, index=["Predicted", "Actual"])
            st.bar_chart(comparison)

            st.write(f"**Actual next bandwidth:** {actual:,.2f}")
            st.write(f"**Prediction error:** {error:,.2f}")

            if prediction > required * 1.15:
                st.warning("⚠️ Predicted demand is significantly higher than current demand.")
            elif prediction < required * 0.85:
                st.info("📉 Predicted demand is significantly lower than current demand.")
            else:
                st.success("✅ Predicted demand is relatively close to current demand.")

        except Exception as e:
            st.error("Prediction failed.")
            st.code(str(e))

# ---------------------- Traffic Analysis ---------------------
elif page == "Traffic Analysis":
    st.subheader("📈 Traffic Analysis")

    tab1, tab2, tab3 = st.tabs([
        "Actual vs Predicted",
        "Prediction Errors",
        "Feature View",
    ])

    with tab1:
        with st.spinner("Generating test-set predictions..."):
            y_test, pred, test_df = calculate_test_predictions(df, model)

        n = min(300, len(y_test))
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.plot(y_test.iloc[:n].to_numpy(), label="Actual")
        ax.plot(pred[:n], label=f"Predicted - {model_name}")
        ax.set_title("Actual vs Predicted Next-Interval 5G Bandwidth")
        ax.set_xlabel("Test observation (5-second interval)")
        ax.set_ylabel("Required Bandwidth")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with tab2:
        with st.spinner("Calculating errors..."):
            y_test, pred, _ = calculate_test_predictions(df, model)
        errors = y_test.to_numpy() - pred

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Error", f"{errors.mean():,.2f}")
        c2.metric("Median Absolute Error", f"{np.median(np.abs(errors)):,.2f}")
        c3.metric("Maximum Absolute Error", f"{np.max(np.abs(errors)):,.2f}")

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.hist(errors, bins=40)
        ax.axvline(0, linestyle="--")
        ax.set_title("Prediction Error Distribution")
        ax.set_xlabel("Actual - Predicted")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with tab3:
        st.dataframe(
            df[[
                "Timestamp",
                "Application_Type",
                "Signal_Strength",
                "Latency",
                "Required_Bandwidth",
                "Allocated_Bandwidth",
                "Resource_Allocation",
                "Target_Next_Bandwidth",
            ]].tail(100),
            use_container_width=True,
            hide_index=True,
        )

# --------------------- Model Performance --------------------
elif page == "Model Performance":
    st.subheader("🤖 Model Performance")

    results = pd.DataFrame([
        ["Persistence Baseline", 163.82, 239407, 489.29, 0.8108],
        ["Linear Regression", 1423.45, 2846570, 1687.17, -1.2492],
        ["Decision Tree", 417.24, 533995, 730.75, 0.5781],
        ["Random Forest", 593.27, 416166, 645.11, 0.6712],
        ["Extra Trees ⭐", 322.32, 206830, 454.79, 0.8366],
        ["Optimized Random Forest", 685.50, 582443, 763.18, 0.5398],
    ], columns=["Model", "MAE", "MSE", "RMSE", "R²"])

    st.dataframe(
        results.style.format({
            "MAE": "{:.2f}",
            "MSE": "{:.2f}",
            "RMSE": "{:.2f}",
            "R²": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        "Best ML model: Extra Trees — RMSE 454.79 and R² 0.8366."
    )

    st.info(
        "Important: this is a regression project, so classification accuracy is "
        "not used. R² is the appropriate goodness-of-fit measure."
    )

# ------------------------- About --------------------------------
elif page == "About Project":
    st.subheader("📘 About the Project")

    st.markdown("""
    ### Machine Learning-Based 5G Traffic Prediction

    **Objective:** predict the bandwidth demand of the next 5-second observation
    from current and historical 5G network information.

    **Dataset:** 15,760-record 5G network-slicing dataset.

    **Candidate models:**
    - Linear Regression
    - Decision Tree
    - Random Forest
    - Extra Trees (additional lightweight benchmark)

    **Evaluation:** MAE, MSE, RMSE and R².

    **Selected model:** Extra Trees.

    **Best observed test result:**
    - MAE: 322.32
    - RMSE: 454.79
    - R²: 0.8366

    The application is intended as a demonstration interface for the trained
    model. It does not replace the notebook used for the formal experiment.
    """)

    st.divider()
    st.caption(
        "Dataset source: Zenodo — Dataset for Proactive Resource Allocation in 5G Network Slicing."
    )
