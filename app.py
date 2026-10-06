import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from dotenv import load_dotenv

# 1. Load local environment configurations (Zero-Cloud policy)
load_dotenv()

# Set Streamlit page settings
st.set_page_config(
    page_title="GhostAudit Sentinel",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ GhostAudit: Local Zero-Cloud Financial Sentinel")
st.markdown("Auditing local transactions on-device using ML anomaly detection.")

# 2. Sidebar configuration
st.sidebar.header("Sentinel Settings")
contamination = st.sidebar.slider(
    "Expected Anomaly Rate (Contamination)",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01
)

uploaded_file = st.sidebar.file_uploader("Upload Financial Ledger (CSV)", type=["csv"])

# Helper function to generate mock ledger if no file is provided
def get_sample_data():
    np.random.seed(42)
    n_samples = 150
    amounts = np.random.exponential(scale=120, size=n_samples)
    # Inject deliberate outliers/anomalies
    amounts[15] = 4500.0
    amounts[42] = 7800.0
    amounts[99] = 9200.0
    
    categories = np.random.choice(["Office Supplies", "Software", "Travel", "Payroll", "Consulting"], size=n_samples)
    
    return pd.DataFrame({
        "Transaction_ID": [f"TXN-{1000 + i}" for i in range(n_samples)],
        "Category": categories,
        "Amount": np.round(amounts, 2)
    })

# 3. Load or generate transaction data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom ledger loaded successfully.")
else:
    st.sidebar.info("Using simulated local transaction data.")
    df = get_sample_data()

# 4. Anomaly Detection Engine (scikit-learn Isolation Forest)
features = df[["Amount"]]
model = IsolationForest(contamination=contamination, random_state=42)
df["Anomaly_Flag"] = model.fit_predict(features)
# IsolationForest outputs -1 for anomalies and 1 for normal data points
df["Status"] = df["Anomaly_Flag"].map({1: "Normal", -1: "Flagged Anomaly"})

# 5. Dashboard Metrics
col1, col2, col3 = st.columns(3)
total_transactions = len(df)
flagged_transactions = (df["Anomaly_Flag"] == -1).sum()
total_spend = df["Amount"].sum()

col1.metric("Total Transactions", total_transactions)
col2.metric("Flagged Anomalies", flagged_transactions)
col3.metric("Total Ledger Volume", f"${total_spend:,.2f}")

# 6. Visualizing Findings (Matplotlib)
st.subheader("Auditing Distribution & Outlier Detection")

fig, ax = plt.subplots(figsize=(10, 4))
normal_points = df[df["Anomaly_Flag"] == 1]
anomaly_points = df[df["Anomaly_Flag"] == -1]

ax.scatter(normal_points.index, normal_points["Amount"], color="#1f77b4", label="Normal", alpha=0.7)
ax.scatter(anomaly_points.index, anomaly_points["Amount"], color="#d62728", label="Flagged Anomaly", s=70, edgecolors="black")

ax.set_xlabel("Transaction Index")
ax.set_ylabel("Amount ($)")
ax.set_title("Local Financial Anomaly Sentinel Scan")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)

st.pyplot(fig)

# 7. Flagged Transactions Table
st.subheader("Flagged High-Risk Transactions")
flagged_df = df[df["Anomaly_Flag"] == -1][["Transaction_ID", "Category", "Amount", "Status"]]

if not flagged_df.empty:
    st.dataframe(flagged_df.style.format({"Amount": "${:,.2f}"}), use_container_width=True)
else:
    st.success("No anomalous transactions detected under current thresholds.")

# 8. Complete Ledger View
with st.expander("View Full Audited Ledger"):
    st.dataframe(df, use_container_width=True)