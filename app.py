# app.py - Streamlit Fraud Detection Dashboard

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# === Load model and scaler ===
model = joblib.load("fraud_model.pkl")

# Title
st.title("💳 Fraud Detection System")
st.markdown("Upload a CSV of transactions and detect fraudulent activity in real-time.")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Raw Input Data")
    st.write(data.head())

    # === Preprocess ===
    if "Time" in data.columns:
        data = data.drop("Time", axis=1)

    if "Amount" in data.columns:
        scaler = StandardScaler()
        data["Amount"] = scaler.fit_transform(data["Amount"].values.reshape(-1, 1))

    # === Predict ===
    probs = model.predict_proba(data)[:, 1]
    predictions = (probs > 0.5).astype(int)

    data["Fraud_Risk_Score"] = probs
    data["Predicted_Fraud"] = predictions

    st.subheader("🔍 Prediction Results")
    st.dataframe(data[["Fraud_Risk_Score", "Predicted_Fraud"]])

    st.subheader("🚨 Top 5 Riskiest Transactions")
    st.dataframe(data.sort_values("Fraud_Risk_Score", ascending=False).head(5))

    # Optionally: download results
    csv_download = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Prediction Results as CSV",
        data=csv_download,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload a CSV file to get started.")
