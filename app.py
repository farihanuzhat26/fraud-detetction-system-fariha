# app.py - Streamlit Fraud Detection Dashboard

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# === Load model and scaler ===
model = joblib.load("fraud_model.pkl")

# Expected features from training (excluding 'Time' and 'Class')
expected_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Title
st.title("\U0001F4B3 Fraud Detection System")
st.markdown("Upload a CSV of transactions and detect fraudulent activity in real-time.")

# Show CSV template download
with open("creditcard.csv", "rb") as f:
    st.download_button(
        label="📄 Download CSV Template",
        data=f,
        file_name="creditcard_template.csv",
        mime="text/csv",
    )

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Raw Input Data")
    st.write(data.head())

    # === Preprocess ===
    drop_cols = ['Time', 'Class']
    for col in drop_cols:
        if col in data.columns:
            data = data.drop(col, axis=1)

    if 'Amount' in data.columns:
        scaler = StandardScaler()
        data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

    # Align to expected column order
    try:
        data = data[expected_columns]
    except KeyError:
        st.error("❌ The uploaded file does not contain the required columns in correct format.")
        st.stop()

    # === Predict ===
    probs = model.predict_proba(data)[:, 1]
    predictions = (probs > 0.5).astype(int)

    data["Fraud_Risk_Score"] = probs
    data["Predicted_Fraud"] = predictions

    st.subheader("\U0001F50D Prediction Results")
    st.dataframe(data[["Fraud_Risk_Score", "Predicted_Fraud"]])

    st.subheader("\U0001F6A8 Top 5 Riskiest Transactions")
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
