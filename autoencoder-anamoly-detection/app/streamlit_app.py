import streamlit as st
import pandas as pd
import sys
import os

# Fix import path for src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import predict_anomaly

st.set_page_config(page_title="Anomaly Detection", layout="wide")

st.title("🔍 Anomaly Detection using Autoencoder")

st.write("Upload a CSV file to detect anomalies based on reconstruction error.")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### 📊 Uploaded Data")
    st.dataframe(df.head())

    if st.button("🚀 Detect Anomalies"):
        # Drop target column if present (VERY IMPORTANT FIX)
        if "Class" in df.columns:
            df = df.drop("Class", axis=1)

        # Run prediction
        result = predict_anomaly(df.values)

        # Add result column
        df["Anomaly"] = result

        st.write("### ✅ Results")
        st.dataframe(df.head())

        # Show anomaly summary
        anomaly_count = df["Anomaly"].sum()
        st.write(f"⚠️ Total anomalies detected: {anomaly_count}")

        # Show only anomalies
        st.write("### 🚨 Anomalies")
        st.dataframe(df[df["Anomaly"] == True])