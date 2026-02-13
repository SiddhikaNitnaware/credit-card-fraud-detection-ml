import streamlit as st
import requests

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter 29 transaction features separated by commas.")

input_text = st.text_area("Transaction Features")

if st.button("Predict Fraud"):

    try:
        features = [float(x.strip()) for x in input_text.split(",")]

        if len(features) != 29:
            st.error("Please enter exactly 29 values.")
        else:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"features": features}
            )

            result = response.json()

            if result["fraud_prediction"] == 1:
                st.error(f"⚠ Fraud Detected\nProbability: {result['fraud_probability']:.4f}")
            else:
                st.success(f"✅ Legitimate Transaction\nProbability: {result['fraud_probability']:.4f}")

    except:
        st.error("Invalid input format.")
