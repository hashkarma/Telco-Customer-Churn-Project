# streamlit_app.py

import streamlit as st
import joblib
import numpy as np

# Load trained model and input feature list
model = joblib.load("rf_churn_model.pkl")
features = joblib.load("model_features.pkl")

st.title("📊 Telco Customer Churn Prediction")

# Create input fields dynamically
user_input = []
for feature in features:
    val = st.number_input(f"{feature}", min_value=0.0)
    user_input.append(val)

# Predict on button click
if st.button("Predict"):
    prediction = model.predict([user_input])[0]
    probability = model.predict_proba([user_input])[0][1]
    st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
    st.info(f"Churn Probability: {probability:.2f}")