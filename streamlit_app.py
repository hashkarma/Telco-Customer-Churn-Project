# Recreate project directory and rewrite updated streamlit app
import os

project_dir = "/mnt/data/TelcoChurnStreamlitProject"
os.makedirs(project_dir, exist_ok=True)

streamlit_path = f"{project_dir}/streamlit_app.py"

streamlit_app_cleaned = '''
import streamlit as st
import joblib
import pandas as pd

# Load model and features
model = joblib.load("rf_churn_model.pkl")
features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("📊 Telco Customer Churn Prediction")
st.markdown("This app predicts if a customer is likely to churn based on their demographic and service details.")

# 1. Raw User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# 2. Create DataFrame
raw_input = {
    'gender': gender,
    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}
input_df = pd.DataFrame([raw_input])

# 3. One-hot encode + align with training features
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=features, fill_value=0)

# 4. Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]
    st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
    st.info(f"Churn Probability: {probability:.2f}")
'''

# Write the improved streamlit app
with open(streamlit_path, "w") as f:
    f.write(streamlit_app_cleaned)

streamlit_path