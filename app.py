import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Telecom Churn Risk Analyzer",
    page_icon="📉",
    layout="wide"
)

@st.cache_resource
def load_assets():
    pipeline = joblib.load("model/churn_pipeline.pkl")
    X_ref = joblib.load("model/model_features.pkl")
    return pipeline, X_ref

pipeline, X_ref = load_assets()

st.title("📉 Telecom Customer Churn Risk Analyzer")
st.markdown("**XGBoost Boosting & Dynamic SHAP Interpretability**")

st.sidebar.header("Customer Profile Settings")

def user_input_features():
    tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    total_charges = tenure * monthly_charges
    
    payment_method = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

    data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
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
        'PaperlessBilling': paperless,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Subscriber Inputs")
    st.dataframe(input_df.T, height=430)

with col2:
    st.subheader("Risk Analytics")
    if st.button("Calculate Churn Risk", type="primary"):
        proba = pipeline.predict_proba(input_df)[0][1]
        
        st.metric(label="Calculated Churn Probability", value=f"{proba * 100:.1f}%")
        
        if proba > 0.65:
            st.error("⚠️ **High Risk**: Customer profile matches high-churn historical clusters.")
        elif proba > 0.35:
            st.warning("⚠️ **Moderate Risk**: Customer shows elevated risk behavior.")
        else:
            st.success("✅ **Low Risk**: High retention stability indicated.")

        st.markdown("### 🔍 Key Risk Factors (SHAP Explanation)")
        
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['classifier']
        
        X_ref_transformed = preprocessor.transform(X_ref)
        input_transformed = preprocessor.transform(input_df)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_ref_transformed)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(explainer(input_transformed)[0], max_display=7, show=False)
        st.pyplot(fig)