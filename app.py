import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import streamlit.components.v1 as components

BASE = Path(__file__).parent

# -----------------------------
# PAGE CONFIG & SIMPLE STYLES
# -----------------------------
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

st.markdown("""
<style>
body { background-color: #f5f7fa; }
.title-text {
    font-size: 28px;
    font-weight: 700;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 0.3rem;
}
.subtitle-text {
    font-size: 14px;
    color: #7f8c8d;
    text-align: center;
    margin-bottom: 1.5rem;
}
.result-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #ffffff;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.12);
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title-text'>📊 Customer Churn Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Enter customer details to predict whether they are likely to churn.</div>", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_data
def load_model():
    data = pickle.load(open(BASE / "model" / "Customer_Churn_Prediction_Model.pkl", "rb"))
    model = data["model"]
    feature_names = data["feature_names"]
    return model, feature_names

model, feature_names = load_model()

# -----------------------------
# INPUT FORM
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=1000, value=12)

with col2:
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col3:
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

MonthlyCharges = st.slider("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
TotalCharges = st.slider("Total Charges", min_value=0.0, max_value=10000.0, value=500.0, step=10.0)

# -----------------------------
# ENCODING – MATCHES LabelEncoder ON TELCO DATA
# -----------------------------

# Binary yes/no
yes_no = {"No": 0, "Yes": 1}

# gender: LabelEncoder on ['Female','Male'] → Female:0, Male:1
gender_map = {"Female": 0, "Male": 1}

# MultipleLines: ['No', 'No phone service', 'Yes'] sorted → 'No':0, 'No phone service':1, 'Yes':2
multiple_lines_map = {"No": 0, "No phone service": 1, "Yes": 2}

# InternetService: ['DSL','Fiber optic','No'] → DSL:0, Fiber optic:1, No:2
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}

# Ternary internet-related: ['No','No internet service','Yes'] → 'No':0, 'No internet service':1, 'Yes':2
tri_internet_map = {"No": 0, "No internet service": 1, "Yes": 2}

# Contract: ['Month-to-month','One year','Two year'] (already sorted)
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}

# PaymentMethod sorted:
# ['Bank transfer (automatic)','Credit card (automatic)','Electronic check','Mailed check']
payment_map = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3
}

def build_input_df():
    encoded = {
        "gender": gender_map[gender],
        "SeniorCitizen": yes_no[SeniorCitizen],           # originally 0/1
        "Partner": yes_no[Partner],
        "Dependents": yes_no[Dependents],
        "tenure": tenure,
        "PhoneService": yes_no[PhoneService],
        "MultipleLines": multiple_lines_map[MultipleLines],
        "InternetService": internet_map[InternetService],
        "OnlineSecurity": tri_internet_map[OnlineSecurity],
        "OnlineBackup": tri_internet_map[OnlineBackup],
        "DeviceProtection": tri_internet_map[DeviceProtection],
        "TechSupport": tri_internet_map[TechSupport],
        "StreamingTV": tri_internet_map[StreamingTV],
        "StreamingMovies": tri_internet_map[StreamingMovies],
        "Contract": contract_map[Contract],
        "PaperlessBilling": yes_no[PaperlessBilling],
        "PaymentMethod": payment_map[PaymentMethod],
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }
    # Ensure columns in same order as training
    return pd.DataFrame([encoded], columns=feature_names)

# -----------------------------
# PREDICT
# -----------------------------
st.markdown("### 🚀 Predict")

if st.button("Predict Churn"):
    try:
        X = build_input_df()
        proba = float(model.predict_proba(X)[0][1])
        pred = int(model.predict(X)[0])
    except Exception as e:
        st.error("Something went wrong during prediction. Check model and feature mapping.")
        st.exception(e)
    else:
        churn_text = "Customer Will CHURN ❌" if pred == 1 else "Customer Will NOT Churn ✅"
        color = "#e74c3c" if pred == 1 else "#2ecc71"

        result_html = f"""
        <div class="result-card">
            <h3 style="color:{color}; margin-bottom: 10px;">{churn_text}</h3>
            <h4 style="margin-top: 0;">Probability: <span style="color:{color}; font-weight:700;">{proba:.2f}</span></h4>
        </div>
        """

        st.markdown(result_html, unsafe_allow_html=True)

        # simple circular gauge
        svg = f"""
        <div style='display:flex;justify-content:center;margin-top:10px;'>
          <svg width="160" height="160">
              <circle cx="80" cy="80" r="65" stroke="#dfe6e9" stroke-width="12" fill="none"/>
              <circle cx="80" cy="80" r="65"
                  stroke="{color}"
                  stroke-width="12"
                  fill="none"
                  stroke-dasharray="{int(proba*408)} 408"
                  transform="rotate(-90 80 80)"
              />
          </svg>
        </div>
        """
        components.html(svg, height=200)
