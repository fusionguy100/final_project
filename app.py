import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import joblib

# Configure page
st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")


# Load data
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_telecom_data.csv")


df = load_data()

# Load images
feature_img = Image.open("feature_importance.png")
conf_img = Image.open("confusion_matrix.png")
roc_img = Image.open("roc_curve.png")

# Load model and encoders
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Title and project summary
st.markdown("# 📊 Customer Churn Prediction Dashboard")
st.markdown("A simple, interactive dashboard to understand and visualize customer churn predictions.")
st.markdown("---")

# How to Use
st.markdown("## 🧭 How to Use This Dashboard")
st.markdown("""
This dashboard helps you explore predictions made by a machine learning model trained to detect telecom customer churn.

You can:
- Understand which customer traits increase churn risk (Feature Importance)
- See how accurate the model is (Confusion Matrix and ROC Curve)
- Read model performance metrics (Classification Report)
- Test your own customer scenarios with the prediction form below
""")

# Project summary
st.markdown("### 📌 Project Summary")
st.markdown("""
This dashboard uses a logistic regression model to predict customer churn in a telecom company.
It highlights which customers are most likely to leave and why, helping the business take early action.

- **Model Type**: Logistic Regression  
- **Goal**: Predict customer churn before it happens  
- **AUC Score**: **0.83** (good prediction accuracy)
""")

st.markdown("---")
st.markdown("## 🔢 Key Metrics")

# Feature Importance
st.markdown("### 🔍 Feature Importance")
st.markdown("""
This chart shows which customer attributes matter most in predicting churn.
- **Negative values** = reduce churn risk  
- **Positive values** = increase churn risk  
- Example: Longer tenure reduces churn risk. Month-to-month contracts increase it.
""")
st.image(feature_img, caption="Feature Importance from Logistic Regression")

# Confusion Matrix
st.markdown("### 📊 Confusion Matrix")
st.markdown("""
This table shows how well the model predicted churn:
- **True Positive (185)**: Model correctly predicted the customer would churn  
- **True Negative (922)**: Model correctly predicted the customer would stay  
- **False Positive (111)**: Model incorrectly predicted churn  
- **False Negative (189)**: Model missed predicting actual churn

> ✅ **Higher values in the diagonals = better model performance**
""")
st.image(conf_img, caption="Confusion Matrix")

# ROC Curve
st.markdown("### 📈 ROC Curve (Receiver Operating Characteristic)")
st.markdown("""
This curve shows how well the model separates churners from non-churners across all probability thresholds.

- **AUC = 0.83** means the model has an 83% chance of correctly ranking a random churner higher than a random non-churner.
- A perfect model would have an AUC of 1.0  
- The dashed line shows a random guess (AUC = 0.5)

> ✅ **Higher curve and AUC score = better performance**
""")
st.image(roc_img, caption="ROC Curve (AUC: 0.83)")

# Classification Report
st.markdown("### 📄 Classification Report")
st.markdown("""
This report shows precision, recall, and F1-score for both churn and non-churn predictions.

- **Precision**: How many predicted churns were correct  
- **Recall**: How many actual churns were correctly predicted  
- **F1-score**: Harmonic average of precision and recall

> ✅ Use this to judge the balance between false alarms and missed churns
""")
with open("classification_report.txt", "r") as f:
    st.text(f.read())

# Success Benchmarks
st.markdown("## ✅ Project Success Benchmarks")
st.markdown("""
| Metric                             | Goal     | Achieved | Status |
|------------------------------------|----------|----------|--------|
| AUC Score                          | ≥ 0.80   | **0.83** | ✅ Met |
| Accuracy                           | ≥ 75%    | **0.79 (79%)**       | ✅ Met |
| Feature Interpretability           | Clear    | ✔️       | ✅ Met |
| Usable Dashboard                   | Yes      | ✔️       | ✅ Met |
| Churn Risk + Trends Visualized     | Yes      | ✔️       | ✅ Met |
""")

# Input form for prediction
st.markdown("## 🧪 Try a Churn Prediction")
st.markdown("Use the form below to input customer data and see the predicted churn probability.")

with st.form("predict_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1400.0)

    submitted = st.form_submit_button("Predict Churn")

    if submitted:
        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        for col in input_data.select_dtypes(include='object').columns:
            le = label_encoders[col]
            input_data[col] = le.transform(input_data[col])

        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0][1]
        prediction = "Yes" if prob >= 0.5 else "No"

        st.markdown(f"### 🔍 Prediction: **{prediction}**")
        st.markdown(f"**Churn Probability:** {prob:.2f}")

# Footer
st.markdown("---")
st.markdown(
    "This dashboard was created as part of a capstone project to demonstrate explainable machine learning for customer churn.")
