import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import joblib

# Page setup
st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")

# Load data and models
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_telecom_data.csv")

df = load_data()
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Load visuals
feature_img = Image.open("feature_importance.png")
conf_img = Image.open("confusion_matrix.png")
roc_img = Image.open("roc_curve.png")

# Title and intro
st.title("\U0001F4CA Customer Churn Prediction Dashboard")
st.markdown("""
Welcome! This dashboard helps visualize and explore customer churn predictions from a trained machine learning model.

**You can:**
- See what factors influence churn
- Review model performance visually
- Test new customer inputs
- Explore and filter real data
""")

st.markdown("---")

# Project Summary
with st.expander(" Project Summary"):
    st.markdown("""
    **Model Used**: Logistic Regression  
    **Data Source**: Kaggle Telco Customer Churn Dataset  
    **AUC Score**: 0.83

    The goal is to identify which customers are most likely to leave so the business can intervene earlier.
    """)

# Feature Importance
st.header("Feature Importance")
st.image(feature_img, caption="Feature Importance (Positive = Increases Churn Risk)")
st.caption("For example, customers with month-to-month contracts are more likely to churn.")

# Confusion Matrix
st.header("Confusion Matrix")
st.image(conf_img, caption="Confusion Matrix")
st.caption("High numbers on the diagonal mean more accurate predictions.")

# ROC Curve
st.header("ROC Curve")
st.image(roc_img, caption="ROC Curve (AUC = 0.83)")
st.caption("The curve shows how well the model separates churn vs. no churn across thresholds.")

# Classification Report
st.header("Classification Report")
with open("classification_report.txt", "r") as f:
    st.text(f.read())
st.caption("Precision = fewer false alarms, Recall = fewer missed churns")

st.markdown("---")

# Filter and explore data
st.subheader("Explore Customer Data")
st.markdown("Use filters below to preview data by customer attributes.")
st.caption("Note: This is just a preview of data, not all data will be shown!")


contract_type = st.selectbox("Contract Type", ["All"] + sorted(df["Contract"].unique()))
internet = st.selectbox("Internet Service", ["All"] + sorted(df["InternetService"].unique()))
gender = st.selectbox("Gender", ["All"] + sorted(df["gender"].unique()))

filtered = df.copy()
if contract_type != "All":
    filtered = filtered[filtered["Contract"] == contract_type]
if internet != "All":
    filtered = filtered[filtered["InternetService"] == internet]
if gender != "All":
    filtered = filtered[filtered["gender"] == gender]

st.dataframe(filtered.head(10))

# Manual input section
st.markdown("---")
st.subheader("Predict Churn for a New Customer")
st.markdown("Fill in the fields below to simulate a churn prediction.")

# Helper input fields (add only a few for simplicity, use label_encoders to match encoded format)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=150.0, value=70.0, help="Monthly amount charged to the customer")
tenure = st.slider("Tenure (months)", 0, 72, 12, help="Number of months the customer has stayed")
contract_input = st.selectbox("Contract Type", options=list(label_encoders['Contract'].classes_))

# Encode manually
user_data = pd.DataFrame({
    'MonthlyCharges': [monthly_charges],
    'tenure': [tenure],
    'Contract': [label_encoders['Contract'].transform([contract_input])[0]]
})

# Add missing features with default/filler values (for demonstration purposes)
for col in df.columns:
    if col not in user_data.columns and col != 'Churn':
        user_data[col] = df[col].mode()[0]

# Reorder and scale
user_data = user_data[df.drop('Churn', axis=1).columns]
user_scaled = scaler.transform(user_data)

if st.button("Predict Churn"):
    prediction = model.predict(user_scaled)[0]
    prob = model.predict_proba(user_scaled)[0][1]
    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer likely to stay. (Probability of churn: {prob:.2f})")

st.markdown("---")
st.caption("Created for the WGU Capstone Project: Predicting Customer Churn")
st.caption("Created by Jacob Newell")
