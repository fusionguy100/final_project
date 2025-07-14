import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

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

# Title
st.markdown("# 📊 Customer Churn Prediction Dashboard")
st.markdown("A simple, interactive dashboard to understand and visualize customer churn predictions.")
st.markdown("---")

# How to Use
st.markdown("## 🧭 How to Use This Dashboard")
st.markdown("""
This dashboard helps you explore predictions made by a machine learning model trained to detect telecom customer churn.

You can:
- Understand which customer traits increase churn risk
- Review model accuracy through visual explanations
- Preview and filter real customer data to identify trends

Each section below includes short explanations to help you interpret what you’re seeing.
""")

# Project Summary
st.markdown("### 🔍 Project Summary")
st.markdown("""
This dashboard uses a logistic regression model to predict churn.  
It shows the top reasons customers are likely to leave and offers clear visuals for decision-making.

- **Model Type**: Logistic Regression  
- **Goal**: Predict which customers are at risk of churning  
- **AUC Score**: **0.83** (Strong performance)
""")

st.markdown("---")
st.markdown("## 🔢 Key Metrics")

# Feature Importance
st.markdown("### 🔧 Feature Importance")
st.markdown("""
This chart shows which customer traits most influence churn risk.

- **Negative values**: reduce risk of churn  
- **Positive values**: increase risk  
- Example: Month-to-month contracts increase risk. Long tenure reduces it.
""")
st.image(feature_img, caption="Feature Importance")

# Confusion Matrix
st.markdown("### 📊 Confusion Matrix")
st.markdown("""
This matrix shows how well the model predicted churn.

- **True Positives (TP)**: Correct churn predictions  
- **True Negatives (TN)**: Correct no-churn predictions  
- **False Positives (FP)**: Predicted churn but customer stayed  
- **False Negatives (FN)**: Predicted no churn but customer left

✅ **Goal**: High numbers on the diagonal = good performance
""")
st.image(conf_img, caption="Confusion Matrix")

# ROC Curve
st.markdown("### 📈 ROC Curve (Receiver Operating Characteristic)")
st.markdown("""
This curve shows the model’s ability to separate churners from non-churners.

- **Higher curve = better**  
- AUC = 0.83 means there's an 83% chance the model ranks a random churner higher than a random non-churner  
- The dashed line (AUC = 0.5) means random guessing

✅ **Goal**: Stay far above the dashed line
""")
st.image(roc_img, caption="ROC Curve (AUC: 0.83)")

# Classification Report
st.markdown("### 📄 Classification Report")
st.markdown("""
This report summarizes precision, recall, and F1-score.

- **Precision**: Of the predicted churns, how many actually churned  
- **Recall**: Of all actual churns, how many were correctly predicted  
- **F1-score**: Balance between precision and recall

✅ Use this report to check if the model is making more false alarms or missing actual churns
""")
with open("classification_report.txt", "r") as f:
    st.text(f.read())

# Project Success Benchmarks
st.markdown("## ✅ Project Success Benchmarks")
st.markdown("""
| Metric                             | Goal     | Achieved        | Status |
|------------------------------------|----------|------------------|--------|
| AUC Score                          | ≥ 0.80   | **0.83**         | ✅ Met |
| Accuracy                           | ≥ 75%    | **0.79 (79%)**   | ✅ Met |
| Feature Interpretability           | Clear    | ✔️               | ✅ Met |
| Usable Dashboard                   | Yes      | ✔️               | ✅ Met |
| Churn Risk + Trends Visualized     | Yes      | ✔️               | ✅ Met |
""")

# Filtered Data Section
st.markdown("---")
st.markdown("## 🔍 Explore and Filter Customer Data")
st.markdown("""
You can explore real cleaned customer data and filter it by traits like contract type, internet service, or gender.  
This helps you analyze trends in churn risk and see how certain features might relate to outcomes.
""")

# Filters
contract_type = st.selectbox("Select Contract Type", options=["All"] + sorted(df["Contract"].unique()))
internet_service = st.selectbox("Select Internet Service", options=["All"] + sorted(df["InternetService"].unique()))
gender = st.selectbox("Select Gender", options=["All"] + sorted(df["gender"].unique()))

# Apply filters
filtered_df = df.copy()
if contract_type != "All":
    filtered_df = filtered_df[filtered_df["Contract"] == contract_type]
if internet_service != "All":
    filtered_df = filtered_df[filtered_df["InternetService"] == internet_service]
if gender != "All":
    filtered_df = filtered_df[filtered_df["gender"] == gender]

st.markdown("### 📋 Filtered Data Preview")
st.dataframe(filtered_df.head(10))

# Footer
st.markdown("---")
st.markdown("This dashboard was created as part of a capstone project to demonstrate explainable machine learning for customer churn.")
