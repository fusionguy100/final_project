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

# How to use
st.markdown("## 🧭 How to Use This Dashboard")
st.markdown("""
1. Explore churn prediction results using graphs and metrics.
2. Learn which features contribute most to churn risk.
3. Use the classification report to evaluate model performance.
4. Ideal for business users and evaluators who want quick insights.
""")

# Project Summary
st.markdown("## 🧠 Project Summary")
st.markdown("""
This dashboard uses a logistic regression model to predict customer churn in a telecom company.
It highlights which customers are most likely to leave and why, helping the business take early action.

- **Model Type**: Logistic Regression  
- **Goal**: Predict customer churn before it happens  
- **AUC Score**: **0.83** (good predictive accuracy)
""")
st.markdown("---")

# Feature Importance
st.markdown("## 🔍 Feature Importance")
st.markdown("""
This chart shows which customer features are most important for predicting churn.  
- **Negative values** = reduce churn risk  
- **Positive values** = increase churn risk  
- Example: Customers with longer tenure tend to stay; month-to-month contracts increase churn
""")
st.image(feature_img, caption="Feature Importance from Logistic Regression")

# Confusion Matrix
st.markdown("## 📊 Confusion Matrix")
st.markdown("""
The confusion matrix shows how well the model predicted churn vs. reality:  
- **922** were correctly predicted as not churning  
- **185** were correctly predicted as churning  
- **111** were incorrectly predicted as churning  
- **189** were missed churns (predicted to stay, but left)

Use this to see how accurate the model’s decisions were.
""")
st.image(conf_img, caption="Confusion Matrix")

# ROC Curve
st.markdown("## 📈 ROC Curve (AUC: 0.83)")
st.markdown("""
The ROC curve tells how good the model is at separating churners from non-churners.  
- The **higher the curve**, the better.  
- The **AUC score of 0.83** means the model has solid predictive power.
""")
st.image(roc_img, caption="ROC Curve with AUC Score")

# Classification Report
st.markdown("## 📄 Classification Report")
st.markdown("""
This report shows precision, recall, and F1-score for both churn and non-churn classes.

- **Precision**: % of churn predictions that were correct  
- **Recall**: % of actual churns correctly caught  
- **F1-score**: Combines precision and recall  

```plaintext
              precision    recall  f1-score   support

     0           0.83      0.89      0.86      1033
     1           0.62      0.49      0.55       374

accuracy                         0.79      1407
macro avg        0.73      0.69      0.71      1407
weighted avg     0.78      0.79      0.78      1407
