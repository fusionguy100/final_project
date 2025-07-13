import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64

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

# Title and description
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("This interactive dashboard explains and visualizes a logistic regression model used to predict customer churn in telecom.")

# Dropdown filter
st.sidebar.header("🔎 Filter Data")
contract_filter = st.sidebar.multiselect("Contract Type", options=df["Contract"].unique(), default=df["Contract"].unique())
payment_filter = st.sidebar.multiselect("Payment Method", options=df["PaymentMethod"].unique(), default=df["PaymentMethod"].unique())

filtered_df = df[
    (df["Contract"].isin(contract_filter)) &
    (df["PaymentMethod"].isin(payment_filter))
]

st.markdown("---")
st.subheader("🔢 Feature Importance")
st.image(feature_img, caption="Which features most influence churn.")

st.subheader("📊 Confusion Matrix")
st.image(conf_img)

st.subheader("📈 ROC Curve")
st.image(roc_img)

st.subheader("📄 Classification Report")
with open("classification_report.txt", "r") as f:
    st.text(f.read())

st.subheader("🧮 Filtered Data Preview")
st.dataframe(filtered_df)

# Download buttons
def generate_download_link(df, filename, label):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'

st.markdown("### 📥 Download Options")
st.markdown(generate_download_link(filtered_df, "filtered_data.csv", "📄 Download Filtered Data as CSV"), unsafe_allow_html=True)
st.markdown(generate_download_link(df, "cleaned_telecom_data.csv", "📂 Download Full Cleaned Dataset"), unsafe_allow_html=True)

st.markdown("---")
st.markdown("Made with ❤️ for WGU Capstone")

