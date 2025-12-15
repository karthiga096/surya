import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.write("Predict diabetes using Logistic Regression")

# ----------------------------
# Default Dataset (Auto Loaded)
# ----------------------------
data = {
    "age": [50, 31, 32, 21, 33, 30, 26, 29],
    "mass": [33.6, 26.6, 23.3, 28.1, 43.1,]()
