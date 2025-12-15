import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.write("Predict diabetes using Logistic Regression")

# ----------------------------
# Default Dataset (FIXED)
# ----------------------------
data = {
    "age":   [50, 31, 32, 21, 33, 30, 26, 29],
    "mass":  [33.6, 26.6, 23.3, 28.1, 43.1, 30.1, 25.6, 28.7],
    "insu":  [0, 0, 0, 0, 0, 0, 0, 0],
    "plas":  [148, 85, 183, 89, 137, 116, 78, 115],
    "class": [
        "tested_positive",
        "tested_negative",
        "tested_positive",
        "tested_negative",
        "tested_positive",
        "tested_negative",
        "tested_negative",
        "tested_negative"
    ]
}

df = pd.DataFrame(data)

# ----------------------------
# Features & Target
# ----------------------------
X = df[['age', 'mass', 'insu', 'plas']]
y = df['class']

# ----------------------------
# Train Model
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ----------------------------
# User Input
# ----------------------------
st.subheader("Enter Patient Details")

age = st.number_input("Age", min_value=1, max_value=120)
mass = st.number_input("Body Mass Index (BMI)", min_value=0.0)
insu = st.number_input("Insulin Level", min_value=0.0)
plas = st.number_input("Plasma Glucose Level", min_value=0.0)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):
    user_data = pd.DataFrame(
        [[age, mass, insu, plas]],
        columns=['age', 'mass', 'insu', 'plas']
    )

    prediction = model.predict(user_data)[0]

    if prediction == "tested_positive":
        st.error("⚠️ Diabetes Positive")
    else:
        st.success("✅ Diabetes Negative")

# ----------------------------
# Accuracy
# ----------------------------
acc = model.score(X, y)
st.write(f"📊 Model Accuracy: **{acc:.2f}**")
