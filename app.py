import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# App title
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🩺 Diabetes Prediction using Logistic Regression")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_excel("diabetes.xlsx")

df = load_data()

# Features and target
X = df[['age', 'mass', 'insu', 'plas']]
y = df['class']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.subheader("Enter Patient Details")

# User inputs
age = st.number_input("Age", min_value=1, max_value=120, value=25)
mass = st.number_input("Body Mass Index (BMI)", min_value=0.0, value=25.0)
insu = st.number_input("Insulin Level", min_value=0.0, value=80.0)
plas = st.number_input("Plasma Glucose Level", min_value=0.0, value=120.0)

# Predict button
if st.button("Predict"):
    user_data = pd.DataFrame(
        [[age, mass, insu, plas]],
        columns=['age', 'mass', 'insu', 'plas']
    )

    prediction = model.predict(user_data)[0]

    if prediction == "tested_positive":
        st.error("⚠️ Result: Diabetes Positive")
    else:
        st.success("✅ Result: Diabetes Negative")

# Model accuracy
st.markdown("---")
accuracy = model.score(X, y)
st.write(f"📊 **Model Accuracy:** {accuracy:.2f}")

