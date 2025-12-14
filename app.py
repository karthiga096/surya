import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Prediction App")

st.write("Upload the diabetes dataset (Excel file)")

# File uploader
uploaded_file = st.file_uploader("Upload diabetes Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Read Excel
    df = pd.read_excel(uploaded_file)

    st.success("File uploaded successfully!")

    # Features & target
    X = df[['age', 'mass', 'insu', 'plas']]
    y = df['class']

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    st.subheader("Enter Patient Details")

    age = st.number_input("Age", min_value=1, max_value=120)
    mass = st.number_input("Body Mass Index (BMI)", min_value=0.0)
    insu = st.number_input("Insulin Level", min_value=0.0)
    plas = st.number_input("Plasma Glucose Level", min_value=0.0)

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

    # Accuracy
    acc = model.score(X, y)
    st.write(f"📊 Model Accuracy: **{acc:.2f}**")

else:
    st.warning("Please upload the Excel file to continue.")
