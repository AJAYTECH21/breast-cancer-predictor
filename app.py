import streamlit as st
import joblib
import numpy as np

# Load model
with open("model.pkl", "rb") as file:
    model = joblib.load(file)

st.title("Breast Cancer Predictor")

# 30 features
features = []
for i in range(30):
    value = st.number_input(f"Feature {i + 1}", value=0.0)
    features.append(value)

if st.button("Predict"):
    prediction = model.predict([features])[0]
    st.write("Prediction:", "Malignant" if prediction == 0 else "Benign")