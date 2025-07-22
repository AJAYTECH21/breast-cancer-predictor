import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Breast Cancer Predictor")

mean_radius = st.number_input("Mean Radius")
mean_texture = st.number_input("Mean Texture")
mean_perimeter = st.number_input("Mean Perimeter")
mean_area = st.number_input("Mean Area")

if st.button("Predict"):
    features = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area]])
    result = model.predict(features)
    if result[0] == 1:
        st.error("The prediction: Malignant (Cancerous)")
    else:
        st.success("The prediction: Benign (Non-Cancerous)")