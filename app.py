import streamlit as st
import joblib
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as file:
    model = joblib.load(file)

# App title and description
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")
st.title("🔬 Breast Cancer Predictor")
st.markdown("Use this app to predict whether a tumor is **Malignant** or **Benign** based on 30 diagnostic features.")

# Sidebar for input options
st.sidebar.header("🧪 Input Feature Options")
input_mode = st.sidebar.radio("Select Input Mode:", ("Manual Input", "Use Example"))

# Feature names from the breast cancer dataset
feature_names = [
    "Radius (mean)", "Texture (mean)", "Perimeter (mean)", "Area (mean)", "Smoothness (mean)",
    "Compactness (mean)", "Concavity (mean)", "Concave points (mean)", "Symmetry (mean)", "Fractal dimension (mean)",
    "Radius (SE)", "Texture (SE)", "Perimeter (SE)", "Area (SE)", "Smoothness (SE)",
    "Compactness (SE)", "Concavity (SE)", "Concave points (SE)", "Symmetry (SE)", "Fractal dimension (SE)",
    "Radius (worst)", "Texture (worst)", "Perimeter (worst)", "Area (worst)", "Smoothness (worst)",
    "Compactness (worst)", "Concavity (worst)", "Concave points (worst)", "Symmetry (worst)", "Fractal dimension (worst)"
]

# Example values for breast cancer dataset (benign sample)
example_values = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.0064, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]

# Input collection
features = []
st.subheader("📥 Input Diagnostic Features")
st.markdown("Enter values for each feature:")

for i in range(30):
    label = f"{i + 1}. {feature_names[i]}"
    if input_mode == "Manual Input":
        value = st.number_input(label, min_value=0.0, value=0.0, step=0.01)
    else:
        value = st.number_input(label, min_value=0.0, value=example_values[i], step=0.01)
    features.append(value)

# Predict and display result
if st.button("🧠 Predict"):
    prediction = model.predict([features])[0]
    prediction_proba = model.predict_proba([features])[0]
    st.markdown("### 🧾 Prediction Result")
    st.success(f"**Prediction:** {'Benign' if prediction == 1 else 'Malignant'}")
    st.info(f"🔍 Confidence: Benign {prediction_proba[1]:.2%} | Malignant {prediction_proba[0]:.2%}")

    if prediction == 0:
        st.warning("⚠️ This result indicates a possible malignant tumor. Please consult a medical professional.")
    else:
        st.balloons()
        st.success("🎉 This result suggests a benign tumor. Continue regular checkups!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Dataset: Breast Cancer Wisconsin (Diagnostic)")
st.markdown("**Designed by Ajay**")