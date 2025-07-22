
import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# App Title
st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")
st.title("🔬 Breast Cancer Prediction App")
st.markdown("This ML app predicts whether a tumor is **benign or malignant** based on medical data.")

st.markdown("---")

# Sidebar Info
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This app uses a machine learning model (Random Forest Classifier) to detect the presence of breast cancer using 30 diagnostic features.  
Made by [Your Name]  
GitHub: [github.com/yourprofile](https://github.com/yourprofile)
""")

# Input fields in two columns
features = []
labels = [
    "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean",
    "Compactness Mean", "Concavity Mean", "Concave Points Mean", "Symmetry Mean", "Fractal Dimension Mean",
    "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE",
    "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
    "Radius Worst", "Texture Worst", "Perimeter Worst", "Area Worst", "Smoothness Worst",
    "Compactness Worst", "Concavity Worst", "Concave Points Worst", "Symmetry Worst", "Fractal Dimension Worst"
]

st.markdown("### Enter Diagnostic Features:")

col1, col2 = st.columns(2)
for i, label in enumerate(labels):
    if i % 2 == 0:
        with col1:
            value = st.number_input(label, min_value=0.0, step=0.01, format="%.2f")
    else:
        with col2:
            value = st.number_input(label, min_value=0.0, step=0.01, format="%.2f")
    features.append(value)

# Predict button
if st.button("🔍 Predict"):
    input_data = np.array([features])
    prediction = model.predict(input_data)

    st.markdown("### 🧾 Result:")
    if prediction[0] == 0:
        st.success("✅ The tumor is **Benign** (non-cancerous)")
    else:
        st.error("❌ The tumor is **Malignant** (cancerous)")

# Footer
st.markdown("---")
st.markdown("🧠 *This app is for educational purposes only.*")
st.markdown("Made by [Your Name] | GitHub: [Your GitHub Link]")
