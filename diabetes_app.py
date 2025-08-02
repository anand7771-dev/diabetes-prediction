import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# App title
st.title("Diabetes Prediction App")
st.write("Fill in the patient details below:")

# Input form
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300)
bp = st.number_input("Blood Pressure", min_value=0, max_value=200)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
insulin = st.number_input("Insulin", min_value=0, max_value=1000)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, format="%.3f")
age = st.number_input("Age", min_value=1, max_value=120)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ The patient is **likely to have diabetes.**")
    else:
        st.success("✅ The patient is **not likely to have diabetes.**")
