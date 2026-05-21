# app.py
import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model1 = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict the likelihood of diabetes.")

# Create user input fields
pregnancies = st.slider("Pregnancies", 0, 17, 1)
glucose = st.slider("Glucose Level", 0, 200, 120)
blood_pressure = st.slider("Blood Pressure", 0, 122, 70)
skin_thickness = st.slider("Skin Thickness", 0, 99, 20)
insulin = st.slider("Insulin", 0, 846, 79)
bmi = st.slider("BMI", 0.0, 67.1, 32.0)
dpf = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.5)
age = st.slider("Age", 21, 81, 30)

# Predict button
if st.button("Predict"):
    # Prepare the input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                            insulin, bmi, dpf, age]])
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model1.predict(input_scaled)
    probability = model1.predict_proba(input_scaled)[0][1]
    
    # Display result
    st.subheader("Result:")
    if prediction[0] == 1:
        st.error(f"⚠️ High risk of Diabetes! (Probability: {probability*100:.2f}%)")
    else:
        st.success(f"✅ Low risk of Diabetes. (Probability: {probability*100:.2f}%)")