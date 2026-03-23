import streamlit as st
import pandas as pd
import joblib

# Load saved objects
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('heart_disease_scaler.pkl')
expected_features = joblib.load('heart_disease_features.pkl')

# App title
st.title('Heart Disease Prediction by nbk')

st.markdown('Enter the following details to predict the likelihood of heart disease:')

# User Inputs
age = st.slider('Age', 18, 100, 50)
sex = st.selectbox("Sex", ['Male', 'Female'])
chest_pain_type = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
resting_blood_pressure = st.number_input('Resting Blood Pressure', 80, 200, 120)
cholesterol = st.number_input('Cholesterol (mg/dL)', 100, 400, 200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ['Yes', 'No'])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
max_heart_rate = st.slider('Max Heart Rate', 60, 220, 150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina", ['Yes', 'No'])
oldpeak = st.slider('Oldpeak (ST Depression)', 0.0, 6.0, 1.0)
st_slope = st.selectbox("Slope of ST Segment", ['Up', 'Flat', 'Down'])

# Prediction button
if st.button('Predict'):

    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [1 if sex == 'Male' else 0],
        'chest_pain_type': [chest_pain_type],
        'resting_blood_pressure': [resting_blood_pressure],
        'cholesterol': [cholesterol],
        'fasting_blood_sugar': [1 if fasting_blood_sugar == 'Yes' else 0],
        'resting_ecg': [resting_ecg],
        'max_heart_rate': [max_heart_rate],
        'exercise_induced_angina': [1 if exercise_induced_angina == 'Yes' else 0],
        'oldpeak': [oldpeak],
        'st_slope': [st_slope]
    })

    #Convert categorical columns to dummy variables
    input_data = pd.get_dummies(input_data)

    #Align with training features
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns
    input_data = input_data[expected_features]

    # Scale input
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)[0][1]

    # Output
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠️ High risk of heart disease\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low risk of heart disease\n\nProbability: {probability:.2f}")