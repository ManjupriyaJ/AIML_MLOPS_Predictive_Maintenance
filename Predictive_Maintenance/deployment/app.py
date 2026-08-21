import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="ManjupriyaJ/Predictive-Maintenance", filename="best_predictive_maintenance_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Machine Failure Prediction App")
st.write("""
This application predicts the likelihood of potential failures of Machine based on its historical and real-time engine sensor data.
Please enter the sensor data below to get a prediction.
""")

# User input
engine_rpm = st.number_input(
    "Engine RPM",
    min_value=50.0,
    max_value=5000.0,
    value=61.0,
    step=1.0
)

lub_oil_pressure = st.number_input(
    "Lub Oil Pressure",
    min_value=0.001,
    value=0.003,
    step=0.001
)

fuel_pressure = st.number_input(
    "Fuel Pressure",
    min_value=0.0,
    value=0.003,
    step=0.001
)

coolant_pressure = st.number_input(
    "Coolant Pressure",
    min_value=0.001,
    value=0.003,
    step=0.001
)

lub_oil_temp = st.number_input(
    "Lub Oil Temp",
    min_value=10.0,
    value=2.0,
    step=1.0
)

coolant_temp = st.number_input(
    "Coolant Temp",
    min_value=20.0,
    value=50.0,
    step=1.0
)
numeric_features = ['engine_rpm', 'lub_oil_pressure', 'fuel_pressure', 'coolant_pressure', 'lub_oil_temp', 'coolant_temp']
input_data = pd.DataFrame([{
    "engine_rpm": engine_rpm,
    "lub_oil_pressure": lub_oil_pressure,
    "fuel_pressure": fuel_pressure,
    "coolant_pressure": coolant_pressure,
    "lub_oil_temp": lub_oil_temp,
    "coolant_temp": coolant_temp
}])

if st.button("Machine Failure Prediction"):
    prediction = model.predict(input_data)[0]
    result = "Machine is Abnormal/Faulty!!" if prediction == 1 else "Machine is Normal/Active!!"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
