import streamlit as st
import pandas as pd
import joblib

# Load model & encoders
model, feature_columns = joblib.load("best_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
median_values = joblib.load("median_values.pkl")
unique_values = joblib.load("unique_values.pkl")

st.set_page_config(page_title="Crop Yield Prediction", layout="centered")

st.title("🌾 Crop Yield Prediction")

# Dropdowns
crop = st.selectbox("Select Crop", unique_values["crops"])
state = st.selectbox("Select State", unique_values["states"])
season = st.selectbox("Select Season", unique_values["seasons"])

year = st.number_input("Crop Year (Optional)", value=int(median_values["Crop_Year"]))
rainfall = st.number_input("Annual Rainfall (Optional)", value=float(median_values["Annual_Rainfall"]))
fertilizer = st.number_input("Fertilizer Use (Optional)", value=float(median_values["Fertilizer"]))
pesticide = st.number_input("Pesticide Use (Optional)", value=float(median_values["Pesticide"]))

if st.button("Predict Yield"):
    input_data = {
        "Crop": [label_encoders["Crop"].transform([crop])[0]],
        "State": [label_encoders["State"].transform([state])[0]],
        "Season": [label_encoders["Season"].transform([season])[0]],
        "Crop_Year": [year],
        "Annual_Rainfall": [rainfall],
        "Fertilizer": [fertilizer],
        "Pesticide": [pesticide],
    }

    df = pd.DataFrame(input_data)

    prediction = model.predict(df)[0]

    st.success(f"🌾 Predicted Yield: **{prediction:.2f} tons per hectare**")
