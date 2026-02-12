import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Crop Yield Prediction")

st.title("🌾 AI Crop Yield Prediction")

df = pd.read_csv("crop_yield.csv")

for col in ["Crop", "State"]:
    df[col] = df[col].str.strip().str.title()
df["Season"] = df["Season"].str.strip()

df["Yield"] = df["Production"] / df["Area"]

label_encoders = {}
for col in ["Crop", "State", "Season"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df[["Crop", "State", "Season", "Crop_Year", "Annual_Rainfall"]]
y = df["Yield"]

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

crop = st.selectbox("Crop", label_encoders["Crop"].classes_)
state = st.selectbox("State", label_encoders["State"].classes_)
season = st.selectbox("Season", label_encoders["Season"].classes_)
year = st.number_input("Crop Year", 2000, 2050, 2024)
rain = st.number_input("Annual Rainfall", 0.0, 5000.0, 1000.0)

if st.button("Predict"):
    c = label_encoders["Crop"].transform([crop])[0]
    s = label_encoders["State"].transform([state])[0]
    se = label_encoders["Season"].transform([season])[0]

    pred = model.predict([[c, s, se, year, rain]])
    st.success(f"🌱 Predicted Yield: {pred[0]:.2f} tons/hectare")
