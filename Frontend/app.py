
import streamlit as st
import requests
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Smart Agriculture System",
    page_icon="🌱",
    layout="wide"
)

# Title
st.title("🌱 AI-Powered Smart Agriculture System")
st.subheader("Early Crop Stress Prediction")

st.write(
    "Enter soil and environmental parameters "
    "to predict crop health using Machine Learning."
)

API_URL = "http://127.0.0.1:8000/predict"

# Sidebar
st.sidebar.header("Model Selection")

model = st.sidebar.selectbox(
    "Choose Machine Learning Model",
    ["Random Forest", "Decision Tree"]
)

model_name = (
    "random_forest"
    if model == "Random Forest"
    else "decision_tree"
)

st.sidebar.info("Select a model to predict crop health.")

# Input fields
st.header("🌿 Enter Crop Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    soil_moisture = st.number_input(
        "Soil Moisture", value=45.0
    )
    ambient_temp = st.number_input(
        "Ambient Temperature", value=28.0
    )
    soil_temp = st.number_input(
        "Soil Temperature", value=25.0
    )

with col2:
    humidity = st.number_input(
        "Humidity", value=60.0
    )
    light = st.number_input(
        "Light Intensity", value=500.0
    )
    soil_ph = st.number_input(
        "Soil pH", value=6.5
    )

with col3:
    nitrogen = st.number_input(
        "Nitrogen Level", value=40.0
    )
    phosphorus = st.number_input(
        "Phosphorus Level", value=30.0
    )
    potassium = st.number_input(
        "Potassium Level", value=35.0
    )

# Prediction button
if st.button("🔍 Predict Crop Health", use_container_width=True):

    input_data = {
        "Soil_Moisture": soil_moisture,
        "Ambient_Temperature": ambient_temp,
        "Soil_Temperature": soil_temp,
        "Humidity": humidity,
        "Light_Intensity": light,
        "Soil_pH": soil_ph,
        "Nitrogen_Level": nitrogen,
        "Phosphorus_Level": phosphorus,
        "Potassium_Level": potassium
    }

    try:
        response = requests.post(
            API_URL,
            params={"model": model_name},
            json=input_data,
            timeout=15
        )

        response.raise_for_status()
        result = response.json()

        st.subheader("📊 Prediction Result")

        if result["crop_health"] == "Healthy":
            st.success("🌱 Crop Health: HEALTHY")
        else:
            st.error("⚠️ Crop Health: STRESSED")

        st.write("**Selected Model:**", model)
        st.write(
            "**Prediction:**",
            result["crop_health"]
        )

    except requests.exceptions.RequestException as e:
        st.error(f"Backend connection error: {e}")

# Model comparison
st.divider()
st.header("📈 Model Performance Comparison")

comparison = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy": [99.58, 99.58]
})

st.dataframe(comparison, use_container_width=True)

st.bar_chart(
    comparison.set_index("Model")["Accuracy"]
)

st.caption(
    "Note: Accuracy values are based on the "
    "project's held-out test dataset."
)
