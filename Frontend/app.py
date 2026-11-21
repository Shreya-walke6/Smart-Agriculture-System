
import streamlit as st
import requests
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101c17, #172820);
    color: #f1f5f2;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

[data-testid="stSidebar"] {
    background: #14251d;
    border-right: 1px solid #31513d;
}

.hero {
    background: linear-gradient(120deg, #1d4933, #286344);
    padding: 32px;
    border-radius: 18px;
    margin-bottom: 25px;
    border: 1px solid #477c57;
}

.hero h1 {
    color: white;
    font-size: 34px;
    margin-bottom: 8px;
}

.hero p {
    color: #d4ead9;
    font-size: 16px;
}

.section-title {
    color: #a8e6b5;
    font-size: 23px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

.info-card {
    background: #20382b;
    border: 1px solid #365c43;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 12px;
}

.info-card h3 {
    color: #b5e8c0;
    font-size: 15px;
    margin-bottom: 8px;
}

.info-card h2 {
    color: white;
    font-size: 28px;
    margin: 0;
}

.result-healthy {
    background: #174b32;
    border: 1px solid #3a9b65;
    color: #c5f5d5;
    padding: 22px;
    border-radius: 14px;
    font-size: 21px;
    font-weight: bold;
}

.result-stressed {
    background: #512a2a;
    border: 1px solid #b45c5c;
    color: #ffd0d0;
    padding: 22px;
    border-radius: 14px;
    font-size: 21px;
    font-weight: bold;
}

div.stButton > button {
    background: linear-gradient(90deg, #287447, #39965d);
    color: white;
    border: none;
    border-radius: 10px;
    height: 50px;
    font-weight: 700;
    font-size: 16px;
}

div.stButton > button:hover {
    background: #43a66b;
    color: white;
    border: 1px solid #9be2ad;
}

div[data-testid="stMetric"] {
    background: #20382b;
    border: 1px solid #365c43;
    padding: 15px;
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #9bb5a2;
    padding: 20px;
    border-top: 1px solid #365c43;
    margin-top: 35px;
}
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000/predict"

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 🌱 Smart Agriculture")
st.sidebar.caption("AI Crop Health Monitoring")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Model Selection")

model = st.sidebar.selectbox(
    "Choose Machine Learning Model",
    ["Random Forest", "Decision Tree"]
)

model_name = (
    "random_forest"
    if model == "Random Forest"
    else "decision_tree"
)

st.sidebar.success(f"Selected Model: {model}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Project Information")
st.sidebar.write("**Models:** Decision Tree & Random Forest")
st.sidebar.write("**Output:** Healthy / Stressed")
st.sidebar.write("**Platform:** Streamlit + FastAPI")

st.sidebar.info(
    "This project predicts crop health using "
    "soil and environmental parameters."
)

# ---------------- HERO HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🌿 AI-Powered Smart Agriculture System</h1>
    <p>Early Crop Stress Prediction using Machine Learning</p>
    <p>Monitor environmental conditions and predict crop health intelligently.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- PERFORMANCE CARDS ----------------
st.markdown('<div class="section-title">📊 Model Performance Overview</div>',
            unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    <div class="info-card">
        <h3>🌳 Decision Tree Accuracy</h3>
        <h2>99.58%</h2>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="info-card">
        <h3>🌲 Random Forest Accuracy</h3>
        <h2>99.58%</h2>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="info-card">
        <h3>🧪 Input Features</h3>
        <h2>09</h2>
    </div>
    """, unsafe_allow_html=True)

# ---------------- CROP INPUTS ----------------
st.markdown('<div class="section-title">🌾 Enter Crop Parameters</div>',
            unsafe_allow_html=True)

st.write("Enter the soil and environmental measurements below.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 💧 Soil Conditions")

    soil_moisture = st.number_input(
        "Soil Moisture", value=45.0
    )

    soil_temp = st.number_input(
        "Soil Temperature", value=25.0
    )

    soil_ph = st.number_input(
        "Soil pH", value=6.5
    )

with col2:
    st.markdown("#### 🌤️ Environment")

    ambient_temp = st.number_input(
        "Ambient Temperature", value=28.0
    )

    humidity = st.number_input(
        "Humidity", value=60.0
    )

    light = st.number_input(
        "Light Intensity", value=500.0
    )

with col3:
    st.markdown("#### 🧪 Soil Nutrients")

    nitrogen = st.number_input(
        "Nitrogen Level", value=40.0
    )

    phosphorus = st.number_input(
        "Phosphorus Level", value=30.0
    )

    potassium = st.number_input(
        "Potassium Level", value=35.0
    )

st.markdown("---")

# ---------------- PREDICTION ----------------
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
        with st.spinner("Analyzing crop parameters..."):

            response = requests.post(
                API_URL,
                params={"model": model_name},
                json=input_data,
                timeout=15
            )

            response.raise_for_status()
            result = response.json()

        prediction = result["crop_health"]

        st.markdown(
            '<div class="section-title">🔬 Crop Health Prediction Result</div>',
            unsafe_allow_html=True
        )

        if prediction == "Healthy":
            st.markdown("""
            <div class="result-healthy">
                🌱 CROP HEALTH: HEALTHY
                <p style="font-size:15px; font-weight:normal;">
                The selected model predicts a healthy crop condition.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="result-stressed">
                ⚠️ CROP HEALTH: STRESSED
                <p style="font-size:15px; font-weight:normal;">
                The selected model predicts a stressed crop condition.
                </p>
            </div>
            """, unsafe_allow_html=True)

        r1, r2 = st.columns(2)

        with r1:
            st.metric("Selected ML Model", model)

        with r2:
            st.metric("Predicted Crop Health", prediction)

    except requests.exceptions.RequestException as e:
        st.error(f"Backend connection error: {e}")

    except (KeyError, ValueError):
        st.error("Unexpected prediction response from backend.")

# ---------------- MODEL COMPARISON ----------------
st.markdown("---")

st.markdown(
    '<div class="section-title">📈 Model Performance Comparison</div>',
    unsafe_allow_html=True
)

comparison = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy (%)": [99.58, 99.58]
})

chart_col, table_col = st.columns([1.5, 1])

with chart_col:
    st.markdown("#### Accuracy Comparison")
    st.bar_chart(
        comparison.set_index("Model"),
        y="Accuracy (%)"
    )

with table_col:
    st.markdown("#### Performance Details")
    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "Accuracy values are based on the held-out test dataset. "
    "The dataset contains simulated sensor observations and "
    "these results do not establish real-farm performance."
)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    🌱 AI-Powered Smart Agriculture System
    <br>
    Early Crop Stress Prediction | Machine Learning Project
</div>
""", unsafe_allow_html=True)
