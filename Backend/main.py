
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="AI-Powered Smart Agriculture System",
    description="Crop Health Prediction API",
    version="1.0"
)

# Load trained ML models
dt_model = joblib.load("decision_tree.pkl")
rf_model = joblib.load("random_forest.pkl")


# Input features
class CropData(BaseModel):
    Soil_Moisture: float
    Ambient_Temperature: float
    Soil_Temperature: float
    Humidity: float
    Light_Intensity: float
    Soil_pH: float
    Nitrogen_Level: float
    Phosphorus_Level: float
    Potassium_Level: float


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Smart Agriculture API is running successfully!"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: CropData, model: str = "random_forest"):

    input_data = pd.DataFrame([data.model_dump()])

    if model == "decision_tree":
        prediction = dt_model.predict(input_data)[0]

    elif model == "random_forest":
        prediction = rf_model.predict(input_data)[0]

    else:
        return {"error": "Invalid model selected"}

    status = "Stressed" if int(prediction) == 1 else "Healthy"

    return {
        "crop_health": status,
        "prediction": int(prediction),
        "model_used": model
    }
