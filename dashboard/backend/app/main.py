from pathlib import Path
from typing import Any
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
app = FastAPI(title="AI Prediction Studio API", version="0.1.0", description="Validated API gateway for crop, fertilizer, and diabetes ML predictions.")
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Authorization"])

class CropInput(BaseModel):
    N: float = Field(ge=0, le=200)
    P: float = Field(ge=0, le=200)
    K: float = Field(ge=0, le=250)
    temperature: float = Field(ge=-10, le=60)
    humidity: float = Field(ge=0, le=100)
    ph: float = Field(ge=0, le=14)
    rainfall: float = Field(ge=0, le=5000)

class FertilizerInput(BaseModel):
    Temperature: float = Field(ge=-10, le=60)
    Humidity: float = Field(ge=0, le=100)
    Moisture: float = Field(ge=0, le=100)
    **kwargs: Any

class DiabetesInput(BaseModel):
    Pregnancies: float = Field(ge=0, le=20)
    Glucose: float = Field(ge=0, le=300)
    BloodPressure: float = Field(ge=0, le=200)
    SkinThickness: float = Field(ge=0, le=100)
    Insulin: float = Field(ge=0, le=900)
    BMI: float = Field(ge=0, le=80)
    DiabetesPedigreeFunction: float = Field(ge=0, le=3)
    Age: float = Field(ge=1, le=120)

def load_model(name: str):
    path = MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Model artifact '{name}.joblib' is not installed yet. Follow dashboard/README.md to train and configure the model.")
    try:
        return joblib.load(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load model artifact '{name}': {exc}") from exc

@app.get("/")
def root():
    return {"service": "AI Prediction Studio API", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "ai-prediction-studio-api", "models": {n: (MODEL_DIR / f"{n}.joblib").exists() for n in ("crop", "fertilizer", "diabetes")}}

@app.post("/api/crop/predict")
def predict_crop(payload: CropInput):
    model = load_model("crop")
    frame = pd.DataFrame([payload.model_dump()])
    try:
        prediction = model.predict(frame)[0]
        return {"prediction": str(prediction), "model": "crop"}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Crop model input mismatch: {exc}") from exc

@app.post("/api/fertilizer/predict")
def predict_fertilizer(payload: dict[str, Any]):
    required = {"Temperature", "Humidity", "Moisture", "Soil Type", "Crop Type", "Nitrogen", "Potassium", "Phosphorous"}
    missing = sorted(required - payload.keys())
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required fields: {', '.join(missing)}")
    for key in ("Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"):
        try:
            payload[key] = float(payload[key])
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f"'{key}' must be numeric")
    if not -10 <= payload["Temperature"] <= 60 or not 0 <= payload["Humidity"] <= 100 or not 0 <= payload["Moisture"] <= 100:
        raise HTTPException(status_code=422, detail="Temperature, humidity, or moisture is outside supported input limits")
    model = load_model("fertilizer")
    try:
        prediction = model.predict(pd.DataFrame([payload]))[0]
        return {"prediction": str(prediction), "model": "fertilizer"}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Fertilizer model input mismatch: {exc}") from exc

@app.post("/api/diabetes/predict")
def predict_diabetes(payload: DiabetesInput):
    model = load_model("diabetes")
    try:
        prediction = int(model.predict(pd.DataFrame([payload.model_dump()]))[0])
        return {"prediction": "Higher model-indicated risk" if prediction == 1 else "Lower model-indicated risk", "class": prediction, "model": "diabetes", "disclaimer": "Educational screening output only; not a medical diagnosis."}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Diabetes model input mismatch: {exc}") from exc
