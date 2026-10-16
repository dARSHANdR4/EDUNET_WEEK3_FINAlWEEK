import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_reports_service_and_models():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert set(response.json()["models"]) == {"crop", "fertilizer", "diabetes"}

def test_crop_rejects_invalid_ph():
    response = client.post("/api/crop/predict", json={"N":90,"P":42,"K":43,"temperature":22,"humidity":80,"ph":99,"rainfall":200})
    assert response.status_code == 422

def test_diabetes_rejects_invalid_age():
    response = client.post("/api/diabetes/predict", json={"Pregnancies":1,"Glucose":120,"BloodPressure":70,"SkinThickness":20,"Insulin":79,"BMI":25.8,"DiabetesPedigreeFunction":0.5,"Age":0})
    assert response.status_code == 422

def test_fertilizer_rejects_missing_fields():
    response = client.post("/api/fertilizer/predict", json={"Temperature":26})
    assert response.status_code == 422
