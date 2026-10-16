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


def test_crop_prediction_when_artifact_is_installed():
    from pathlib import Path
    model_path = Path(__file__).resolve().parents[1] / "models" / "crop.joblib"
    if not model_path.exists():
        import pytest
        pytest.skip("Training artifacts are created in the model-integration CI job")
    response = client.post("/api/crop/predict", json={"N":90,"P":42,"K":43,"temperature":22,"humidity":80,"ph":6.5,"rainfall":203})
    assert response.status_code == 200, response.text
    assert isinstance(response.json()["prediction"], str) and response.json()["prediction"]

def test_fertilizer_prediction_when_artifact_is_installed():
    from pathlib import Path
    model_path = Path(__file__).resolve().parents[1] / "models" / "fertilizer.joblib"
    if not model_path.exists():
        import pytest
        pytest.skip("Training artifacts are created in the model-integration CI job")
    response = client.post("/api/fertilizer/predict", json={"Temperature":26,"Humidity":52,"Moisture":38,"Soil Type":"Sandy","Crop Type":"Maize","Nitrogen":37,"Potassium":0,"Phosphorous":0})
    assert response.status_code == 200, response.text
    assert isinstance(response.json()["prediction"], str) and response.json()["prediction"]

def test_diabetes_prediction_when_artifact_is_installed():
    from pathlib import Path
    model_path = Path(__file__).resolve().parents[1] / "models" / "diabetes.joblib"
    if not model_path.exists():
        import pytest
        pytest.skip("Training artifacts are created in the model-integration CI job")
    response = client.post("/api/diabetes/predict", json={"Pregnancies":5,"Glucose":166,"BloodPressure":72,"SkinThickness":19,"Insulin":175,"BMI":25.8,"DiabetesPedigreeFunction":0.587,"Age":51})
    assert response.status_code == 200, response.text
    assert response.json()["class"] in (0, 1)
    assert "not a medical diagnosis" in response.json()["disclaimer"]



def test_diabetes_zero_sentinels_are_marked_missing():
    import pandas as pd
    from app.preprocessing import mark_diabetes_zeros_missing

    frame = pd.DataFrame([{
        "Pregnancies": 0, "Glucose": 0, "BloodPressure": 0,
        "SkinThickness": 0, "Insulin": 0, "BMI": 0,
        "DiabetesPedigreeFunction": 0.2, "Age": 30,
    }])
    transformed = mark_diabetes_zeros_missing(frame)
    for column in ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"):
        assert pd.isna(transformed.loc[0, column])
    assert transformed.loc[0, "Pregnancies"] == 0
    assert transformed.loc[0, "DiabetesPedigreeFunction"] == 0.2
    assert transformed.loc[0, "Age"] == 30
    # The helper must not mutate the caller's original input.
    assert frame.loc[0, "Glucose"] == 0


def test_diabetes_preprocessing_preserves_nonzero_measurements():
    import pandas as pd
    from app.preprocessing import mark_diabetes_zeros_missing

    frame = pd.DataFrame([{
        "Pregnancies": 1, "Glucose": 120, "BloodPressure": 70,
        "SkinThickness": 20, "Insulin": 79, "BMI": 25.8,
        "DiabetesPedigreeFunction": 0.5, "Age": 33,
    }])
    transformed = mark_diabetes_zeros_missing(frame)
    assert transformed.equals(frame)
