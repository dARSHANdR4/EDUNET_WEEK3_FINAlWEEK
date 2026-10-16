"""Train exportable inference pipelines from the existing project datasets.

Example (from dashboard/backend):
  python train_models.py --crop-data ../../Crop_recommendation.csv --fertilizer-data "../../Fertilizer Prediction.csv" --diabetes-data ./data/diabetes.csv

All preprocessing is saved inside each pipeline so API inference uses the exact fitted transforms.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(__file__).resolve().parent / "models"

CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FERT_NUMERIC = ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
FERT_CATEGORICAL = ["Soil Type", "Crop Type"]
FERT_FEATURES = FERT_NUMERIC + FERT_CATEGORICAL
DIABETES_FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

def evaluate_and_fit(name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, output: Path) -> None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    evaluation_pipeline = joblib.loads(joblib.dumps(pipeline))
    evaluation_pipeline.fit(X_train, y_train)
    score = accuracy_score(y_test, evaluation_pipeline.predict(X_test))
    # Refit on all available labeled rows for deployment after held-out evaluation.
    pipeline.fit(X, y)
    joblib.dump(pipeline, output)
    print(f"{name}: held-out accuracy={score:.4f}; rows={len(X)}; saved={output}")

def load_crop(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    target = "label" if "label" in df.columns else "crop"
    missing = set(CROP_FEATURES + [target]) - set(df.columns)
    if missing:
        raise ValueError(f"Crop dataset missing columns: {sorted(missing)}")
    X = df[CROP_FEATURES].apply(pd.to_numeric, errors="raise")
    return X, df[target].astype(str)

def load_fertilizer(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    # Source dataset has legacy spelling/spacing; normalize to the API contract.
    df.columns = [str(c).strip() for c in df.columns]
    rename = {"Temparature": "Temperature", "Temperature": "Temperature", "Humidity": "Humidity", "Moisture": "Moisture", "Fertilizer Name": "Fertilizer Name"}
    df = df.rename(columns=rename)
    if "Temperature" not in df.columns and "Temparature" in df.columns:
        df = df.rename(columns={"Temparature": "Temperature"})
    target = "Fertilizer Name"
    missing = set(FERT_FEATURES + [target]) - set(df.columns)
    if missing:
        raise ValueError(f"Fertilizer dataset missing columns: {sorted(missing)}")
    X = df[FERT_FEATURES].copy()
    for c in FERT_NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="raise")
    for c in FERT_CATEGORICAL:
        X[c] = X[c].astype(str).str.strip()
    return X, df[target].astype(str)

def load_diabetes(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    target = "Outcome"
    missing = set(DIABETES_FEATURES + [target]) - set(df.columns)
    if missing:
        raise ValueError(f"Diabetes dataset missing columns: {sorted(missing)}")
    X = df[DIABETES_FEATURES].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(df[target], errors="raise").astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Diabetes Outcome must contain only 0/1 labels")
    return X, y

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-data", type=Path, required=True)
    parser.add_argument("--fertilizer-data", type=Path, required=True)
    parser.add_argument("--diabetes-data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_crop(args.crop_data)
    crop = Pipeline([("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1))])
    evaluate_and_fit("crop", crop, X, y, args.model_dir / "crop.joblib")

    X, y = load_fertilizer(args.fertilizer_data)
    prep = ColumnTransformer([
        ("numeric", "passthrough", FERT_NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), FERT_CATEGORICAL),
    ])
    fertilizer = Pipeline([("preprocess", prep), ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1))])
    evaluate_and_fit("fertilizer", fertilizer, X, y, args.model_dir / "fertilizer.joblib")

    X, y = load_diabetes(args.diabetes_data)
    diabetes = Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="linear"))])
    evaluate_and_fit("diabetes", diabetes, X, y, args.model_dir / "diabetes.joblib")

if __name__ == "__main__":
    main()
