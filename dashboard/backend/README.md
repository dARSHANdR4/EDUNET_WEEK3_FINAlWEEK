# Backend model setup

Install dependencies from this directory:

```bash
python -m pip install -r requirements.txt
```

Put a legally usable Pima diabetes dataset at `data/diabetes.csv` with columns `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`. Do not commit the dataset unless licensing and redistribution permission are confirmed.

Train the three saved pipelines from this directory:

```bash
python train_models.py --crop-data ../../Crop_recommendation.csv --fertilizer-data "../../Fertilizer Prediction.csv" --diabetes-data data/diabetes.csv
```

The two crop/fertilizer CSV paths above point to the existing EDUNET repository root when run from this folder. Adjust paths if your checkout differs. The command reports held-out accuracy and saves the fitted inference pipelines into `models/`.

Run the API:

```bash
uvicorn app.main:app --reload --port 8000
```

Run checks:

```bash
pytest -q
```

Never deploy without training all three artifacts and testing the real endpoints. Diabetes output is educational screening only, not a diagnosis.
