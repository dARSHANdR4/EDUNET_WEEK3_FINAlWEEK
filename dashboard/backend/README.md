<p align="center"><img src="https://raw.githubusercontent.com/dARSHANdR4/dARSHANdR4/main/assets/minecraft-emerald-divider.svg?v=4" width="100%" alt="Animated emerald pixel divider"></p>

<div align="center">

# 🧪 AI Prediction Studio · Backend

**FastAPI middleware for crop, fertilizer, and educational diabetes predictions.**

[![Health](https://img.shields.io/badge/LIVE_API-Render-245b32?style=for-the-badge&logo=render&logoColor=white)](https://ai-prediction-studio-api.onrender.com/health)
[![API docs](https://img.shields.io/badge/API_DOCS-OpenAPI-245b32?style=for-the-badge)](https://ai-prediction-studio-api.onrender.com/docs)
[**Dashboard ↗**](https://ai-prediction-studio.pages.dev/)

</div>

<p align="center"><img src="https://raw.githubusercontent.com/dARSHANdR4/dARSHANdR4/main/assets/minecraft-emerald-divider.svg?v=4" width="100%" alt="Animated emerald pixel divider"></p>

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

## 📊 Evaluation report

The training script writes `models/evaluation_report.json` with holdout and stratified cross-validation metrics. The CI workflow also publishes this file as a downloadable artifact. See the [summary of measured results and limitations](EVALUATION.md).

These are dataset-level estimates, not proof of real-world agricultural performance or clinical validity. The fertilizer dataset is small; the diabetes classifier requires further independent validation.

Run the API:

```bash
uvicorn app.main:app --reload --port 8000
```

Run checks:

```bash
pytest -q
```

Never deploy without training all three artifacts and testing the real endpoints. Diabetes output is educational screening only, not a diagnosis.


<p align="center"><img src="https://raw.githubusercontent.com/dARSHANdR4/dARSHANdR4/main/assets/minecraft-emerald-divider.svg?v=4" width="100%" alt="Animated emerald pixel divider"></p>

<div align="center">

**Built by [dARSHANdR4](https://github.com/dARSHANdR4) · Voxel Builder / Developer / Automation**

</div>
