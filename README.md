<p align="center"><img src="https://raw.githubusercontent.com/dARSHANdR4/dARSHANdR4/main/assets/minecraft-emerald-divider.svg?v=4" width="100%" alt="Animated emerald pixel divider"></p>

<div align="center">

# 🌱 AI Prediction Studio · Crop & Fertilizer ML

**A combined prediction workspace and the original EDUNET agricultural ML notebooks.**

[![Live dashboard](https://img.shields.io/badge/LIVE_DASHBOARD-Cloudflare_Pages-245b32?style=for-the-badge&logo=cloudflarepages&logoColor=white)](https://ai-prediction-studio.pages.dev/)
[![Backend API](https://img.shields.io/badge/BACKEND-Render-245b32?style=for-the-badge&logo=render&logoColor=white)](https://ai-prediction-studio-api.onrender.com/health)
[![API docs](https://img.shields.io/badge/API_DOCS-OpenAPI-245b32?style=for-the-badge)](https://ai-prediction-studio-api.onrender.com/docs)
[**Developer profile ↗**](https://github.com/dARSHANdR4)

</div>

<p align="center"><img src="https://raw.githubusercontent.com/dARSHANdR4/dARSHANdR4/main/assets/minecraft-emerald-divider.svg?v=4" width="100%" alt="Animated emerald pixel divider"></p>

## 🚀 Live project

Open the deployed [**AI Prediction Studio dashboard**](https://ai-prediction-studio.pages.dev/). It combines three modules in one interface:

- **Crop recommendation** — uses soil nutrients and environmental inputs to suggest a crop.
- **Fertilizer recommendation** — suggests a fertilizer category from soil and crop inputs.
- **Diabetes screening** — educational screening output from health measurements; not a diagnosis.

The API is separately hosted on Render:
- Health and model readiness: [`/health`](https://ai-prediction-studio-api.onrender.com/health)
- Interactive API documentation: [`/docs`](https://ai-prediction-studio-api.onrender.com/docs)

## 🧱 Repository map

| Path | Purpose |
|---|---|
| `dashboard/frontend/` | React + Vite frontend |
| `dashboard/backend/app/` | FastAPI middleware and prediction endpoints |
| `dashboard/backend/train_models.py` | Model training and evaluation reporting |
| `dashboard/backend/EVALUATION.md` | Dataset metrics and model limitations |
| `Crop_recommendation.csv` | Crop recommendation source dataset |
| `Fertilizer Prediction.csv` | Fertilizer source dataset |
| `FINALPROJ.ipynb`, `Untitled1.ipynb` | Original notebook-based project work |

## 🔌 API routes

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service/model readiness |
| POST | `/api/crop/predict` | Crop suggestion |
| POST | `/api/fertilizer/predict` | Fertilizer-category suggestion |
| POST | `/api/diabetes/predict` | Educational screening output |

## 🛠️ Run locally

The combined dashboard's setup, environment variables, and deployment configuration are documented in the [dashboard README](dashboard/README.md).

At a high level:
1. Start the FastAPI backend from `dashboard/backend/` using its documented Python requirements and environment variables.
2. Start the React frontend from `dashboard/frontend/`.
3. Set `VITE_API_BASE_URL` to the backend URL and configure `CORS_ORIGINS` on the backend for the frontend origin.

## 📊 Model evaluation

The evaluation pipeline reports holdout-set metrics and stratified cross-validation results. See [`dashboard/backend/EVALUATION.md`](dashboard/backend/EVALUATION.md) and download the CI evaluation artifact for machine-readable metrics.

**Interpretation matters:** scores on the supplied datasets do not establish real-world farm performance or clinical validity. The fertilizer dataset is very small and requires more verified examples. Validate crop and fertilizer recommendations against local agronomic data and qualified guidance before acting on them.

## ⚠️ Safety and limitations

- Crop results are suggestions, not yield or suitability guarantees.
- Fertilizer output recommends a category; it does not determine a safe dose or replace soil testing.
- Diabetes output is educational only. A lower-risk result does not rule out diabetes and must not replace clinical testing.

<p align="center"><img src="https://raw.githubusercontent.com/dARSHANdR4/dARSHANdR4/main/assets/minecraft-emerald-divider.svg?v=4" width="100%" alt="Animated emerald pixel divider"></p>

<div align="center">

**Built and maintained by [dARSHANdR4](https://github.com/dARSHANdR4) · Voxel Builder / Developer / Automation**

</div>


## Copyright and permissions

Copyright © 2026 dARSHANdR4. All rights reserved.

Unless otherwise stated, original project code and assets are reserved and no open-source licence is granted. Third-party materials remain subject to their respective licences. See [COPYRIGHT](./COPYRIGHT) for the full notice.
