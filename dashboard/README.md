<p align="center"><img src="https://raw.githubusercontent.com/dARSHANdR4/dARSHANdR4/main/assets/minecraft-emerald-divider.svg?v=4" width="100%" alt="Animated emerald pixel divider"></p>

<div align="center">

# 🌱 AI Prediction Studio

**Crop recommendations · Fertilizer suggestions · Educational diabetes screening**

[![Live dashboard](https://img.shields.io/badge/LIVE_DASHBOARD-Cloudflare_Pages-245b32?style=for-the-badge&logo=cloudflarepages&logoColor=white)](https://ai-prediction-studio.pages.dev/)
[![Backend](https://img.shields.io/badge/API-Render-245b32?style=for-the-badge&logo=render&logoColor=white)](https://ai-prediction-studio-api.onrender.com/health)
[![API docs](https://img.shields.io/badge/API_DOCS-OpenAPI-245b32?style=for-the-badge)](https://ai-prediction-studio-api.onrender.com/docs)

</div>

- **Dashboard:** https://ai-prediction-studio.pages.dev/
- **Backend health:** https://ai-prediction-studio-api.onrender.com/health
- **Interactive API docs:** https://ai-prediction-studio-api.onrender.com/docs

The live dashboard combines crop recommendation, fertilizer recommendation, and educational diabetes screening. Model evaluation is dataset-based; it does not establish real-world agricultural validity or clinical safety. See [model evaluation notes](backend/EVALUATION.md).

**Safety:** Crop and fertilizer results are suggestions, not guarantees. Diabetes screening is educational only and is not a diagnosis or a substitute for medical testing.

---

# AI Prediction Studio

A combined React + FastAPI workspace for the existing crop, fertilizer, and diabetes ML projects.

## Architecture

- `frontend/`: React + Vite responsive dashboard.
- `backend/app/main.py`: FastAPI middleware, validation, CORS, and prediction routes.
- `backend/models/`: trained, versioned `.joblib` model pipelines (generated locally; never commit private data or model artifacts by default).

## Current status

The dashboard is deployed to Cloudflare Pages and the FastAPI service is deployed on Render. The service health response has reported all three models loaded, and browser smoke tests have returned predictions from the crop, fertilizer, and diabetes modules. GitHub Actions checks cover the frontend build, backend tests, model training/evaluation, and Docker build. These checks verify technical behaviour; they do not prove real-world agricultural accuracy or clinical validity. The production container trains and packages model pipelines during its Docker build, so generated `.joblib` files do not need to be committed to Git.

## API routes

- `GET /health` — service and artifact readiness.
- `POST /api/crop/predict` — N, P, K, temperature, humidity, pH, rainfall.
- `POST /api/fertilizer/predict` — temperature, humidity, moisture, soil type, crop type, nitrogen, potassium, phosphorous.
- `POST /api/diabetes/predict` — pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, age.

OpenAPI docs are at `/docs` when the backend runs.

## Run locally

### Frontend
```bash
cd dashboard/frontend
npm install
cp .env.example .env
npm run dev
```

### Backend
```bash
cd dashboard/backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Configure `VITE_API_BASE_URL` in the frontend env file if the API runs elsewhere. Configure `CORS_ORIGINS` on the API to include the deployed frontend origin.

## Model integration checklist

1. The CI pipeline trains and tests the crop, fertilizer, and diabetes inference pipelines using the source project CSVs plus a public Pima diabetes dataset.
2. The Render Docker build uses the **repository root as Docker context** and `dashboard/backend/Dockerfile` as its Dockerfile. It trains and bundles `.joblib` files inside the image; model files and the downloaded diabetes dataset do not need to be committed.
3. Deploy the frontend to Cloudflare Pages: connect this repository, set root directory to `dashboard/frontend`, build command `npm run build`, output directory `dist`, and set `VITE_API_BASE_URL` to the deployed Render API URL (no trailing slash).
4. Deploy the API on Render as a Docker web service: Docker context/root directory `.` (repository root), Dockerfile path `dashboard/backend/Dockerfile`, health check path `/health`. After Cloudflare Pages gives the frontend URL, set Render environment variable `CORS_ORIGINS` to that exact origin (for example `https://your-project.pages.dev`, no trailing slash).
5. After any deployment or configuration change, re-test `/health`, `/docs`, and all three prediction routes from the live UI. A successful CI run is not a substitute for production smoke tests.

## Safety

The diabetes output is educational screening only, not diagnosis or medical advice. Crop and fertilizer outputs are model recommendations, not agronomic guarantees; confirm decisions against local conditions and qualified advice.


<p align="center"><img src="https://raw.githubusercontent.com/dARSHANdR4/dARSHANdR4/main/assets/minecraft-emerald-divider.svg?v=4" width="100%" alt="Animated emerald pixel divider"></p>

<div align="center">

**Built by [dARSHANdR4](https://github.com/dARSHANdR4) · Voxel Builder / Developer / Automation**

</div>
