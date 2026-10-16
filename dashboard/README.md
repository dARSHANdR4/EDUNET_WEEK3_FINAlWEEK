# AI Prediction Studio

A combined React + FastAPI workspace for the existing crop, fertilizer, and diabetes ML projects.

## Architecture

- `frontend/`: React + Vite responsive dashboard.
- `backend/app/main.py`: FastAPI middleware, validation, CORS, and prediction routes.
- `backend/models/`: trained, versioned `.joblib` model pipelines (generated locally; never commit private data or model artifacts by default).

## Current status

The UI and API contract are scaffolded. The API returns a clear HTTP 503 until the corresponding trained artifact is installed. This is intentional: CI passing on the source projects does not itself mean inference artifacts are available. Do not describe the dashboard as production-ready until model pipelines are trained, saved, and end-to-end tested.

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

1. For crop and fertilizer, create reusable training scripts from the source EDUNET notebook that preserve its exact feature order, encoders, and target labels. Save each *complete fitted preprocessing + model pipeline* as `crop.joblib` and `fertilizer.joblib`.
2. For diabetes, use the same eight Pima dataset features and fit the `StandardScaler` only on training data, then save a single sklearn `Pipeline(StandardScaler, SVC(kernel="linear"))` as `diabetes.joblib`. Do not fit a scaler on the full dataset before splitting.
3. Put the trained artifacts in `dashboard/backend/models/` locally or configure `MODEL_DIR` on the backend host. Keep datasets and sensitive artifacts out of Git unless their licensing and privacy status is confirmed.
4. Run API tests and browser-based end-to-end tests for all three modules, including validation failures and missing-artifact behavior.
5. Deploy the frontend to Cloudflare Pages (root directory `dashboard/frontend`, build command `npm run build`, output `dist`) and deploy the API to Render using `dashboard/backend/Dockerfile`. Set `VITE_API_BASE_URL` at frontend build time and `CORS_ORIGINS` server-side.

## Safety

The diabetes output is educational screening only, not diagnosis or medical advice. Crop and fertilizer outputs are model recommendations, not agronomic guarantees; confirm decisions against local conditions and qualified advice.
