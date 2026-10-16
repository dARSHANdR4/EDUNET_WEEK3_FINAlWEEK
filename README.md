# Crop and Fertilizer Recommendation System

Educational machine-learning project with crop and fertilizer recommendation work in `FINALPROJ.ipynb`.

## Repository files and datasets

Keep the notebook and both CSV datasets in the repository root unless you update the notebook paths:

- `Crop_recommendation.csv`
- `Fertilizer Prediction.csv`

The earlier README referenced a `data/` directory, which may not match the current repository layout. If you prefer a `data/` folder, move the CSV files there and update the notebook's `read_csv` paths accordingly.

## Setup and launch

Use Python 3.10 or newer.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m pip install jupyter
jupyter notebook FINALPROJ.ipynb
```

Run notebook cells from top to bottom. If a dataset-loading cell fails, verify that the filename and path match the files in the repository.

## Scope and limitations

This is a learning/demo project, not professional agricultural advice. Validate recommendations locally with an agricultural expert before making crop or fertilizer decisions.
