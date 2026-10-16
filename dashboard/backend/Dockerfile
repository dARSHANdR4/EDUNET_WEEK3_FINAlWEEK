FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Build from the repository root so the original project CSVs are available.
COPY dashboard/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard/backend/app ./app
COPY dashboard/backend/train_models.py ./train_models.py
COPY dashboard/backend/models/.gitkeep ./models/.gitkeep
RUN mkdir -p /tmp/datasets

COPY ["Crop_recommendation.csv", "/tmp/datasets/Crop_recommendation.csv"]
COPY ["Fertilizer Prediction.csv", "/tmp/datasets/Fertilizer Prediction.csv"]

# Fetch the public Pima dataset during image build; don't store it in the repo.
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv', '/tmp/datasets/diabetes.csv')"

# Train and package complete inference pipelines into the container image.
RUN python train_models.py \
    --crop-data "/tmp/datasets/Crop_recommendation.csv" \
    --fertilizer-data "/tmp/datasets/Fertilizer Prediction.csv" \
    --diabetes-data "/tmp/datasets/diabetes.csv" \
    --model-dir /app/models

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
