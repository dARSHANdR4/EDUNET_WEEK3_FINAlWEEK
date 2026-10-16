## Live AI Prediction Studio

- **Dashboard:** https://ai-prediction-studio.pages.dev/
- **Backend health:** https://ai-prediction-studio-api.onrender.com/health
- **Interactive API docs:** https://ai-prediction-studio-api.onrender.com/docs

The live dashboard combines crop recommendation, fertilizer recommendation, and educational diabetes screening. Model evaluation is dataset-based; it does not establish real-world agricultural validity or clinical safety. See [model evaluation notes](dashboard/backend/EVALUATION.md).

**Safety:** Crop and fertilizer results are suggestions, not guarantees. Diabetes screening is educational only and is not a diagnosis or a substitute for medical testing.

---

# EDUNET_WEEK3_FINAlWEEK

# Crop and Fertilizer Recommendation System

## Overview

This project implements two machine learning models to assist in agricultural decision-making:
1.  **Crop Recommendation:** Predicts the most suitable crop to grow based on soil nutrient levels (N, P, K) and environmental conditions (temperature, humidity, pH, rainfall).
2.  **Fertilizer Recommendation:** Recommends the appropriate fertilizer based on soil conditions, crop type, and nutrient levels.

The models are trained on provided datasets and use Decision Tree Classifiers. The project also demonstrates data preprocessing, model evaluation, and model persistence using `pickle`.

## Features

-   **Crop Recommendation:**
    -   Input: N, P, K, temperature, humidity, pH, rainfall.
    -   Output: Recommended crop.
-   **Fertilizer Recommendation:**
    -   Input: Temperature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous.
    -   Output: Recommended fertilizer.

## Technology Stack

-   **Language:** Python 3.x
-   **Libraries:**
    -   Pandas: For data manipulation and analysis.
    -   NumPy: For numerical operations.
    -   Matplotlib & Seaborn: For data visualization.
    -   Scikit-learn: For machine learning tasks (model building, preprocessing, evaluation).
    -   Pickle (or Joblib): For model persistence.
-   **Environment:** Jupyter Notebook (recommended for exploration and training).

## Dataset Description

Two datasets are used:
1.  `data/Crop_recommendation.csv`: Contains soil nutrient data, environmental factors, and the corresponding recommended crop.
2.  `data/Fertilizer Prediction.csv`: Contains soil conditions, crop types, nutrient values, and the corresponding recommended fertilizer.

## Methodology / Workflow

The project follows these general steps for each recommendation task:

1.  **Data Loading and Exploration (EDA):**
    -   Load CSV data into Pandas DataFrames.
    -   Perform initial checks: shape, data types, missing values, duplicates.
    -   Visualize feature distributions and correlations.
2.  **Data Preprocessing:**
    -   **Target Encoding:** Convert categorical target variables (`label` for crops, `Fertilizer Name` for fertilizers) into numerical representations using dictionary mapping or `LabelEncoder`.
    -   **Feature Encoding:** Convert categorical input features (e.g., `Soil Type`, `Crop Type` in the fertilizer dataset) into numerical representations using `LabelEncoder`.
    -   **Train-Test Split:** Split the data into training (80%) and testing (20%) sets.
    -   **Feature Scaling:** Apply `StandardScaler` to numerical features on the training data and transform the test data.
3.  **Model Selection & Training:**
    -   Utilize `DecisionTreeClassifier` for both tasks.
    -   Train the model on the preprocessed training data.
4.  **Model Evaluation:**
    -   Evaluate model performance on the test set using metrics like accuracy and classification reports.
5.  **Model Persistence:**
    -   Save the trained models and fitted preprocessors (Scalers, LabelEncoders) using `pickle` to `.sav` files for later use.
6.  **Predictive System:**
    -   Develop Python functions to load the saved models/preprocessors and make predictions on new input data.
