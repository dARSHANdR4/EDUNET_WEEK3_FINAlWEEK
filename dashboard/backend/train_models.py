"""Train inference pipelines and produce reproducible validation reports.

The hold-out split and stratified cross-validation are dataset-level estimates,
not evidence of real-world/field/clinical accuracy. Preprocessing stays inside
each saved pipeline so inference uses the fitted transforms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"

CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FERT_NUMERIC = ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
FERT_CATEGORICAL = ["Soil Type", "Crop Type"]
FERT_FEATURES = FERT_NUMERIC + FERT_CATEGORICAL
DIABETES_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]


def dataset_diagnostics(name: str, X: pd.DataFrame) -> dict:
    diagnostics = {"rows": int(len(X)), "features": int(X.shape[1]), "missing_values_by_feature": {}}
    diagnostics["missing_values_by_feature"] = {
        str(k): int(v) for k, v in X.isna().sum().items() if int(v) > 0
    }
    # In the Pima dataset, zero values for these measurements can mean missing
    # measurements. Report them explicitly; don't silently change clinical data.
    if name == "diabetes":
        zero_suspect = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        diagnostics["zero_values_in_measurements"] = {
            col: int((pd.to_numeric(X[col], errors="coerce") == 0).sum())
            for col in zero_suspect
        }
    return diagnostics


def evaluate_and_fit(
    name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, output: Path
) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    evaluation_pipeline = clone(pipeline)
    evaluation_pipeline.fit(X_train, y_train)
    y_pred = evaluation_pipeline.predict(X_test)
    labels = sorted(pd.Series(y).unique().tolist())

    # Stratified CV provides a less split-sensitive estimate. Scores remain
    # internal-dataset estimates and must not be presented as field accuracy.
    min_class_count = int(pd.Series(y).value_counts().min())
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        raise ValueError(f"{name}: each target class needs at least two rows for stratified CV")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
    }
    if name == "diabetes":
        # Class 1 is the positive (diabetes) label in this dataset.
        from sklearn.metrics import make_scorer
        scoring["sensitivity"] = make_scorer(recall_score, pos_label=1, zero_division=0)
        scoring["specificity"] = make_scorer(recall_score, pos_label=0, zero_division=0)
        scoring["roc_auc"] = "roc_auc"

    cv_scores = cross_validate(
        clone(pipeline),
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise",
    )

    binary_diagnostics = None
    if name == "diabetes":
        # Explicitly calculate clinically relevant screening metrics. These are
        # dataset metrics only, not evidence of clinical safety.
        matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(v) for v in matrix.ravel())
        decision_scores = evaluation_pipeline.decision_function(X_test)
        binary_diagnostics = {
            "positive_label": 1,
            "negative_label": 0,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "sensitivity_recall_positive_class": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
            "specificity_recall_negative_class": float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, decision_scores)),
        }

    report = {
        "model": name,
        "evaluation_scope": "internal dataset only; not independent real-world or clinical validation",
        "dataset": dataset_diagnostics(name, X),
        "split": {"method": "stratified hold-out", "test_size": 0.2, "random_state": 42,
                  "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "holdout_metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "classification_report": classification_report(
                y_test, y_pred, labels=labels, output_dict=True, zero_division=0
            ),
            "confusion_matrix_labels": [str(label) for label in labels],
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        },
        "diabetes_binary_diagnostics": binary_diagnostics,
        "stratified_cross_validation": {
            "folds": n_splits,
            "accuracy_mean": float(cv_scores["test_accuracy"].mean()),
            "accuracy_std": float(cv_scores["test_accuracy"].std()),
            "balanced_accuracy_mean": float(cv_scores["test_balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(cv_scores["test_balanced_accuracy"].std()),
            "macro_f1_mean": float(cv_scores["test_macro_f1"].mean()),
            "macro_f1_std": float(cv_scores["test_macro_f1"].std()),
            **({
                "sensitivity_mean": float(cv_scores["test_sensitivity"].mean()),
                "sensitivity_std": float(cv_scores["test_sensitivity"].std()),
                "specificity_mean": float(cv_scores["test_specificity"].mean()),
                "specificity_std": float(cv_scores["test_specificity"].std()),
                "roc_auc_mean": float(cv_scores["test_roc_auc"].mean()),
                "roc_auc_std": float(cv_scores["test_roc_auc"].std()),
            } if name == "diabetes" else {}),
        },
    }

    # Refit the deployment pipeline on all labeled rows only after evaluation.
    pipeline.fit(X, y)
    joblib.dump(pipeline, output)
    print(
        f"{name}: holdout accuracy={report['holdout_metrics']['accuracy']:.4f}; "
        f"holdout macro-F1={report['holdout_metrics']['macro_f1']:.4f}; "
        f"{n_splits}-fold CV macro-F1={report['stratified_cross_validation']['macro_f1_mean']:.4f}; "
        f"rows={len(X)}; saved={output}"
    )
    return report


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
    df.columns = [str(c).strip() for c in df.columns]
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
    crop = Pipeline([("model", RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
    ))])
    reports = [evaluate_and_fit("crop", crop, X, y, args.model_dir / "crop.joblib")]

    X, y = load_fertilizer(args.fertilizer_data)
    prep = ColumnTransformer([
        ("numeric", "passthrough", FERT_NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), FERT_CATEGORICAL),
    ])
    fertilizer = Pipeline([
        ("preprocess", prep),
        ("model", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
        )),
    ])
    reports.append(evaluate_and_fit(
        "fertilizer", fertilizer, X, y, args.model_dir / "fertilizer.joblib"
    ))

    X, y = load_diabetes(args.diabetes_data)
    diabetes = Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="linear"))])
    reports.append(evaluate_and_fit(
        "diabetes", diabetes, X, y, args.model_dir / "diabetes.joblib"
    ))

    report_path = args.model_dir / "evaluation_report.json"
    report_path.write_text(json.dumps({
        "notice": (
            "Metrics are estimates on the supplied datasets only. They do not establish "
            "real-world field performance or clinical validity."
        ),
        "models": reports,
    }, indent=2), encoding="utf-8")
    print(f"Evaluation report saved to {report_path}")


if __name__ == "__main__":
    main()
