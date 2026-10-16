"""Preprocessing shared by diabetes training and inference.

In the Pima dataset, zero values for several measured clinical features commonly
represent unavailable measurements rather than plausible physiological values.
They are treated as missing and imputed using medians learned only from training
folds by the surrounding sklearn Pipeline. This is dataset-specific and must be
revalidated before use with other populations or clinical workflows.
"""
import pandas as pd

DIABETES_ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def mark_diabetes_zeros_missing(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with zero sentinels marked missing for selected measurements."""
    result = frame.copy()
    columns = [column for column in DIABETES_ZERO_AS_MISSING if column in result.columns]
    if columns:
        result.loc[:, columns] = result.loc[:, columns].replace(0, float("nan"))
    return result
