"""Load the insurance data and turn it into numbers a model can use."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]

# The two feature sets the project compares.
BASE_FEATURES = [
    "age",
    "bmi",
    "children",
    "is_smoker",
    "is_male",
    "region_northwest",
    "region_southeast",
    "region_southwest",
]
# Adds what insurers know from experience: smoking costs far more when the person
# is also obese, and medical costs rise faster as people get older.
RISK_FEATURES = BASE_FEATURES + ["smoker_obese", "age_squared"]


def load_data(path):
    """Read the CSV file and check that it has the columns the project needs."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Data file is missing columns: {missing}")
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


def add_features(df):
    """Turn text columns into 0 or 1 values and add the two risk features."""
    out = df.copy()
    out["is_smoker"] = (out["smoker"].str.lower() == "yes").astype(int)
    out["is_male"] = (out["sex"].str.lower() == "male").astype(int)
    # Northeast is the reference region, so it gets no column of its own.
    for region in ["northwest", "southeast", "southwest"]:
        out[f"region_{region}"] = (out["region"].str.lower() == region).astype(int)
    # A BMI of 30 or more is the standard cutoff for obesity.
    out["smoker_obese"] = out["is_smoker"] * (out["bmi"] >= 30).astype(int)
    out["age_squared"] = out["age"] ** 2
    return out


def split_features(df, feature_names):
    """Return the input columns (X) and the value to predict (y)."""
    return df[feature_names], df["charges"]
