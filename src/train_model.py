import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "real_aqi_data.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "aqi_model.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "pm25",
    "hour",
    "day",
    "month",
]

TARGET = "aqi"


# ============================================================
# LOAD AND CLEAN DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("LOADING REAL OPENAQ DATA")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\n\n"
            "Pehle real data download karo."
        )

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset loaded: {len(df)} rows")
    print(f"Available columns: {list(df.columns)}")

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Sort chronologically if datetime is available.
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
            utc=True,
        )

        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime")

    # Ensure all model columns contain numeric values.
    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Remove missing or invalid values.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_columns)

    # Basic valid-value filtering.
    df = df[
        (df["pm25"] >= 0)
        & (df["hour"].between(0, 23))
        & (df["day"].between(1, 31))
        & (df["month"].between(1, 12))
        & (df["aqi"].between(0, 500))
    ].copy()

    df = df.reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(
            "Training ke liye sufficient valid records nahi hain."
        )

    print(f"Valid records after cleaning: {len(df)}")

    return df


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    df = load_data()

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # First 80% for training, last 20% for testing.
    # This avoids randomly mixing earlier and later hours.
    split_index = int(len(df) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("\n" + "=" * 60)
    print("TRAINING INFORMATION")
    print("=" * 60)

    print(f"\nTotal records : {len(df)}")
    print(f"Training rows : {len(X_train)}")
    print(f"Testing rows  : {len(X_test)}")
    print(f"Model features: {FEATURES}")

    print("\nTraining Random Forest...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Model training completed.")

    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(f"\nMAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False,
    )

    print("\nFeature Importance:")
    print(importance_df.to_string(index=False))

    # ========================================================
    # SAVE MODEL
    # ========================================================

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("\n" + "=" * 60)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nModel location:\n{MODEL_PATH}")

    # ========================================================
    # SAVE EVALUATION METRICS
    # ========================================================

    metrics = {
        "model": "Random Forest Regressor",
        "dataset": "Real OpenAQ hourly PM2.5 data",
        "target": "PM2.5-derived AQI proxy",
        "features": FEATURES,
        "total_records": int(len(df)),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "split_method": "Chronological 80/20",
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "feature_importance": {
            row["Feature"]: round(
                float(row["Importance"]),
                6,
            )
            for _, row in importance_df.iterrows()
        },
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print("\nEvaluation metrics saved to:")
    print(METRICS_PATH)

    print("\nFeatures expected by model:")
    print(FEATURES)

    print("\nTraining complete.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    train_model()