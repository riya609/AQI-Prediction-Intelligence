import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


# -----------------------------
# PROJECT PATHS
# -----------------------------

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

data_path = os.path.join(
    project_root,
    "data",
    "aqi_historical.csv"
)

models_folder = os.path.join(
    project_root,
    "models"
)

os.makedirs(
    models_folder,
    exist_ok=True
)


# -----------------------------
# LOAD DATA
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(data_path)

print(f"Dataset loaded: {len(df)} records")


# -----------------------------
# FEATURES
# -----------------------------

features = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "temperature",
    "humidity",
    "wind_speed",
    "hour",
    "day",
    "month"
]

target = "aqi"


X = df[features]

y = df[target]


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# -----------------------------
# RANDOM FOREST MODEL
# -----------------------------

print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# -----------------------------
# PREDICTION
# -----------------------------

predictions = model.predict(X_test)


# -----------------------------
# MODEL EVALUATION
# -----------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# -----------------------------
# SAVE MODEL
# -----------------------------

model_path = os.path.join(
    models_folder,
    "aqi_model.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\n==============================")
print("MODEL SAVED SUCCESSFULLY")
print("==============================")

print(model_path)
