import os
import numpy as np
import pandas as pd

# Reproducible results
np.random.seed(42)

# Number of historical records
n = 5000

# Create hourly timestamps
dates = pd.date_range(
    start="2025-01-01",
    periods=n,
    freq="h"
)

# -----------------------------
# WEATHER FEATURES
# -----------------------------

temperature = np.random.uniform(8, 42, n)

humidity = np.random.uniform(25, 90, n)

wind_speed = np.random.uniform(1, 25, n)

# -----------------------------
# POLLUTANT FEATURES
# -----------------------------

pm25 = np.random.uniform(10, 180, n)

pm10 = pm25 * np.random.uniform(1.3, 2.5, n)

no2 = np.random.uniform(5, 100, n)

so2 = np.random.uniform(2, 60, n)

co = np.random.uniform(100, 1500, n)

o3 = np.random.uniform(10, 150, n)

# -----------------------------
# TIME FEATURES
# -----------------------------

hour = dates.hour

day = dates.day

month = dates.month

# -----------------------------
# AQI CALCULATION
# -----------------------------

# Simplified training target.
# A proper pollutant-specific AQI calculation
# will be integrated into the final version.

aqi = (
    pm25 * 2.2
    + pm10 * 0.35
    + no2 * 0.25
    + so2 * 0.15
    + co * 0.01
    + o3 * 0.20
)

# Add realistic variation
aqi += np.random.normal(0, 10, n)

# Keep AQI within a reasonable range
aqi = np.clip(aqi, 0, 500)

# -----------------------------
# CREATE DATAFRAME
# -----------------------------

df = pd.DataFrame({
    "datetime": dates,
    "pm25": pm25,
    "pm10": pm10,
    "no2": no2,
    "so2": so2,
    "co": co,
    "o3": o3,
    "temperature": temperature,
    "humidity": humidity,
    "wind_speed": wind_speed,
    "hour": hour,
    "day": day,
    "month": month,
    "aqi": aqi
})

# -----------------------------
# CREATE DATA FOLDER
# -----------------------------

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

data_folder = os.path.join(
    project_root,
    "data"
)

os.makedirs(data_folder, exist_ok=True)

# -----------------------------
# SAVE DATASET
# -----------------------------

file_path = os.path.join(
    data_folder,
    "aqi_historical.csv"
)

df.to_csv(
    file_path,
    index=False
)

print("Dataset generated successfully!")
print(f"Records: {len(df)}")
print(f"Saved at: {file_path}")

print("\nFirst 5 records:")
print(df.head())