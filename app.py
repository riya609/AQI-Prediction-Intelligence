import os
import requests
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================================================
# CPCB AQI CALCULATION
# =========================================================

CPCB_BREAKPOINTS = {
    "pm25": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500)
    ],

    "pm10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 1000, 401, 500)
    ],

    "no2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 1000, 401, 500)
    ],

    "so2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 3000, 401, 500)
    ],

    "o3": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500)
    ],

    # CO is in mg/m³ for CPCB breakpoints
    "co": [
        (0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10, 101, 200),
        (10.1, 17, 201, 300),
        (17.1, 34, 301, 400),
        (34.1, 50, 401, 500)
    ]
}


def calculate_sub_index(concentration, pollutant):

    if pd.isna(concentration):
        return None

    concentration = float(concentration)

    breakpoints = CPCB_BREAKPOINTS[pollutant]

    for c_low, c_high, i_low, i_high in breakpoints:

        if c_low <= concentration <= c_high:

            sub_index = (
                ((i_high - i_low) / (c_high - c_low))
                * (concentration - c_low)
                + i_low
            )

            return round(sub_index)

    return 500


def calculate_cpcb_aqi(row):

    values = {
        "pm25": row["pm25"],
        "pm10": row["pm10"],
        "no2": row["no2"],
        "so2": row["so2"],
        "o3": row["o3"],

        # Open-Meteo gives CO in µg/m³.
        # CPCB breakpoint is mg/m³.
        "co": row["co"] / 1000
    }

    sub_indices = {}

    for pollutant, value in values.items():

        index = calculate_sub_index(
            value,
            pollutant
        )

        if index is not None:
            sub_indices[pollutant] = index

    # CPCB requires at least 3 pollutants
    # including PM2.5 or PM10.
    particulate_available = (
        "pm25" in sub_indices or
        "pm10" in sub_indices
    )

    if len(sub_indices) < 3 or not particulate_available:
        return None, sub_indices

    overall_aqi = max(
        sub_indices.values()
    )

    return overall_aqi, sub_indices

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AQI Intelligence",
    page_icon="🌍",
    layout="wide"
)


# =========================================================
# CITY COORDINATES
# =========================================================

cities = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867)
}


# =========================================================
# LOAD ML MODEL
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "aqi_model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception:
    model = None
    model_loaded = False


# =========================================================
# AQI CATEGORY
# =========================================================

def aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    elif aqi <= 200:
        return "Unhealthy"

    elif aqi <= 300:
        return "Very Unhealthy"

    else:
        return "Hazardous"


# =========================================================
# AQI COLOR
# =========================================================

def aqi_color(aqi):

    if aqi <= 50:
        return "🟢"

    elif aqi <= 100:
        return "🟡"

    elif aqi <= 150:
        return "🟠"

    elif aqi <= 200:
        return "🔴"

    elif aqi <= 300:
        return "🟣"

    else:
        return "⚫"


# =========================================================
# FETCH AIR QUALITY
# =========================================================

def get_air_quality(latitude, longitude):

    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&hourly=pm2_5,pm10,carbon_monoxide,"
        "nitrogen_dioxide,sulphur_dioxide,ozone"
        "&forecast_days=2"
        "&timezone=auto"
    )

    response = requests.get(url, timeout=15)

    response.raise_for_status()

    return response.json()


# =========================================================
# FETCH WEATHER
# =========================================================

def get_weather(latitude, longitude):

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&hourly=temperature_2m,"
        "relative_humidity_2m,"
        "wind_speed_10m"
        "&forecast_days=2"
        "&timezone=auto"
    )

    response = requests.get(url, timeout=15)

    response.raise_for_status()

    return response.json()


# =========================================================
# CREATE DATAFRAME
# =========================================================

def create_dataframe(air_data, weather_data):

    air = air_data["hourly"]
    weather = weather_data["hourly"]

    air_df = pd.DataFrame({
        "datetime": pd.to_datetime(air["time"]),
        "pm25": air["pm2_5"],
        "pm10": air["pm10"],
        "co": air["carbon_monoxide"],
        "no2": air["nitrogen_dioxide"],
        "so2": air["sulphur_dioxide"],
        "o3": air["ozone"]
    })

    weather_df = pd.DataFrame({
        "datetime": pd.to_datetime(weather["time"]),
        "temperature": weather["temperature_2m"],
        "humidity": weather["relative_humidity_2m"],
        "wind_speed": weather["wind_speed_10m"]
    })

    df = air_df.merge(
        weather_df,
        on="datetime",
        how="left"
    )

    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month

    return df


# =========================================================
# MAIN UI
# =========================================================

st.title("🌍 Real-Time AQI Prediction & Pollution Intelligence")

st.markdown(
    "### Live monitoring + Machine Learning based AQI prediction"
)

st.sidebar.header("🎛️ Dashboard Controls")

selected_city = st.sidebar.selectbox(
    "Select City",
    list(cities.keys())
)

latitude, longitude = cities[selected_city]


# =========================================================
# FETCH DATA
# =========================================================

try:

    with st.spinner("Fetching live air-quality and weather data..."):

        air_data = get_air_quality(
            latitude,
            longitude
        )

        weather_data = get_weather(
            latitude,
            longitude
        )

        df = create_dataframe(
            air_data,
            weather_data
        )

except Exception as e:

    st.error(
        "Unable to fetch live data. Please check your internet connection."
    )

    st.stop()


# =========================================================
# CURRENT DATA
# =========================================================

current = df.iloc[0]

current_pm25 = float(current["pm25"])
current_pm10 = float(current["pm10"])

# CPCB-based AQI
current_aqi, current_sub_indices = calculate_cpcb_aqi(
    current
)

if current_aqi is None:
    # Fallback only when enough pollutant data
    # is not available.
    current_aqi = round(current_pm25 * 4)

category = aqi_category(current_aqi)

icon = aqi_color(current_aqi)


# =========================================================
# CITY HEADER
# =========================================================

st.subheader(f"📍 {selected_city}")

st.caption(
    f"Coordinates: {latitude:.4f}, {longitude:.4f}"
)


# =========================================================
# CURRENT METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Current AQI",
    current_aqi
)

col2.metric(
    "PM2.5",
    f"{current_pm25:.1f} µg/m³"
)

col3.metric(
    "PM10",
    f"{current_pm10:.1f} µg/m³"
)

col4.metric(
    "Temperature",
    f"{current['temperature']:.1f} °C"
)

st.info(
    f"{icon} Current AQI Category: **{category}**"
)


# =========================================================
# POLLUTANTS
# =========================================================

st.subheader("🧪 Current Pollution Levels")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "NO₂",
    f"{current['no2']:.1f}"
)

col2.metric(
    "SO₂",
    f"{current['so2']:.1f}"
)

col3.metric(
    "CO",
    f"{current['co']:.1f}"
)

col4.metric(
    "O₃",
    f"{current['o3']:.1f}"
)


# =========================================================
# ML PREDICTION
# =========================================================

st.divider()

st.header("🔮 AI-Based AQI Prediction")

if not model_loaded:

    st.error(
        "ML model not found. Please make sure models/aqi_model.pkl exists."
    )

else:

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

    prediction_data = df[features].copy()

    predictions = model.predict(
        prediction_data
    )

    predictions = pd.Series(
        predictions
    ).clip(
        lower=0,
        upper=500
    )

    df["predicted_aqi"] = predictions

    # First 24 hours
    forecast_df = df.head(24).copy()

    # -----------------------------------------------------
    # PREDICTION CARDS
    # -----------------------------------------------------

    p1 = round(float(forecast_df.iloc[1]["predicted_aqi"]))
    p3 = round(float(forecast_df.iloc[3]["predicted_aqi"]))
    p6 = round(float(forecast_df.iloc[6]["predicted_aqi"]))
    p12 = round(float(forecast_df.iloc[12]["predicted_aqi"]))
    p24 = round(float(forecast_df.iloc[23]["predicted_aqi"]))

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Next 1 Hour",
        p1
    )

    col2.metric(
        "Next 3 Hours",
        p3
    )

    col3.metric(
        "Next 6 Hours",
        p6
    )

    col4.metric(
        "Next 12 Hours",
        p12
    )

    col5.metric(
        "Next 24 Hours",
        p24
    )

    # -----------------------------------------------------
    # PREDICTION GRAPH
    # -----------------------------------------------------

    st.subheader("📈 24-Hour AQI Forecast")

    fig_prediction = go.Figure()

    fig_prediction.add_trace(
        go.Scatter(
            x=forecast_df["datetime"],
            y=forecast_df["predicted_aqi"],
            mode="lines+markers",
            name="AI Predicted AQI"
        )
    )

    fig_prediction.add_hline(
        y=100,
        line_dash="dash",
        annotation_text="Moderate Limit"
    )

    fig_prediction.add_hline(
        y=150,
        line_dash="dash",
        annotation_text="Unhealthy for Sensitive Groups"
    )

    fig_prediction.add_hline(
        y=200,
        line_dash="dash",
        annotation_text="Unhealthy"
    )

    fig_prediction.update_layout(
        xaxis_title="Time",
        yaxis_title="Predicted AQI",
        height=450,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_prediction,
        use_container_width=True
    )

    # -----------------------------------------------------
    # PREDICTION SUMMARY
    # -----------------------------------------------------

    average_prediction = round(
        forecast_df["predicted_aqi"].mean()
    )

    maximum_prediction = round(
        forecast_df["predicted_aqi"].max()
    )

    minimum_prediction = round(
        forecast_df["predicted_aqi"].min()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "24-Hour Average",
        average_prediction
    )

    col2.metric(
        "Expected Maximum",
        maximum_prediction
    )

    col3.metric(
        "Expected Minimum",
        minimum_prediction
    )


# =========================================================
# POLLUTION INTELLIGENCE
# =========================================================

st.divider()

st.header("🧠 Pollution Intelligence")

pollutant_names = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2": "NO₂",
    "so2": "SO₂",
    "co": "CO",
    "o3": "O₃"
}

# Find pollutant with highest AQI sub-index
if current_sub_indices:

    dominant_key = max(
        current_sub_indices,
        key=current_sub_indices.get
    )

    dominant_pollutant = pollutant_names.get(
        dominant_key,
        dominant_key
    )

    dominant_subindex = current_sub_indices[
        dominant_key
    ]

else:

    dominant_pollutant = "Unavailable"
    dominant_subindex = 0


col1, col2 = st.columns(2)


# ---------------------------------------------------------
# DOMINANT POLLUTANT
# ---------------------------------------------------------

with col1:

    st.subheader("🔬 Dominant Pollutant")

    st.metric(
        "Pollutant",
        dominant_pollutant
    )

    st.metric(
        "AQI Sub-Index",
        dominant_subindex
    )

    st.write(
        f"**{dominant_pollutant}** is currently contributing "
        f"the highest pollutant sub-index of **{dominant_subindex}**."
    )


# ---------------------------------------------------------
# POLLUTION STATUS
# ---------------------------------------------------------

with col2:

    st.subheader("⚠️ Pollution Status")

    if current_aqi <= 50:

        st.success(
            "Air quality is currently Good."
        )

    elif current_aqi <= 100:

        st.info(
            "Air quality is currently Moderate."
        )

    elif current_aqi <= 150:

        st.warning(
            "Sensitive groups should consider reducing "
            "prolonged outdoor exposure."
        )

    elif current_aqi <= 200:

        st.error(
            "Air quality is currently Unhealthy."
        )

    elif current_aqi <= 300:

        st.error(
            "Air quality is Very Unhealthy."
        )

    else:

        st.error(
            "Air quality is Hazardous."
        )


# ---------------------------------------------------------
# POLLUTANT SUB-INDEX TABLE
# ---------------------------------------------------------

st.subheader("📊 Pollutant-wise AQI Contribution")

if current_sub_indices:

    contribution_data = []

    for key, value in current_sub_indices.items():

        contribution_data.append({
            "Pollutant": pollutant_names.get(
                key,
                key
            ),
            "AQI Sub-Index": value
        })

    contribution_df = pd.DataFrame(
        contribution_data
    )

    contribution_df = contribution_df.sort_values(
        "AQI Sub-Index",
        ascending=False
    )

    st.dataframe(
        contribution_df,
        use_container_width=True,
        hide_index=True
    )

    
# =========================================================
# AQI TREND
# =========================================================

st.divider()

st.subheader("📊 AQI Prediction Trend")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["datetime"],
        y=df["predicted_aqi"],
        mode="lines",
        name="Predicted AQI"
    )
)

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="AQI",
    height=400
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# PM2.5 TREND
# =========================================================

st.subheader("🌫️ PM2.5 Trend")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=df["datetime"],
        y=df["pm25"],
        mode="lines",
        name="PM2.5"
    )
)

fig2.update_layout(
    xaxis_title="Time",
    yaxis_title="PM2.5 (µg/m³)",
    height=350
)

st.plotly_chart(
    fig2,
    use_container_width=True
)


# =========================================================
# WEATHER
# =========================================================

st.subheader("🌦️ Weather Conditions")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Temperature",
    f"{current['temperature']:.1f} °C"
)

col2.metric(
    "Humidity",
    f"{current['humidity']:.0f}%"
)

col3.metric(
    "Wind Speed",
    f"{current['wind_speed']:.1f} km/h"
)


# =========================================================
# LIVE DATA
# =========================================================

with st.expander("📋 View Live Data"):

    st.dataframe(
        df,
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AQI Intelligence | Real-Time Air Quality Monitoring & ML Prediction"
)