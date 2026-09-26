import os
import json
import requests
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AQI Intelligence",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# CITY COORDINATES
# ============================================================

CITIES = {
    "Delhi": {
        "latitude": 28.6139,
        "longitude": 77.2090
    },

    "Mumbai": {
        "latitude": 19.0760,
        "longitude": 72.8777
    },

    "Bengaluru": {
        "latitude": 12.9716,
        "longitude": 77.5946
    },

    "Kolkata": {
        "latitude": 22.5726,
        "longitude": 88.3639
    },

    "Chennai": {
        "latitude": 13.0827,
        "longitude": 80.2707
    },

    "Hyderabad": {
        "latitude": 17.3850,
        "longitude": 78.4867
    }
}


# ============================================================
# MODEL PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "aqi_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# AQI CATEGORY
# ============================================================

def aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderately Polluted"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


# ============================================================
# CPCB BREAKPOINTS
# ============================================================

CPCB_BREAKPOINTS = {

    "pm25": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 1000, 401, 500)
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
        (1601, 10000, 401, 500)
    ],

    "o3": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 10000, 401, 500)
    ],

    "co": [
        (0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10, 101, 200),
        (10.1, 17, 201, 300),
        (17.1, 34, 301, 400),
        (34.1, 100, 401, 500)
    ]
}


# ============================================================
# SUB-INDEX CALCULATION
# ============================================================

def calculate_sub_index(
    concentration,
    pollutant
):

    if concentration is None:
        return None

    try:
        concentration = float(concentration)
    except:
        return None

    breakpoints = CPCB_BREAKPOINTS.get(
        pollutant
    )

    if not breakpoints:
        return None

    for (
        low_conc,
        high_conc,
        low_aqi,
        high_aqi
    ) in breakpoints:

        if (
            concentration >= low_conc
            and concentration <= high_conc
        ):

            aqi = (
                (
                    high_aqi - low_aqi
                )
                /
                (
                    high_conc - low_conc
                )
            ) * (
                concentration - low_conc
            ) + low_aqi

            return round(aqi)

    return 500


# ============================================================
# CPCB STYLE AQI
# ============================================================

def calculate_cpcb_aqi(data):

    sub_indices = {}

    # PM2.5
    if pd.notna(data.get("pm25")):

        sub_indices["pm25"] = calculate_sub_index(
            data["pm25"],
            "pm25"
        )

    # PM10
    if pd.notna(data.get("pm10")):

        sub_indices["pm10"] = calculate_sub_index(
            data["pm10"],
            "pm10"
        )

    # NO2
    if pd.notna(data.get("no2")):

        sub_indices["no2"] = calculate_sub_index(
            data["no2"],
            "no2"
        )

    # SO2
    if pd.notna(data.get("so2")):

        sub_indices["so2"] = calculate_sub_index(
            data["so2"],
            "so2"
        )

    # O3
    if pd.notna(data.get("o3")):

        sub_indices["o3"] = calculate_sub_index(
            data["o3"],
            "o3"
        )

    # CO
    if pd.notna(data.get("co")):

        # Open-Meteo CO is usually µg/m³.
        # CPCB breakpoint is mg/m³.
        co_mg = float(data["co"]) / 1000

        sub_indices["co"] = calculate_sub_index(
            co_mg,
            "co"
        )

    if not sub_indices:

        return None, {}

    overall_aqi = max(
        sub_indices.values()
    )

    return overall_aqi, sub_indices


# ============================================================
# FETCH AIR QUALITY DATA
# ============================================================

@st.cache_data(ttl=300)
def get_air_quality(
    latitude,
    longitude
):

    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
    )

    params = {

        "latitude": latitude,

        "longitude": longitude,

        "hourly": (
            "pm2_5,"
            "pm10,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone"
        ),

        "forecast_days": 2,

        "timezone": "auto"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# FETCH WEATHER DATA
# ============================================================

@st.cache_data(ttl=300)
def get_weather(
    latitude,
    longitude
):

    url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    params = {

        "latitude": latitude,

        "longitude": longitude,

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m"
        ),

        "forecast_days": 2,

        "timezone": "auto"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# CREATE AIR QUALITY DATAFRAME
# ============================================================

def create_dataframe(
    air_data,
    weather_data
):

    air_hourly = air_data["hourly"]

    weather_hourly = weather_data["hourly"]

    air_df = pd.DataFrame({

        "datetime":
            air_hourly["time"],

        "pm25":
            air_hourly["pm2_5"],

        "pm10":
            air_hourly["pm10"],

        "co":
            air_hourly["carbon_monoxide"],

        "no2":
            air_hourly["nitrogen_dioxide"],

        "so2":
            air_hourly["sulphur_dioxide"],

        "o3":
            air_hourly["ozone"]
    })

    weather_df = pd.DataFrame({

        "datetime":
            weather_hourly["time"],

        "temperature":
            weather_hourly["temperature_2m"],

        "humidity":
            weather_hourly[
                "relative_humidity_2m"
            ],

        "wind_speed":
            weather_hourly["wind_speed_10m"]
    })

    df = pd.merge(

        air_df,

        weather_df,

        on="datetime",

        how="left"
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    df["hour"] = (
        df["datetime"].dt.hour
    )

    df["day"] = (
        df["datetime"].dt.day
    )

    df["month"] = (
        df["datetime"].dt.month
    )

    return df


# ============================================================
# PREDICT AQI USING ML MODEL
# ============================================================

def predict_aqi(df):

    if model is None:

        return None

    # IMPORTANT:
    # The current trained model uses EXACTLY
    # these four features.

    prediction_data = df[
        [
            "pm25",
            "hour",
            "day",
            "month"
        ]
    ].copy()

    predictions = model.predict(
        prediction_data
    )

    predictions = predictions.clip(
        0,
        500
    )

    return predictions


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🎛️ Dashboard Controls"
)

city = st.sidebar.selectbox(
    "Select City",
    list(CITIES.keys())
)

coordinates = CITIES[city]

latitude = coordinates["latitude"]

longitude = coordinates["longitude"]


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌍 AQI Intelligence"
)

st.caption(
    "Real-Time Air Quality Monitoring & "
    "Machine Learning Based AQI Forecasting"
)

st.divider()


# ============================================================
# FETCH DATA
# ============================================================

try:

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
        f"Unable to fetch live data: {e}"
    )

    st.stop()


# ============================================================
# CURRENT DATA
# ============================================================

current = df.iloc[0]

current_aqi, current_sub_indices = (
    calculate_cpcb_aqi(current)
)

if current_aqi is None:

    current_aqi = calculate_sub_index(
        current["pm25"],
        "pm25"
    )

current_category = aqi_category(
    current_aqi
)


# ============================================================
# TOP METRICS
# ============================================================

st.subheader(
    f"📍 Current Air Quality — {city}"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "AQI",
        int(current_aqi)
    )

with col2:

    st.metric(
        "PM2.5",
        f"{current['pm25']:.1f} µg/m³"
    )

with col3:

    st.metric(
        "PM10",
        f"{current['pm10']:.1f} µg/m³"
    )

with col4:

    st.metric(
        "Temperature",
        f"{current['temperature']:.1f} °C"
    )


st.info(
    f"Current AQI category: **{current_category}**"
)


# ============================================================
# CURRENT POLLUTION LEVELS
# ============================================================

st.subheader(
    "🧪 Current Pollution Levels"
)

p1, p2, p3, p4 = st.columns(4)

with p1:

    st.metric(
        "NO₂",
        f"{current['no2']:.1f}"
    )

with p2:

    st.metric(
        "SO₂",
        f"{current['so2']:.1f}"
    )

with p3:

    st.metric(
        "CO",
        f"{current['co']:.1f}"
    )

with p4:

    st.metric(
        "O₃",
        f"{current['o3']:.1f}"
    )


st.divider()


# ============================================================
# ML AQI PREDICTION
# ============================================================

st.header(
    "🔮 AI-Based AQI Prediction"
)

if model is None:

    st.error(
        "AQI model not found. "
        "Please train the model first."
    )

else:

    try:

        # Predict future AQI
        predictions = predict_aqi(df)

        df["predicted_aqi"] = predictions

        # ----------------------------------------------------
        # Prediction cards
        # ----------------------------------------------------

        prediction_indices = {

            "Next 1 Hour": 1,

            "Next 3 Hours": 3,

            "Next 6 Hours": 6,

            "Next 12 Hours": 12,

            "Next 24 Hours": 24
        }

        cols = st.columns(5)

        for i, (
            label,
            index
        ) in enumerate(
            prediction_indices.items()
        ):

            index = min(
                index,
                len(df) - 1
            )

            predicted_value = round(
                float(
                    df.iloc[index][
                        "predicted_aqi"
                    ]
                )
            )

            with cols[i]:

                st.metric(
                    label,
                    predicted_value
                )

        # ----------------------------------------------------
        # Forecast statistics
        # ----------------------------------------------------

        forecast_24 = df.iloc[
            :25
        ].copy()

        average_aqi = round(
            forecast_24[
                "predicted_aqi"
            ].mean()
        )

        maximum_aqi = round(
            forecast_24[
                "predicted_aqi"
            ].max()
        )

        minimum_aqi = round(
            forecast_24[
                "predicted_aqi"
            ].min()
        )

        st.subheader(
            "📊 24-Hour Forecast Summary"
        )

        s1, s2, s3 = st.columns(3)

        with s1:

            st.metric(
                "24-Hour Average",
                average_aqi
            )

        with s2:

            st.metric(
                "Expected Maximum",
                maximum_aqi
            )

        with s3:

            st.metric(
                "Expected Minimum",
                minimum_aqi
            )

        # ----------------------------------------------------
        # AQI FORECAST GRAPH
        # ----------------------------------------------------

        st.subheader(
            "📈 AQI Prediction Trend"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(

                x=forecast_24[
                    "datetime"
                ],

                y=forecast_24[
                    "predicted_aqi"
                ],

                mode="lines+markers",

                name="Predicted AQI"
            )
        )

        fig.add_hline(
            y=50,
            line_dash="dash",
            annotation_text="Good"
        )

        fig.add_hline(
            y=100,
            line_dash="dash",
            annotation_text="Satisfactory"
        )

        fig.add_hline(
            y=200,
            line_dash="dash",
            annotation_text="Moderate"
        )

        fig.add_hline(
            y=300,
            line_dash="dash",
            annotation_text="Poor"
        )

        fig.update_layout(

            xaxis_title="Time",

            yaxis_title="Predicted AQI",

            height=450,

            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"ML prediction error: {e}"
        )

# ============================================================
# PREDICTION INTELLIGENCE & ALERTS
# ============================================================

st.header("🧠 Prediction Intelligence")

if "predicted_aqi" in df.columns:

    current_predicted = float(
        df.iloc[0]["predicted_aqi"]
    )

    future_24 = df.iloc[:25].copy()

    average_predicted = float(
        future_24["predicted_aqi"].mean()
    )

    max_index = future_24[
        "predicted_aqi"
    ].idxmax()

    max_predicted = float(
        future_24.loc[
            max_index,
            "predicted_aqi"
        ]
    )

    max_time = future_24.loc[
        max_index,
        "datetime"
    ]

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    first_value = float(
        future_24.iloc[0]["predicted_aqi"]
    )

    last_value = float(
        future_24.iloc[-1]["predicted_aqi"]
    )

    if last_value > first_value + 10:

        trend = "📈 Increasing"

        trend_message = (
            "The predicted AQI shows an increasing "
            "trend over the forecast period."
        )

    elif last_value < first_value - 10:

        trend = "📉 Decreasing"

        trend_message = (
            "The predicted AQI shows a decreasing "
            "trend over the forecast period."
        )

    else:

        trend = "➡️ Relatively Stable"

        trend_message = (
            "The predicted AQI remains relatively "
            "stable over the forecast period."
        )

    # --------------------------------------------------------
    # INSIGHT CARDS
    # --------------------------------------------------------

    i1, i2, i3 = st.columns(3)

    with i1:

        st.metric(
            "Current ML Prediction",
            round(current_predicted)
        )

    with i2:

        st.metric(
            "24-Hour Average",
            round(average_predicted)
        )

    with i3:

        st.metric(
            "Expected Maximum",
            round(max_predicted)
        )

    st.info(
        f"**Expected Trend:** {trend}\n\n"
        f"{trend_message}"
    )

    # --------------------------------------------------------
    # PM2.5 ANALYSIS
    # --------------------------------------------------------

    current_pm25 = float(
        df.iloc[0]["pm25"]
    )

    future_pm25 = float(
        future_24["pm25"].mean()
    )

    if future_pm25 > current_pm25 + 5:

        pm_message = (
            "PM2.5 is expected to increase during "
            "the forecast period, which may contribute "
            "to higher AQI."
        )

    elif future_pm25 < current_pm25 - 5:

        pm_message = (
            "PM2.5 is expected to decrease during "
            "the forecast period, which may support "
            "lower AQI."
        )

    else:

        pm_message = (
            "PM2.5 is expected to remain relatively "
            "stable during the forecast period."
        )

    st.write(
        f"🔎 **PM2.5 Insight:** {pm_message}"
    )

    # --------------------------------------------------------
    # PEAK AQI TIME
    # --------------------------------------------------------

    st.write(
        f"🕒 **Highest predicted AQI:** "
        f"{round(max_predicted)} at "
        f"{pd.to_datetime(max_time).strftime('%d %b, %H:%M')}"
    )

    # --------------------------------------------------------
    # AUTOMATIC POLLUTION ALERT
    # --------------------------------------------------------

    if max_predicted >= 300:

        st.error(
            "🚨 Severe Pollution Alert: "
            "The model predicts AQI may reach "
            f"{round(max_predicted)} during the next "
            "24 hours."
        )

    elif max_predicted >= 200:

        st.warning(
            "⚠️ High Pollution Alert: "
            "The model predicts AQI may exceed 200 "
            "during the next 24 hours."
        )

    elif max_predicted >= 100:

        st.warning(
            "⚠️ Moderate Pollution Alert: "
            "The forecast includes AQI values above 100."
        )

    else:

        st.success(
            "✅ No high-pollution AQI threshold is "
            "currently predicted for the next 24 hours."
        )

else:

    st.info(
        "Prediction intelligence will appear "
        "after ML forecasting is completed."
    )

# ============================================================
# POLLUTION INTELLIGENCE
# ============================================================

st.header(
    "🧠 Pollution Intelligence"
)

if current_sub_indices:

    pollutant_names = {

        "pm25": "PM2.5",

        "pm10": "PM10",

        "no2": "NO₂",

        "so2": "SO₂",

        "o3": "O₃",

        "co": "CO"
    }

    dominant_key = max(
        current_sub_indices,
        key=current_sub_indices.get
    )

    dominant_name = pollutant_names.get(
        dominant_key,
        dominant_key
    )

    dominant_value = current_sub_indices[
        dominant_key
    ]

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Dominant Pollutant",
            dominant_name
        )

    with c2:

        st.metric(
            "Pollutant AQI Sub-Index",
            dominant_value
        )

    st.write(
        f"**{dominant_name}** is currently "
        f"the largest contributor among the "
        f"available pollutant sub-indices."
    )

    # --------------------------------------------------------
    # SUB-INDEX TABLE
    # --------------------------------------------------------

    table_data = []

    for key, value in current_sub_indices.items():

        table_data.append({

            "Pollutant":
                pollutant_names.get(
                    key,
                    key
                ),

            "AQI Sub-Index":
                value
        })

    sub_index_df = pd.DataFrame(
        table_data
    )

    st.dataframe(
        sub_index_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PM2.5 TREND
# ============================================================

st.header(
    "🌫️ PM2.5 Trend"
)

pm25_df = df.iloc[
    :25
].copy()

fig_pm = go.Figure()

fig_pm.add_trace(
    go.Scatter(

        x=pm25_df[
            "datetime"
        ],

        y=pm25_df[
            "pm25"
        ],

        mode="lines+markers",

        name="PM2.5"
    )
)

fig_pm.update_layout(

    xaxis_title="Time",

    yaxis_title="PM2.5 (µg/m³)",

    height=400,

    hovermode="x unified"
)

st.plotly_chart(
    fig_pm,
    use_container_width=True
)


# ============================================================
# WEATHER
# ============================================================

st.header(
    "🌤️ Weather Conditions"
)

w1, w2, w3 = st.columns(3)

with w1:

    st.metric(
        "Temperature",
        f"{current['temperature']:.1f} °C"
    )

with w2:

    st.metric(
        "Humidity",
        f"{current['humidity']:.0f}%"
    )

with w3:

    st.metric(
        "Wind Speed",
        f"{current['wind_speed']:.1f} km/h"
    )

# ============================================================
# MULTI-CITY AQI COMPARISON
# ============================================================

st.header("🌆 Multi-City AQI Comparison")

st.caption(
    "Live AQI comparison across supported cities"
)


@st.cache_data(ttl=300)
def get_city_current_aqi(city_name):

    city_info = CITIES[city_name]

    try:

        air_data = get_air_quality(
            city_info["latitude"],
            city_info["longitude"]
        )

        weather_data = get_weather(
            city_info["latitude"],
            city_info["longitude"]
        )

        city_df = create_dataframe(
            air_data,
            weather_data
        )

        current_city = city_df.iloc[0]

        city_aqi, _ = calculate_cpcb_aqi(
            current_city
        )

        if city_aqi is None:

            city_aqi = calculate_sub_index(
                current_city["pm25"],
                "pm25"
            )

        return {
            "City": city_name,
            "AQI": round(float(city_aqi)),
            "PM2.5": round(
                float(current_city["pm25"]),
                1
            )
        }

    except Exception:

        return {
            "City": city_name,
            "AQI": None,
            "PM2.5": None
        }


# ------------------------------------------------------------
# FETCH ALL CITY DATA
# ------------------------------------------------------------

city_results = []

for city_name in CITIES.keys():

    result = get_city_current_aqi(
        city_name
    )

    city_results.append(result)


city_comparison_df = pd.DataFrame(
    city_results
)

# Remove cities where data couldn't be retrieved
city_comparison_df = (
    city_comparison_df
    .dropna(subset=["AQI"])
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# COMPARISON TABLE
# ------------------------------------------------------------

if not city_comparison_df.empty:

    display_df = city_comparison_df.copy()

    display_df["Category"] = (
        display_df["AQI"]
        .apply(aqi_category)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # BAR CHART
    # --------------------------------------------------------

    fig_city = go.Figure()

    fig_city.add_trace(
        go.Bar(

            x=city_comparison_df[
                "City"
            ],

            y=city_comparison_df[
                "AQI"
            ],

            text=city_comparison_df[
                "AQI"
            ],

            textposition="auto",

            name="Current AQI"
        )
    )

    fig_city.add_hline(
        y=50,
        line_dash="dash",
        annotation_text="Good"
    )

    fig_city.add_hline(
        y=100,
        line_dash="dash",
        annotation_text="Satisfactory"
    )

    fig_city.add_hline(
        y=200,
        line_dash="dash",
        annotation_text="Moderate"
    )

    fig_city.add_hline(
        y=300,
        line_dash="dash",
        annotation_text="Poor"
    )

    fig_city.update_layout(

        title="Current AQI by City",

        xaxis_title="City",

        yaxis_title="AQI",

        height=450,

        showlegend=False
    )

    st.plotly_chart(
        fig_city,
        use_container_width=True
    )

else:

    st.warning(
        "Unable to retrieve AQI data "
        "for the selected cities."
    )

# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader(
    "🤖 Machine Learning Model"
)

m1, m2 = st.columns(2)

with m1:

    st.write(
        "**Model:** Random Forest Regressor"
    )

    st.write(
        "**Training Dataset:** "
        "Real OpenAQ hourly data"
    )

with m2:

    st.write(
        "**Features:** PM2.5, Hour, Day, Month"
    )

    st.write(
        "**Target:** Project-level AQI proxy"
    )


# ============================================================
# DATA DISCLAIMER
# ============================================================

with st.expander(
    "ℹ️ About this dashboard"
):

    st.write(
        """
        This dashboard combines live air-quality data
        with a machine-learning model trained on
        OpenAQ historical measurements.

        The current ML target is a project-level AQI
        proxy derived primarily from PM2.5. Therefore,
        the ML forecast should not be interpreted as
        an officially validated CPCB AQI forecast.

        The CPCB-style AQI section uses pollutant
        breakpoint calculations for available live
        pollutants. Official CPCB AQI calculation also
        depends on pollutant-specific averaging periods
        and complete pollutant coverage.
        """
    )


# ============================================================
# LIVE DATA
# ============================================================

with st.expander(
    "🔎 View Live Data"
):

    st.dataframe(
        df.head(24),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.header("📊 Model Performance")

METRICS_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "model_metrics.json"
)

if os.path.exists(METRICS_PATH):

    try:

        with open(
            METRICS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            metrics = json.load(file)

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "MAE",
                metrics.get("MAE", "N/A")
            )

        with m2:

            st.metric(
                "RMSE",
                metrics.get("RMSE", "N/A")
            )

        with m3:

            st.metric(
                "R² Score",
                metrics.get("R2", "N/A")
            )

        st.write(
            "**Model:** Random Forest Regressor"
        )

        st.write(
            "**Training Dataset:** Real OpenAQ hourly data"
        )

        st.write(
            "**Features:** PM2.5, Hour, Day, Month"
        )

        st.caption(
            "Note: The current model predicts a "
            "PM2.5-derived project-level AQI proxy. "
            "The reported metrics should not be interpreted "
            "as official CPCB AQI forecasting accuracy."
        )

    except Exception as e:

        st.warning(
            f"Unable to load model metrics: {e}"
        )

else:

    st.warning(
        "Model metrics file not found."
    )