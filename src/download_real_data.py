import os
import requests
import pandas as pd
from dotenv import load_dotenv

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENAQ_API_KEY not found in .env file"
    )

# ============================================================
# OPENAQ CONFIGURATION
# ============================================================

BASE_URL = "https://api.openaq.org/v3"

# New Delhi OpenAQ location
LOCATION_ID = 8118

HEADERS = {
    "X-API-Key": API_KEY
}


# ============================================================
# GET AVAILABLE SENSORS
# ============================================================

def get_sensors():

    url = f"{BASE_URL}/locations/{LOCATION_ID}/sensors"

    print("\nChecking OpenAQ sensors...")

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    results = response.json().get(
        "results",
        []
    )

    print("\nAvailable sensors:")

    selected = {}

    for sensor in results:

        parameter = sensor.get(
            "parameter",
            {}
        )

        sensor_id = sensor.get("id")
        name = parameter.get("name")
        units = parameter.get("units")

        print(
            f"Sensor ID: {sensor_id} | "
            f"Parameter: {name} | "
            f"Units: {units}"
        )

        if name:

            name = name.lower()

            if name in [
                "pm25",
                "pm10",
                "no2",
                "so2",
                "co",
                "o3"
            ]:

                selected[name] = sensor_id

    return selected


# ============================================================
# DOWNLOAD HOURLY DATA
# ============================================================

def get_hourly_data(
    sensor_id,
    pollutant
):

    url = (
        f"{BASE_URL}/sensors/"
        f"{sensor_id}/hours"
    )

    rows = []

    # --------------------------------------------------------
    # Historical date range
    # --------------------------------------------------------

    params = {

        "datetime_from":
            "2025-09-25T00:00:00Z",

        "datetime_to":
            "2026-09-25T23:59:59Z",

        "limit": 1000,

        "page": 1
    }

    MAX_PAGES = 7

    while True:

        print(
            f"  Downloading page "
            f"{params['page']}..."
        )

        # ----------------------------------------------------
        # API REQUEST
        # ----------------------------------------------------

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=60
            )

            response.raise_for_status()

        except requests.exceptions.Timeout:

            print(
                f"  Timeout on page "
                f"{params['page']}."
            )

            print(
                "  Keeping data downloaded "
                "so far."
            )

            break

        except requests.exceptions.RequestException as e:

            print(
                f"  API error on page "
                f"{params['page']}: {e}"
            )

            print(
                "  Keeping data downloaded "
                "so far."
            )

            break

        # ----------------------------------------------------
        # READ RESULTS
        # ----------------------------------------------------

        results = response.json().get(
            "results",
            []
        )

        if not results:

            print(
                "  No more records available."
            )

            break

        print(
            f"  Page {params['page']}: "
            f"{len(results)} records"
        )

        # ----------------------------------------------------
        # PROCESS RECORDS
        # ----------------------------------------------------

        for item in results:

            period = item.get(
                "period",
                {}
            )

            datetime_to = period.get(
                "datetimeTo",
                {}
            )

            timestamp = (
                datetime_to.get("local")
                or datetime_to.get("utc")
            )

            value = item.get("value")

            if (
                timestamp is not None
                and value is not None
            ):

                rows.append({

                    "datetime":
                        timestamp,

                    pollutant:
                        value
                })

        # ----------------------------------------------------
        # STOP CONDITIONS
        # ----------------------------------------------------

        if len(results) < 1000:

            print(
                "  Last page reached."
            )

            break

        if params["page"] >= MAX_PAGES:

            print(
                f"  Maximum page limit "
                f"({MAX_PAGES}) reached."
            )

            break

        params["page"] += 1

    return pd.DataFrame(rows)


# ============================================================
# AQI PROXY
# ============================================================

def calculate_aqi_proxy(row):

    values = []

    # PM2.5
    if pd.notna(row.get("pm25")):

        values.append(
            row["pm25"] * 4
        )

    # PM10
    if pd.notna(row.get("pm10")):

        values.append(
            row["pm10"] * 1.5
        )

    # NO2
    if pd.notna(row.get("no2")):

        values.append(
            row["no2"] * 1.2
        )

    # SO2
    if pd.notna(row.get("so2")):

        values.append(
            row["so2"] * 1.2
        )

    # CO
    if pd.notna(row.get("co")):

        values.append(
            row["co"] / 100
        )

    # O3
    if pd.notna(row.get("o3")):

        values.append(
            row["o3"] * 1.2
        )

    if not values:

        return None

    return round(
        min(
            max(
                max(values),
                0
            ),
            500
        ),
        2
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 60)

    print(
        "OPENAQ REAL AIR QUALITY "
        "DATA DOWNLOADER"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # FIND SENSORS
    # --------------------------------------------------------

    sensors = get_sensors()

    print("\nSelected sensors:")

    print(sensors)

    if not sensors:

        raise RuntimeError(
            "No required pollutant sensors found."
        )

    # --------------------------------------------------------
    # DOWNLOAD DATA
    # --------------------------------------------------------

    all_data = []

    for pollutant, sensor_id in sensors.items():

        print(
            f"\nDownloading "
            f"{pollutant.upper()} "
            f"(Sensor {sensor_id})..."
        )

        try:

            df = get_hourly_data(
                sensor_id,
                pollutant
            )

            if not df.empty:

                print(
                    f"Downloaded "
                    f"{len(df)} records."
                )

                all_data.append(df)

            else:

                print(
                    f"No data found for "
                    f"{pollutant.upper()}."
                )

        except Exception as e:

            print(
                f"Error downloading "
                f"{pollutant.upper()}: {e}"
            )

    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    if not all_data:

        raise RuntimeError(
            "No air-quality data was downloaded."
        )

    # --------------------------------------------------------
    # MERGE POLLUTANTS
    # --------------------------------------------------------

    final_df = all_data[0]

    for df in all_data[1:]:

        final_df = pd.merge(

            final_df,

            df,

            on="datetime",

            how="outer"
        )

    # --------------------------------------------------------
    # DATETIME PROCESSING
    # --------------------------------------------------------

    final_df["datetime"] = pd.to_datetime(
        final_df["datetime"]
    )

    final_df = final_df.sort_values(
        "datetime"
    )

    final_df = final_df.drop_duplicates(
        subset=["datetime"]
    )

    final_df = final_df.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # TIME FEATURES
    # --------------------------------------------------------

    final_df["hour"] = (
        final_df["datetime"].dt.hour
    )

    final_df["day"] = (
        final_df["datetime"].dt.day
    )

    final_df["month"] = (
        final_df["datetime"].dt.month
    )

    # --------------------------------------------------------
    # POLLUTANT COLUMNS
    # --------------------------------------------------------

    pollutants = [

        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
        "o3"
    ]

    existing = [

        pollutant

        for pollutant in pollutants

        if pollutant in final_df.columns
    ]

    # --------------------------------------------------------
    # REMOVE EMPTY ROWS
    # --------------------------------------------------------

    final_df = final_df.dropna(

        subset=existing,

        how="all"
    )

    # --------------------------------------------------------
    # FILL SMALL DATA GAPS
    # --------------------------------------------------------

    if existing:

        final_df[existing] = (

            final_df[existing]

            .interpolate(
                limit_direction="both"
            )
        )

    # --------------------------------------------------------
    # CALCULATE AQI PROXY
    # --------------------------------------------------------

    final_df["aqi"] = final_df.apply(

        calculate_aqi_proxy,

        axis=1
    )

    final_df = final_df.dropna(

        subset=["aqi"]
    )

    # --------------------------------------------------------
    # SAVE DATASET
    # --------------------------------------------------------

    project_root = os.path.dirname(

        os.path.dirname(

            os.path.abspath(__file__)
        )
    )

    data_folder = os.path.join(

        project_root,

        "data"
    )

    os.makedirs(

        data_folder,

        exist_ok=True
    )

    output_path = os.path.join(

        data_folder,

        "real_aqi_data.csv"
    )

    final_df.to_csv(

        output_path,

        index=False
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n")

    print("=" * 60)

    print(
        "REAL DATA DOWNLOAD COMPLETE"
    )

    print("=" * 60)

    print(
        f"\nTotal records: "
        f"{len(final_df)}"
    )

    print(
        f"\nColumns:"
    )

    print(
        list(final_df.columns)
    )

    print(
        f"\nSaved to:"
    )

    print(
        output_path
    )

    print(
        "\nFirst 5 records:"
    )

    print(
        final_df.head()
    )

    print(
        "\nDataset shape:"
    )

    print(
        final_df.shape
    )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()