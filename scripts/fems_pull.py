import os
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from requests.auth import HTTPBasicAuth

# ======================================================
# CONFIG
# ======================================================

ENDPOINT = "https://fems.fs2c.usda.gov/api/ext-climatology/graphql"
DATA_DIR = "data"

USERNAME = os.environ["FEMS_USERNAME"]
API_KEY  = os.environ["FEMS_API_KEY"]
AUTH     = HTTPBasicAuth(USERNAME, API_KEY)

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "FEMS-FULL-REPORT/1.0"
}

# ======================================================
# STATION LISTS
# ======================================================

nm_stations = set(pd.read_csv(os.path.join(DATA_DIR, "stations_nm.csv"), header=None)[0].astype(str))
az_stations = set(pd.read_csv(os.path.join(DATA_DIR, "stations_az.csv"), header=None)[0].astype(str))

station_ids = list(nm_stations | az_stations)
station_ids_csv = ",".join(station_ids)

# ======================================================
# TIME WINDOWS
# ======================================================

now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
last_hour_start = (now_utc - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
last_hour_end   = last_hour_start + timedelta(hours=1) - timedelta(seconds=1)

start_dt_iso = last_hour_start.strftime("%Y-%m-%dT%H:%M:%SZ")
end_dt_iso   = last_hour_end.strftime("%Y-%m-%dT%H:%M:%SZ")
today_str    = now_utc.strftime("%Y-%m-%d")

run_minmax = (now_utc.hour == 0)

# ======================================================
# FULL FEMS FIELD LISTS
# ======================================================

WEATHER_FIELDS = """
station_id station_name wrcc_id latitude longitude elevation station_type
observation_time observation_time_lst display_hour display_hour_lst
masked_observation_time display_date
temperature relative_humidity hourly_precip wind_speed wind_direction
peak_gust_speed peak_gust_dir sol_rad snow_flag observation_type
"""

NFDRS_FIELDS = """
station_name station_id wrcc_id latitude longitude elevation
observation_time observation_time_lst display_hour display_hour_lst
nfdr_date nfdr_time nfdr_type fuel_model fuel_model_version
kbdi one_hr_tl_fuel_moisture ten_hr_tl_fuel_moisture
hun_hr_tl_fuel_moisture thou_hr_tl_fuel_moisture
ignition_component spread_component energy_release_component burning_index
herbaceous_lfi_fuel_moisture woody_lfi_fuel_moisture gsi quality_code
"""

WXMIN_FIELDS = """
station_name station_id wrcc_id latitude longitude elevation
summary_date summary_date_lst observation_type
temperature_min temperature_max relative_humidity_min relative_humidity_max
wind_speed_min wind_speed_max
peak_gust_speed_min peak_gust_speed_max
peak_wind_gust_time peak_wind_gust_time_lst
solar_radiation_max daily_precipitation_total snow_flag
"""

NFDRMIN_FIELDS = """
station_id station_name summary_date summary_date_lst nfdr_type fuel_model
kbdi gsi
ignition_component_max ignition_component_max_time ignition_component_max_time_lst
spread_component_max spread_component_max_time spread_component_max_time_lst
energy_release_component_max energy_release_component_max_time energy_release_component_max_time_lst
burning_index_max burning_index_max_time burning_index_max_time_lst
one_hr_tl_fuel_moisture_min one_hr_tl_fuel_moisture_min_time one_hr_tl_fuel_moisture_min_time_lst
ten_hr_tl_fuel_moisture_min ten_hr_tl_fuel_moisture_min_time ten_hr_tl_fuel_moisture_min_time_lst
hun_hr_tl_fuel_moisture_min hun_hr_tl_fuel_moisture_min_time hun_hr_tl_fuel_moisture_min_time_lst
thou_hr_tl_fuel_moisture_min thou_hr_tl_fuel_moisture_min_time thou_hr_tl_fuel_moisture_min_time_lst
herbaceous_lfi_fuel_moisture woody_lfi_fuel_moisture
latitude longitude elevation
"""

# ======================================================
# GRAPHQL HANDLER
# ======================================================

def gql(query, vars):
    r = requests.post(
        ENDPOINT,
        headers=HEADERS,
        auth=AUTH,
        json={"query": query, "variables": vars},
        timeout=90
    )
    print(f"[HTTP {r.status_code}]")
    j = r.json()
    if "errors" in j:
        raise RuntimeError(j["errors"])
    return j["data"]

# ======================================================
# WEATHER OBSERVATIONS
# ======================================================

Q_WEATHER_OBS = f"""
query WeatherObs($start: DateTime!, $end: DateTime!, $stations: String) {{
    weatherObs(
        startDateTimeRange: $start,
        endDateTimeRange: $end,
        stationIds: $stations,
        hasHistoricData: ALL
    ) {{
        data {{
            {WEATHER_FIELDS}
        }}
    }}
}}
"""

wx_data = gql(Q_WEATHER_OBS, {
    "start": start_dt_iso,
    "end": end_dt_iso,
    "stations": station_ids_csv
})["weatherObs"]["data"]

df_wx = pd.DataFrame(wx_data)

# ======================================================
# NFDRS OBS
# ======================================================

Q_NFDRS_OBS = f"""
query NfdrsObs($start: Date, $end: Date, $stations: String) {{
    nfdrsObs(
        startDateRange: $start,
        endDateRange: $end,
        stationIds: $stations,
        dateTimeFormat: UTC,
        hasHistoricData: ALL
    ) {{
        data {{
            {NFDRS_FIELDS}
        }}
    }}
}}
"""

nfdrs_data = gql(Q_NFDRS_OBS, {
    "start": today_str,
    "end": today_str,
    "stations": station_ids_csv
})["nfdrsObs"]["data"]

df_nfdrs = pd.DataFrame(nfdrs_data)

# ======================================================
# WX MINMAX
# ======================================================

if run_minmax:
    Q_WX_MINMAX = f"""
    query WxMinMax($start: Date!, $end: Date!, $stations: String) {{
        wxMinMax(
            startDate: $start,
            endDate: $end,
            stationIds: $stations,
            hasHistoricData: ALL
        ) {{
            data {{
                {WXMIN_FIELDS}
            }}
        }}
    }}
    """

    wxmin_data = gql(Q_WX_MINMAX, {
        "start": today_str,
        "end": today_str,
        "stations": station_ids_csv
    })["wxMinMax"]["data"]

    df_wxmin = pd.DataFrame(wxmin_data)

else:
    df_wxmin = pd.DataFrame()

# ======================================================
# NFDR MINMAX
# ======================================================

if run_minmax:
    NFDRMIN_FRAMES = []

    for fm in ["V","W","X","Y","Z"]:
        Q_NFDR_MINMAX = f"""
        query NfdrMinMax($start: Date!, $end: Date!, $stations: String, $fm: String) {{
            nfdrMinMax(
                startDate: $start,
                endDate: $end,
                stationIds: $stations,
                fuelModels: $fm,
                hasHistoricData: ALL
            ) {{
                data {{
                    {NFDRMIN_FIELDS}
                }}
            }}
        }}
        """

        nfdrmin_data = gql(Q_NFDR_MINMAX, {
            "start": today_str,
            "end": today_str,
            "stations": station_ids_csv,
            "fm": fm
        })["nfdrMinMax"]["data"]

        NFDRMIN_FRAMES.append(pd.DataFrame(nfdrmin_data))

    df_nfdrmin = pd.concat(NFDRMIN_FRAMES, ignore_index=True)

else:
    df_nfdrmin = pd.DataFrame()

# ======================================================
# SAVE OUTPUT
# ======================================================

os.makedirs(DATA_DIR, exist_ok=True)

df_wx.to_excel(os.path.join(DATA_DIR,"fems_latest_weatherObs.xlsx"), index=False)
df_nfdrs.to_excel(os.path.join(DATA_DIR,"fems_latest_nfdrsObs.xlsx"), index=False)
df_wxmin.to_excel(os.path.join(DATA_DIR,"fems_latest_wxMinMax.xlsx"), index=False)
df_nfdrmin.to_excel(os.path.join(DATA_DIR,"fems_latest_nfdrMinMax.xlsx"), index=False)

with pd.ExcelWriter(os.path.join(DATA_DIR,"fems_full_data.xlsx")) as x:
    df_wx.to_excel(x, "WeatherObs", index=False)
    df_nfdrs.to_excel(x, "NFDRSObs", index=False)
    df_wxmin.to_excel(x, "WxMinMax", index=False)
    df_nfdrmin.to_excel(x, "NFDRMinMax", index=False)

print("FULL FEMS REPORTS EXPORTED EXACTLY AS FEMS SENDS THEM.")
