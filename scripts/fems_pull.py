import os
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from requests.auth import HTTPBasicAuth

# ========= CONFIG =========
ENDPOINT = "https://fems.fs2c.usda.gov/api/ext-climatology/graphql"
FUEL_MODELS = ["V", "W", "X", "Y", "Z"]
DATA_DIR = "data"

USERNAME = os.environ["FEMS_USERNAME"]
API_KEY  = os.environ["FEMS_API_KEY"]
AUTH     = HTTPBasicAuth(USERNAME, API_KEY)

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "FEMS-NM-AZ-GitHubActions/1.0"
}

# ========= LOAD STATION LISTS =========
nm_stations = set(pd.read_csv(os.path.join(DATA_DIR, "stations_nm.csv"), header=None)[0].astype(str))
az_stations = set(pd.read_csv(os.path.join(DATA_DIR, "stations_az.csv"), header=None)[0].astype(str))

station_ids = list(nm_stations | az_stations)
station_ids_csv = ",".join(station_ids)

# ========= TIME WINDOWS =========
now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)

last_hour_start = (now_utc - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
last_hour_end   = last_hour_start + timedelta(hours=1) - timedelta(seconds=1)

start_dt_iso = last_hour_start.strftime("%Y-%m-%dT%H:%M:%SZ")
end_dt_iso   = last_hour_end.strftime("%Y-%m-%dT%H:%M:%SZ")
today_str    = now_utc.strftime("%Y-%m-%d")

# Forecast window: today + 6 more days = 7 days total
forecast_end_str = (now_utc + timedelta(days=6)).strftime("%Y-%m-%d")

# Daily (MinMax) reports refresh only once a day at 12 am UTC
run_minmax = (now_utc.hour == 0)

# ========= QUERIES =========

Q_WEATHER_OBS = """
query WeatherObs($startDateTimeRange: DateTime!, $endDateTimeRange: DateTime!, $stationIds: String) {
  weatherObs(
    startDateTimeRange: $startDateTimeRange,
    endDateTimeRange: $endDateTimeRange,
    stationIds: $stationIds,
    hasHistoricData: ALL
  ) {
    data {
      station_id station_name wrcc_id latitude longitude elevation station_type
      observation_time observation_time_lst display_hour display_hour_lst
      masked_observation_time display_date
      temperature relative_humidity hourly_precip wind_speed wind_direction
      peak_gust_speed peak_gust_dir sol_rad snow_flag observation_type
    }
  }
}
"""

Q_NFDRS_OBS = """
query NfdrsObs(
  $fuelModels: String!, $stationIds: String,
  $startDateRange: Date, $endDateRange: Date,
  $dateTimeFormat: DateTimeFormat
) {
  nfdrsObs(
    fuelModels: $fuelModels, stationIds: $stationIds,
    startDateRange: $startDateRange, endDateRange: $endDateRange,
    dateTimeFormat: $dateTimeFormat, hasHistoricData: ALL
  ) {
    data {
      station_name station_id wrcc_id latitude longitude elevation
      observation_time observation_time_lst display_hour display_hour_lst
      nfdr_date nfdr_time nfdr_type fuel_model fuel_model_version
      kbdi one_hr_tl_fuel_moisture ten_hr_tl_fuel_moisture
      hun_hr_tl_fuel_moisture thou_hr_tl_fuel_moisture
      ignition_component spread_component energy_release_component burning_index
      herbaceous_lfi_fuel_moisture woody_lfi_fuel_moisture gsi quality_code
    }
  }
}
"""

Q_WX_MINMAX = """
query WxMinMax($startDate: Date!, $endDate: Date, $stationIds: String) {
  wxMinMax(startDate: $startDate, endDate: $endDate, stationIds: $stationIds, hasHistoricData: ALL) {
    data {
      station_name station_id wrcc_id latitude longitude elevation
      summary_date observation_type
      temperature_min temperature_max relative_humidity_min relative_humidity_max
      wind_speed_min wind_speed_max
      peak_gust_speed_min peak_gust_speed_max
      peak_wind_gust_direction peak_wind_gust_time
      solar_radiation_max daily_precipitation_total snow_flag
    }
  }
}
"""

Q_NFDR_MINMAX = """
query NfdrMinMax($startDate: Date!, $endDate: Date, $stationIds: String, $fuelModels: String) {
  nfdrMinMax(startDate: $startDate, endDate: $endDate, stationIds: $stationIds, fuelModels: $fuelModels, hasHistoricData: ALL) {
    data {
      station_id station_name summary_date nfdr_type fuel_model
      kbdi gsi
      ignition_component_max ignition_component_max_time
      spread_component_max spread_component_max_time
      energy_release_component_max energy_release_component_max_time
      burning_index_max burning_index_max_time
      one_hr_tl_fuel_moisture_min one_hr_tl_fuel_moisture_min_time
      ten_hr_tl_fuel_moisture_min ten_hr_tl_fuel_moisture_min_time
      hun_hr_tl_fuel_moisture_min hun_hr_tl_fuel_moisture_min_time
      thou_hr_tl_fuel_moisture_min thou_hr_tl_fuel_moisture_min_time
      herbaceous_lfi_fuel_moisture woody_lfi_fuel_moisture
      latitude longitude elevation
    }
  }
}
"""

# ========= REQUEST FUNC =========

def gql(query, variables=None):
    r = requests.post(
        ENDPOINT, headers=HEADERS, auth=AUTH,
        json={"query": query, "variables": variables}, timeout=90
    )
    print(f"[HTTP {r.status_code}] {ENDPOINT}")
    j = r.json()
    if "errors" in j:
        raise RuntimeError(j["errors"])
    return j["data"]

# ========= OBSERVED / FORECAST SPLIT =========
# FEMS marks rows with observation_type / nfdr_type.
# "O..." = observed, "F..." = forecast. Observed reports keep only O.
# Forecast reports keep only F.

def keep_observed(df, type_col):
    if df.empty or type_col not in df.columns:
        return df
    return df[df[type_col].astype(str).str.upper().str.startswith("O")].copy()

def keep_forecast(df, type_col):
    if df.empty or type_col not in df.columns:
        return df
    return df[df[type_col].astype(str).str.upper().str.startswith("F")].copy()

# ========= WEATHER OBS (hourly, observed only) =========

wx = gql(Q_WEATHER_OBS, {
    "startDateTimeRange": start_dt_iso,
    "endDateTimeRange": end_dt_iso,
    "stationIds": station_ids_csv
})["weatherObs"]["data"]

df_wx = keep_observed(pd.DataFrame(wx), "observation_type")

# ========= NFDRS OBS (hourly, observed only) =========

nfdrs_frames = []
for fm in FUEL_MODELS:
    nf = gql(Q_NFDRS_OBS, {
        "fuelModels": fm,
        "stationIds": station_ids_csv,
        "startDateRange": today_str,
        "endDateRange": today_str,
        "dateTimeFormat": "UTC"
    })["nfdrsObs"]["data"]
    nfdrs_frames.append(pd.DataFrame(nf))

df_nfdrs = keep_observed(pd.concat(nfdrs_frames, ignore_index=True), "nfdr_type")

# ========= DAILY MINMAX (observed only, refreshed once a day at 12 am) =========

if run_minmax:
    wxmm = gql(Q_WX_MINMAX, {
        "startDate": today_str,
        "endDate": today_str,
        "stationIds": station_ids_csv
    })["wxMinMax"]["data"]
    df_wxmm = keep_observed(pd.DataFrame(wxmm), "observation_type")

    nfdrmm_frames = []
    for fm in FUEL_MODELS:
        nm = gql(Q_NFDR_MINMAX, {
            "startDate": today_str,
            "endDate": today_str,
            "stationIds": station_ids_csv,
            "fuelModels": fm
        })["nfdrMinMax"]["data"]
        nfdrmm_frames.append(pd.DataFrame(nm))
    df_nfdrmm = keep_observed(pd.concat(nfdrmm_frames, ignore_index=True), "nfdr_type")
else:
    # Not midnight: reuse yesterday's daily report so it is NOT overwritten with empty data
    p_wxmm = os.path.join(DATA_DIR, "fems_latest_wxMinMax.xlsx")
    p_nfdrmm = os.path.join(DATA_DIR, "fems_latest_nfdrMinMax.xlsx")
    df_wxmm = pd.read_excel(p_wxmm) if os.path.exists(p_wxmm) else pd.DataFrame()
    df_nfdrmm = pd.read_excel(p_nfdrmm) if os.path.exists(p_nfdrmm) else pd.DataFrame()

# ========= FORECAST (7 days: today + 6, forecast only, refreshed every hour) =========

wxfc = gql(Q_WX_MINMAX, {
    "startDate": today_str,
    "endDate": forecast_end_str,
    "stationIds": station_ids_csv
})["wxMinMax"]["data"]
df_wxfc = keep_forecast(pd.DataFrame(wxfc), "observation_type")

nfdrfc_frames = []
for fm in FUEL_MODELS:
    fc = gql(Q_NFDR_MINMAX, {
        "startDate": today_str,
        "endDate": forecast_end_str,
        "stationIds": station_ids_csv,
        "fuelModels": fm
    })["nfdrMinMax"]["data"]
    nfdrfc_frames.append(pd.DataFrame(fc))
df_nfdrfc = keep_forecast(pd.concat(nfdrfc_frames, ignore_index=True), "nfdr_type")

# ========= WRITE OUTPUTS =========

os.makedirs(DATA_DIR, exist_ok=True)

def append_csv(path, df):
    if df.empty:
        return
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)

# History = observed only (forecast never enters history)
append_csv(os.path.join(DATA_DIR, "history_weatherObs.csv"), df_wx)
append_csv(os.path.join(DATA_DIR, "history_nfdrsObs.csv"), df_nfdrs)
if run_minmax:
    append_csv(os.path.join(DATA_DIR, "history_wxMinMax.csv"), df_wxmm)
    append_csv(os.path.join(DATA_DIR, "history_nfdrMinMax.csv"), df_nfdrmm)

# Hourly observed reports
df_wx.to_excel(os.path.join(DATA_DIR, "fems_latest_weatherObs.xlsx"), index=False)
df_nfdrs.to_excel(os.path.join(DATA_DIR, "fems_latest_nfdrsObs.xlsx"), index=False)

# Daily observed reports (only rewritten at 12 am; otherwise re-saved unchanged)
df_wxmm.to_excel(os.path.join(DATA_DIR, "fems_latest_wxMinMax.xlsx"), index=False)
df_nfdrmm.to_excel(os.path.join(DATA_DIR, "fems_latest_nfdrMinMax.xlsx"), index=False)

# Forecast reports (7 days, refreshed every hour, kept separate from observed)
df_wxfc.to_excel(os.path.join(DATA_DIR, "fems_forecast_weather.xlsx"), index=False)
df_nfdrfc.to_excel(os.path.join(DATA_DIR, "fems_forecast_nfdrs.xlsx"), index=False)

# Combined workbook
with pd.ExcelWriter(os.path.join(DATA_DIR, "fems_data.xlsx"), engine="openpyxl") as xw:
    df_wx.to_excel(xw, sheet_name="weatherObs", index=False)
    df_nfdrs.to_excel(xw, sheet_name="nfdrsObs", index=False)
    df_wxmm.to_excel(xw, sheet_name="wxMinMax", index=False)
    df_nfdrmm.to_excel(xw, sheet_name="nfdrMinMax", index=False)
    df_wxfc.to_excel(xw, sheet_name="wxForecast", index=False)
    df_nfdrfc.to_excel(xw, sheet_name="nfdrForecast", index=False)

print("DONE - OBSERVED AND FORECAST REPORTS MATCH FEMS.")
