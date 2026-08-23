import os, base64, requests, pandas as pd
from datetime import datetime, timedelta, timezone

# QA endpoint (dash in path)
ENDPOINT = "https://fems-qa.fs2c.usda.gov/api/ext-climatology/graphql"
FUEL_MODELS = ["V", "W", "X", "Y", "Z"]
DATA_DIR = "data"

USERNAME = os.environ["FEMS_USERNAME"]
API_KEY = os.environ["FEMS_API_KEY"]
BASIC = base64.b64encode(f"{USERNAME}:{API_KEY}".encode("ascii")).decode("ascii")
HEADERS = {"Authorization": f"Basic {BASIC}", "Content-Type": "application/json"}

# Load NM+AZ stations (one ID per line, no header)
stations_path = os.path.join(DATA_DIR, "stations.csv")
station_ids = pd.read_csv(stations_path, header=None)[0].astype(str).tolist()
station_ids_csv = ",".join(station_ids)

# Time windows (UTC)
now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
last_hour_start = (now_utc - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
last_hour_end   = (last_hour_start + timedelta(hours=1)) - timedelta(seconds=1)
start_dt_iso = last_hour_start.strftime("%Y-%m-%dT%H:%M:%SZ")
end_dt_iso   = last_hour_end.strftime("%Y-%m-%dT%H:%M:%SZ")
today_str = now_utc.strftime("%Y-%m-%d")

# GraphQL queries
Q_WEATHER_OBS = """
query WeatherObs($startDateTimeRange: DateTime!, $endDateTimeRange: DateTime!, $stationIds: String) {
  weatherObs(startDateTimeRange: $startDateTimeRange, endDateTimeRange: $endDateTimeRange, stationIds: $stationIds, hasHistoricData: ALL) {
    data {
      station_id wrcc_id station_name latitude longitude elevation station_type
      observation_time observation_time_lst display_hour display_hour_lst
      masked_observation_time display_date temperature relative_humidity
      hourly_precip wind_speed wind_direction peak_gust_speed peak_gust_dir
      sol_rad snow_flag observation_type hex t_flag rh_flag pcp_flag ws_flag wa_flag sr_flag gs_flag ga_flag
    }
  }
}
"""

Q_NFDRS_OBS = """
query NfdrsObs($fuelModels: String!, $stationIds: String, $startDateRange: Date, $endDateRange: Date, $startHour: Int, $endHour: Int, $dateTimeFormat: DateTimeFormat) {
  nfdrsObs(fuelModels: $fuelModels, stationIds: $stationIds, startDateRange: $startDateRange, endDateRange: $endDateRange, startHour: $startHour, endHour: $endHour, dateTimeFormat: $dateTimeFormat, hasHistoricData: ALL) {
    data {
      station_name station_id wrcc_id latitude longitude elevation
      observation_time observation_time_lst display_hour display_hour_lst
      nfdr_date nfdr_time nfdr_type fuel_model fuel_model_version kbdi
      one_hr_tl_fuel_moisture ten_hr_tl_fuel_moisture hun_hr_tl_fuel_moisture thou_hr_tl_fuel_moisture
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
      station_name station_id wrcc_id latitude longitude elevation summary_date observation_type
      temperature_min temperature_max relative_humidity_min relative_humidity_max
      wind_speed_min wind_speed_max peak_gust_speed_min peak_gust_speed_max
      peak_wind_gust_direction peak_wind_gust_time solar_radiation_max
      daily_precipitation_total snow_flag
    }
  }
}
"""

Q_NFDR_MINMAX = """
query NfdrMinMax($startDate: Date!, $endDate: Date, $stationIds: String, $fuelModels: String) {
  nfdrMinMax(startDate: $startDate, endDate: $endDate, stationIds: $stationIds, fuelModels: $fuelModels, hasHistoricData: ALL) {
    data {
      station_id station_name summary_date fuel_model nfdr_type kbdi gsi
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

def gql(query, variables):
    r = requests.post(ENDPOINT, json={"query": query, "variables": variables}, headers=HEADERS, timeout=90)
    r.raise_for_status()
    j = r.json()
    if "errors" in j:
        raise RuntimeError(j["errors"])
    return j["data"]

# WEATHER OBS (last hour)
wx = gql(Q_WEATHER_OBS, {
    "startDateTimeRange": start_dt_iso,
    "endDateTimeRange": end_dt_iso,
    "stationIds": station_ids_csv
})["weatherObs"]["data"]
df_wx = pd.DataFrame(wx)

# NFDRS OBS (last hour, loop fuel models)
nfdrs_frames = []
for fm in FUEL_MODELS:
    nf = gql(Q_NFDRS_OBS, {
        "fuelModels": fm,
        "stationIds": station_ids_csv,
        "startDateRange": today_str,
        "endDateRange": today_str,
        "startHour": int(last_hour_start.strftime("%H")),
        "endHour": int((last_hour_start + timedelta(hours=1)).strftime("%H")),
        "dateTimeFormat": "UTC"
    })["nfdrsObs"]["data"]
    nfdrs_frames.append(pd.DataFrame(nf))
df_nfdrs = pd.concat(nfdrs_frames, ignore_index=True) if nfdrs_frames else pd.DataFrame()

# WX MIN/MAX (daily)
wxmm = gql(Q_WX_MINMAX, {
    "startDate": today_str,
    "endDate": today_str,
    "stationIds": station_ids_csv
})["wxMinMax"]["data"]
df_wxmm = pd.DataFrame(wxmm)

# NFDR MIN/MAX (daily; loop fuel models)
nfdrmm_frames = []
for fm in FUEL_MODELS:
    nm = gql(Q_NFDR_MINMAX, {
        "startDate": today_str,
        "endDate": today_str,
        "stationIds": station_ids_csv,
        "fuelModels": fm
    })["nfdrMinMax"]["data"]
    nfdrmm_frames.append(pd.DataFrame(nm))
df_nfdrmm = pd.concat(nfdrmm_frames, ignore_index=True) if nfdrmm_frames else pd.DataFrame()

# WRITE files
os.makedirs(DATA_DIR, exist_ok=True)

def append_csv(path, df):
    if df.empty: return
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)

append_csv(os.path.join(DATA_DIR, "history_weatherObs.csv"), df_wx)
append_csv(os.path.join(DATA_DIR, "history_nfdrsObs.csv"), df_nfdrs)
append_csv(os.path.join(DATA_DIR, "history_wxMinMax.csv"), df_wxmm)
append_csv(os.path.join(DATA_DIR, "history_nfdrMinMax.csv"), df_nfdrmm)

df_wx.to_excel(os.path.join(DATA_DIR, "fems_latest_weatherObs.xlsx"), index=False)
df_nfdrs.to_excel(os.path.join(DATA_DIR, "fems_latest_nfdrsObs.xlsx"), index=False)
df_wxmm.to_excel(os.path.join(DATA_DIR, "fems_latest_wxMinMax.xlsx"), index=False)
df_nfdrmm.to_excel(os.path.join(DATA_DIR, "fems_latest_nfdrMinMax.xlsx"), index=False)

with pd.ExcelWriter(os.path.join(DATA_DIR, "fems_data.xlsx"), engine="openpyxl") as xw:
    df_wx.to_excel(xw, sheet_name="weatherObs", index=False)
    df_nfdrs.to_excel(xw, sheet_name="nfdrsObs", index=False)
    df_wxmm.to_excel(xw, sheet_name="wxMinMax", index=False)
    df_nfdrmm.to_excel(xw, sheet_name="nfdrMinMax", index=False)

print("Done: wrote latest + appended history.")
