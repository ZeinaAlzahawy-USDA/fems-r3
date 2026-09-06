import os
import math
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from requests.auth import HTTPBasicAuth

# ========= CONFIG =========
ENDPOINT         = "https://fems.fs2c.usda.gov/api/ext-climatology/graphql"  # PROD
FUEL_MODELS      = ["V", "W", "X", "Y", "Z"]
DATA_DIR         = "data"
WX_FORECAST_DAYS   = 8   # today + 7 more days = 8 days total
NFDR_FORECAST_DAYS = 7   # today + 6 more days = 7 days total

WX_OUT   = os.path.join(DATA_DIR, "history_forecast_daily_weather.csv")
NFDR_OUT = os.path.join(DATA_DIR, "history_forecast_daily_nfdr.csv")
HOURLY_FORECAST_WX_IN = os.path.join(DATA_DIR, "30days_forecast_hourly_weather.csv")  # read-only, for real VPD

# NFDR fire-danger columns: round to match FEMS's own display (1 decimal, GSI keeps 2)
ROUND_1_COLS = [
    "kbdi",
    "herbaceous_lfi_fuel_moisture", "woody_lfi_fuel_moisture",
    "ignition_component_max", "spread_component_max",
    "energy_release_component_max", "burning_index_max",
    "one_hr_tl_fuel_moisture_min", "ten_hr_tl_fuel_moisture_min",
    "hun_hr_tl_fuel_moisture_min", "thou_hr_tl_fuel_moisture_min",
]
ROUND_2_COLS = ["gsi"]

# Clean column names matching FEMS's own Daily table headers
NFDR_RENAME = {
    "kbdi": "KBDI",
    "gsi": "GSI",
    "herbaceous_lfi_fuel_moisture": "Herb FM",
    "woody_lfi_fuel_moisture": "Woody FM",
    "one_hr_tl_fuel_moisture_min": "Min 1 hr FM",
    "one_hr_tl_fuel_moisture_min_time": "Min 1 hr FM Time",
    "ten_hr_tl_fuel_moisture_min": "Min 10 hr FM",
    "ten_hr_tl_fuel_moisture_min_time": "Min 10 hr FM Time",
    "hun_hr_tl_fuel_moisture_min": "Min 100 hr FM",
    "hun_hr_tl_fuel_moisture_min_time": "Min 100 hr FM Time",
    "thou_hr_tl_fuel_moisture_min": "Min 1000 hr FM",
    "thou_hr_tl_fuel_moisture_min_time": "Min 1000 hr FM Time",
    "ignition_component_max": "Max IC",
    "ignition_component_max_time": "Max IC Time",
    "spread_component_max": "Max SC",
    "spread_component_max_time": "Max SC Time",
    "energy_release_component_max": "Max ERC",
    "energy_release_component_max_time": "Max ERC Time",
    "burning_index_max": "Max BI",
    "burning_index_max_time": "Max BI Time",
}

# Clean column names matching FEMS's own Daily Weather table headers
WX_RENAME = {
    "temperature_min": "Min Temp F",
    "temperature_max": "Max Temp F",
    "relative_humidity_min": "Min RH pct",
    "relative_humidity_max": "Max RH pct",
    "wind_speed_min": "Min Wind mph",
    "wind_speed_max": "Max Wind mph",
    "peak_gust_speed_min": "Min Wind Gust mph",
    "peak_gust_speed_max": "Max Wind Gust mph",
    "peak_wind_gust_direction": "Max Wind Gust Dir",
    "peak_wind_gust_time": "Max Wind Gust Time",
    "solar_radiation_max": "Max SR Wm2",
    "daily_precipitation_total": "Precip Total in",
    "snow_flag": "SF",
}
# VPD isn't a real column we rename — it's calculated below and named directly

USERNAME = os.environ["FEMS_USERNAME"]
API_KEY  = os.environ["FEMS_API_KEY"]
AUTH     = HTTPBasicAuth(USERNAME, API_KEY)

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "FEMS-NM-AZ-GitHubActions/1.0"
}

# ========= STATIONS (AZ + NM) =========
def load_stations(fname):
    path = os.path.join(DATA_DIR, fname)
    return pd.read_csv(path, header=None)[0].astype(str).tolist()

station_ids     = load_stations("stations_az.csv") + load_stations("stations_nm.csv")
station_ids_csv = ",".join(station_ids)

# ========= TIME WINDOW =========
# Anchored to Arizona/Mountain time (no daylight saving), starting today.
# Weather and nfdr use different day-counts, so each gets its own end date.
now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
now_mst = now_utc - timedelta(hours=7)
today_mst_date = now_mst.date()

wx_start_date   = today_mst_date.strftime("%Y-%m-%d")
wx_end_date     = (today_mst_date + timedelta(days=WX_FORECAST_DAYS - 1)).strftime("%Y-%m-%d")
nfdr_start_date = today_mst_date.strftime("%Y-%m-%d")
nfdr_end_date   = (today_mst_date + timedelta(days=NFDR_FORECAST_DAYS - 1)).strftime("%Y-%m-%d")

# pull_date_mst: when this run happened, in Arizona/Mountain Standard Time
pull_date_mst = now_mst.strftime("%Y-%m-%d %H:%M:%S")

# ========= QUERIES =========
Q_WX_MINMAX = """
query WxMinMax($startDate: Date!, $endDate: Date, $stationIds: String) {
  wxMinMax(startDate: $startDate, endDate: $endDate, stationIds: $stationIds, hasHistoricData: ALL) {
    data {
      station_name station_id wrcc_id latitude longitude elevation summary_date observation_type
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
      station_id station_name wrcc_id summary_date nfdr_type fuel_model quality_code
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

# ========= REQUEST HELPER =========
def gql(query, variables=None):
    r = requests.post(
        ENDPOINT,
        headers=HEADERS,
        auth=AUTH,
        json={"query": query, "variables": variables},
        timeout=300
    )
    print(f"[HTTP {r.status_code}] {ENDPOINT}")

    ct = r.headers.get("content-type", "")
    j = r.json() if "application/json" in ct else {"errors": [{"message": r.text[:300]}]}

    if j.get("errors"):
        raise RuntimeError(j["errors"])
    r.raise_for_status()

    return j["data"]

# ========= DIAGNOSTICS =========
def report(label, df, type_col):
    print(f"--- {label} ---")
    print(f"rows: {len(df)}")
    if not df.empty and type_col in df.columns:
        print(f"{type_col} distribution:\n{df[type_col].value_counts(dropna=False)}")

# ========= ROUNDING (matches how FEMS itself rounds, not Python's binary-float rounding) =========
def round_half_up(x, decimals):
    if pd.isna(x):
        return x
    quantum = Decimal(1).scaleb(-decimals)
    return float(Decimal(str(x)).quantize(quantum, rounding=ROUND_HALF_UP))

# ========= VPD (Vapor Pressure Deficit) — FEMS doesn't expose this as an API field.
# The hourly forecast file already calculates VPD for every forecast hour, so we
# read it, group by station + target date, and take the real max/min — same method
# used for observed daily. The important difference here: forecast_hourly.py is a
# pure append-forever log (every day's 7-day outlook gets added, duplicates and all),
# so we must filter to ONLY today's freshly-pulled batch before grouping — otherwise
# we'd mix today's forecast for a date with yesterday's now-stale forecast for that
# same date.
def compute_daily_vpd_from_hourly(hourly_path, today_date):
    empty = pd.DataFrame(columns=["station_id", "_date_key", "Max VPD", "Min VPD"])
    if not os.path.exists(hourly_path):
        print(f"{os.path.basename(hourly_path)} not found — Max/Min VPD will be blank this run")
        return empty

    df_h = pd.read_csv(hourly_path, dtype=str)
    required = {"vpd", "observation_date_lst", "Pull_date_mst", "station_id"}
    if df_h.empty or not required.issubset(df_h.columns):
        print("Hourly forecast file missing required columns — Max/Min VPD will be blank this run")
        return empty

    pull_dates = pd.to_datetime(df_h["Pull_date_mst"], errors="coerce").dt.date
    df_h = df_h[pull_dates == today_date].copy()
    if df_h.empty:
        print("No hourly forecast rows from today's pull found yet — Max/Min VPD will be blank this run")
        return empty

    df_h["vpd"] = pd.to_numeric(df_h["vpd"], errors="coerce")
    df_h["_date_key"] = pd.to_datetime(df_h["observation_date_lst"], errors="coerce").dt.date
    grouped = df_h.groupby(["station_id", "_date_key"])["vpd"].agg(["max", "min"]).reset_index()
    return grouped.rename(columns={"max": "Max VPD", "min": "Min VPD"})

# ========= VPD FALLBACK FOR TODAY =========
# forecast_hourly.py starts tomorrow, so it never has data for today — meaning
# compute_daily_vpd_from_hourly() above can never fill in today's row. Rather than
# leave today blank, fall back to the daily-extremes approximation (pairing the
# day's max temp with min RH, and vice versa) for today only. Every other day
# (tomorrow through the end of the window) still uses the accurate hourly-derived
# value above — this fallback only fires where that's genuinely unavailable.
def calc_vpd_pa_approx(temp_f, rh_pct):
    if pd.isna(temp_f) or pd.isna(rh_pct):
        return None
    temp_c = (float(temp_f) - 32) * 5 / 9
    svp_kpa = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    vpd_kpa = svp_kpa * (1 - float(rh_pct) / 100)
    return math.floor(vpd_kpa * 1000)

# ========= CLEAN TIME COLUMNS =========
# FEMS's Max/Min "_time" fields are a bare hour number (e.g. 16, 18, 17), not a full
# timestamp. Format that as "16:00" to match FEMS's own display. Anything that
# doesn't look like a bare hour or a parseable time is kept as-is.
def clean_time_column(df, col):
    if col not in df.columns:
        return df

    def format_one(val):
        if pd.isna(val) or str(val).strip() == "":
            return val
        s = str(val).strip()

        try:
            hour = int(float(s))
            if 0 <= hour <= 23:
                return f"{hour:02d}:00"
        except (ValueError, TypeError):
            pass

        if s.isdigit() and len(s) in (3, 4):
            hh, mm = int(s[:-2]), int(s[-2:])
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return f"{hh:02d}:{mm:02d}"

        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            return dt.strftime("%H:%M:%S")

        return s

    df[col] = df[col].apply(format_one)
    return df

# ========= PULL: WEATHER MINMAX (8 future days) =========
wxmm = gql(
    Q_WX_MINMAX,
    {"startDate": wx_start_date, "endDate": wx_end_date, "stationIds": station_ids_csv}
)["wxMinMax"]["data"]
df_wx = pd.DataFrame(wxmm)
report("weather minmax pull (raw)", df_wx, "observation_type")

# Forecast only: keep just type == F
if not df_wx.empty:
    df_wx = df_wx[df_wx["observation_type"] == "F"].copy()
    df_wx["Pull_date_mst"] = pull_date_mst

    # Max/Min VPD: merged in from today's fresh hourly-forecast-derived daily extremes
    daily_vpd = compute_daily_vpd_from_hourly(HOURLY_FORECAST_WX_IN, today_mst_date)
    df_wx["station_id"] = df_wx["station_id"].astype(str)
    df_wx["_date_key"] = pd.to_datetime(df_wx["summary_date"], errors="coerce").dt.date
    df_wx = df_wx.merge(daily_vpd, on=["station_id", "_date_key"], how="left")
    df_wx = df_wx.drop(columns=["_date_key"], errors="ignore")

    # Fill in today's row (the one date the accurate method above can never cover)
    # using the daily-extremes approximation instead of leaving it blank
    missing_max = df_wx["Max VPD"].isna()
    missing_min = df_wx["Min VPD"].isna()
    df_wx.loc[missing_max, "Max VPD"] = df_wx.loc[missing_max].apply(
        lambda r: calc_vpd_pa_approx(r.get("temperature_max"), r.get("relative_humidity_min")), axis=1)
    df_wx.loc[missing_min, "Min VPD"] = df_wx.loc[missing_min].apply(
        lambda r: calc_vpd_pa_approx(r.get("temperature_min"), r.get("relative_humidity_max")), axis=1)

    df_wx = clean_time_column(df_wx, "peak_wind_gust_time")
    df_wx = df_wx.rename(columns=WX_RENAME)
report("weather minmax pull (forecast only)", df_wx, "observation_type")

# ========= PULL: NFDR MINMAX (7 future days, per fuel model) =========
nfdr_frames = []
for fm in FUEL_MODELS:
    nm = gql(
        Q_NFDR_MINMAX,
        {"startDate": nfdr_start_date, "endDate": nfdr_end_date, "stationIds": station_ids_csv, "fuelModels": fm}
    )["nfdrMinMax"]["data"]
    print(f"fuel model {fm}: {len(nm)} rows")
    nfdr_frames.append(pd.DataFrame(nm))

df_nfdr = pd.concat(nfdr_frames, ignore_index=True) if nfdr_frames else pd.DataFrame()
report("nfdr minmax pull (raw)", df_nfdr, "nfdr_type")

# Forecast only: keep just type == F
if not df_nfdr.empty:
    df_nfdr = df_nfdr[df_nfdr["nfdr_type"] == "F"].copy()
    df_nfdr["Pull_date_mst"] = pull_date_mst

    for col in ["ignition_component_max_time", "spread_component_max_time",
                "energy_release_component_max_time", "burning_index_max_time",
                "one_hr_tl_fuel_moisture_min_time", "ten_hr_tl_fuel_moisture_min_time",
                "hun_hr_tl_fuel_moisture_min_time", "thou_hr_tl_fuel_moisture_min_time"]:
        df_nfdr = clean_time_column(df_nfdr, col)

    for col in ROUND_1_COLS:
        if col in df_nfdr.columns:
            df_nfdr[col] = pd.to_numeric(df_nfdr[col], errors="coerce").apply(lambda x: round_half_up(x, 1))
    for col in ROUND_2_COLS:
        if col in df_nfdr.columns:
            df_nfdr[col] = pd.to_numeric(df_nfdr[col], errors="coerce").apply(lambda x: round_half_up(x, 2))

    df_nfdr = df_nfdr.rename(columns=NFDR_RENAME)
report("nfdr minmax pull (forecast only)", df_nfdr, "nfdr_type")

# ========= APPEND FOREVER =========
# No re-sync, no dedup, no retention cutoff — every day's fresh 7-day outlook is
# simply added on top and kept permanently. Duplicates for the same future date
# across different pull-days are expected and intentional.
def append_forever(path, df_new):
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(path):
        df_old = pd.read_csv(path, dtype=str)
    else:
        df_old = pd.DataFrame()

    df_new = df_new.astype(str) if not df_new.empty else df_new
    combined = pd.concat([df_old, df_new], ignore_index=True)

    combined.to_csv(path, index=False)
    print(f"{os.path.basename(path)}: total rows now {len(combined)}")

append_forever(WX_OUT, df_wx)
append_forever(NFDR_OUT, df_nfdr)

print("Done: forecast_daily.")
