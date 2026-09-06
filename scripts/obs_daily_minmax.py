import os
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from requests.auth import HTTPBasicAuth

# ========= CONFIG =========
ENDPOINT    = "https://fems.fs2c.usda.gov/api/ext-climatology/graphql"  # PROD
FUEL_MODELS = ["V", "W", "X", "Y", "Z"]
DATA_DIR    = "data"

RECHECK_DAYS   = 60    # re-check window on every run
RETENTION_DAYS = 365   # keep 1 year of history

WX_OUT   = os.path.join(DATA_DIR, "history_daily_weatherminmax.csv")
NFDR_OUT = os.path.join(DATA_DIR, "history_daily_nfdrminmax.csv")
HOURLY_WX_IN = os.path.join(DATA_DIR, "history_hourly_weather.csv")  # read-only source for real daily VPD

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
# (FEMS shows "Min 1 hr FM/Time" as one combined cell; we store value + time
# as two columns, so we split that into "Min 1 hr FM" and "Min 1 hr FM Time")
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
now_utc      = datetime.utcnow().replace(tzinfo=timezone.utc)
window_start = now_utc - timedelta(days=RECHECK_DAYS)
cutoff_1yr   = now_utc - timedelta(days=RETENTION_DAYS)

start_date        = window_start.strftime("%Y-%m-%d")
end_date          = now_utc.strftime("%Y-%m-%d")
window_start_date = window_start.date()
cutoff_1yr_date   = cutoff_1yr.date()

# pull_date_mst: when this run happened, in Arizona/Mountain Standard Time (no daylight saving)
pull_date_mst = (now_utc - timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S")

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
# Python's round(9.45, 1) gives 9.4 because 9.45 can't be stored exactly in binary.
# Going through Decimal(str(x)) preserves the value as returned by FEMS, so .45 rounds up to .5 as expected.
def round_half_up(x, decimals):
    if pd.isna(x):
        return x
    quantum = Decimal(1).scaleb(-decimals)
    return float(Decimal(str(x)).quantize(quantum, rounding=ROUND_HALF_UP))

# ========= VPD (Vapor Pressure Deficit) — FEMS doesn't expose this as an API field.
# Our first attempt approximated daily Max/Min VPD by pairing the day's temp extreme
# with the day's RH extreme — that turned out inaccurate (off by up to 16% against
# FEMS's real numbers), because the hottest hour and driest hour don't always coincide.
# The accurate method: the hourly weather report already calculates VPD for every
# single hour and keeps a full year of it. So instead of re-pulling hourly data here,
# we read that file, group it by station + day, and take the real max/min — this is
# the same hour-by-hour method FEMS itself uses.
def compute_daily_vpd_from_hourly(hourly_path):
    empty = pd.DataFrame(columns=["station_id", "_date_key", "Max VPD", "Min VPD"])
    if not os.path.exists(hourly_path):
        print(f"{os.path.basename(hourly_path)} not found — Max/Min VPD will be blank this run")
        return empty

    df_h = pd.read_csv(hourly_path, dtype=str)
    if df_h.empty or "vpd" not in df_h.columns or "observation_date_lst" not in df_h.columns:
        print("Hourly weather file missing vpd or observation_date_lst — Max/Min VPD will be blank this run")
        return empty

    df_h["vpd"] = pd.to_numeric(df_h["vpd"], errors="coerce")
    df_h["_date_key"] = pd.to_datetime(df_h["observation_date_lst"], errors="coerce").dt.date
    grouped = df_h.groupby(["station_id", "_date_key"])["vpd"].agg(["max", "min"]).reset_index()
    return grouped.rename(columns={"max": "Max VPD", "min": "Min VPD"})

# ========= CLEAN TIME COLUMNS =========
# FEMS's Max/Min "_time" fields turn out to be a bare hour number (e.g. 16, 18, 17) —
# not a full timestamp like the hourly report's fields. Format that as "16:00" to
# match FEMS's own display (they show "76.6/16:00" — hour precision, no minutes).
# Anything that doesn't look like a bare hour or a parseable time is kept as-is,
# so we never silently blank real data.
def clean_time_column(df, col):
    if col not in df.columns:
        return df

    def format_one(val):
        if pd.isna(val) or str(val).strip() == "":
            return val
        s = str(val).strip()

        # Bare hour number: "16", "16.0", 16 -> "16:00"
        try:
            hour = int(float(s))
            if 0 <= hour <= 23:
                return f"{hour:02d}:00"
        except (ValueError, TypeError):
            pass

        # Military-style time with no separator: "1730" -> "17:30", "930" -> "09:30"
        # (checked before the generic parser below, which would misread "1730" as the year 1730)
        if s.isdigit() and len(s) in (3, 4):
            hh, mm = int(s[:-2]), int(s[-2:])
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return f"{hh:02d}:{mm:02d}"

        # Fallback: a full/partial timestamp, in case FEMS ever sends one
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            return dt.strftime("%H:%M:%S")

        # Unrecognized shape — keep the original text rather than lose it
        return s

    df[col] = df[col].apply(format_one)
    return df

# ========= PULL: WEATHER MINMAX (60-day window) =========
wxmm = gql(
    Q_WX_MINMAX,
    {"startDate": start_date, "endDate": end_date, "stationIds": station_ids_csv}
)["wxMinMax"]["data"]
df_wx = pd.DataFrame(wxmm)
report("weather minmax pull (raw)", df_wx, "observation_type")

# Observed only: keep all non-F rows
if not df_wx.empty:
    df_wx = df_wx[df_wx["observation_type"] != "F"].copy()
    df_wx["Pull_date_mst"] = pull_date_mst

    # Max/Min VPD: merged in from the real hourly-derived daily extremes
    daily_vpd = compute_daily_vpd_from_hourly(HOURLY_WX_IN)
    df_wx["station_id"] = df_wx["station_id"].astype(str)
    df_wx["_date_key"] = pd.to_datetime(df_wx["summary_date"], errors="coerce").dt.date
    df_wx = df_wx.merge(daily_vpd, on=["station_id", "_date_key"], how="left")
    df_wx = df_wx.drop(columns=["_date_key"], errors="ignore")

    df_wx = clean_time_column(df_wx, "peak_wind_gust_time")
    df_wx = df_wx.rename(columns=WX_RENAME)
report("weather minmax pull (observed only)", df_wx, "observation_type")

# ========= PULL: NFDR MINMAX (60-day window, per fuel model) =========
nfdr_frames = []
for fm in FUEL_MODELS:
    nm = gql(
        Q_NFDR_MINMAX,
        {"startDate": start_date, "endDate": end_date, "stationIds": station_ids_csv, "fuelModels": fm}
    )["nfdrMinMax"]["data"]
    print(f"fuel model {fm}: {len(nm)} rows")
    nfdr_frames.append(pd.DataFrame(nm))

df_nfdr = pd.concat(nfdr_frames, ignore_index=True) if nfdr_frames else pd.DataFrame()
report("nfdr minmax pull (raw)", df_nfdr, "nfdr_type")

# Observed only: keep all non-F rows
if not df_nfdr.empty:
    df_nfdr = df_nfdr[df_nfdr["nfdr_type"] != "F"].copy()
    df_nfdr["Pull_date_mst"] = pull_date_mst

    # Clean the raw ISO _time columns down to just the clock time
    for col in ["ignition_component_max_time", "spread_component_max_time",
                "energy_release_component_max_time", "burning_index_max_time",
                "one_hr_tl_fuel_moisture_min_time", "ten_hr_tl_fuel_moisture_min_time",
                "hun_hr_tl_fuel_moisture_min_time", "thou_hr_tl_fuel_moisture_min_time"]:
        df_nfdr = clean_time_column(df_nfdr, col)

    # Round fire-danger numbers to match FEMS's own display
    for col in ROUND_1_COLS:
        if col in df_nfdr.columns:
            df_nfdr[col] = pd.to_numeric(df_nfdr[col], errors="coerce").apply(lambda x: round_half_up(x, 1))
    for col in ROUND_2_COLS:
        if col in df_nfdr.columns:
            df_nfdr[col] = pd.to_numeric(df_nfdr[col], errors="coerce").apply(lambda x: round_half_up(x, 2))

    # Clean column names to match FEMS's table headers
    df_nfdr = df_nfdr.rename(columns=NFDR_RENAME)
report("nfdr minmax pull (observed only)", df_nfdr, "nfdr_type")

# ========= SYNC LOGIC =========
# Drop stored rows inside the re-check window, insert fresh pull, keep older rows.
# Then trim anything older than 1 year. Daily data only needs a date (no time-of-day) to sync on.
def sync_history(path, df_new, date_col):
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(path):
        df_old = pd.read_csv(path, dtype=str)
    else:
        df_old = pd.DataFrame()

    def build_date(df):
        if df.empty or date_col not in df.columns:
            return pd.Series([pd.NaT] * len(df), index=df.index)
        return pd.to_datetime(df[date_col], errors="coerce").dt.date

    if not df_old.empty:
        old_dates = build_date(df_old)
        keep_old  = df_old[(old_dates < window_start_date) & (old_dates >= cutoff_1yr_date)]
        print(f"{os.path.basename(path)}: kept {len(keep_old)} rows outside window, "
              f"replaced {len(df_old) - len(keep_old)} rows inside window/expired")
    else:
        keep_old = df_old

    df_new = df_new.astype(str) if not df_new.empty else df_new
    combined = pd.concat([keep_old, df_new], ignore_index=True)

    if not combined.empty:
        sort_dates = build_date(combined)
        combined = combined.assign(_sort=sort_dates).sort_values("_sort").drop(columns="_sort")

    combined.to_csv(path, index=False)
    print(f"{os.path.basename(path)}: total rows now {len(combined)}")

sync_history(WX_OUT, df_wx, "summary_date")
sync_history(NFDR_OUT, df_nfdr, "summary_date")

print("Done: obs_daily_minmax.")
