import os
import math
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from requests.auth import HTTPBasicAuth

# ========= CONFIG =========
ENDPOINT    = "https://fems.fs2c.usda.gov/api/ext-climatology/graphql"  # PROD
FUEL_MODELS = ["V", "W", "X", "Y", "Z"]
DATA_DIR    = "data"

RECHECK_DAYS   = 30    # re-sync window on every run
RETENTION_DAYS = 365   # keep 1 year of history

# NFDR fire-danger columns: round to match FEMS's own display (1 decimal, GSI keeps 2)
ROUND_1_COLS = [
    "one_hr_tl_fuel_moisture", "ten_hr_tl_fuel_moisture",
    "hun_hr_tl_fuel_moisture", "thou_hr_tl_fuel_moisture",
    "kbdi", "herbaceous_lfi_fuel_moisture", "woody_lfi_fuel_moisture",
    "ignition_component", "spread_component",
    "energy_release_component", "burning_index"
]
ROUND_2_COLS = ["gsi"]

# Clean column names matching FEMS's own table headers
NFDR_RENAME = {
    "one_hr_tl_fuel_moisture": "1 hr FM",
    "ten_hr_tl_fuel_moisture": "10 hr FM",
    "hun_hr_tl_fuel_moisture": "100 hr FM",
    "thou_hr_tl_fuel_moisture": "1000 hr FM",
    "kbdi": "KBDI",
    "herbaceous_lfi_fuel_moisture": "Herb FM",
    "woody_lfi_fuel_moisture": "Woody FM",
    "gsi": "GSI",
    "ignition_component": "IC",
    "spread_component": "SC",
    "energy_release_component": "ERC",
    "burning_index": "BI",
}

WX_OUT   = os.path.join(DATA_DIR, "history_hourly_weather.csv")
NFDR_OUT = os.path.join(DATA_DIR, "history_hourly_nfdr.csv")

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

# ========= TIME WINDOW (UTC) =========
now_utc      = datetime.utcnow().replace(tzinfo=timezone.utc)
window_start = (now_utc - timedelta(days=RECHECK_DAYS)).replace(minute=0, second=0, microsecond=0)
cutoff_1yr   = now_utc - timedelta(days=RETENTION_DAYS)

start_dt_iso = window_start.strftime("%Y-%m-%dT%H:%M:%SZ")
end_dt_iso   = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
start_date   = window_start.strftime("%Y-%m-%d")
end_date     = now_utc.strftime("%Y-%m-%d")

# pull_date: when this run happened in Arizona/Mountain Standard Time (no daylight saving),
# displayed in the same timestamp layout as the FEMS masked_observation_time field
pull_date = (now_utc - timedelta(hours=7)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

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
      station_id wrcc_id station_name latitude longitude elevation station_type
      zoom_level hex
      observation_time observation_time_lst display_hour display_hour_lst
      masked_observation_time display_date
      temperature relative_humidity hourly_precip wind_speed wind_direction
      peak_gust_speed peak_gust_dir sol_rad snow_flag observation_type
      t_flag rh_flag pcp_flag ws_flag wa_flag sr_flag gs_flag ga_flag
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
# Going through Decimal(str(x)) preserves the value as typed/returned by FEMS, so .45 rounds up to .5 as expected.
def round_half_up(x, decimals):
    if pd.isna(x):
        return x
    quantum = Decimal(1).scaleb(-decimals)
    return float(Decimal(str(x)).quantize(quantum, rounding=ROUND_HALF_UP))

# ========= VPD (Vapor Pressure Deficit) — FEMS doesn't expose this as an API field,
# so we calculate it ourselves from temperature (F) and relative humidity (%).
# Result is in Pa. FEMS truncates the decimal rather than rounding it (confirmed by
# comparing real values — our rounding was always exactly 1 Pa too high), so we
# truncate here too, to match their display exactly.
def calc_vpd_pa(temp_f, rh_pct):
    if pd.isna(temp_f) or pd.isna(rh_pct):
        return None
    temp_c = (float(temp_f) - 32) * 5 / 9
    svp_kpa = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    vpd_kpa = svp_kpa * (1 - float(rh_pct) / 100)
    return math.floor(vpd_kpa * 1000)

# ========= PULL: WEATHER (30-day window) =========
wx = gql(
    Q_WEATHER_OBS,
    {
        "startDateTimeRange": start_dt_iso,
        "endDateTimeRange": end_dt_iso,
        "stationIds": station_ids_csv
    }
)["weatherObs"]["data"]
df_wx = pd.DataFrame(wx)
report("weather pull (raw)", df_wx, "observation_type")

# Observed only: keep all non-F rows
if not df_wx.empty:
    df_wx = df_wx[df_wx["observation_type"] != "F"].copy()
    df_wx["pull_date"] = pull_date

    # Calculate VPD from temperature + relative humidity (not an available FEMS field)
    df_wx["vpd"] = df_wx.apply(lambda r: calc_vpd_pa(r.get("temperature"), r.get("relative_humidity")), axis=1)
report("weather pull (observed only)", df_wx, "observation_type")

# ========= PULL: NFDR (30-day window, per fuel model) =========
nfdr_frames = []
for fm in FUEL_MODELS:
    nf = gql(
        Q_NFDRS_OBS,
        {
            "fuelModels": fm,
            "stationIds": station_ids_csv,
            "startDateRange": start_date,
            "endDateRange": end_date,
            "dateTimeFormat": "UTC"
        }
    )["nfdrsObs"]["data"]
    print(f"fuel model {fm}: {len(nf)} rows")
    nfdr_frames.append(pd.DataFrame(nf))

df_nfdr = pd.concat(nfdr_frames, ignore_index=True) if nfdr_frames else pd.DataFrame()
report("nfdr pull (raw)", df_nfdr, "nfdr_type")

# Observed only: keep all non-F rows
if not df_nfdr.empty:
    df_nfdr = df_nfdr[df_nfdr["nfdr_type"] != "F"].copy()
    df_nfdr["pull_date"] = pull_date

    # Round fire-danger numbers to match FEMS's own display
    for col in ROUND_1_COLS:
        if col in df_nfdr.columns:
            df_nfdr[col] = pd.to_numeric(df_nfdr[col], errors="coerce").apply(lambda x: round_half_up(x, 1))
    for col in ROUND_2_COLS:
        if col in df_nfdr.columns:
            df_nfdr[col] = pd.to_numeric(df_nfdr[col], errors="coerce").apply(lambda x: round_half_up(x, 2))

    # Clean column names to match FEMS's table headers
    df_nfdr = df_nfdr.rename(columns=NFDR_RENAME)
report("nfdr pull (observed only)", df_nfdr, "nfdr_type")

# ========= SYNC LOGIC =========
# Drop stored rows inside the re-check window, insert fresh pull, keep older rows.
# Then trim anything older than 1 year.
def sync_history(path, df_new, timestamp_col):
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(path):
        df_old = pd.read_csv(path, dtype=str)

        # Migrate the former MST pull timestamp to the new name and display layout.
        if "Pull_date_mst" in df_old.columns:
            old_pull_mst = pd.to_datetime(df_old["Pull_date_mst"], errors="coerce")
            migrated_pull_date = old_pull_mst.dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            if "pull_date" in df_old.columns:
                df_old["pull_date"] = df_old["pull_date"].fillna(migrated_pull_date)
            else:
                df_old["pull_date"] = migrated_pull_date
            df_old = df_old.drop(columns=["Pull_date_mst"])
    else:
        df_old = pd.DataFrame()

    def build_dt(df):
        if df.empty:
            return pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns, UTC]")

        if timestamp_col in df.columns:
            parsed = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
        else:
            parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

        # Preserve history from CSVs produced before raw FEMS timestamps were retained.
        legacy_date_col = "observation_date_lst"
        legacy_time_col = "observation_time_lst_mst"
        if legacy_date_col in df.columns and legacy_time_col in df.columns:
            legacy_local = pd.to_datetime(
                df[legacy_date_col].astype(str) + " " + df[legacy_time_col].astype(str),
                errors="coerce"
            )
            legacy_utc = (legacy_local + pd.Timedelta(hours=7)).dt.tz_localize("UTC")
            parsed = parsed.fillna(legacy_utc)

        return parsed

    if not df_old.empty:
        old_times = build_dt(df_old)
        keep_old  = df_old[(old_times < window_start) & (old_times >= cutoff_1yr)]
        print(f"{os.path.basename(path)}: kept {len(keep_old)} rows outside window, "
              f"replaced {len(df_old) - len(keep_old)} rows inside window/expired")
    else:
        keep_old = df_old

    df_new = df_new.astype(str) if not df_new.empty else df_new
    combined = pd.concat([keep_old, df_new], ignore_index=True)

    if not combined.empty:
        sort_times = build_dt(combined)
        combined = combined.assign(_sort=sort_times).sort_values("_sort").drop(columns="_sort")

    combined.to_csv(path, index=False)
    print(f"{os.path.basename(path)}: total rows now {len(combined)}")

sync_history(WX_OUT, df_wx, "observation_time_lst")
sync_history(NFDR_OUT, df_nfdr, "observation_time_lst")

print("Done: obs_hourly.")
