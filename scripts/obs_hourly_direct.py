import math
import os
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from time import sleep

import pandas as pd
import requests
from arcgis.features import FeatureLayerCollection
from arcgis.gis import GIS
from requests.auth import HTTPBasicAuth

# ========= CONFIG =========
ENDPOINT = "https://fems.fs2c.usda.gov/api/ext-climatology/graphql"
PORTAL_URL = "https://nifc.maps.arcgis.com"
SCRIPT_VERSION = "5-owner-identity-check"
DATA_DIR = "data"
AGOL_FOLDER = "Southwest FEMS Direct Connect"
AGOL_OWNER = "zalzahawy_nifc"
GROUP_ID = "3a8d33917b694e2d8587f301e1739a2f"
FUEL_MODELS = ["V", "W", "X", "Y", "Z"]

RECHECK_DAYS = 30
RETENTION_DAYS = 365
EDIT_BATCH_SIZE = 500
EDIT_RETRIES = 3

WX_TITLE = "History hourly weather"
WX_SERVICE_NAME = "southwest_fems_history_hourly_weather"
NFDR_TITLE = "History hourly nfdr"
NFDR_SERVICE_NAME = "southwest_fems_history_hourly_nfdr"

ROUND_1_COLS = [
    "one_hr_tl_fuel_moisture", "ten_hr_tl_fuel_moisture",
    "hun_hr_tl_fuel_moisture", "thou_hr_tl_fuel_moisture",
    "kbdi", "herbaceous_lfi_fuel_moisture", "woody_lfi_fuel_moisture",
    "ignition_component", "spread_component",
    "energy_release_component", "burning_index",
]
ROUND_2_COLS = ["gsi"]

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

WX_COLUMNS = [
    "station_id", "wrcc_id", "station_name", "latitude", "longitude", "elevation",
    "station_type", "zoom_level", "hex", "observation_time", "observation_time_lst",
    "display_hour", "display_hour_lst", "masked_observation_time", "display_date",
    "temperature", "relative_humidity", "hourly_precip", "wind_speed", "wind_direction",
    "peak_gust_speed", "peak_gust_dir", "sol_rad", "snow_flag", "observation_type",
    "t_flag", "rh_flag", "pcp_flag", "ws_flag", "wa_flag", "sr_flag", "gs_flag",
    "ga_flag", "pull_date", "vpd",
]

NFDR_COLUMNS = [
    "station_name", "station_id", "wrcc_id", "latitude", "longitude", "elevation",
    "observation_time", "observation_time_lst", "display_hour", "display_hour_lst",
    "nfdr_date", "nfdr_time", "nfdr_type", "fuel_model", "fuel_model_version",
    "KBDI", "1 hr FM", "10 hr FM", "100 hr FM", "1000 hr FM", "IC", "SC", "ERC",
    "BI", "Herb FM", "Woody FM", "GSI", "quality_code", "pull_date",
]

NUMERIC_COLUMNS = {
    "latitude", "longitude", "elevation", "zoom_level", "temperature",
    "relative_humidity", "hourly_precip", "wind_speed", "wind_direction",
    "peak_gust_speed", "peak_gust_dir", "sol_rad", "vpd", "KBDI", "1 hr FM",
    "10 hr FM", "100 hr FM", "1000 hr FM", "IC", "SC", "ERC", "BI",
    "Herb FM", "Woody FM", "GSI",
}
DATE_COLUMNS = {"observation_time", "pull_date"}

FEMS_USERNAME = os.environ["FEMS_USERNAME"]
FEMS_API_KEY = os.environ["FEMS_API_KEY"]
AGOL_CLIENT_ID = os.environ["AGOL_CLIENT_ID"]
AGOL_CLIENT_SECRET = os.environ["AGOL_CLIENT_SECRET"]
FEMS_AUTH = HTTPBasicAuth(FEMS_USERNAME, FEMS_API_KEY)

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "FEMS-NM-AZ-GitHubActions/1.0",
}

# ========= VERIFY ARCGIS BEFORE THE LARGE FEMS PULL =========
print(f"Direct-connect script version: {SCRIPT_VERSION}")
print(f"Connecting to ArcGIS organization: {PORTAL_URL}")
gis = GIS(PORTAL_URL, client_id=AGOL_CLIENT_ID, client_secret=AGOL_CLIENT_SECRET)

me = None
try:
    me = gis.users.me
except Exception as exc:
    print(f"Could not resolve signed-in user: {exc}")
print(f"Signed in as: {me.username if me else 'APP-ONLY TOKEN (no user identity)'}")
if me is None or me.username != AGOL_OWNER:
    raise RuntimeError(
        "ArcGIS token is not acting as the owner account "
        f"({AGOL_OWNER}). Check that AGOL_CLIENT_ID/AGOL_CLIENT_SECRET are from "
        "'Southwest FEMS GitHub Owner Automation' (app auth, all owner privileges)."
    )

folder_id = gis._portal.get_folder_id(AGOL_OWNER, AGOL_FOLDER)
if folder_id is None:
    raise RuntimeError(
        f"ArcGIS folder '{AGOL_FOLDER}' was not found for owner {AGOL_OWNER}."
    )
print(f"ArcGIS owner-folder access verified: {AGOL_OWNER}/{AGOL_FOLDER}")

# ========= STATIONS (AZ + NM) =========
def load_stations(fname):
    path = os.path.join(DATA_DIR, fname)
    return pd.read_csv(path, header=None)[0].astype(str).tolist()


station_ids = load_stations("stations_az.csv") + load_stations("stations_nm.csv")
station_ids_csv = ",".join(station_ids)

# ========= TIME WINDOW (UTC) =========
now_utc = datetime.now(timezone.utc)
window_start = (now_utc - timedelta(days=RECHECK_DAYS)).replace(minute=0, second=0, microsecond=0)
cutoff_1yr = now_utc - timedelta(days=RETENTION_DAYS)

start_dt_iso = window_start.strftime("%Y-%m-%dT%H:%M:%SZ")
end_dt_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
start_date = window_start.strftime("%Y-%m-%d")
end_date = now_utc.strftime("%Y-%m-%d")
pull_date = now_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")

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

# ========= REQUEST HELPERS =========
def gql(query, variables=None):
    response = requests.post(
        ENDPOINT,
        headers=HEADERS,
        auth=FEMS_AUTH,
        json={"query": query, "variables": variables},
        timeout=300,
    )
    print(f"[HTTP {response.status_code}] {ENDPOINT}")

    content_type = response.headers.get("content-type", "")
    payload = response.json() if "application/json" in content_type else {
        "errors": [{"message": response.text[:300]}]
    }

    if payload.get("errors"):
        raise RuntimeError(payload["errors"])
    response.raise_for_status()
    return payload["data"]


def report(label, df, type_col):
    print(f"--- {label} ---")
    print(f"rows: {len(df)}")
    if not df.empty and type_col in df.columns:
        print(f"{type_col} distribution:\n{df[type_col].value_counts(dropna=False)}")


def round_half_up(value, decimals):
    if pd.isna(value):
        return value
    quantum = Decimal(1).scaleb(-decimals)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def calc_vpd_pa(temp_f, rh_pct):
    if pd.isna(temp_f) or pd.isna(rh_pct):
        return None
    temp_c = (float(temp_f) - 32) * 5 / 9
    svp_kpa = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    vpd_kpa = svp_kpa * (1 - float(rh_pct) / 100)
    return math.floor(vpd_kpa * 1000)


# ========= PULL: WEATHER (30-day window) =========
weather_rows = gql(
    Q_WEATHER_OBS,
    {
        "startDateTimeRange": start_dt_iso,
        "endDateTimeRange": end_dt_iso,
        "stationIds": station_ids_csv,
    },
)["weatherObs"]["data"]
df_wx = pd.DataFrame(weather_rows)
report("weather pull (raw)", df_wx, "observation_type")

if not df_wx.empty:
    df_wx = df_wx[df_wx["observation_type"] != "F"].copy()
    df_wx["pull_date"] = pull_date
    df_wx["vpd"] = df_wx.apply(
        lambda row: calc_vpd_pa(row.get("temperature"), row.get("relative_humidity")),
        axis=1,
    )
report("weather pull (observed only)", df_wx, "observation_type")

# ========= PULL: NFDR (30-day window, per fuel model) =========
nfdr_frames = []
for fuel_model in FUEL_MODELS:
    nfdr_rows = gql(
        Q_NFDRS_OBS,
        {
            "fuelModels": fuel_model,
            "stationIds": station_ids_csv,
            "startDateRange": start_date,
            "endDateRange": end_date,
            "dateTimeFormat": "UTC",
        },
    )["nfdrsObs"]["data"]
    print(f"fuel model {fuel_model}: {len(nfdr_rows)} rows")
    nfdr_frames.append(pd.DataFrame(nfdr_rows))

df_nfdr = pd.concat(nfdr_frames, ignore_index=True) if nfdr_frames else pd.DataFrame()
report("nfdr pull (raw)", df_nfdr, "nfdr_type")

if not df_nfdr.empty:
    df_nfdr = df_nfdr[df_nfdr["nfdr_type"] != "F"].copy()
    df_nfdr["pull_date"] = pull_date

    for column in ROUND_1_COLS:
        if column in df_nfdr.columns:
            df_nfdr[column] = pd.to_numeric(df_nfdr[column], errors="coerce").apply(
                lambda value: round_half_up(value, 1)
            )
    for column in ROUND_2_COLS:
        if column in df_nfdr.columns:
            df_nfdr[column] = pd.to_numeric(df_nfdr[column], errors="coerce").apply(
                lambda value: round_half_up(value, 2)
            )

    df_nfdr = df_nfdr.rename(columns=NFDR_RENAME)
report("nfdr pull (observed only)", df_nfdr, "nfdr_type")

# ========= ARCGIS DIRECT-CONNECT HELPERS =========
def safe_field_name(alias, used_names):
    name = re.sub(r"[^A-Za-z0-9_]", "_", alias).strip("_").lower()
    if not name or name[0].isdigit():
        name = f"f_{name}"
    name = name[:60]

    candidate = name
    suffix = 2
    while candidate.lower() in used_names or candidate.upper() in {"OBJECTID", "SHAPE"}:
        candidate = f"{name[:55]}_{suffix}"
        suffix += 1

    used_names.add(candidate.lower())
    return candidate


def build_field_definitions(columns):
    fields = [
        {
            "name": "OBJECTID",
            "type": "esriFieldTypeOID",
            "alias": "OBJECTID",
            "nullable": False,
            "editable": False,
        }
    ]
    used_names = {"objectid", "shape"}

    for column in columns:
        field = {
            "name": safe_field_name(column, used_names),
            "alias": column,
            "nullable": True,
            "editable": True,
        }
        if column in DATE_COLUMNS:
            field["type"] = "esriFieldTypeDate"
        elif column in NUMERIC_COLUMNS:
            field["type"] = "esriFieldTypeDouble"
        else:
            field["type"] = "esriFieldTypeString"
            field["length"] = 512
        fields.append(field)

    return fields


def layer_definition(title, columns):
    return {
        "id": 0,
        "name": title,
        "type": "Feature Layer",
        "displayField": "station_name",
        "description": "FEMS observed hourly data maintained by GitHub Actions.",
        "geometryType": "esriGeometryPoint",
        "objectIdField": "OBJECTID",
        "globalIdField": "",
        "fields": build_field_definitions(columns),
        "indexes": [
            {
                "name": "observation_time_idx",
                "fields": "observation_time",
                "isAscending": True,
                "isUnique": False,
            }
        ],
        "extent": {
            "xmin": -180,
            "ymin": -90,
            "xmax": 180,
            "ymax": 90,
            "spatialReference": {"wkid": 4326},
        },
        "spatialReference": {"wkid": 4326},
        "capabilities": "Query,Create,Update,Delete,Editing",
        "drawingInfo": {
            "renderer": {
                "type": "simple",
                "symbol": {
                    "type": "esriSMS",
                    "style": "esriSMSCircle",
                    "color": [31, 121, 180, 200],
                    "size": 6,
                    "outline": {"color": [255, 255, 255, 255], "width": 0.5},
                },
            }
        },
    }


def find_feature_service(gis, title):
    candidates = gis.content.search(
        query=f'title:"{title}" AND type:"Feature Service" AND owner:{AGOL_OWNER}',
        max_items=100,
    )
    matches = [item for item in candidates if item.title == title]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple ArcGIS feature services found with title: {title}")
    return matches[0] if matches else None


def get_or_create_layer(gis, title, service_name, columns):
    item = find_feature_service(gis, title)
    if item is None:
        print(f"Creating ArcGIS hosted feature layer: {title}")
        create_params = {
            "name": service_name,
            "serviceDescription": "FEMS observed hourly data maintained by GitHub Actions.",
            "hasStaticData": False,
            "maxRecordCount": 5000,
            "supportedQueryFormats": "JSON",
            "capabilities": "Query,Create,Update,Delete,Editing",
            "spatialReference": {"wkid": 4326},
            "initialExtent": {
                "xmin": -180,
                "ymin": -90,
                "xmax": 180,
                "ymax": 90,
                "spatialReference": {"wkid": 4326},
            },
        }
        item_id = gis._portal.create_service(
            name=service_name,
            service_type="featureService",
            create_params=create_params,
            owner=AGOL_OWNER,
            folder=AGOL_FOLDER,
            tags="FEMS,Southwest,Direct Connect,Hourly,Observed",
            snippet="FEMS observed hourly data updated directly by GitHub Actions.",
        )
        if not item_id:
            raise RuntimeError(f"ArcGIS did not create the hosted service: {title}")
        item = gis.content.get(item_id)
        item.update(item_properties={"title": title})
        FeatureLayerCollection.fromitem(item).manager.add_to_definition(
            {"layers": [layer_definition(title, columns)]}
        )
        item = gis.content.get(item.id)
        item.share(groups=[GROUP_ID])
        print(f"Created and shared {title}; item ID: {item.id}")
    else:
        print(f"Using existing ArcGIS hosted feature layer: {title} ({item.id})")

    if not item.layers:
        FeatureLayerCollection.fromitem(item).manager.add_to_definition(
            {"layers": [layer_definition(title, columns)]}
        )
        item = gis.content.get(item.id)

    return item.layers[0]


def field_map_for_layer(layer, columns):
    fields = layer.properties.fields
    aliases = {field.get("alias", field["name"]): field["name"] for field in fields}
    names = {field["name"]: field["name"] for field in fields}
    field_map = {}

    for column in columns:
        if column in aliases:
            field_map[column] = aliases[column]
        elif column in names:
            field_map[column] = names[column]
        else:
            raise RuntimeError(f"ArcGIS layer is missing field for column: {column}")

    return field_map


def normalize_attribute(value, column):
    if value is None or pd.isna(value):
        return None
    if column in DATE_COLUMNS:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        return None if pd.isna(parsed) else int(parsed.timestamp() * 1000)
    if column in NUMERIC_COLUMNS:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return str(value)


def dataframe_to_features(df, layer, columns):
    if df.empty:
        return []

    frame = df.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[columns]

    field_map = field_map_for_layer(layer, columns)
    features = []
    for row in frame.to_dict(orient="records"):
        attributes = {
            field_map[column]: normalize_attribute(row.get(column), column)
            for column in columns
        }
        feature = {"attributes": attributes}

        longitude = normalize_attribute(row.get("longitude"), "longitude")
        latitude = normalize_attribute(row.get("latitude"), "latitude")
        if longitude is not None and latitude is not None:
            feature["geometry"] = {
                "x": longitude,
                "y": latitude,
                "spatialReference": {"wkid": 4326},
            }
        features.append(feature)

    return features


def chunks(values, size):
    for index in range(0, len(values), size):
        yield values[index:index + size]


def delete_refresh_and_expired_rows(layer):
    window_sql = window_start.strftime("%Y-%m-%d %H:%M:%S")
    cutoff_sql = cutoff_1yr.strftime("%Y-%m-%d %H:%M:%S")
    where = (
        f"observation_time >= TIMESTAMP '{window_sql}' "
        f"OR observation_time < TIMESTAMP '{cutoff_sql}'"
    )
    result = layer.query(where=where, return_ids_only=True)
    object_ids = result.get("objectIds") or []

    for batch in chunks(object_ids, EDIT_BATCH_SIZE):
        delete_result = layer.delete_features(deletes=",".join(str(value) for value in batch))
        failures = [entry for entry in delete_result.get("deleteResults", []) if not entry.get("success")]
        if failures:
            raise RuntimeError(f"ArcGIS delete failed: {failures[:3]}")

    print(f"Deleted {len(object_ids)} rows inside refresh window or beyond retention")


def add_features_in_batches(layer, features):
    added = 0
    for batch in chunks(features, EDIT_BATCH_SIZE):
        last_error = None
        for attempt in range(1, EDIT_RETRIES + 1):
            try:
                result = layer.edit_features(adds=batch, rollback_on_failure=True)
                failures = [entry for entry in result.get("addResults", []) if not entry.get("success")]
                if failures:
                    raise RuntimeError(f"ArcGIS add failed: {failures[:3]}")
                added += len(batch)
                break
            except Exception as exc:
                last_error = exc
                if attempt == EDIT_RETRIES:
                    raise
                print(f"ArcGIS add attempt {attempt} failed; retrying")
                sleep(10)
        if last_error and added == 0:
            raise last_error

    print(f"Added {added} refreshed rows")


def sync_arcgis_layer(layer, df, columns):
    delete_refresh_and_expired_rows(layer)
    features = dataframe_to_features(df, layer, columns)
    add_features_in_batches(layer, features)
    print(f"ArcGIS layer count: {layer.query(where='1=1', return_count_only=True)}")


# ========= SYNC DIRECTLY TO ARCGIS =========
weather_layer = get_or_create_layer(gis, WX_TITLE, WX_SERVICE_NAME, WX_COLUMNS)
nfdr_layer = get_or_create_layer(gis, NFDR_TITLE, NFDR_SERVICE_NAME, NFDR_COLUMNS)

sync_arcgis_layer(weather_layer, df_wx, WX_COLUMNS)
sync_arcgis_layer(nfdr_layer, df_nfdr, NFDR_COLUMNS)

print("Done: obs_hourly_direct.")
