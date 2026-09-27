import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ========= CONFIG =========
DATA_DIR = os.environ.get("FEMS_DATA_DIR", "data")
NFDR_IN = os.path.join(DATA_DIR, "history_daily_nfdrminmax.csv")
FDRA_IN = os.path.join(DATA_DIR, "r3_fdra.csv")
PERCENTILE_IN = os.path.join(DATA_DIR, "r3_fdra_percentile_historical.csv")
OUTPUT = os.path.join(DATA_DIR, "obs_nfdr_daily_fdra.csv")
ARCGIS_PORTAL_URL = os.environ.get(
    "ARCGIS_PORTAL_URL", "https://nifc.maps.arcgis.com"
)
ARCGIS_FDRA_ITEM_ID = os.environ.get(
    "ARCGIS_FDRA_ITEM_ID", "66e86d16806046c098a0e2860cb520bc"
)

SOURCE_TO_AVERAGE = {
    "Min 1 hr FM": "AvgMin1HrFM",
    "Min 10 hr FM": "AvgMin10HrFM",
    "Min 100 hr FM": "AvgMin100HrFM",
    "Min 1000 hr FM": "AvgMin1000HrFM",
    "Woody FM": "AvgWoodFM",
    "KBDI": "AvgKBDI",
    "GSI": "AvgGSI",
    "Herb FM": "AvgHerbFM",
    "Max IC": "AvgMaxIC",
    "Max ERC": "AvgMaxERC",
    "Max SC": "AvgMaxSC",
    "Max BI": "AvgMaxBI",
}

# Historical percentile thresholds retain their existing column names.
AVERAGE_TO_HISTORICAL = {
    "AvgMin1HrFM": "Avg1HrFM",
    "AvgMin10HrFM": "Avg10HrFM",
    "AvgMin100HrFM": "Avg100HrFM",
    "AvgMin1000HrFM": "Avg1000HrFM",
    "AvgWoodFM": "AvgWoodFM",
    "AvgKBDI": "AvgKBDI",
    "AvgGSI": "AvgGSI",
    "AvgHerbFM": "AvgHerbFM",
    "AvgMaxIC": "AvgIC",
    "AvgMaxERC": "AvgERC",
    "AvgMaxSC": "AvgSC",
    "AvgMaxBI": "AvgBI",
}
AVERAGE_TO_PERCENTILE = {
    "AvgMin1HrFM": "PctMin1HrFM",
    "AvgMin10HrFM": "PctMin10HrFM",
    "AvgMin100HrFM": "PctMin100HrFM",
    "AvgMin1000HrFM": "PctMin1000HrFM",
    "AvgWoodFM": "PctWoodFM",
    "AvgKBDI": "PctKBDI",
    "AvgGSI": "PctGSI",
    "AvgHerbFM": "PctHerbFM",
    "AvgMaxIC": "PctMaxIC",
    "AvgMaxERC": "PctMaxERC",
    "AvgMaxSC": "PctMaxSC",
    "AvgMaxBI": "PctMaxBI",
}

OUTPUT_COLUMNS = [
    "SIG_RAWS",
    "FDRA_ZONE",
    "Zone",
    "FDRA",
    "Shape_Area",
    "Shape_Length",
    "ObservationDate",
    "NFDRType",
    "FuelModel",
    "StationsUsed",
    *SOURCE_TO_AVERAGE.values(),
    *AVERAGE_TO_PERCENTILE.values(),
]


def normalize_station_id(value):
    """Match FEMS IDs such as 020608 with historical IDs such as 20608."""
    value = str(value).strip()
    if not value:
        return ""
    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]
    return value.lstrip("0") or "0"


def require_columns(df, columns, source_name):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source_name} is missing columns: {', '.join(missing)}")


def percentile_for_value(value, thresholds, percentiles):
    """Return the highest historical percentile threshold at or below value."""
    if pd.isna(value):
        return pd.NA

    valid = pd.DataFrame(
        {"threshold": pd.to_numeric(thresholds, errors="coerce"),
         "percentile": pd.to_numeric(percentiles, errors="coerce")}
    ).dropna().sort_values(["threshold", "percentile"])
    if valid.empty:
        return pd.NA

    position = np.searchsorted(valid["threshold"].to_numpy(), float(value), side="right") - 1
    if position < 0:
        return int(valid["percentile"].min())
    return int(valid.iloc[position]["percentile"])


def require_environment(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def get_arcgis_access_token(client_id, refresh_token):
    """Exchange the MFA-authorized refresh token for a temporary access token."""
    response = requests.post(
        f"{ARCGIS_PORTAL_URL}/sharing/rest/oauth2/token",
        data={
            "client_id": client_id,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "expiration": 120,
            "f": "json",
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if "access_token" not in payload:
        error = payload.get("error", payload)
        raise RuntimeError(
            "ArcGIS refresh-token authentication failed. The token may have "
            f"expired or been revoked: {error}"
        )
    return payload["access_token"], payload.get("username", "unknown user")


def inspect_arcgis_upload(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("ArcGIS upload file contains no data rows")

    observation_dates = {
        row.get("ObservationDate", "").strip()
        for row in rows
        if row.get("ObservationDate", "").strip()
    }
    if len(observation_dates) != 1:
        raise ValueError(
            "ArcGIS upload file must contain exactly one ObservationDate; found "
            + ", ".join(sorted(observation_dates))
        )
    return len(rows), observation_dates.pop()


def overwrite_arcgis_table():
    """Overwrite the hosted table when GitHub supplies the OAuth secrets."""
    client_id = os.environ.get("AGOL_CLIENT_ID", "").strip()
    refresh_token = os.environ.get("AGOL_REFRESH_TOKEN", "").strip()
    if not client_id and not refresh_token:
        print("ArcGIS refresh skipped: OAuth secrets were not supplied")
        return
    client_id = require_environment("AGOL_CLIENT_ID")
    refresh_token = require_environment("AGOL_REFRESH_TOKEN")

    # Imported only for the GitHub publishing run; local calculations do not need it.
    from arcgis.features import FeatureLayerCollection
    from arcgis.gis import GIS

    source_count, observation_date = inspect_arcgis_upload(OUTPUT)
    access_token, username = get_arcgis_access_token(client_id, refresh_token)
    gis = GIS(ARCGIS_PORTAL_URL, token=access_token)
    item = gis.content.get(ARCGIS_FDRA_ITEM_ID)
    if item is None:
        raise RuntimeError(f"ArcGIS item was not found: {ARCGIS_FDRA_ITEM_ID}")

    collection = FeatureLayerCollection.fromitem(item)
    result = collection.manager.overwrite(str(Path(OUTPUT).resolve()))
    if not result or not result.get("success"):
        raise RuntimeError(f"ArcGIS overwrite failed: {result}")

    collection = FeatureLayerCollection.fromitem(gis.content.get(ARCGIS_FDRA_ITEM_ID))
    sublayers = list(collection.tables) or list(collection.layers)
    if not sublayers:
        raise RuntimeError("The overwritten ArcGIS item contains no table or layer")
    published_count = sublayers[0].query(return_count_only=True)
    if published_count != source_count:
        raise RuntimeError(
            f"ArcGIS row-count verification failed: CSV has {source_count}, "
            f"hosted table has {published_count}"
        )

    print(
        f"ArcGIS table refreshed successfully as {username}: item "
        f"{ARCGIS_FDRA_ITEM_ID}, {published_count} rows for {observation_date}"
    )


def main():
    for path in (NFDR_IN, FDRA_IN, PERCENTILE_IN):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required input not found: {path}")

    nfdr = pd.read_csv(NFDR_IN, dtype=str, keep_default_na=False)
    fdra = pd.read_csv(FDRA_IN, dtype=str, keep_default_na=False)
    historical = pd.read_csv(PERCENTILE_IN, dtype=str, keep_default_na=False)

    require_columns(
        nfdr,
        ["station_id", "summary_date", "nfdr_type", "fuel_model", *SOURCE_TO_AVERAGE],
        NFDR_IN,
    )
    require_columns(
        fdra,
        [
            "SIG_RAWS",
            "FDRA_ZONE",
            "Zone",
            "FDRA",
            "FM_Long",
            "Shape_Area",
            "Shape_Length",
        ],
        FDRA_IN,
    )
    require_columns(
        historical,
        ["FDRA_ZONE", "Percentiles", *AVERAGE_TO_HISTORICAL.values()],
        PERCENTILE_IN,
    )

    # Ignore trailing blank rows and use the configured 19 FDRA records only.
    fdra = fdra[
        fdra["FDRA_ZONE"].str.strip().ne("")
        & fdra["SIG_RAWS"].str.strip().ne("")
        & fdra["FM_Long"].str.strip().ne("")
    ].copy()

    nfdr["_station_key"] = nfdr["station_id"].map(normalize_station_id)
    nfdr["_date"] = pd.to_datetime(nfdr["summary_date"], errors="coerce").dt.date
    nfdr["_fuel_model"] = nfdr["fuel_model"].str.strip().str.upper()
    nfdr = nfdr[nfdr["_date"].notna() & nfdr["nfdr_type"].str.upper().ne("F")].copy()

    for source_column in SOURCE_TO_AVERAGE:
        nfdr[source_column] = pd.to_numeric(nfdr[source_column], errors="coerce")

    historical["Percentiles"] = pd.to_numeric(historical["Percentiles"], errors="coerce")
    for historical_column in AVERAGE_TO_HISTORICAL.values():
        historical[historical_column] = pd.to_numeric(
            historical[historical_column], errors="coerce"
        )

    output_frames = []
    for _, zone in fdra.iterrows():
        station_ids = [
            normalize_station_id(value)
            for value in str(zone["SIG_RAWS"]).split(",")
            if str(value).strip()
        ]
        expected = len(set(station_ids))
        fuel_model = zone["FM_Long"].strip().upper()

        selected = nfdr[
            nfdr["_station_key"].isin(station_ids)
            & nfdr["_fuel_model"].eq(fuel_model)
        ].copy()
        if selected.empty:
            print(f"WARNING: no NFDR rows for {zone['FDRA_ZONE']}")
            continue

        # Collapse duplicate station/date rows before calculating zone averages.
        station_date = selected.groupby(
            ["_date", "_station_key"], as_index=False
        ).agg(
            {
                **{column: "mean" for column in SOURCE_TO_AVERAGE},
                "nfdr_type": lambda values: "|".join(
                    sorted({str(value).strip() for value in values if str(value).strip()})
                ),
            }
        )

        station_counts = station_date.groupby("_date")["_station_key"].nunique()
        metric_counts = station_date.groupby("_date")[list(SOURCE_TO_AVERAGE)].count()
        complete_dates = station_counts[
            (station_counts == expected) & metric_counts.eq(expected).all(axis=1)
        ].index
        complete = station_date[station_date["_date"].isin(complete_dates)].copy()
        if complete.empty:
            print(f"WARNING: no complete station dates for {zone['FDRA_ZONE']}")
            continue

        daily = complete.groupby("_date", as_index=False).agg(
            {
                **{column: "mean" for column in SOURCE_TO_AVERAGE},
                "_station_key": "nunique",
                "nfdr_type": lambda values: "|".join(
                    sorted(
                        {
                            item
                            for value in values
                            for item in str(value).split("|")
                            if item
                        }
                    )
                ),
            }
        )
        daily = daily.rename(
            columns={
                "_date": "ObservationDate",
                "_station_key": "StationsUsed",
                "nfdr_type": "NFDRType",
                **SOURCE_TO_AVERAGE,
            }
        )

        daily.insert(0, "FDRA", zone["FDRA"])
        daily.insert(0, "Zone", zone["Zone"])
        daily.insert(0, "FDRA_ZONE", zone["FDRA_ZONE"])
        daily.insert(0, "SIG_RAWS", zone["SIG_RAWS"])
        daily["Shape_Area"] = pd.to_numeric(zone["Shape_Area"], errors="coerce")
        daily["Shape_Length"] = pd.to_numeric(
            zone["Shape_Length"], errors="coerce"
        )
        daily["FuelModel"] = fuel_model
        daily["ObservationDate"] = daily["ObservationDate"].map(
            lambda value: value.isoformat()
        )

        zone_history = historical[
            historical["FDRA_ZONE"].str.strip().eq(zone["FDRA_ZONE"].strip())
        ]
        if zone_history.empty:
            raise ValueError(f"No percentile rows for {zone['FDRA_ZONE']}")

        for average_column, percentile_column in AVERAGE_TO_PERCENTILE.items():
            daily[percentile_column] = daily[average_column].apply(
                lambda value, column=average_column: percentile_for_value(
                    value,
                    zone_history[AVERAGE_TO_HISTORICAL[column]],
                    zone_history["Percentiles"],
                )
            )

        for average_column in SOURCE_TO_AVERAGE.values():
            daily[average_column] = daily[average_column].round(2)
        output_frames.append(daily[OUTPUT_COLUMNS])

        print(
            f"{zone['FDRA_ZONE']}: {len(daily)} complete dates, "
            f"{expected} stations"
        )

    if not output_frames:
        raise RuntimeError("No complete FDRA daily averages were produced")

    output = pd.concat(output_frames, ignore_index=True)
    latest_date = output["ObservationDate"].max()
    output = output[output["ObservationDate"].eq(latest_date)].copy()
    output = output.sort_values("FDRA_ZONE").reset_index(drop=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    output.to_csv(OUTPUT, index=False)
    print(
        f"Created {OUTPUT}: {len(output)} rows for {latest_date} "
        f"across {output['FDRA_ZONE'].nunique()} FDRAs"
    )
    overwrite_arcgis_table()


if __name__ == "__main__":
    main()
