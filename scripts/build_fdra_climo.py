"""
build_fdra_climo.py — ONE-TIME pull of 2005–2024 daily NFDR history from FEMS,
turned into the "Average" and "Historical max" lines for the website charts.

For each FDRA in data/r3_fdra.csv:
  1. Pull daily observed Max ERC and Max BI for its SIG_RAWS stations (fuel model = FM_Long).
  2. FDRA daily value = average of its stations that reported that day.
  3. For each calendar day (Jan 1 … Dec 31): average and maximum of that day across all years.

Output: data/r3_fdra_daily_climo.csv
  FDRA_ZONE, doy, month_day, erc_avg, erc_max, bi_avg, bi_max, years
Raw station data is also saved: data/history_2005_2024_daily_erc_bi.csv

Uses the same FEMS endpoint and secrets as the existing pipeline. Does not change any other file.
Run it from GitHub Actions: "FDRA History 2005-2024 (one time)".
"""
import os
import time
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

ENDPOINT = "https://fems.fs2c.usda.gov/api/ext-climatology/graphql"  # PROD
DATA_DIR = os.environ.get("FEMS_DATA_DIR", "data")
START_YEAR = int(os.environ.get("CLIMO_START_YEAR", "2005"))
END_YEAR = int(os.environ.get("CLIMO_END_YEAR", "2024"))
OUT = os.path.join(DATA_DIR, "r3_fdra_daily_climo.csv")
RAW_OUT = os.path.join(DATA_DIR, f"history_{START_YEAR}_{END_YEAR}_daily_erc_bi.csv")

AUTH = HTTPBasicAuth(os.environ["FEMS_USERNAME"], os.environ["FEMS_API_KEY"])
HEADERS = {"Content-Type": "application/json", "Accept": "application/json",
           "User-Agent": "FEMS-NM-AZ-GitHubActions/1.0"}

Q_NFDR_MINMAX = """
query NfdrMinMax($startDate: Date!, $endDate: Date, $stationIds: String, $fuelModels: String) {
  nfdrMinMax(startDate: $startDate, endDate: $endDate, stationIds: $stationIds, fuelModels: $fuelModels, hasHistoricData: ALL) {
    data { station_id summary_date nfdr_type fuel_model energy_release_component_max burning_index_max }
  }
}
"""


def norm_id(v):
    v = str(v).strip()
    if v.endswith(".0") and v[:-2].isdigit():
        v = v[:-2]
    return v.lstrip("0") or "0"


def rnd(x, places=1):
    if pd.isna(x):
        return None
    q = Decimal(1).scaleb(-places)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def gql(variables, tries=4):
    for attempt in range(1, tries + 1):
        try:
            r = requests.post(ENDPOINT, headers=HEADERS, auth=AUTH,
                              json={"query": Q_NFDR_MINMAX, "variables": variables}, timeout=600)
            ct = r.headers.get("content-type", "")
            j = r.json() if "application/json" in ct else {"errors": [{"message": r.text[:300]}]}
            if j.get("errors"):
                raise RuntimeError(j["errors"])
            r.raise_for_status()
            return j["data"]["nfdrMinMax"]["data"] or []
        except Exception as e:
            print(f"  attempt {attempt} failed: {e}")
            if attempt == tries:
                raise
            time.sleep(10 * attempt)


def main():
    fdra = pd.read_csv(os.path.join(DATA_DIR, "r3_fdra.csv"), dtype=str).dropna(subset=["FDRA_ZONE"])
    fdra = fdra[fdra["FDRA_ZONE"].str.strip() != ""]
    fdra["sig"] = fdra["SIG_RAWS"].map(lambda s: [norm_id(x) for x in str(s).split(",") if x.strip()])
    fdra["fm"] = fdra["FM_Long"].fillna("Y").str.strip().replace("", "Y")

    rows = []
    for fm, group in fdra.groupby("fm"):
        stations = sorted({s for lst in group["sig"] for s in lst})
        print(f"Fuel model {fm}: {len(stations)} stations")
        for year in range(START_YEAR, END_YEAR + 1):
            data = gql({"startDate": f"{year}-01-01", "endDate": f"{year}-12-31",
                        "stationIds": ",".join(stations), "fuelModels": fm})
            print(f"  {year}: {len(data)} rows")
            rows.extend(data)
            time.sleep(2)

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("FEMS returned no historical rows — nothing written")
    raw = raw[raw["nfdr_type"] == "O"].copy()
    raw["sid"] = raw["station_id"].map(norm_id)
    raw["erc"] = pd.to_numeric(raw["energy_release_component_max"], errors="coerce")
    raw["bi"] = pd.to_numeric(raw["burning_index_max"], errors="coerce")
    raw = raw.drop_duplicates(["sid", "summary_date", "fuel_model"], keep="last")
    raw[["sid", "summary_date", "fuel_model", "erc", "bi"]].to_csv(RAW_OUT, index=False)
    print(f"Saved raw history: {RAW_OUT} ({len(raw)} rows)")

    out = []
    for _, r in fdra.iterrows():
        sub = raw[(raw["sid"].isin(r["sig"])) & (raw["fuel_model"] == r["fm"])]
        if sub.empty:
            print(f"  {r['FDRA_ZONE']}: no history")
            continue
        daily = sub.groupby("summary_date")[["erc", "bi"]].mean().reset_index()
        d = pd.to_datetime(daily["summary_date"])
        daily = daily[~((d.dt.month == 2) & (d.dt.day == 29))]      # drop Feb 29 so every year has 365 days
        d = pd.to_datetime(daily["summary_date"])
        daily["month_day"] = d.dt.strftime("%m-%d")
        daily["year"] = d.dt.year
        clim = daily.groupby("month_day").agg(erc_avg=("erc", "mean"), erc_max=("erc", "max"),
                                              bi_avg=("bi", "mean"), bi_max=("bi", "max"),
                                              years=("year", "nunique")).reset_index()
        clim["doy"] = pd.to_datetime("2001-" + clim["month_day"]).dt.dayofyear
        clim.insert(0, "FDRA_ZONE", r["FDRA_ZONE"].strip())
        out.append(clim)
        print(f"  {r['FDRA_ZONE']}: {daily['year'].nunique()} years")

    res = pd.concat(out, ignore_index=True).sort_values(["FDRA_ZONE", "doy"])
    for c in ["erc_avg", "erc_max", "bi_avg", "bi_max"]:
        res[c] = res[c].map(lambda v: rnd(v, 1))
    res[["FDRA_ZONE", "doy", "month_day", "erc_avg", "erc_max", "bi_avg", "bi_max", "years"]].to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {res['FDRA_ZONE'].nunique()} FDRAs, {len(res)} rows ({START_YEAR}-{END_YEAR})")


if __name__ == "__main__":
    main()
