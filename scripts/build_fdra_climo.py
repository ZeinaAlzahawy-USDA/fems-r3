"""
build_fdra_climo.py — ONE-TIME pull of 2005–2024 daily NFDR history from FEMS,
turned into the "Average" and "Historical max" lines for the website charts.

For each FDRA in data/r3_fdra.csv:
  1. Pull daily observed Max ERC and Max BI for its SIG_RAWS stations (fuel model = FM_Long).
  2. FDRA daily value = average of its stations that reported that day.
  3. For each calendar day (Jan 1 … Dec 31): average and maximum of that day across all years.

Outputs:
  data/r3_fdra_daily_climo.csv      FDRA_ZONE, doy, month_day, erc_avg, erc_max, bi_avg, bi_max, years
  data/r3_fdra_percentile_extra.csv FDRA_ZONE, Percentiles (0-100), SFDI, VPD, HDWI
      (2005-2024 percentiles for the indices the FDRA percentile file does not have; same formulas as the
       Kaibab FDOP workbook: SFDI = ERC x BI, VPD from Max T and Min RH, HDWI = VPD/100 x max wind (m/s))
  data/r3_fdra_percentile_historical.csv  every FDRA, percentiles 1-100 of the daily FDRA averages for
      1/10/100/1000-hr FM, woody/herb FM, KBDI, GSI, IC, ERC, SC, BI (same method as the original file:
      days on which every FDRA station reported, averaged, linear percentiles, 2 decimals)
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
EXTRA_OUT = os.path.join(DATA_DIR, "r3_fdra_percentile_extra.csv")
RAW_OUT = os.path.join(DATA_DIR, f"history_{START_YEAR}_{END_YEAR}_daily_erc_bi.csv")
PCT_OUT = os.path.join(DATA_DIR, "r3_fdra_percentile_historical.csv")
# FEMS field -> column name in r3_fdra_percentile_historical.csv
PCT_FIELDS = {
    "one_hr_tl_fuel_moisture_min": "Avg1HrFM", "ten_hr_tl_fuel_moisture_min": "Avg10HrFM",
    "hun_hr_tl_fuel_moisture_min": "Avg100HrFM", "thou_hr_tl_fuel_moisture_min": "Avg1000HrFM",
    "woody_lfi_fuel_moisture": "AvgWoodFM", "kbdi": "AvgKBDI", "gsi": "AvgGSI",
    "herbaceous_lfi_fuel_moisture": "AvgHerbFM", "ignition_component_max": "AvgIC",
    "energy_release_component_max": "AvgERC", "spread_component_max": "AvgSC", "burning_index_max": "AvgBI",
}
PCT_INFO_COLS = ["FDRA", "Zone", "FDRA_ID", "MFD_Thresh", "LFD_Threshold", "SIG_RAWS", "Analysis_Period",
                 "FM_Short", "FM_Long", "FDRA_ZONE"]

AUTH = HTTPBasicAuth(os.environ["FEMS_USERNAME"], os.environ["FEMS_API_KEY"])
HEADERS = {"Content-Type": "application/json", "Accept": "application/json",
           "User-Agent": "FEMS-NM-AZ-GitHubActions/1.0"}

Q_NFDR_MINMAX = """
query NfdrMinMax($startDate: Date!, $endDate: Date, $stationIds: String, $fuelModels: String) {
  nfdrMinMax(startDate: $startDate, endDate: $endDate, stationIds: $stationIds, fuelModels: $fuelModels, hasHistoricData: ALL) {
    data { station_id summary_date nfdr_type fuel_model energy_release_component_max burning_index_max
           ignition_component_max spread_component_max kbdi gsi
           one_hr_tl_fuel_moisture_min ten_hr_tl_fuel_moisture_min hun_hr_tl_fuel_moisture_min
           thou_hr_tl_fuel_moisture_min herbaceous_lfi_fuel_moisture woody_lfi_fuel_moisture }
  }
}
"""


Q_WX_MINMAX = """
query WxMinMax($startDate: Date!, $endDate: Date, $stationIds: String) {
  wxMinMax(startDate: $startDate, endDate: $endDate, stationIds: $stationIds, hasHistoricData: ALL) {
    data { station_id summary_date observation_type temperature_max relative_humidity_min wind_speed_max }
  }
}
"""


def vpd_pa(tmax_f, rhmin):
    """Same Magnus formula as the Kaibab FDOP workbook (and build_web_data.py)."""
    import numpy as np
    tc = (tmax_f - 32) * 5 / 9
    es = 6.112 * np.exp((17.67 * tc) / (tc + 243.5))
    return (es - (rhmin / 100) * es) * 100


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


def gql(variables, tries=4, query=Q_NFDR_MINMAX, field="nfdrMinMax"):
    for attempt in range(1, tries + 1):
        try:
            r = requests.post(ENDPOINT, headers=HEADERS, auth=AUTH,
                              json={"query": query, "variables": variables}, timeout=600)
            ct = r.headers.get("content-type", "")
            j = r.json() if "application/json" in ct else {"errors": [{"message": r.text[:300]}]}
            if j.get("errors"):
                raise RuntimeError(j["errors"])
            r.raise_for_status()
            return j["data"][field]["data"] or []
        except Exception as e:
            print(f"  attempt {attempt} failed: {e}")
            if attempt == tries:
                raise
            time.sleep(10 * attempt)


def build_percentile_history(fdra, raw):
    """Percentiles 1-100 for every FDRA with its current stations and fuel model.
    Same method as the original r3_fdra_percentile_historical.csv (checked: it reproduces that file exactly):
    for each index, the days on which every FDRA station reported are averaged, then linear percentiles."""
    import numpy as np
    pcts = list(range(1, 101))
    period = f"1/1/{START_YEAR} - 12/31/{END_YEAR}"
    old_path = PCT_OUT
    old = pd.read_csv(old_path, dtype=str) if os.path.exists(old_path) else None
    frames = []
    for _, r in fdra.iterrows():
        key = r["FDRA_ZONE"].strip()
        sig = sorted(set(r["sig"]))
        sub = raw[(raw["sid"].isin(sig)) & (raw["fuel_model"] == r["fm"])]
        out = pd.DataFrame({"Percentiles": pcts})
        for c in PCT_INFO_COLS:
            out[c] = r.get(c, "") if c not in ("FDRA_ZONE", "Analysis_Period", "FM_Long") else None
        out["FDRA_ZONE"], out["Analysis_Period"], out["FM_Long"] = key, period, r["fm"]
        out["SIG_RAWS"] = ", ".join(sig)
        days_used = {}
        for field, col in PCT_FIELDS.items():
            s = sub.dropna(subset=[field])
            n = s.groupby("summary_date")["sid"].nunique()
            daily = s.groupby("summary_date")[field].mean()
            daily = daily[n[n == len(sig)].index]           # only days on which every FDRA station reported
            days_used[col] = len(daily)
            out[col] = [round(float(v), 2) for v in np.percentile(daily, pcts)] if len(daily) else None
        frames.append(out)
        print(f"  {key}: percentiles from {days_used['AvgERC']} complete days (ERC)")
    res = pd.concat(frames, ignore_index=True)[[*PCT_INFO_COLS, "Percentiles", *PCT_FIELDS.values()]]

    # Self-check against the previous file for FDRAs whose stations and fuel model did not change
    if old is not None:
        norm = lambda s: ", ".join(sorted({x.strip().split(".")[0].lstrip("0") for x in str(s).split(",") if x.strip()}))
        checked, worst = 0, {}
        for key, o in old.groupby("FDRA_ZONE"):
            n_ = res[res["FDRA_ZONE"] == key.strip()]
            if n_.empty or norm(o["SIG_RAWS"].iloc[0]) != n_["SIG_RAWS"].iloc[0] or str(o["FM_Long"].iloc[0]).strip() != n_["FM_Long"].iloc[0]:
                continue
            checked += 1
            o = o.sort_values("Percentiles", key=lambda x: x.astype(int))
            for col in PCT_FIELDS.values():
                d = (pd.to_numeric(o[col]).values - n_[col].values)
                worst[col] = max(worst.get(col, 0), float(np.nanmax(np.abs(d))))
        print(f"Self-check against the previous percentile file ({checked} unchanged FDRAs), largest difference per column:")
        for col, d in worst.items():
            print(f"  {col}: {d:.2f}{'' if d <= 0.02 else '   <-- differs from the previous file'}")
    res.to_csv(PCT_OUT, index=False)
    print(f"Wrote {PCT_OUT}: {res['FDRA_ZONE'].nunique()} FDRAs x 100 percentiles")


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
    for c in PCT_FIELDS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.drop_duplicates(["sid", "summary_date", "fuel_model"], keep="last")
    raw[["sid", "summary_date", "fuel_model", "erc", "bi", *[c for c in PCT_FIELDS if c not in
         ("energy_release_component_max", "burning_index_max")]]].to_csv(RAW_OUT, index=False)
    print(f"Saved raw history: {RAW_OUT} ({len(raw)} rows)")

    build_percentile_history(fdra, raw)

    # ---- weather history (for VPD / HDWI) ----
    all_stations = sorted({s for lst in fdra["sig"] for s in lst})
    wrows = []
    for year in range(START_YEAR, END_YEAR + 1):
        data = gql({"startDate": f"{year}-01-01", "endDate": f"{year}-12-31", "stationIds": ",".join(all_stations)},
                   query=Q_WX_MINMAX, field="wxMinMax")
        print(f"  weather {year}: {len(data)} rows")
        wrows.extend(data)
        time.sleep(2)
    wx = pd.DataFrame(wrows)
    if not wx.empty:
        wx = wx[wx["observation_type"] == "O"].copy()
        wx["sid"] = wx["station_id"].map(norm_id)
        for c in ["temperature_max", "relative_humidity_min", "wind_speed_max"]:
            wx[c] = pd.to_numeric(wx[c], errors="coerce")
        wx = wx.drop_duplicates(["sid", "summary_date"], keep="last")

    out, extra = [], []
    for _, r in fdra.iterrows():
        sub = raw[(raw["sid"].isin(r["sig"])) & (raw["fuel_model"] == r["fm"])]
        if sub.empty:
            print(f"  {r['FDRA_ZONE']}: no history")
            continue
        daily = sub.groupby("summary_date")[["erc", "bi"]].mean().reset_index()

        # SFDI / VPD / HDWI percentiles over every day 2005-2024 (Jan 1 - Dec 31)
        if not wx.empty:
            w = wx[wx["sid"].isin(r["sig"])].groupby("summary_date")[["temperature_max", "relative_humidity_min", "wind_speed_max"]].mean()
            dd = daily.set_index("summary_date").join(w, how="inner")
            dd["SFDI"] = dd["erc"] * dd["bi"]
            dd["VPD"] = vpd_pa(dd["temperature_max"], dd["relative_humidity_min"])
            dd["HDWI"] = (dd["VPD"] / 100) * (dd["wind_speed_max"] * 0.447)
            pct = list(range(0, 101))
            ex = pd.DataFrame({"FDRA_ZONE": r["FDRA_ZONE"].strip(), "Percentiles": pct})
            for c in ["SFDI", "VPD", "HDWI"]:
                vals = dd[c].dropna()
                ex[c] = [rnd(v, 2) for v in vals.quantile([p / 100 for p in pct]).values] if len(vals) else None
            extra.append(ex)

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
    if extra:
        pd.concat(extra, ignore_index=True).to_csv(EXTRA_OUT, index=False)
        print(f"Wrote {EXTRA_OUT}: SFDI / VPD / HDWI percentiles for {len(extra)} FDRAs")


if __name__ == "__main__":
    main()
