"""
build_web_data.py — turns the release CSVs into small JSON files for the website.

Reads (from FEMS_DATA_DIR, default "data"):
  history_daily_nfdrminmax.csv        (obs-history release)
  history_daily_weatherminmax.csv     (obs-history release)
  history_hourly_weather.csv          (obs-history release, last 7 days used)
  obs_nfdr_daily_fdra.csv             (obs-history release, optional)
  history_forecast_daily_nfdr.csv     (forecast-data release, latest pull used)
  r3_fdra.csv, r3_fdra_percentile_historical.csv, stations_az.csv, stations_nm.csv (repo)

Writes (to WEB_OUT_DIR, default "site/data"):
  meta.json                 stations + FDRAs + latest values (loads first, small)
  fdra/<key>.json           season + forecast daily FDRA averages, percentile lines
  stations/<id>.json        daily NFDR per fuel model, forecast, daily wx, 7-day hourly wx

This script only reads the existing outputs. It does not change any pipeline file.
"""
import json
import os
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("FEMS_DATA_DIR", "data")
OUT_DIR = os.environ.get("WEB_OUT_DIR", os.path.join("site", "data"))
HOURLY_DAYS = 7

P = lambda name: os.path.join(DATA_DIR, name)

# Daily NFDR columns kept for the site (FEMS display labels -> short JSON keys)
NFDR_COLS = {
    "Max ERC": "erc", "Max BI": "bi", "Max IC": "ic", "Max SC": "sc",
    "Min 1 hr FM": "fm1", "Min 10 hr FM": "fm10", "Min 100 hr FM": "fm100",
    "Min 1000 hr FM": "fm1000", "KBDI": "kbdi", "GSI": "gsi",
    "Herb FM": "herb", "Woody FM": "woody",
}
# FDRA daily average column -> historical percentile column
FDRA_INDEX = {
    "erc": ("AvgMaxERC", "PctMaxERC", "AvgERC"),
    "bi": ("AvgMaxBI", "PctMaxBI", "AvgBI"),
    "ic": ("AvgMaxIC", "PctMaxIC", "AvgIC"),
    "sc": ("AvgMaxSC", "PctMaxSC", "AvgSC"),
    "fm1": ("AvgMin1HrFM", "PctMin1HrFM", "Avg1HrFM"),
    "fm10": ("AvgMin10HrFM", "PctMin10HrFM", "Avg10HrFM"),
    "fm100": ("AvgMin100HrFM", "PctMin100HrFM", "Avg100HrFM"),
    "fm1000": ("AvgMin1000HrFM", "PctMin1000HrFM", "Avg1000HrFM"),
    "kbdi": ("AvgKBDI", "PctKBDI", "AvgKBDI"),
    "gsi": ("AvgGSI", "PctGSI", "AvgGSI"),
    "herb": ("AvgHerbFM", "PctHerbFM", "AvgHerbFM"),
    "woody": ("AvgWoodFM", "PctWoodFM", "AvgWoodFM"),
}
WX_DAILY_COLS = {
    "Min Temp F": "tmin", "Max Temp F": "tmax", "Min RH pct": "rhmin",
    "Max RH pct": "rhmax", "Max Wind mph": "wind", "Max Wind Gust mph": "gust",
    "Precip Total in": "precip", "Max VPD": "vpdmax",
}
WX_HOURLY_COLS = {
    "temperature": "t", "relative_humidity": "rh", "wind_speed": "ws",
    "wind_direction": "wd", "peak_gust_speed": "gs", "hourly_precip": "p",
    "sol_rad": "sr", "vpd": "vpd",
}
REF_PCTS = [50, 80, 90, 97]

# Fire danger classes by percentile (from the legend the team uses). Change here only —
# the website reads these from meta.json.
CLASSES = [
    {"name": "Low", "min": 0, "max": 34, "color": "#a9d18e"},
    {"name": "Moderate", "min": 35, "max": 64, "color": "#9dc3e6"},
    {"name": "High", "min": 65, "max": 84, "color": "#ffe699"},
    {"name": "Very High", "min": 85, "max": 97, "color": "#f4b183"},
    {"name": "Extreme", "min": 98, "max": 100, "color": "#ff8f8f"},
]
FIRE_DANGER_INDEX = "erc"          # used for the ERC method, and as the fallback when VPD/SFDI are missing
# Fire Danger method:
#   "matrix" = VPD class x SFDI class, the same grid as the Kaibab FDOP workbook (Prevention tab).
#              Each FDRA's VPD and SFDI classes come from its own 2005-2024 percentiles and the CLASSES legend.
#   "erc"    = ERC percentile class only.
FIRE_DANGER_METHOD = "matrix"
# Rows = VPD class (Low..Extreme), columns = SFDI class (Low..Extreme); 0=Low 1=Moderate 2=High 3=Very High 4=Extreme
FD_MATRIX = [
    [0, 0, 0, 1, 1],
    [0, 1, 1, 1, 2],
    [1, 1, 2, 2, 3],
    [1, 2, 3, 3, 4],
    [2, 3, 3, 4, 4],
]
BREAKPOINT_INDICES = ["erc", "bi"]  # shown in the Fire Danger Breakpoints table
# Fuel moistures: a wetter fuel has a HIGHER historical percentile, which is LOWER danger.
# The report shows these as dryness percentiles (100 - pct) so the colors mean danger, like the CA dashboard.
#   GSI too: higher GSI = greener = lower danger. (Same as the Kaibab FDOP workbook's percentile tables.)
INVERT_PCT = ["fm100", "fm1000", "gsi"]
# SFDI / VPD / HDWI percentiles come from the one-time 2005-2024 history pull (r3_fdra_percentile_extra.csv).
EXTRA_PCT_FILE = "r3_fdra_percentile_extra.csv"
EXTRA_INDICES = {"sfdi": "SFDI", "vpd": "VPD", "hdwi": "HDWI"}

# Daily report columns: (key, label, group, has_percentile)
REPORT_COLS = [
    ("erc", "ERC", "Fuels", True), ("bi", "BI", "Fuels", True),
    ("sfdi", "SFDI", "Fuels", True), ("vpd", "VPD", "Fuels", True), ("hdwi", "HDWI", "Fuels", True),
    ("fm100", "100HR", "Fuels", True), ("fm1000", "1000HR", "Fuels", True),
    ("kbdi", "KBDI", "Fuels", True), ("gsi", "GSI", "Fuels", True),
    ("tmax", "Max T", "Weather", False), ("tmin", "Min T", "Weather", False),
    ("wind", "Max Wind", "Weather", False), ("gust", "Max Gust", "Weather", False),
    ("rhmin", "Min RH", "Weather", False), ("precip", "Precip", "Weather", False),
]


def rnd(x, places=1):
    """Decimal ROUND_HALF_UP, same rule as the pipeline. Returns None for blanks."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        return None
    q = Decimal(1).scaleb(-places)
    v = float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))
    return int(v) if places == 0 else v


def norm_id(v):
    v = str(v).strip()
    if v.endswith(".0") and v[:-2].isdigit():
        v = v[:-2]
    return v.lstrip("0") or "0"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def load_station_list(name):
    path = P(name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [norm_id(line.split(",")[0]) for line in f if line.strip()]


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"), allow_nan=False)


def vpd_pa(tmax_f, rhmin):
    """Daily VPD (Pa) from Max Temp (F) and Min RH (%), same Magnus formula as the Kaibab FDOP workbook."""
    if tmax_f is None or rhmin is None or pd.isna(tmax_f) or pd.isna(rhmin):
        return None
    tc = (float(tmax_f) - 32) * 5 / 9
    es = 6.112 * np.exp((17.67 * tc) / (tc + 243.5))
    return (es - (float(rhmin) / 100) * es) * 100


def sfdi_of(erc, bi):
    """Severe Fire Danger Index = ERC x BI."""
    return None if erc is None or bi is None or pd.isna(erc) or pd.isna(bi) else float(erc) * float(bi)


def hdwi_of(vpd, wind_mph):
    """Hot-Dry-Windy Index as in the workbook: VPD/100 x wind (m/s)."""
    return None if vpd is None or wind_mph is None or pd.isna(vpd) or pd.isna(wind_mph) else (float(vpd) / 100) * (float(wind_mph) * 0.447)


def class_of_pct(p):
    """Index into CLASSES for a percentile (0=Low ... 4=Extreme)."""
    if p is None:
        return None
    for i, c in enumerate(CLASSES):
        if c["min"] <= p <= c["max"]:
            return i
    return len(CLASSES) - 1 if p > CLASSES[-1]["max"] else 0


def fire_danger_class(res):
    """Returns (class index, method used) for one report row."""
    if FIRE_DANGER_METHOD == "matrix":
        cv, cs = class_of_pct(res.get("vpd_pct")), class_of_pct(res.get("sfdi_pct"))
        if cv is not None and cs is not None:
            return FD_MATRIX[cv][cs], "matrix"
    c = class_of_pct(res.get(FIRE_DANGER_INDEX + "_pct"))
    return c, (FIRE_DANGER_INDEX if c is not None else None)


def percentile_for(value, thresholds):
    """Highest historical percentile whose threshold is <= value (same rule as obs_nfdr_daily_fdra.py)."""
    if value is None or not thresholds:
        return None
    vals = np.array([t for t, _ in thresholds])
    pos = np.searchsorted(vals, float(value), side="right") - 1
    if pos < 0:
        return int(thresholds[0][1])
    return int(thresholds[pos][1])


def main():
    now = datetime.now(timezone.utc)

    # ---------- stations ----------
    az = set(load_station_list("stations_az.csv"))
    nm = set(load_station_list("stations_nm.csv"))

    nfdr = pd.read_csv(P("history_daily_nfdrminmax.csv"), dtype={"station_id": str})
    nfdr["sid"] = nfdr["station_id"].map(norm_id)
    nfdr = nfdr[nfdr["nfdr_type"] == "O"]
    nfdr = nfdr.sort_values("pull_date" if "pull_date" in nfdr else "summary_date")
    nfdr = nfdr.drop_duplicates(["sid", "summary_date", "fuel_model"], keep="last")
    latest_obs_date = nfdr["summary_date"].max()

    fc = pd.read_csv(P("history_forecast_daily_nfdr.csv"), dtype={"station_id": str})
    fc["sid"] = fc["station_id"].map(norm_id)
    fc = fc[fc["nfdr_type"] == "F"]
    latest_pull = fc["Pull_date_mst"].max()
    fc = fc[(fc["Pull_date_mst"] == latest_pull) & (fc["summary_date"] > latest_obs_date)]
    fc = fc.drop_duplicates(["sid", "summary_date", "fuel_model"], keep="last")

    wxd = pd.read_csv(P("history_daily_weatherminmax.csv"), dtype={"station_id": str})
    wxd["sid"] = wxd["station_id"].map(norm_id)
    wxd = wxd[wxd["observation_type"] == "O"].drop_duplicates(["sid", "summary_date"], keep="last")

    fwx_path = P("history_forecast_daily_weather.csv")
    fwx = pd.DataFrame(columns=["sid", "summary_date", *WX_DAILY_COLS.keys()])
    if os.path.exists(fwx_path):
        fwx = pd.read_csv(fwx_path, dtype={"station_id": str})
        fwx["sid"] = fwx["station_id"].map(norm_id)
        fwx = fwx[(fwx["Pull_date_mst"] == fwx["Pull_date_mst"].max()) & (fwx["observation_type"] == "F")]
        fwx = fwx.drop_duplicates(["sid", "summary_date"], keep="last")
    # "Today" is the first forecast day; yesterday is the latest observed day
    fc_days = sorted(fc["summary_date"].unique())
    today = fc_days[0] if fc_days else None

    hourly_path = P("history_hourly_weather.csv")
    wxh = None
    if os.path.exists(hourly_path):
        wxh = pd.read_csv(hourly_path, dtype={"station_id": str},
                          usecols=["station_id", "observation_type", "display_hour_lst", "pull_date",
                                   *WX_HOURLY_COLS.keys()])
        wxh["sid"] = wxh["station_id"].map(norm_id)
        wxh = wxh[wxh["observation_type"] == "O"]
        wxh["hour"] = wxh["display_hour_lst"].str.slice(0, 16)  # 2026-08-18T13:00
        cutoff = (pd.to_datetime(wxh["hour"]).max() - timedelta(days=HOURLY_DAYS)).strftime("%Y-%m-%dT%H:%M")
        # Observed daily Max VPD is blank in history_daily_weatherminmax.csv right now,
        # so fill it from the hourly file (daily max of hourly VPD) where it is missing.
        hv = wxh.assign(summary_date=wxh["hour"].str.slice(0, 10))
        hv = hv.groupby(["sid", "summary_date"])["vpd"].max().rename("vpd_from_hourly").reset_index()
        wxd = wxd.merge(hv, on=["sid", "summary_date"], how="left")
        wxd["Max VPD"] = pd.to_numeric(wxd["Max VPD"], errors="coerce").fillna(wxd["vpd_from_hourly"])
        wxh = wxh[wxh["hour"] > cutoff].sort_values("pull_date")
        wxh = wxh.drop_duplicates(["sid", "hour"], keep="last").sort_values("hour")

    # ---------- FDRAs ----------
    fdra = pd.read_csv(P("r3_fdra.csv"), dtype=str).dropna(subset=["FDRA_ZONE"])
    fdra = fdra[fdra["FDRA_ZONE"].str.strip() != ""]
    hist = pd.read_csv(P("r3_fdra_percentile_historical.csv"))
    extra_path = P(EXTRA_PCT_FILE)
    extra = pd.read_csv(extra_path) if os.path.exists(extra_path) else None
    climo_path = P("r3_fdra_daily_climo.csv")   # optional: from the one-time 2005-2024 history pull
    climo = pd.read_csv(climo_path) if os.path.exists(climo_path) else None
    fdra_obs_path = P("obs_nfdr_daily_fdra.csv")
    fdra_official = pd.read_csv(fdra_obs_path) if os.path.exists(fdra_obs_path) else None

    station_fdras = {}
    fdra_list = []
    for _, r in fdra.iterrows():
        key = r["FDRA_ZONE"].strip()
        sig = [norm_id(s) for s in str(r["SIG_RAWS"]).split(",") if s.strip()]
        fm = (r.get("FM_Long") or "Y").strip() or "Y"
        for s in sig:
            station_fdras.setdefault(s, []).append(key)

        h = hist[hist["FDRA_ZONE"].str.strip() == key].sort_values("Percentiles")
        thresholds = {}
        refs = {}
        for k, (_, _, hcol) in FDRA_INDEX.items():
            pairs = sorted(zip(h[hcol].astype(float), h["Percentiles"].astype(int)))
            thresholds[k] = pairs
            row = {int(p): v for v, p in pairs}
            refs[k] = {str(p): rnd(row.get(p), 2) for p in REF_PCTS}
        for k, col in EXTRA_INDICES.items():
            thresholds[k] = []
            if extra is not None:
                e = extra[extra["FDRA_ZONE"].str.strip() == key]
                thresholds[k] = sorted(zip(e[col].astype(float), e["Percentiles"].astype(int)))

        # FDRA daily averages from station data (season + forecast)
        def daily_avg(df, kind):
            sub = df[(df["sid"].isin(sig)) & (df["fuel_model"] == fm)]
            out = []
            for d, g in sub.groupby("summary_date"):
                row = {"d": d, "n": int(len(g)), "kind": kind}
                for src, k in NFDR_COLS.items():
                    v = pd.to_numeric(g[src], errors="coerce").mean()
                    row[k] = rnd(v, 2)
                for k in ("erc", "bi"):
                    row[k + "_pct"] = percentile_for(row[k], thresholds[k])
                out.append(row)
            return out

        season = daily_avg(nfdr, "O")
        forecast = daily_avg(fc, "F")

        # Latest = official FDRA file when it has this FDRA, else computed
        latest = None
        if fdra_official is not None:
            o = fdra_official[fdra_official["FDRA_ZONE"].str.strip() == key]
            if len(o):
                o = o.sort_values("ObservationDate").iloc[-1]
                latest = {"d": o["ObservationDate"], "n": int(o["StationsUsed"]), "source": "obs_nfdr_daily_fdra"}
                for k, (acol, pcol, _) in FDRA_INDEX.items():
                    latest[k] = rnd(o[acol], 2)
                    latest[k + "_pct"] = rnd(o[pcol], 0)
        if latest is None and season:
            s = season[-1]
            latest = {**s, "source": "computed"}
            for k in FDRA_INDEX:
                latest[k + "_pct"] = percentile_for(s.get(k), thresholds[k])

        # ---------- daily report table (Yesterday / Today / Tomorrow / 5-day averages) ----------
        def day_values(nf, wx, days):
            """FDRA values for the given days, following the Kaibab FDOP workbook:
            each day = average of the FDRA's stations; VPD from that day's averaged Max T and Min RH;
            several days = average of the daily values, SFDI = avg ERC x avg BI, HDWI from avg VPD and avg wind,
            precip = total over the days."""
            s_n = nf[(nf["sid"].isin(sig)) & (nf["fuel_model"] == fm) & (nf["summary_date"].isin(days))]
            s_w = wx[(wx["sid"].isin(sig)) & (wx["summary_date"].isin(days))]
            if s_n.empty and s_w.empty:
                return None
            src = {"erc": "Max ERC", "bi": "Max BI", "fm100": "Min 100 hr FM",
                   "fm1000": "Min 1000 hr FM", "kbdi": "KBDI", "gsi": "GSI"}
            wsrc = {"tmax": "Max Temp F", "tmin": "Min Temp F", "wind": "Max Wind mph",
                    "gust": "Max Wind Gust mph", "rhmin": "Min RH pct", "precip": "Precip Total in"}
            daily = pd.DataFrame(index=sorted(days))
            for k, c in src.items():
                daily[k] = pd.to_numeric(s_n[c], errors="coerce").groupby(s_n["summary_date"]).mean()
            for k, c in wsrc.items():
                daily[k] = pd.to_numeric(s_w[c], errors="coerce").groupby(s_w["summary_date"]).mean() if len(s_w) else np.nan
            daily["vpd"] = [vpd_pa(t, h) for t, h in zip(daily["tmax"], daily["rhmin"])]
            daily["vpd"] = pd.to_numeric(daily["vpd"], errors="coerce")
            avg = daily.mean(numeric_only=True)
            out = {}
            for k in [*src, "vpd", *[w for w in wsrc if w != "precip"]]:
                out[k] = avg.get(k)
            out["precip"] = daily["precip"].sum(min_count=1) if "precip" in daily else None
            out["sfdi"] = sfdi_of(out["erc"], out["bi"])
            out["hdwi"] = hdwi_of(out["vpd"], out["wind"])
            res = {}
            for k, v in out.items():
                res[k] = rnd(v, 2 if k in ("gsi", "precip") else 0)
                if k in thresholds:
                    pct = percentile_for(v, thresholds[k]) if (v is not None and pd.notna(v)) else None
                    res[k + "_pct"] = (100 - pct) if (pct is not None and k in INVERT_PCT) else pct
            res["fd"], res["fd_method"] = fire_danger_class(res)
            return res

        obs_days = sorted(nfdr["summary_date"].unique())
        future = [d for d in fc_days if today and d > today]
        report = []
        for label, nf, wx, days in [
            ("Yesterday", nfdr, wxd, obs_days[-1:]),
            ("Today", fc, fwx, [today] if today else []),
            ("Tomorrow", fc, fwx, future[:1]),
            ("5-Day Average", nfdr, wxd, obs_days[-5:]),            # last 5 observed days
            ("5-Day Outlook", fc, fwx, ([today] if today else []) + future[:4]),   # today + next 4 days
        ]:
            report.append({"label": label, "days": days, "type": "O" if nf is nfdr else "F",
                           "values": day_values(nf, wx, days) if days else None})

        breakpoints = []
        for c in CLASSES:
            row = {"class": c["name"], "range": f'{c["min"]}-{c["max"]}%'}
            for k in [*BREAKPOINT_INDICES, *[x for x in EXTRA_INDICES if thresholds[x]]]:
                lo = next((v for v, p in thresholds[k] if p >= max(c["min"], 1)), None)
                hi = next((v for v, p in reversed(thresholds[k]) if p <= c["max"]), None)
                row[k] = [rnd(lo, 0), rnd(hi, 0)]
            breakpoints.append(row)

        item = {
            "report": report, "breakpoints": breakpoints,
            "key": key, "slug": slug(key), "name": r["FDRA"].strip(), "zone": r["Zone"].strip(),
            "fdra_id": r["FDRA_ID"], "fuel_model": fm, "sig_raws": sig,
            "mfd": r.get("MFD_Thresh"), "lfd": r.get("LFD_Threshold"),
            "analysis_period": r.get("Analysis_Period"), "latest": latest,
            "erc_refs": refs["erc"], "bi_refs": refs["bi"],
        }
        fdra_list.append(item)
        fdra_file = {"key": key, "fuel_model": fm, "refs": refs, "season": season, "forecast": forecast}
        if climo is not None:
            c = climo[climo["FDRA_ZONE"].str.strip() == key].sort_values("doy")
            if len(c):
                fdra_file["climo"] = [{"doy": int(r.doy), "erc_avg": rnd(r.erc_avg, 1), "erc_max": rnd(r.erc_max, 1),
                                       "bi_avg": rnd(r.bi_avg, 1), "bi_max": rnd(r.bi_max, 1)} for r in c.itertuples()]
        write_json(os.path.join(OUT_DIR, "fdra", item["slug"] + ".json"), fdra_file)

    # ---------- per-station files ----------
    info_cols = ["station_name", "wrcc_id", "latitude", "longitude", "elevation"]
    info = nfdr.sort_values("summary_date").groupby("sid")[info_cols].last()
    stations = []
    for sid, inf in info.iterrows():
        state = "AZ" if sid in az else "NM" if sid in nm else ""
        sn = nfdr[nfdr["sid"] == sid]
        by_fm, fc_fm, latest = {}, {}, {}
        for fm, g in sn.groupby("fuel_model"):
            g = g.sort_values("summary_date")
            by_fm[fm] = [{"d": d, **{k: rnd(v, 1) for k, v in zip(NFDR_COLS.values(), vals)}}
                         for d, *vals in g[["summary_date", *NFDR_COLS.keys()]].itertuples(index=False)]
            if by_fm[fm]:
                latest[fm] = by_fm[fm][-1]
        for fm, g in fc[fc["sid"] == sid].groupby("fuel_model"):
            g = g.sort_values("summary_date")
            fc_fm[fm] = [{"d": d, **{k: rnd(v, 1) for k, v in zip(NFDR_COLS.values(), vals)}}
                         for d, *vals in g[["summary_date", *NFDR_COLS.keys()]].itertuples(index=False)]
        def wx_rows(df):
            df = df.sort_values("summary_date")
            rows = []
            for d, *vals in df[["summary_date", *WX_DAILY_COLS.keys()]].itertuples(index=False):
                row = {"d": d, **{k: rnd(v, 2 if k == "precip" else 0) for k, v in zip(WX_DAILY_COLS.values(), vals)}}
                row["vpdmax"] = rnd(vpd_pa(row.get("tmax"), row.get("rhmin")), 0)   # same VPD formula as the FDRA tables
                rows.append(row)
            return rows
        wx_daily = wx_rows(wxd[wxd["sid"] == sid])
        wx_fc = wx_rows(fwx[fwx["sid"] == sid]) if len(fwx) else []
        wx_hourly = []
        if wxh is not None:
            wh = wxh[wxh["sid"] == sid]
            wx_hourly = [{"h": h, **{k: rnd(v, 2 if k == "p" else 0) for k, v in zip(WX_HOURLY_COLS.values(), vals)}}
                         for h, *vals in wh[["hour", *WX_HOURLY_COLS.keys()]].itertuples(index=False)]
        write_json(os.path.join(OUT_DIR, "stations", sid + ".json"),
                   {"id": sid, "nfdr": by_fm, "forecast": fc_fm, "wx_daily": wx_daily, "wx_forecast": wx_fc, "wx_hourly": wx_hourly})

        last_wx = wx_hourly[-1] if wx_hourly else None
        stations.append({
            "id": sid, "name": inf["station_name"], "wrcc": inf["wrcc_id"],
            "lat": rnd(inf["latitude"], 5), "lon": rnd(inf["longitude"], 5),
            "elev": rnd(inf["elevation"], 0), "state": state,
            "fdras": station_fdras.get(sid, []),
            "latest": {fm: {"d": v["d"], "erc": v["erc"], "bi": v["bi"], "fm100": v["fm100"],
                            "fm1000": v["fm1000"], "kbdi": v["kbdi"]} for fm, v in latest.items()},
            "wx": last_wx,
        })

    write_json(os.path.join(OUT_DIR, "meta.json"), {
        "generated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "latest_obs_date": latest_obs_date,
        "forecast_pull_mst": latest_pull,
        "today": today,
        "classes": CLASSES, "fire_danger_index": FIRE_DANGER_INDEX,
        "fire_danger_method": FIRE_DANGER_METHOD, "fd_matrix": FD_MATRIX,
        "report_cols": [{"key": k, "label": l, "group": g, "pct": p} for k, l, g, p in REPORT_COLS],
        "breakpoint_indices": BREAKPOINT_INDICES + ([*EXTRA_INDICES] if extra is not None else []), "inverted_pct": INVERT_PCT,
        "stations": stations,
        "fdras": fdra_list,
    })
    print(f"Wrote {len(stations)} stations and {len(fdra_list)} FDRAs to {OUT_DIR} "
          f"(obs through {latest_obs_date}, forecast pull {latest_pull})")


if __name__ == "__main__":
    main()
