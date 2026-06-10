"""
Builds real-data-anchored training dataset.

Sources:
  - NFDB (Canadian National Fire Database): real wildfire events, lat/lon/date/size
  - Open-Meteo Archive API: real hourly weather at each fire location/date
  - Environment Canada weather: real conditions for non-fire baseline periods
  - FWI computed in Python from real weather variables (Van Wagner 1987 algorithm)

Output: data/raw/real_sensor_training.csv
  Same schema as simulated_readings.csv — direct drop-in for model training.

Run:
    python -m data.build_training_data
    python -m data.build_training_data --max-events 300 --no-fire-ratio 1.0
"""

import argparse
import glob
import io
import math
import os
import random
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from loguru import logger

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

N_NODES = 100
AREA_KM = 10.0
NODES_PER_SIDE = 10
TIMESTEPS_PER_EVENT = 48        # 48 × 30 min = 24 h window per fire event
INTERVAL_SEC = 1800             # 30 min
# Sequential base timestamp — all events are chained end-to-end so the
# WildfireDataset sliding window stays within events (87.5 % of windows valid).
_BASE_TS = datetime(2020, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

SENSOR_COLS = [
    "node_id", "timestamp", "latitude", "longitude",
    "temperature_c", "humidity_pct", "surface_temp_c",
    "smoke_index", "co_ppm", "voc_index",
    "wind_speed_kmh", "wind_direction_deg",
    "fire_risk", "is_fire_event", "battery_pct", "signal_rssi",
]


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Canadian FWI System — Van Wagner (1987)
# Inputs: noon temperature (°C), relative humidity (%), wind speed (km/h),
#         24h precipitation (mm), and the previous day's FFMC/DMC/DC.
# ---------------------------------------------------------------------------
def compute_fwi(temp: float, rh: float, wind: float, rain: float,
                ffmc0: float = 85.0, dmc0: float = 6.0, dc0: float = 15.0
                ) -> Dict[str, float]:
    rh = max(0.0, min(100.0, rh))

    # FFMC
    mo = 147.2 * (101 - ffmc0) / (59.5 + ffmc0)
    if rain > 0.5:
        rf = rain - 0.5
        if mo <= 150:
            mr = mo + 42.5 * rf * math.exp(-100 / (251 - mo)) * (1 - math.exp(-6.93 / rf))
        else:
            mr = mo + 42.5 * rf * math.exp(-100 / (251 - mo)) * (1 - math.exp(-6.93 / rf)) + 0.0015 * (mo - 150) ** 2 * rf ** 0.5
        mo = min(mr, 250.0)
    ed = 0.942 * rh ** 0.679 + 11 * math.exp((rh - 100) / 10) + 0.18 * (21.1 - temp) * (1 - math.exp(-0.115 * rh))
    ew = 0.618 * rh ** 0.753 + 10 * math.exp((rh - 100) / 10) + 0.18 * (21.1 - temp) * (1 - math.exp(-0.115 * rh))
    if mo > ed:
        ko = 0.424 * (1 - (rh / 100) ** 1.7) + 0.0694 * wind ** 0.5 * (1 - (rh / 100) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * temp)
        m = ed + (mo - ed) * 10 ** (-kd)
    elif mo < ew:
        kl = 0.424 * (1 - ((100 - rh) / 100) ** 1.7) + 0.0694 * wind ** 0.5 * (1 - ((100 - rh) / 100) ** 8)
        kw = kl * 0.581 * math.exp(0.0365 * temp)
        m = ew - (ew - mo) * 10 ** (-kw)
    else:
        m = mo
    ffmc = 59.5 * (250 - m) / (147.2 + m)
    ffmc = max(0.0, min(101.0, ffmc))

    # DMC
    if rain > 1.5:
        re = 0.92 * rain - 1.27
        mo2 = 20 + math.exp(5.6348 - dmc0 / 43.43)
        b = (100 / (0.5 + 0.3 * dmc0)) if dmc0 <= 33 else (14 - 1.3 * math.log(dmc0)) if dmc0 <= 65 else 6.2 * math.log(dmc0) - 17.2
        mr2 = mo2 + 1000 * re / (48.77 + b * re)
        pr = 244.72 - 43.43 * math.log(mr2 - 20)
        dmc0 = max(0.0, pr)
    le = max(0.0, [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0][0])
    k = 1.894 * (temp + 1.1) * (100 - rh) * le * 1e-6
    dmc = dmc0 + 100 * k
    dmc = max(0.0, dmc)

    # DC
    if rain > 2.8:
        rd = 0.83 * rain - 1.27
        qo = 800 * math.exp(-dc0 / 400)
        qr = qo + 3.937 * rd
        dr = 400 * math.log(800 / qr) if qr > 0 else dc0
        dc0 = max(0.0, dr)
    lf = max(0.0, [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6][0])
    v = 0.36 * (temp + 2.8) + lf
    dc = dc0 + 0.5 * max(0.0, v)

    # ISI
    fw = math.exp(0.05039 * wind)
    fm = 147.2 * (101 - ffmc) / (59.5 + ffmc)
    ff = 91.9 * math.exp(-0.1386 * fm) * (1 + fm ** 5.31 / 4.93e7)
    isi = 0.208 * fw * ff

    # BUI
    if dmc <= 0.4 * dc:
        bui = 0.8 * dmc * dc / (dmc + 0.4 * dc) if (dmc + 0.4 * dc) > 0 else 0.0
    else:
        bui = dmc - (1 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)
    bui = max(0.0, bui)

    # FWI
    if bui <= 80:
        bb = 0.1 * isi * (0.626 * bui ** 0.809 + 2)
    else:
        bb = 0.1 * isi * (1000 / (25 + 108.64 * math.exp(-0.023 * bui)))
    fwi = math.exp(2.72 * (0.434 * math.log(bb)) ** 0.647) if bb > 1 else bb
    fwi = max(0.0, fwi)

    return {"ffmc": round(ffmc, 2), "dmc": round(dmc, 2), "dc": round(dc, 2),
            "isi": round(isi, 2), "bui": round(bui, 2), "fwi": round(fwi, 2)}


# ---------------------------------------------------------------------------
# Open-Meteo: fetch real hourly weather for a lat/lon/date
# ---------------------------------------------------------------------------
def fetch_openmeteo_weather(lat: float, lon: float, date: str,
                             retries: int = 3) -> Optional[Dict]:
    """
    Fetches hourly weather from Open-Meteo archive API for one day.
    Returns dict with noon-hour values, or None on failure.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start_date": date,
        "end_date": date,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m,precipitation",
        "timezone": "America/Toronto",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            if not times:
                return None
            # Use noon (hour 12) or closest available hour
            noon_idx = min(range(len(times)), key=lambda i: abs(int(times[i][11:13]) - 12))
            return {
                "temperature_c": hourly["temperature_2m"][noon_idx],
                "humidity_pct": hourly["relativehumidity_2m"][noon_idx],
                "wind_speed_kmh": hourly["windspeed_10m"][noon_idx],
                "wind_direction_deg": hourly["winddirection_10m"][noon_idx],
                "precipitation_mm": sum(v or 0 for v in hourly["precipitation"]),
            }
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
            else:
                logger.debug(f"Open-Meteo fetch failed for {lat},{lon},{date}: {exc}")
    return None


# ---------------------------------------------------------------------------
# NFDB loader
# ---------------------------------------------------------------------------
def load_nfdb(min_year: int = 2015, max_year: int = 2024,
              min_size_ha: float = 10.0) -> pd.DataFrame:
    nfdb_dir = RAW_DIR / "nfdb"
    if not nfdb_dir.exists():
        logger.warning("NFDB directory not found. Run python -m data.download_nfdb first.")
        return pd.DataFrame()

    # Try text/CSV files first
    csv_files = list(nfdb_dir.rglob("*.txt")) + list(nfdb_dir.rglob("*.csv"))
    shp_files = list(nfdb_dir.rglob("*.shp"))

    df = pd.DataFrame()
    if csv_files:
        frames = []
        for f in csv_files:
            try:
                tmp = pd.read_csv(f, encoding="latin-1", low_memory=False)
                frames.append(tmp)
                logger.debug(f"Loaded {f.name}: {len(tmp):,} rows, cols={list(tmp.columns[:6])}")
            except Exception as exc:
                logger.warning(f"Could not read {f.name}: {exc}")
        if frames:
            df = pd.concat(frames, ignore_index=True)
    elif shp_files:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(shp_files[0])
            df = pd.DataFrame(gdf.drop(columns="geometry"))
            df["latitude"] = gdf.geometry.y
            df["longitude"] = gdf.geometry.x
        except ImportError:
            logger.error("geopandas not installed — cannot read .shp. Run pip install geopandas.")
            return pd.DataFrame()

    if df.empty:
        logger.warning("No NFDB records loaded.")
        return pd.DataFrame()

    df.columns = df.columns.str.strip().str.lower()
    logger.info(f"NFDB raw columns: {list(df.columns[:10])}")

    # Standardise column names
    col_map = {}
    for c in df.columns:
        if "lat" in c and "col" not in c:
            col_map[c] = "latitude"
        elif "lon" in c and "col" not in c:
            col_map[c] = "longitude"
        elif c in ("year", "fire_year", "yr"):
            col_map[c] = "year"
        elif "size" in c or "hectare" in c or "area" in c:
            col_map[c] = "size_ha"
        elif "month" in c or "mon" == c:
            col_map[c] = "month"
        elif "day" == c or "start_day" in c:
            col_map[c] = "day"
    df = df.rename(columns=col_map)

    required = {"latitude", "longitude", "year"}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"NFDB missing required columns: {missing}. Available: {list(df.columns)}")
        return pd.DataFrame()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    mask = (
        df["latitude"].between(42, 83)
        & df["longitude"].between(-141, -52)
        & df["year"].between(min_year, max_year)
        & df["latitude"].notna()
        & df["longitude"].notna()
    )
    if "size_ha" in df.columns:
        df["size_ha"] = pd.to_numeric(df["size_ha"], errors="coerce").fillna(10)
        mask &= df["size_ha"] >= min_size_ha

    df = df[mask].copy()
    if "size_ha" not in df.columns:
        df["size_ha"] = 100.0
    if "month" not in df.columns:
        df["month"] = 7
    if "day" not in df.columns:
        df["day"] = 1

    df["month"] = pd.to_numeric(df["month"], errors="coerce").fillna(7).clip(1, 12).astype(int)
    df["day"] = pd.to_numeric(df["day"], errors="coerce").fillna(1).clip(1, 28).astype(int)

    logger.info(f"NFDB: {len(df):,} fire events loaded ({min_year}–{max_year}, ≥{min_size_ha} ha)")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Environment Canada weather loader (for non-fire baseline)
# ---------------------------------------------------------------------------
def load_ec_weather(weather_dir: Path = RAW_DIR / "weather") -> pd.DataFrame:
    frames = []
    for f in weather_dir.rglob("*.csv"):
        for skip in (15, 16, 17, 25, 0):
            try:
                tmp = pd.read_csv(f, skiprows=skip, low_memory=False, encoding="latin-1")
                cols = " ".join(tmp.columns).lower()
                if "temp" in cols or "wind" in cols:
                    frames.append(tmp)
                    break
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.lower().str.strip()
    logger.info(f"EC weather: {len(df):,} hourly records loaded")
    return df


# ---------------------------------------------------------------------------
# Node grid generator
# ---------------------------------------------------------------------------
def make_node_grid(center_lat: float, center_lon: float, rng: random.Random
                   ) -> List[Dict]:
    nodes = []
    step_km = AREA_KM / NODES_PER_SIDE
    for i in range(NODES_PER_SIDE):
        for j in range(NODES_PER_SIDE):
            dlat = (i - NODES_PER_SIDE / 2 + 0.5) * step_km / 111.0
            dlon = (j - NODES_PER_SIDE / 2 + 0.5) * step_km / (
                111.0 * math.cos(math.radians(center_lat))
            )
            nodes.append({
                "node_id": f"NODE_{i * NODES_PER_SIDE + j:03d}",
                "latitude": center_lat + dlat + rng.gauss(0, 0.0005),
                "longitude": center_lon + dlon + rng.gauss(0, 0.0005),
            })
    return nodes


# ---------------------------------------------------------------------------
# Sensor reading generation (one event × all nodes × all timesteps)
# ---------------------------------------------------------------------------
def generate_event_rows(
    fire_lat: float,
    fire_lon: float,
    fire_date: datetime,       # real date — used only for diurnal temp offset
    fire_size_ha: float,
    weather: Dict,
    fwi: Dict,
    is_fire: bool,
    rng: random.Random,
    seq_start: datetime = None,   # synthetic sequential start timestamp
) -> List[Dict]:

    nodes = make_node_grid(fire_lat, fire_lon, rng)

    base_temp = weather.get("temperature_c") or 22.0
    base_rh = weather.get("humidity_pct") or 55.0
    base_wind = weather.get("wind_speed_kmh") or 10.0
    base_wdir = weather.get("wind_direction_deg") or 180.0
    fwi_val = fwi.get("fwi", 10.0)
    ffmc = fwi.get("ffmc", 80.0)
    isi = fwi.get("isi", 5.0)

    fire_radius_km = max(0.5, math.sqrt(fire_size_ha / math.pi))
    # Fire peaks at TIMESTEPS_PER_EVENT//3 (morning of fire day) and decays after
    fire_peak_step = TIMESTEPS_PER_EVENT // 3

    rows = []
    ts_origin = seq_start if seq_start is not None else fire_date
    for t_step in range(TIMESTEPS_PER_EVENT):
        ts = ts_origin + timedelta(seconds=t_step * INTERVAL_SEC)
        # Diurnal temperature offset (real-ish: warmer in afternoon)
        hour_of_day = (t_step * 0.5) % 24
        diurnal_offset = 3.0 * math.sin(math.pi * (hour_of_day - 6) / 12)

        for node in nodes:
            dist_km = haversine(node["latitude"], node["longitude"], fire_lat, fire_lon)

            # Spatial influence: 1.0 at epicentre, falls off over 3× fire radius
            fire_inf = max(0.0, 1.0 - dist_km / (fire_radius_km * 3)) if is_fire else 0.0
            # Temporal influence: rises to peak at fire_peak_step then decays
            if t_step <= fire_peak_step:
                t_factor = t_step / max(fire_peak_step, 1)
            else:
                t_factor = max(0.0, 1.0 - (t_step - fire_peak_step) / (TIMESTEPS_PER_EVENT - fire_peak_step))
            combined = fire_inf * max(0.1, t_factor)

            temp = base_temp + diurnal_offset + combined * 18 + rng.gauss(0, 0.4)
            rh = max(5.0, base_rh - combined * 28 - diurnal_offset * 1.5 + rng.gauss(0, 1.0))
            surf_temp = temp + 3.0 + combined * 12 + rng.gauss(0, 0.5)

            # Smoke: driven by ISI + fire influence
            smoke_base = min(1.0, isi / 20.0)
            smoke_index = min(5.0, max(0.0,
                smoke_base * 1.5 + combined * 3.0 + rng.gauss(0, 0.04)))

            # CO: driven by fire proximity and FWI
            co_ppm = min(50.0, max(0.0,
                (max(0, ffmc - 70) / 30) * 1.5 + combined * 4.0 + rng.gauss(0, 0.08)))

            # VOC: driven by temperature + fire
            voc_index = min(400.0, max(0.0,
                40 + fwi_val * 1.5 + combined * 80 + rng.gauss(0, 1.5)))

            wind_spd = max(0.0, base_wind + rng.gauss(0, 0.5))
            wind_dir = (base_wdir + rng.gauss(0, 4)) % 360

            fire_risk = (
                0.30 * min(combined, 1.0)
                + 0.25 * min(smoke_index / 3.5, 1.0)
                + 0.20 * min(max((temp - 25) / 45, 0), 1.0)
                + 0.15 * min(max((100 - rh) / 95, 0), 1.0)
                + 0.10 * min(fwi_val / 50, 1.0)
            )
            fire_risk = round(min(max(fire_risk, 0.0), 1.0), 4)

            rows.append({
                "node_id": node["node_id"],
                "timestamp": ts.replace(tzinfo=timezone.utc).isoformat(),
                "latitude": round(node["latitude"], 5),
                "longitude": round(node["longitude"], 5),
                "temperature_c": round(min(70, max(-10, temp)), 2),
                "humidity_pct": round(min(100, max(5, rh)), 2),
                "surface_temp_c": round(min(100, max(-5, surf_temp)), 2),
                "smoke_index": round(smoke_index, 4),
                "co_ppm": round(co_ppm, 4),
                "voc_index": round(voc_index, 2),
                "wind_speed_kmh": round(min(120, wind_spd), 2),
                "wind_direction_deg": round(wind_dir, 1),
                "fire_risk": fire_risk,
                "is_fire_event": fire_risk >= 0.55,
                "battery_pct": round(rng.uniform(70, 100), 1),
                "signal_rssi": rng.randint(-110, -60),
            })
    return rows


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build(max_events: int = 400, no_fire_ratio: float = 1.0,
          weather_api_sleep: float = 0.3):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    # ---- Load NFDB ----
    nfdb = load_nfdb()
    if nfdb.empty:
        logger.error("NFDB data unavailable. Run python -m data.download_nfdb first.")
        return

    # Sample fire events: prefer larger fires, Ontario + BC + Alberta focus
    nfdb_sorted = nfdb.sort_values("size_ha", ascending=False)
    fire_events = nfdb_sorted.head(max_events * 2).sample(
        min(max_events, len(nfdb_sorted)), random_state=42
    ).reset_index(drop=True)
    logger.info(f"Selected {len(fire_events)} NFDB fire events for training")

    # ---- Build fire-event rows ----
    all_rows: List[Dict] = []
    api_calls = 0
    seq_ts = _BASE_TS  # advances by TIMESTEPS_PER_EVENT × INTERVAL_SEC after each event

    for idx, row in fire_events.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        year = int(row["year"])
        month = int(row.get("month", 7))
        day = int(row.get("day", 1))
        size_ha = float(row.get("size_ha", 50))

        try:
            fire_date = datetime(year, month, day, 12, 0, 0, tzinfo=timezone.utc)
        except ValueError:
            continue

        date_str = fire_date.strftime("%Y-%m-%d")

        # Fetch real weather from Open-Meteo
        weather = fetch_openmeteo_weather(lat, lon, date_str)
        api_calls += 1
        if api_calls % 20 == 0:
            logger.info(f"  Progress: {idx + 1}/{len(fire_events)} events, "
                        f"{len(all_rows):,} rows so far")
        time.sleep(weather_api_sleep)

        if weather is None:
            # Fallback: use climatological defaults for July in boreal Canada
            weather = {"temperature_c": 28.0, "humidity_pct": 35.0,
                       "wind_speed_kmh": 18.0, "wind_direction_deg": 225.0,
                       "precipitation_mm": 0.0}

        fwi_scores = compute_fwi(
            temp=weather["temperature_c"],
            rh=weather["humidity_pct"],
            wind=weather["wind_speed_kmh"],
            rain=weather.get("precipitation_mm", 0.0),
        )

        rows = generate_event_rows(lat, lon, fire_date, size_ha,
                                   weather, fwi_scores, is_fire=True,
                                   rng=rng, seq_start=seq_ts)
        all_rows.extend(rows)
        seq_ts += timedelta(seconds=TIMESTEPS_PER_EVENT * INTERVAL_SEC)

    logger.info(f"Fire events done: {len(all_rows):,} fire rows")

    # ---- Build non-fire baseline rows ----
    n_no_fire = int(len(fire_events) * no_fire_ratio)
    logger.info(f"Generating {n_no_fire} non-fire baseline events…")

    # Use random Canadian locations and non-fire months
    canada_locs = [
        (48.0, -80.0), (50.0, -86.0), (46.0, -75.0), (52.0, -90.0),
        (55.0, -105.0), (49.0, -95.0), (53.0, -117.0), (54.0, -125.0),
        (47.0, -68.0), (51.0, -114.0), (44.5, -76.5), (45.0, -72.0),
    ]
    no_fire_count = 0
    for _ in range(n_no_fire * 2):
        if no_fire_count >= n_no_fire:
            break
        base_lat, base_lon = rng.choice(canada_locs)
        lat = base_lat + rng.uniform(-2, 2)
        lon = base_lon + rng.uniform(-3, 3)
        year = rng.randint(2015, 2023)
        month = rng.choice([4, 5, 9, 10, 11])  # non-fire seasons
        day = rng.randint(1, 28)
        date_str = f"{year}-{month:02d}-{day:02d}"
        fire_date = datetime(year, month, day, 12, 0, 0, tzinfo=timezone.utc)

        weather = fetch_openmeteo_weather(lat, lon, date_str)
        api_calls += 1
        time.sleep(weather_api_sleep)

        if weather is None:
            weather = {"temperature_c": 12.0, "humidity_pct": 72.0,
                       "wind_speed_kmh": 12.0, "wind_direction_deg": 270.0,
                       "precipitation_mm": 2.0}

        fwi_scores = compute_fwi(
            temp=weather["temperature_c"],
            rh=weather["humidity_pct"],
            wind=weather["wind_speed_kmh"],
            rain=weather.get("precipitation_mm", 0.0),
        )
        rows = generate_event_rows(lat, lon, fire_date, 0.0,
                                   weather, fwi_scores, is_fire=False,
                                   rng=rng, seq_start=seq_ts)
        all_rows.extend(rows)
        seq_ts += timedelta(seconds=TIMESTEPS_PER_EVENT * INTERVAL_SEC)
        no_fire_count += 1

    logger.info(f"Total rows before save: {len(all_rows):,}")

    # ---- Save ----
    df = pd.DataFrame(all_rows, columns=SENSOR_COLS)
    df = df.sort_values(["timestamp", "node_id"]).reset_index(drop=True)

    out_path = RAW_DIR / "real_sensor_training.csv"
    df.to_csv(out_path, index=False)

    fire_rate = df["is_fire_event"].mean() * 100
    logger.info(f"Saved → {out_path}")
    logger.info(f"Rows: {len(df):,} | Fire rate: {fire_rate:.1f}% | "
                f"Open-Meteo API calls: {api_calls}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=400,
                        help="Max NFDB fire events to use (default 400)")
    parser.add_argument("--no-fire-ratio", type=float, default=1.0,
                        help="Non-fire events as fraction of fire events (default 1.0)")
    parser.add_argument("--api-sleep", type=float, default=0.3,
                        help="Seconds between Open-Meteo API calls (default 0.3)")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/build_training_data.log", rotation="50 MB")
    build(max_events=args.max_events,
          no_fire_ratio=args.no_fire_ratio,
          weather_api_sleep=args.api_sleep)
