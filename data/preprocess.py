"""
Merges all downloaded datasets into a single processed feature CSV
ready for model training.

Run: python -m data.preprocess
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_simulated(path: Path = RAW_DIR / "simulated_readings.csv") -> pd.DataFrame:
    if not path.exists():
        logger.warning(f"Simulated CSV not found at {path}. Run the simulator first.")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info(f"Loaded simulated readings: {len(df):,} rows")
    return df


def load_fwi(fwi_dir: Path = RAW_DIR / "fwi") -> pd.DataFrame:
    frames = []
    for f in fwi_dir.glob("*.csv"):
        try:
            tmp = pd.read_csv(f, parse_dates=["rep_date"], dayfirst=False, low_memory=False)
            frames.append(tmp)
        except Exception as exc:
            logger.warning(f"Skipping {f.name}: {exc}")
    if not frames:
        logger.warning("No FWI CSVs found.")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.lower().str.strip()
    logger.info(f"Loaded FWI data: {len(df):,} rows")
    return df


def load_weather(weather_dir: Path = RAW_DIR / "weather") -> pd.DataFrame:
    frames = []
    for f in weather_dir.rglob("*.csv"):
        try:
            tmp = pd.read_csv(f, skiprows=15, parse_dates=["Date/Time (LST)"])
            frames.append(tmp)
        except Exception:
            pass
    if not frames:
        logger.warning("No weather CSVs found.")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded weather data: {len(df):,} rows")
    return df


def add_temporal_features(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    dt = df[time_col]
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    df["is_afternoon"] = ((dt.dt.hour >= 12) & (dt.dt.hour <= 18)).astype(int)
    return df


def normalize(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    exclude_cols = set(exclude_cols or ["node_id", "timestamp", "is_fire_event", "fire_risk"])
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    df[num_cols] = (df[num_cols] - df[num_cols].mean()) / (df[num_cols].std() + 1e-6)
    return df


def load_real(path: Path = RAW_DIR / "real_sensor_training.csv") -> pd.DataFrame:
    if not path.exists():
        logger.warning(f"Real training data not found at {path}. "
                       "Run python -m data.build_training_data first.")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info(f"Loaded real-anchored training data: {len(df):,} rows")
    return df


def preprocess(use_real: bool = True, use_simulated: bool = True):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames = []

    if use_real:
        real_df = load_real()
        if not real_df.empty:
            frames.append(real_df)

    if use_simulated:
        sim_df = load_simulated()
        if not sim_df.empty:
            frames.append(sim_df)

    if not frames:
        logger.error("No data to process. Run the simulator and/or build_training_data first.")
        return

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined dataset: {len(df):,} rows "
                f"({'real + simulated' if len(frames) == 2 else 'real only' if use_real else 'simulated only'})")

    df = add_temporal_features(df, "timestamp")

    sensor_cols = ["temperature_c", "humidity_pct", "smoke_index", "co_ppm", "wind_speed_kmh"]
    for col in sensor_cols:
        if col in df.columns:
            q_low = df[col].quantile(0.01)
            q_hi = df[col].quantile(0.99)
            df[col] = df[col].clip(q_low, q_hi)

    out_path = PROCESSED_DIR / "features.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved processed features → {out_path} ({len(df):,} rows)")

    fire_rate = df["is_fire_event"].mean() * 100
    logger.info(f"Fire event rate: {fire_rate:.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-real", action="store_true", help="Skip real training data")
    parser.add_argument("--no-simulated", action="store_true", help="Skip simulated data")
    args = parser.parse_args()
    logger.add("logs/preprocess.log")
    preprocess(use_real=not args.no_real, use_simulated=not args.no_simulated)
