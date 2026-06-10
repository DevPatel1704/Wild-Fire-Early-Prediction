"""
Download Environment Canada historical hourly weather data for Ontario stations.
Uses the bulk download API — no authentication required.

Run: python -m data.download_weather --province ON --years 2019 2020 2021 2022 2023
"""

import argparse
import csv
import io
import os
import time
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

RAW_DIR = Path("data/raw/weather")

# Known station IDs near Ontario forest areas (Parry Sound, Sudbury, North Bay area)
ONTARIO_STATION_IDS = [
    48370,  # Parry Sound
    6158,   # North Bay
    6155,   # Sudbury
    4300,   # Peterborough
    4303,   # Lindsay
    6053,   # Huntsville
]

EC_BULK_URL = (
    "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
    "?format=csv&stationID={station_id}&Year={year}&Month={month}"
    "&timeframe=1&submit=Download+Data"
)


def download_station_month(station_id: int, year: int, month: int, dest_dir: Path) -> bool:
    dest = dest_dir / f"station_{station_id}_{year}_{month:02d}.csv"
    if dest.exists():
        return True

    url = EC_BULK_URL.format(station_id=station_id, year=year, month=month)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        if len(resp.content) < 500:  # empty response
            return False
        dest.write_bytes(resp.content)
        return True
    except Exception as exc:
        logger.warning(f"  Failed {station_id}/{year}/{month}: {exc}")
        return False


def download_weather(station_ids=None, years=None, months=None):
    station_ids = station_ids or ONTARIO_STATION_IDS
    years = years or [2021, 2022, 2023]
    months = months or list(range(1, 13))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total = len(station_ids) * len(years) * len(months)
    success = 0

    with tqdm(total=total, desc="Downloading weather data") as bar:
        for sid in station_ids:
            sid_dir = RAW_DIR / f"station_{sid}"
            sid_dir.mkdir(exist_ok=True)
            for year in years:
                for month in months:
                    ok = download_station_month(sid, year, month, sid_dir)
                    if ok:
                        success += 1
                    bar.update(1)
                    time.sleep(0.3)  # be polite to EC servers

    logger.info(f"Weather download complete: {success}/{total} files saved to {RAW_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023])
    parser.add_argument("--province", default="ON")
    args = parser.parse_args()

    logger.add("logs/download_weather.log")
    download_weather(years=args.years)
