"""
Download NASA FIRMS active fire data for Canada (VIIRS 375m + MODIS).
Run: python -m data.download_firms
"""

import os
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

RAW_DIR = Path("data/raw/firms")

# NRT (Near Real Time) — last 1 year, no key needed
FIRMS_NRT = {
    "viirs_snpp_canada_1yr": "https://firms.modaps.eosdis.nasa.gov/data/country/csv/VIIRS_SNPP_NRT/1/CAN.csv",
    "modis_canada_1yr": "https://firms.modaps.eosdis.nasa.gov/data/country/csv/MODIS_NRT/1/CAN.csv",
}

# Archive requires free NASA Earthdata account and map key
FIRMS_ARCHIVE_PAGE = "https://firms.modaps.eosdis.nasa.gov/download/"


def download_firms_nrt():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in FIRMS_NRT.items():
        dest = RAW_DIR / f"{name}.csv"
        if dest.exists():
            logger.info(f"{dest} already exists — skipping.")
            continue
        try:
            logger.info(f"Downloading {url}")
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            logger.info(f"Saved → {dest}")
        except Exception as exc:
            logger.error(f"Failed to download {name}: {exc}")

    logger.info(
        f"FIRMS NRT download complete.\n"
        f"For historical archive (2012–2023), visit: {FIRMS_ARCHIVE_PAGE}\n"
        f"Select VIIRS S-NPP 375m, Country=Canada, then download CSV."
    )


if __name__ == "__main__":
    logger.add("logs/download_firms.log")
    download_firms_nrt()
