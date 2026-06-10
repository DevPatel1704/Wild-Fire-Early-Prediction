"""
Download Canadian Forest Fire Weather Index (FWI) observation data.
Provides pre-computed fire danger scores: FFMC, DMC, DC, ISI, BUI, FWI.

Run: python -m data.download_fwi
"""

import os
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

RAW_DIR = Path("data/raw/fwi")

FWI_BASE_URL = "https://cwfis.cfs.nrcan.gc.ca/downloads/fwi_obs/"

# Direct links for FWI station observations (public, no auth needed)
FWI_FILES = {
    "fwi_obs_2021.csv": f"{FWI_BASE_URL}cwfis_allstn2021.csv",
    "fwi_obs_2022.csv": f"{FWI_BASE_URL}cwfis_allstn2022.csv",
    "fwi_obs_2023.csv": f"{FWI_BASE_URL}cwfis_allstn2023.csv",
}

# Gridded spatial FWI
FWI_GRIDDED_URL = "https://cwfis.cfs.nrcan.gc.ca/datamart/download/fwi"


def download_fwi():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for fname, url in FWI_FILES.items():
        dest = RAW_DIR / fname
        if dest.exists():
            logger.info(f"{dest} exists — skipping.")
            continue
        try:
            logger.info(f"Downloading {url}")
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=fname) as bar:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            logger.info(f"Saved → {dest}")
        except Exception as exc:
            logger.error(f"Failed {fname}: {exc}")

    logger.info("FWI download complete.")
    logger.info(f"For gridded FWI (spatial NetCDF), visit: {FWI_GRIDDED_URL}")


if __name__ == "__main__":
    logger.add("logs/download_fwi.log")
    download_fwi()
