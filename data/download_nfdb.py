"""
Download Canadian National Fire Database (NFDB) fire point and polygon data.
Run: python -m data.download_nfdb
"""

import os
import zipfile
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

RAW_DIR = Path("data/raw")

NFDB_BASE = "https://cwfis.cfs.nrcan.gc.ca/downloads/nfdb/fire_pnt/current_version"
NFDB_URLS = {
    "nfdb_point_txt": f"{NFDB_BASE}/NFDB_point_txt.zip",   # CSV/text format (13 MB)
}


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    try:
        logger.info(f"Downloading {url}")
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in resp.iter_content(chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
        logger.info(f"Saved → {dest}")
        return True
    except Exception as exc:
        logger.error(f"Download failed ({url}): {exc}")
        return False


def download_nfdb():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    nfdb_dir = RAW_DIR / "nfdb"
    nfdb_dir.mkdir(exist_ok=True)

    for name, url in NFDB_URLS.items():
        zip_path = nfdb_dir / f"{name}.zip"
        if zip_path.exists():
            logger.info(f"{zip_path} already exists — skipping.")
        else:
            if not download_file(url, zip_path):
                continue

        # Unzip
        extract_dir = nfdb_dir / name
        if not extract_dir.exists():
            logger.info(f"Extracting {zip_path}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            logger.info(f"Extracted to {extract_dir}")

    logger.info("NFDB download complete.")


if __name__ == "__main__":
    logger.add("logs/download_nfdb.log")
    download_nfdb()
