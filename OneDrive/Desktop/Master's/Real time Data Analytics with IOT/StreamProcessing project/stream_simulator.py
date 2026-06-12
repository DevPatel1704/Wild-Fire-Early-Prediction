# stream_simulator.py
# ENGR 5785G - Real-Time Data Analytics for IoT
# Assignment: Real-Time Stream Processing - Scenario B
# Student: Dev Patel
#
# Simulates a live IoT data stream by splitting the prepared CSV into 10 files
# (one per 2-minute window) and dropping them into the watched directory one at a time.
# Spark's readStream picks up each new file on the next trigger, which makes it
# behave like sensor data arriving in real time.
#
# Usage:
#   python stream_simulator.py              # default 4 sec delay between files
#   python stream_simulator.py --delay 6   # custom delay

import argparse
import os
import time
from datetime import datetime

import pandas as pd

INPUT_CSV    = "data/patients_streaming.csv"
WATCH_DIR    = "data/stream_input"
N_WINDOWS    = 10
ROWS_TOTAL   = 50_000
ROWS_PER_WIN = ROWS_TOTAL // N_WINDOWS


def clear_watch_dir():
    # remove any leftover csv files from a previous run before dropping new ones
    if os.path.isdir(WATCH_DIR):
        for fname in os.listdir(WATCH_DIR):
            if fname.endswith(".csv"):
                os.remove(os.path.join(WATCH_DIR, fname))


def main(delay: float):
    if not os.path.isfile(INPUT_CSV):
        print(f"[ERROR] '{INPUT_CSV}' not found.  Run prepare_data.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    os.makedirs(WATCH_DIR, exist_ok=True)
    clear_watch_dir()

    print(f"[Simulator] {len(df):,} records  |  {N_WINDOWS} windows  "
          f"|  {ROWS_PER_WIN:,} rows/file  |  delay = {delay}s")
    print(f"[Simulator] Dropping files into: {os.path.abspath(WATCH_DIR)}\n")

    for win in range(N_WINDOWS):
        start = win * ROWS_PER_WIN
        # last window gets any remaining rows in case of rounding
        end   = start + ROWS_PER_WIN if win < N_WINDOWS - 1 else len(df)
        batch = df.iloc[start:end]

        # include timestamp in filename so files sort correctly and don't overwrite
        fname = f"window_{win:02d}_{datetime.now().strftime('%H%M%S%f')}.csv"
        fpath = os.path.join(WATCH_DIR, fname)
        batch.to_csv(fpath, index=False)

        ts_min = batch["timestamp"].min()
        ts_max = batch["timestamp"].max()
        print(f"[{datetime.now().strftime('%H:%M:%S')}]  "
              f"Window {win:02d}  |  {fname}  "
              f"({end - start:,} rows,  {ts_min} to {ts_max})")

        if win < N_WINDOWS - 1:
            time.sleep(delay)

    print(f"\n[Simulator] All {N_WINDOWS} windows delivered.  "
          f"Spark will finalize remaining output within ~{int(delay)}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoMT stream simulator")
    parser.add_argument("--delay", type=float, default=4.0,
                        help="Seconds between file drops (default: 4)")
    args = parser.parse_args()
    main(args.delay)
