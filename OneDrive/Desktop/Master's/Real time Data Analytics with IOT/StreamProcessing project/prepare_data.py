# prepare_data.py
# ENGR 5785G - Real-Time Data Analytics for IoT
# Assignment: Real-Time Stream Processing - Scenario B
# Student: Dev Patel
#
# This script reads the original IoMT dataset (Excel file) and converts it
# into a CSV that's ready for streaming simulation.
#
# The main challenge was making sure every patient shows up in every window.
# If I just assigned timestamps sequentially, patient 1 would be in window 0,
# patient 2 in window 0, etc and then there would be no consecutive window alerts.
# The fix was grouping by N_PATIENTS at a time so all 500 patients cycle through
# each window together.

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_FILE  = "patients_data_with_alerts.xlsx"
OUTPUT_CSV = "data/patients_streaming.csv"
STREAM_DIR = "data/stream_input"

N_PATIENTS  = 500
N_WINDOWS   = 10
BASE_TIME   = datetime(2024, 6, 11, 9, 0, 0)
WINDOW_SECS = 120  # 2 minutes per window


def build_timestamps(n_rows: int) -> list[datetime]:
    # assigns a timestamp to every row such that all 500 patients appear in every window
    # the key formula:  win = (row_index // N_PATIENTS) % N_WINDOWS
    # this groups rows in blocks of 500 and cycles them through windows 0-9
    rows_per_window = n_rows // N_WINDOWS
    readings_per_patient_per_window = rows_per_window // N_PATIENTS

    timestamps = []
    window_counts = [0] * N_WINDOWS   # tracks how many rows have been placed in each window

    for i in range(n_rows):
        win = (i // N_PATIENTS) % N_WINDOWS
        pos = window_counts[win]
        window_counts[win] += 1
        # spread readings evenly within the 2-minute window
        frac = pos / max(rows_per_window, 1)
        secs = win * WINDOW_SECS + frac * (WINDOW_SECS - 1)
        timestamps.append(BASE_TIME + timedelta(seconds=secs))
    return timestamps


def main():
    print("Loading patient monitoring dataset...")
    df = pd.read_excel(DATA_FILE)
    print(f"  Rows: {len(df):,}   Columns: {len(df.columns)}")

    # rename columns by position because the excel headers sometimes have weird encoding
    col_map = {
        df.columns[0]: "patient_number",
        df.columns[1]: "heart_rate",
        df.columns[2]: "spo2",
        df.columns[3]: "sys_bp",
        df.columns[4]: "dia_bp",
        df.columns[5]: "body_temp",
        df.columns[6]: "fall_detection",
        df.columns[7]: "predicted_disease",
    }
    df = df.rename(columns=col_map)

    # create patient_id 1-500 by cycling through the dataset rows
    df["patient_id"] = (df.index % N_PATIENTS) + 1
    df["timestamp"]  = build_timestamps(len(df))

    out = df[["timestamp", "patient_id", "heart_rate", "spo2",
              "sys_bp", "dia_bp", "body_temp",
              "fall_detection", "predicted_disease"]].copy()

    # sort by timestamp so stream_simulator can just slice rows in order
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs("data/stream_input", exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    high_hr = (df["heart_rate"] > 100).sum()
    print(f"\nDataset saved to: {OUTPUT_CSV}")
    print(f"  Records        : {len(out):,}")
    print(f"  Unique patients: {out['patient_id'].nunique()}")
    print(f"  Windows        : {N_WINDOWS}  x  2 min  "
          f"(09:00 to {(BASE_TIME + timedelta(minutes=N_WINDOWS*2)).strftime('%H:%M')})")
    print(f"  Rows per window: {len(out) // N_WINDOWS:,}")
    print(f"  HR > 100 bpm   : {high_hr:,}  ({100*high_hr/len(df):.1f}%)")
    print("\nReady.  Run stream_simulator.py to start dropping files.")


if __name__ == "__main__":
    main()
