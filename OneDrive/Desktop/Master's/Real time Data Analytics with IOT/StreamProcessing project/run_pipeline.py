# run_pipeline.py
# ENGR 5785G - Real-Time Data Analytics for IoT
# Assignment: Real-Time Stream Processing - Scenario B
# Student: Dev Patel
#
# Single entry point to run the whole pipeline end-to-end with one command.
# Does all the setup steps automatically so you don't have to remember the order:
#   1. Check that patients_streaming.csv exists (runs prepare_data.py if not)
#   2. Clear old checkpoint files and stream_input folder
#   3. Drop all 10 window files into the watched directory at once
#   4. Start the Spark streaming job (runs and exits automatically when done)
#
# Usage:
#   python run_pipeline.py

import os
import sys
import shutil
import subprocess

PROJECT    = os.path.dirname(os.path.abspath(__file__))
DATA_CSV   = os.path.join(PROJECT, "data", "patients_streaming.csv")
STREAM_DIR = os.path.join(PROJECT, "data", "stream_input")
CHECKPOINT = "C:/spark_ckpt"


def step(n, msg):
    print(f"\n[{n}/4] {msg}")


def main():
    os.chdir(PROJECT)

    # step 1 - make sure the prepared CSV exists
    step(1, "Checking dataset...")
    if not os.path.isfile(DATA_CSV):
        print("      patients_streaming.csv not found - generating now...")
        subprocess.run([sys.executable, "prepare_data.py"], check=True)
    else:
        print("      patients_streaming.csv ready.")

    # step 2 - clean up any leftover state from previous runs
    step(2, "Clearing old checkpoints and stream files...")
    if os.path.isdir(CHECKPOINT):
        shutil.rmtree(CHECKPOINT, ignore_errors=True)
    os.makedirs(CHECKPOINT, exist_ok=True)
    if os.path.isdir(STREAM_DIR):
        for f in os.listdir(STREAM_DIR):
            if f.endswith(".csv"):
                os.remove(os.path.join(STREAM_DIR, f))
    os.makedirs(STREAM_DIR, exist_ok=True)
    print("      Done.")

    # step 3 - drop all 10 window files at once (delay=0 since spark uses availableNow)
    step(3, "Dropping all 10 window files into stream_input/...")
    subprocess.run(
        [sys.executable, "stream_simulator.py", "--delay", "0"],
        check=True
    )

    # step 4 - start spark, output goes straight to this terminal
    step(4, "Starting Spark pipeline (will process all files then exit)...\n")
    result = subprocess.run([sys.executable, "-u", "spark_streaming_job.py"])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
