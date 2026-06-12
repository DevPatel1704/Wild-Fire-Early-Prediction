# spark_streaming_job.py
# ENGR 5785G - Real-Time Data Analytics for IoT
# Assignment: Real-Time Stream Processing - Scenario B (Hospital Patient Monitoring)
# Student: Dev Patel
#
# This script sets up a Spark Structured Streaming pipeline that reads ICU patient
# heart rate data from a watched folder, groups readings into 2-minute tumbling windows,
# and fires a clinical alert whenever a patient has high average HR in two back-to-back windows.
#
# I chose tumbling windows because each reading should only belong to one time period -
# this avoids double counting that you'd get with sliding windows. Also makes it easier
# to check "two consecutive periods" since the boundaries are clean (exactly 120 sec apart).
#
# State is needed in two places:
#   1. Spark's internal watermark state - it keeps partial window aggregates in memory
#      until the watermark moves past the window end, then finalizes and emits.
#   2. My _window_history dict - tracks which patients had high HR in previous windows
#      so I can detect when it happens two times in a row.

import os
import sys
import shutil
import threading
from datetime import datetime

# need to set these before importing pyspark otherwise spark won't find java/hadoop
os.environ["JAVA_HOME"]      = "C:/Program Files/Amazon Corretto/jdk17.0.19_10"
os.environ["HADOOP_HOME"]    = "C:/hadoop_spark"
os.environ["PYSPARK_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg, col, count, round as spark_round, window
)
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType,
    StructField, StructType, TimestampType,
)

# --- config values ---
WATCH_DIR      = "data/stream_input"
CHECKPOINT_DIR = "C:/spark_ckpt"
WINDOW_SIZE    = "2 minutes"
WATERMARK      = "2 minutes"
HR_THRESHOLD   = 100   # bpm - anything above this is considered elevated

# wipe old checkpoint each time so we always start fresh from batch 0
# without this spark resumes from where it left off and shows empty results
shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# define the schema manually so spark doesn't have to scan files to infer types
# also makes timestamp parsing reliable
PATIENT_SCHEMA = StructType([
    StructField("timestamp",         TimestampType(), True),
    StructField("patient_id",        IntegerType(),   True),
    StructField("heart_rate",        IntegerType(),   True),
    StructField("spo2",              IntegerType(),   True),
    StructField("sys_bp",            IntegerType(),   True),
    StructField("dia_bp",            IntegerType(),   True),
    StructField("body_temp",         DoubleType(),    True),
    StructField("fall_detection",    StringType(),    True),
    StructField("predicted_disease", StringType(),    True),
])

# this dict stores recent high-HR window results per patient on the driver side
# key = patient_id, value = list of (window_start, avg_hr) tuples
# I keep max 3 entries per patient so memory doesn't grow forever
_state_lock    = threading.Lock()
_window_history: dict[int, list[tuple]] = {}


def _check_consecutive(patient_id: int, w_start, avg_hr: float):
    # checks if this patient now has two consecutive high-HR windows
    # returns (True, win1_data, win2_data) if alert should fire, else (False, None, None)
    history = _window_history.get(patient_id, [])

    if avg_hr > HR_THRESHOLD:
        history.append((w_start, avg_hr))
        history.sort(key=lambda x: x[0])
        _window_history[patient_id] = history[-3:]   # only keep last 3 windows

        # check each pair of consecutive entries - if gap is ~120 sec they are adjacent windows
        for i in range(len(_window_history[patient_id]) - 1):
            t1, h1 = _window_history[patient_id][i]
            t2, h2 = _window_history[patient_id][i + 1]
            gap = (t2 - t1).total_seconds()
            if abs(gap - 120) <= 10:   # allow 10 sec tolerance for floating point
                return True, (t1, h1), (t2, h2)
    else:
        # HR came back to normal so reset this patient's history
        _window_history.pop(patient_id, None)

    return False, None, None


def process_batch(batch_df, batch_id: int):
    # this function is called by foreachBatch for every micro-batch
    # batch_df contains the aggregated window results (one row per patient per window)

    # using count() here instead of rdd.isEmpty() because rdd triggers python workers
    # which caused crashes - count() stays on the jvm side
    if batch_df.count() == 0:
        return

    rows = batch_df.collect()
    alerts = []

    # need a lock because foreachBatch can theoretically run from multiple threads
    with _state_lock:
        for row in rows:
            triggered, prev, curr = _check_consecutive(
                row.patient_id, row.window.start, float(row.avg_hr)
            )
            if triggered:
                alerts.append((row.patient_id, prev, curr))

    # print a table showing all window results for this batch
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    W   = 78
    print(f"\n{'-' * W}")
    print(f"  Micro-batch {batch_id:<3}  |  {now}  |  "
          f"{len(rows)} window results processed")
    print(f"{'-' * W}")
    print(f"  {'PID':>5}  {'Window Start':>19}  {'Window End':>19}  "
          f"{'Avg HR':>7}  {'Reads':>5}  Status")
    print(f"  {'-'*5}  {'-'*19}  {'-'*19}  {'-'*7}  {'-'*5}  {'-'*16}")

    for r in sorted(rows, key=lambda x: (x.window.start, x.patient_id)):
        flag = "** HIGH **" if r.avg_hr > HR_THRESHOLD else "Normal"
        w_s  = r.window.start.strftime("%Y-%m-%d %H:%M:%S")
        w_e  = r.window.end.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {r.patient_id:>5}  {w_s:>19}  {w_e:>19}  "
              f"{r.avg_hr:>7.1f}  {r.reading_count:>5}  {flag}")

    # if any patients triggered the consecutive-window rule, print clinical alerts
    if alerts:
        border = "=" * W
        print(f"\n{border}")
        print("  !!!  CLINICAL ALERT  -  SUSTAINED ELEVATED HEART RATE  !!!")
        print(border)
        for pid, (t1, h1), (t2, h2) in alerts:
            print(f"\n  Patient ID  : {pid}")
            print(f"  Window 1    : {t1.strftime('%Y-%m-%d %H:%M:%S')}"
                  f"   Avg HR = {h1:.1f} bpm  [ HIGH ]")
            print(f"  Window 2    : {t2.strftime('%Y-%m-%d %H:%M:%S')}"
                  f"   Avg HR = {h2:.1f} bpm  [ HIGH ]")
            print(f"  Condition   : HR > {HR_THRESHOLD} bpm sustained across "
                  f"2 consecutive 2-minute windows")
            print(f"  Action      : Immediate clinical review required")
            print(f"  {'-' * (W - 2)}")
        print(border)
    else:
        elev = sum(1 for r in rows if r.avg_hr > HR_THRESHOLD)
        print(f"\n  No sustained-HR alerts this batch.  "
              f"({elev} single-window elevated reading(s) noted.)")


# build the spark session - running local[1] to keep memory usage low on this machine
# shuffle partitions set to 1 since we're running locally with a small dataset
spark = (
    SparkSession.builder
    .master("local[1]")
    .appName("ICU_HeartRate_Monitor_ScenarioB")
    .config("spark.sql.shuffle.partitions", "1")
    .config("spark.driver.memory", "512m")
    .config("spark.memory.offHeap.enabled", "false")
    .config("spark.driver.extraJavaOptions",
            "-Dhadoop.home.dir=C:/hadoop_spark"
            " -Djava.library.path=C:/hadoop_spark/bin"
            " -XX:+UseSerialGC"
            " -XX:MaxDirectMemorySize=64m"
            " -XX:ReservedCodeCacheSize=32m"
            " -XX:MaxMetaspaceSize=128m"
            " -XX:TieredStopAtLevel=1")
    # RawLocalFileSystem avoids hadoop trying to create .crc checksum files locally
    .config("spark.hadoop.fs.file.impl",
            "org.apache.hadoop.fs.RawLocalFileSystem")
    .config("spark.hadoop.fs.AbstractFileSystem.file.impl",
            "org.apache.hadoop.fs.local.RawLocalFs")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# set up the streaming read - watching the stream_input folder for new csv files
# maxFilesPerTrigger=1 means spark picks up one window file per batch, simulating real-time feed
raw_stream = (
    spark.readStream
    .schema(PATIENT_SCHEMA)
    .option("header", "true")
    .option("maxFilesPerTrigger", "1")
    .csv(WATCH_DIR)
)

# apply watermark and group by 2-minute tumbling window + patient_id
# withWatermark tells spark to wait up to 2 minutes for late data before closing a window
# window() with only one duration arg = tumbling window (no overlap)
windowed_hr = (
    raw_stream
    .withWatermark("timestamp", WATERMARK)
    .groupBy(
        window(col("timestamp"), WINDOW_SIZE),
        col("patient_id"),
    )
    .agg(
        spark_round(avg("heart_rate"), 2).alias("avg_hr"),
        count("heart_rate").alias("reading_count"),
    )
)

# filter the aggregated stream to only show patients with avg HR above threshold
# this is the "filtered output stream" part of the assignment
high_hr_stream = (
    windowed_hr
    .filter(col("avg_hr") > HR_THRESHOLD)
    .selectExpr(
        "patient_id",
        "window.start  AS window_start",
        "window.end    AS window_end",
        "avg_hr",
        "reading_count",
        f"'HR_EXCEEDS_{HR_THRESHOLD}_BPM' AS alert_type",
    )
)

# Query 1 - stateful alert pipeline using foreachBatch
# foreachBatch lets me run custom python logic on each batch result
# this is where the consecutive-window check happens using _window_history
q_main = (
    windowed_hr
    .writeStream
    .outputMode("update")
    .foreachBatch(process_batch)
    .option("checkpointLocation", f"{CHECKPOINT_DIR}/main")
    .trigger(availableNow=True)
    .start()
)

# Query 2 - write the filtered high-HR stream to console
# outputMode update means only show rows that changed in this batch
q_alerts = (
    high_hr_stream
    .writeStream
    .outputMode("update")
    .format("console")
    .option("truncate", "false")
    .option("numRows", "30")
    .option("checkpointLocation", f"{CHECKPOINT_DIR}/alerts")
    .trigger(availableNow=True)
    .start()
)

# print some info so we know the job started correctly
W = 78
print("\n" + "=" * W)
print("  Spark Structured Streaming  -  ICU Heart Rate Monitor")
print("  Scenario B  |  Tumbling 2-min Window  |  Sustained HR Alert")
print(f"  Window size  : {WINDOW_SIZE}  (tumbling)")
print(f"  Watermark    : {WATERMARK}")
print(f"  HR threshold : {HR_THRESHOLD} bpm  (sustained across 2 windows)")
print(f"  Watch dir    : {os.path.abspath(WATCH_DIR)}")
print("  Trigger      : availableNow (processes all files then exits)")
print("=" * W)
print("  Processing all files in data/stream_input/ ...")
print("=" * W + "\n")

try:
    q_main.awaitTermination()
    q_alerts.awaitTermination()
    print("\n" + "=" * W)
    print("  Pipeline complete.  All windows processed.")
    print("=" * W + "\n")
except KeyboardInterrupt:
    print("\n[Shutdown] Stopping streaming queries...")
    q_main.stop()
    q_alerts.stop()
finally:
    spark.stop()
