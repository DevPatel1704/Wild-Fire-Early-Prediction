# ENGR 5785G - Real-Time Data Analytics for IoT
# Assignment: Real-Time Stream Processing - Scenario B

**Student:** Dev Patel  
**Student ID:** 101042729  
**Scenario:** B - Hospital ICU Patient Monitoring  
**Dataset used:** IoMT Health Monitoring (50,000 rows, 500 patients) from Kaggle

---

## What this project does

This pipeline reads ICU patient heart rate data from a folder (like a live sensor feed),
groups it into 2-minute tumbling windows, and fires an alert if a patient's average
heart rate is above 100 bpm in **two windows in a row**. The idea is to catch patients
who are consistently in distress, not just a one-off spike.

Here's the rough flow of data through the pipeline:

```
CSV files dropped into data/stream_input/
        |
        v
readStream watches the folder for new files
        |
        v
withWatermark("timestamp", "2 minutes")   <- handles late data
        |
        v
groupBy(2-min tumbling window, patient_id)
        |
        v
avg(heart_rate)
        |
        |-- filter(avg_hr > 100) --> console sink  [filtered output stream - Query 2]
        |
        '-- foreachBatch --> check consecutive windows --> CLINICAL ALERT
```

---

## Files in this project

```
StreamProcessing project/
├── patients_data_with_alerts.xlsx   <- original dataset from Kaggle
├── prepare_data.py                  <- converts the Excel file to a streaming CSV
├── stream_simulator.py              <- drops window files into the watched folder one by one
├── spark_streaming_job.py           <- the main Spark Structured Streaming pipeline
├── run_pipeline.py                  <- runs everything with one command
├── requirements.txt
├── README.md
└── data/
    ├── patients_streaming.csv       <- created by prepare_data.py (gitignored)
    └── stream_input/                <- Spark watches this folder (gitignored)
```

---

## What you need installed

- Python 3.9 or newer (I used 3.13)
- **Java 17** - this is important, Spark 4.0 doesn't work with Java 21+. I used Amazon Corretto 17:
  ```
  winget install Amazon.Corretto.17.JDK
  ```
- PySpark 4.0.0 (installed via pip below)
- On Windows you also need the Hadoop winutils stub (see setup step 2)

---

## Setup steps

**Step 1 - install the python packages:**
```
pip install -r requirements.txt
```

**Step 2 - Hadoop stub for Windows:**

Spark on Windows needs `winutils.exe` and `hadoop.dll` to run locally. I put
`winutils.exe` (from cdarlint/winutils on GitHub) into `C:\hadoop_spark\bin\` and
compiled a minimal DLL stub with MinGW. If you're on Windows and get a
`NativeIO` error, this is why.

**Step 3 - check the paths at the top of spark_streaming_job.py:**

```python
os.environ["JAVA_HOME"]   = "C:/Program Files/Amazon Corretto/jdk17.0.19_10"
os.environ["HADOOP_HOME"] = "C:/hadoop_spark"
```

Change these to match wherever Java 17 is installed on your machine.

**Step 4 - prepare the dataset:**
```
python prepare_data.py
```

This reads `patients_data_with_alerts.xlsx` and creates `data/patients_streaming.csv`
with timestamps spread across 10 x 2-minute windows so the simulator can feed them
in one at a time.

---

## How to run it

Just run this one command and it handles everything:

```
python run_pipeline.py
```

It will:
1. Check if the prepared CSV exists, run prepare_data.py if not
2. Clear any old checkpoint files from previous runs
3. Drop all 10 window CSV files into the watched folder
4. Start Spark - it processes everything and exits on its own when done

You should see output like this:

```
[1/4] Checking dataset...
[2/4] Clearing old checkpoints and stream files...
[3/4] Dropping all 10 window files into stream_input/...
[4/4] Starting Spark pipeline...

==============================================================================
  Spark Structured Streaming  -  ICU Heart Rate Monitor
  Scenario B  |  Tumbling 2-min Window  |  Sustained HR Alert
  Window size  : 2 minutes  (tumbling)
  Watermark    : 2 minutes
  HR threshold : 100 bpm  (sustained across 2 windows)
==============================================================================

------------------------------------------------------------------------------
  Micro-batch 0    |  2024-06-11 09:26:03  |  500 window results processed
------------------------------------------------------------------------------
    PID         Window Start           Window End   Avg HR  Reads  Status
  -----  -------------------  -------------------  -------  -----  ----------------
      1  2024-06-11 09:00:00  2024-06-11 09:02:00     98.9     10  Normal
      3  2024-06-11 09:00:00  2024-06-11 09:02:00    103.4     10  ** HIGH **
      4  2024-06-11 09:00:00  2024-06-11 09:02:00    110.7     10  ** HIGH **
...
==============================================================================
  !!!  CLINICAL ALERT  -  SUSTAINED ELEVATED HEART RATE  !!!
==============================================================================

  Patient ID  : 4
  Window 1    : 2024-06-11 09:00:00   Avg HR = 110.7 bpm  [ HIGH ]
  Window 2    : 2024-06-11 09:02:00   Avg HR = 108.0 bpm  [ HIGH ]
  Condition   : HR > 100 bpm sustained across 2 consecutive 2-minute windows
  Action      : Immediate clinical review required
...
==============================================================================
  Pipeline complete.  All windows processed.
==============================================================================
```

There's also a second streaming query (Query 2) that uses a console sink to print
the filtered output — this shows all windows where avg HR exceeded 100, one batch at a time:

```
-------------------------------------------
Batch: 1
-------------------------------------------
+----------+-------------------+-------------------+------+-------------+------------------+
|patient_id|window_start       |window_end         |avg_hr|reading_count|alert_type        |
+----------+-------------------+-------------------+------+-------------+------------------+
|3         |2024-06-11 09:00:00|2024-06-11 09:02:00|103.4 |10           |HR_EXCEEDS_100_BPM|
|4         |2024-06-11 09:00:00|2024-06-11 09:02:00|110.7 |10           |HR_EXCEEDS_100_BPM|
...
```

---

## Written Explanation

### Why I chose tumbling windows

I went with tumbling windows because the whole point of Scenario B is to detect
*sustained* elevated heart rate - not just a random spike that happens once. A tumbling
window cuts time into fixed non-overlapping 2-minute chunks, so each chunk is a clean
independent snapshot of what was happening during that period.

I thought about using sliding windows but that would've made this way harder to get right.
With sliding windows, the same reading ends up in multiple windows, so if I checked two
"consecutive" windows they could both contain the exact same spike. That's not really two
separate periods of elevated HR - it's the same data counted twice. Tumbling windows avoid
this because each reading belongs to exactly one window.

Also the consecutive-window check is a lot simpler with tumbling windows. Since each
window is exactly 120 seconds and they don't overlap, I just need to check if two
high-HR entries in my history are 120 seconds apart (I added a small ±10 sec tolerance
for floating point stuff). If I used sliding windows I'd have to figure out some
other way to define "consecutive" and it would get messy fast.

### Where the pipeline requires state

State comes up in two places in this pipeline:

**1. Spark's watermark state (inside the engine)**

The `withWatermark("timestamp", "2 minutes")` call makes Spark hold partial aggregation
results in memory for each open window. It needs to do this because data might arrive
slightly late, so it can't just close a window the instant it fills up. It keeps those
running sums in state until the watermark (basically the latest event time minus 2 min)
moves past the window's end, then it finalizes the average and emits the result.

So while window 09:00-09:02 is being filled, Spark is maintaining that partial sum
as state. Once enough data has come in to push the watermark past 09:02, Spark closes
that window, computes the avg HR, emits it downstream, and cleans up the state.

**2. My per-patient window history (my own state dict)**

I also maintain my own state on the driver side using a plain Python dictionary:

```python
_window_history = {}  # patient_id -> list of (window_start, avg_hr) tuples
```

I need this because a single window result doesn't tell me anything about the previous
window - I have to remember it myself across batches. After each micro-batch I update
the dict: if a patient's avg HR in this window was above 100, I add it to their history
and check if any two entries are 120 seconds apart (meaning two consecutive windows were
both high). If I find that, I fire the alert. If a patient's HR comes back to normal in
a window I reset their history so the next alert has to be genuinely two new bad windows.

Without this dict the consecutive-window alert is impossible to implement because Spark's
streaming engine doesn't carry that kind of application-level memory between batches on
its own.

---

## About the dataset

I used the IoMT Health Monitoring dataset from Kaggle which has 50,000 ICU patient records
with heart rate, SpO2, blood pressure, body temperature, and a fall detection flag.

For this project I assigned those records to 500 simulated patients and spread them across
10 consecutive 2-minute windows (09:00 to 09:20), with 10 readings per patient per window.
The key thing I had to figure out was making sure all 500 patients appear in every window,
not just the first few windows. The fix was using `win = (i // N_PATIENTS) % N_WINDOWS`
to assign timestamps instead of just `win = i % N_WINDOWS`. That way each patient has data
in all 10 windows and the consecutive-window alert actually fires.

About 54.6% of the heart rate readings exceed 100 bpm, so there are plenty of patients
that trigger the alert during the run.
