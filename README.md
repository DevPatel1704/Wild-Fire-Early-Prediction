# 🔥 Wildfire IoT Early Warning System

> **Real-Time Data Analytics with IoT** — Ontario Tech University  
> Group: Dev · Dhruv · Priyanka · Slesha · Rashmi

A full-stack real-time wildfire risk detection system that simulates 100 IoT forest sensor nodes, streams readings through Apache Kafka, runs spatial-temporal deep learning inference (GAT-LSTM), stores data in InfluxDB and SQLite, and visualises live fire risk on an interactive web dashboard.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Modules](#modules)
  - [Sensor Simulator](#sensor-simulator)
  - [Deep Learning Model](#deep-learning-model-gat-lstm)
  - [Real-Time Pipeline](#real-time-pipeline)
  - [Data Ingestion (Kafka)](#data-ingestion-kafka)
  - [Storage Layer](#storage-layer)
  - [API Server](#api-server)
  - [React Dashboard](#react-dashboard)
  - [Training Dataset](#training-dataset)
- [Dataset Features](#dataset-features)
- [Alert Thresholds](#alert-thresholds)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Docker Services](#docker-services)

---

## Overview

This system simulates an IoT sensor network deployed in a Canadian forest environment. 100 nodes — each modelling a physical ESP32 device with 6 attached sensors — transmit environmental readings every 30 seconds. A **Graph Attention Network + LSTM** model analyses 6 minutes of historical readings across all nodes simultaneously, exploiting both **spatial relationships** (correlated changes between neighbouring nodes) and **temporal patterns** (how readings evolve over time). When fire risk crosses 0.80, an alert is published. Everything streams through Apache Kafka and is visualised live on a React dashboard.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SENSOR SIMULATOR                               │
│  100 nodes · 10×10 km grid · Ontario (44.0°N, 78.95°W)            │
│  5 fire scenarios injected at +6h, +18h, +30h, +50h, +68h         │
│  tick() every 30 seconds → JSON sensor readings                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ JSON, key = node_id
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    KAFKA BROKER (localhost:29092)                    │
│  sensor.raw        [4 partitions]  ← raw 30s readings              │
│  sensor.aggregated [2 partitions]  ← 1-min window stats            │
│  fire.alerts       [1 partition]   ← risk ≥ 0.80                   │
│  drone.commands    [1 partition]   ← reserved                       │
└──────────┬──────────────────────────────────┬───────────────────────┘
           │ group: api-server                 │ group: wildfire-pipeline
           ▼                                   ▼
┌─────────────────────┐           ┌────────────────────────────────────┐
│   FastAPI Server    │           │       Stream Processor             │
│   port 8000         │           │                                    │
│                     │           │  SensorAggregator                  │
│  In-memory state:   │           │  NodeBuffer (60s window/node)      │
│  node_status{}      │           │  → mean / max / std per sensor     │
│  active_alerts{}    │           │  → 6-step history matrix (6, 8)    │
│  kafka/influx/model │           │                                    │
│                     │           │  Every 60 seconds:                 │
│  REST endpoints     │           │  GAT-LSTM inference                │
│  WebSocket /ws/live │           │  (batch, 100, 6, 25) → risk scores │
│  5-second broadcast │           │  risk ≥ 0.80 → fire.alerts         │
└──────────┬──────────┘           └────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SQLite  data/wildfire.db          InfluxDB  localhost:8086         │
│  sensor_readings (raw)             sensor_reading measurement       │
│  sensor_aggregates (JSON blob)     sensor_aggregate measurement     │
│  fire_alerts (+ acknowledged)      fire_alert measurement           │
└─────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  REACT DASHBOARD  localhost:3000                     │
│                                                                     │
│  ┌────────────┐  ┌──────────────────────────┐  ┌────────────────┐  │
│  │ Sensor     │  │   Leaflet Dark Map        │  │ Alert Panel    │  │
│  │ Panel      │  │   CartoDB tiles           │  │ Active alerts  │  │
│  │            │  │                           │  │ Color-coded    │  │
│  │ Nodes OK   │  │   🔴 CRITICAL  ≥ 0.90    │  │ Acknowledge    │  │
│  │ High Risk  │  │   🟠 HIGH      ≥ 0.80    │  │ button         │  │
│  │ Alerts     │  │   🟡 MODERATE  ≥ 0.65    │  │                │  │
│  │ Health     │  │   🟢 LOW       < 0.65    │  │                │  │
│  │ Kafka/DB   │  │   Dimmed = offline        │  │                │  │
│  │ Top Risk   │  │                           │  │                │  │
│  │ Gauges     │  │                           │  │                │  │
│  └────────────┘  └──────────────────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Streaming** | Apache Kafka 7.5 + Zookeeper |
| **Deep Learning** | PyTorch, PyTorch Geometric (GAT-LSTM) |
| **API** | FastAPI, Uvicorn, WebSockets |
| **Storage** | InfluxDB 2.7, SQLite (SQLAlchemy) |
| **Frontend** | React 18, Leaflet, Chart.js |
| **Infrastructure** | Docker Compose |
| **Data Processing** | NumPy, Pandas, SciPy |
| **Monitoring** | Kafka UI, InfluxDB UI |
| **Logging** | Loguru |
| **Testing** | Pytest, pytest-asyncio, HTTPX |

---

## Project Structure

```
Main Project/
├── api/                        # FastAPI REST + WebSocket server
│   ├── main.py                 # App setup, Kafka consumer thread, WebSocket broadcast
│   ├── schemas.py              # Pydantic models (SensorReading, FireAlert, etc.)
│   └── routes/
│       ├── sensors.py          # /sensors/* endpoints
│       ├── alerts.py           # /alerts/* endpoints
│       └── predictions.py      # /predictions/* endpoints
│
├── model/                      # Deep learning model
│   ├── gat_layer.py            # Single-head & multi-head GAT implementation
│   ├── gat_lstm.py             # Full GAT-LSTM spatial-temporal model
│   ├── dataset.py              # PyTorch WildfireDataset with sliding windows
│   ├── train.py                # Training script (AdamW, BCELoss, AUC checkpoint)
│   ├── predict.py              # FireRiskPredictor inference wrapper
│   └── checkpoints/            # Saved model weights (gat_lstm_best.pt)
│
├── pipeline/                   # Real-time stream processing
│   ├── stream_processor.py     # Orchestrates aggregation + inference + alerting
│   └── aggregator.py           # 1-min tumbling window per node (NodeBuffer)
│
├── simulator/                  # IoT sensor network simulator
│   ├── sensor_node.py          # SensorNode (ESP32 model, 6 sensors, fire drift)
│   ├── fire_scenario.py        # FireScenario (ignition, spread, influence)
│   ├── network.py              # SensorNetwork (100-node grid, fire application)
│   └── run_simulator.py        # Entry point (real-time / fast / CSV export modes)
│
├── data/                       # Data download & preprocessing
│   ├── download_firms.py       # NASA FIRMS satellite fire detections
│   ├── download_weather.py     # Environment Canada weather data
│   ├── download_fwi.py         # Canadian FWI data
│   ├── download_nfdb.py        # National Fire Database (NFDB)
│   ├── build_training_data.py  # Builds real-world anchored training CSV
│   └── preprocess.py           # Merges sources, adds temporal features, normalises
│
├── ingestion/                  # Kafka topic management
│   ├── topics.py               # Topic definitions + ensure_topics()
│   ├── producer.py             # SensorProducer (gzip, acks=all, retry)
│   └── consumer.py             # SensorConsumer (auto-commit, batch)
│
├── storage/                    # Persistent storage writers
│   ├── sqlite_writer.py        # SQLiteWriter (3 tables, thread-safe)
│   └── influxdb_writer.py      # InfluxDBWriter (3 measurements, graceful fallback)
│
├── drone/                      # Simulated drone dispatcher (software-only)
│   ├── dispatcher.py           # Listens to fire.alerts, computes flight plans
│   └── flight_plan.py          # Lawnmower survey pattern generator
│
├── dashboard/                  # React frontend
│   ├── src/
│   │   ├── App.jsx             # Main component, layout, polling + WebSocket
│   │   ├── index.js            # Entry point
│   │   ├── services/api.js     # Axios API client + WebSocket helper
│   │   └── components/
│   │       ├── Map.jsx         # Leaflet map with colour-coded node markers
│   │       ├── SensorPanel.jsx # Left sidebar (network status, top risk list)
│   │       ├── AlertPanel.jsx  # Right sidebar (active alerts, acknowledge)
│   │       └── RiskGauge.jsx   # Chart.js doughnut risk gauge
│   └── package.json
│
├── tests/                      # Pytest unit tests
├── logs/                       # Runtime logs (gitignored)
├── docker-compose.yml          # Kafka, Zookeeper, InfluxDB, Kafka-UI, API
├── Dockerfile.api              # API container (python:3.11-slim)
├── requirements.txt            # Python dependencies
├── setup.bat                   # One-shot setup script (Windows)
└── .env                        # Environment variables (copy from .env.example)
```

---

## Modules

### Sensor Simulator

Simulates 100 ESP32 nodes on a 10×10 km grid centred on Ontario (44.0°N, 78.95°W).

**Sensors modelled per node:**

| Physical Sensor | Measurement | Range |
|---|---|---|
| DHT22 | Air temperature | −10 to 70 °C |
| DHT22 | Relative humidity | 5 to 100 % |
| MLX90614 | Surface temperature (IR) | −5 to 100 °C |
| MQ-2 | Smoke index | 0 to 5 |
| BME680 | Carbon monoxide | 0 to 100 PPM |
| BME680 | VOC index | 0 to 500 |
| Anemometer | Wind speed | 0 to 120 km/h |
| Compass/GPS | Wind direction | 0 to 360 ° |

**Fire influence on sensors** (when `fire_influence` ∈ [0, 1]):

```
temperature    +=  0.02 + fi×1.2   ± Gaussian noise
humidity       -=  0.01 + fi×0.8   ± Gaussian noise
surface_temp   +=  0.03 + fi×1.5   ± Gaussian noise
smoke_index    +=  fi×0.15          ± Gaussian noise
co_ppm         +=  fi×0.08          ± Gaussian noise
voc_index      +=  fi×5.0           ± Gaussian noise
wind_speed     +=  fi×0.3           ± Gaussian noise
```

**Reliability simulation:**
- 3% chance per tick to go offline (LoRaWAN packet loss)
- Offline for 1–4 ticks
- Battery drains 0.001–0.003% per reading
- RSSI randomly sampled in [−110, −60] dBm

**Five default fire scenarios:**

| ID | Offset from centre | Start | Spread Rate | Wind |
|---|---|---|---|---|
| fire_001 | +0.02°N, +0.015°E | +6 h | 0.06 km/min | 260° |
| fire_002 | −0.03°N, +0.030°E | +18 h | 0.04 km/min | 310° |
| fire_003 | +0.04°N, −0.025°E | +30 h | 0.07 km/min | 240° |
| fire_004 | −0.01°N, −0.020°E | +50 h | 0.05 km/min | 290° |
| fire_005 | +0.035°N, +0.040°E | +68 h | 0.08 km/min | 270° |

---

### Deep Learning Model (GAT-LSTM)

A spatial-temporal hybrid combining a Graph Attention Network with an LSTM.

**Architecture:**

```
Input: (batch, N=100 nodes, T=6 timesteps, F=25 features)
       ↓
For each timestep t = 0 … 5:
  MultiHeadGAT (4 heads, hidden=64)
  → (batch, N, 64)
       ↓
Stack → (batch, N, 6, 64)
Reshape → (batch×100, 6, 64)
       ↓
LSTM (2 layers, hidden=128, batch_first=True)
Take last hidden state → (batch×100, 128)
Reshape → (batch, 100, 128)
       ↓
Linear(128→64) → ReLU → Dropout(0.2)
Linear(64→1)   → Sigmoid
       ↓
Output: (batch, N=100, 1)  — fire risk score per node [0, 1]
```

**GAT attention mechanism** (per head):

```
Wh   = W × h                                   (linear transform)
e_ij = LeakyReLU(α=0.2)(a_src·Wh_i + a_dst·Wh_j)   (attention score)
       masked to -1e9 where adj[i,j] = 0
α_ij = softmax_j(e_ij)                          (normalised weights)
h'_i = ELU( Σ_j α_ij × Wh_j )                 (aggregated features)
```

**Graph connectivity:** Two nodes are connected if their Haversine distance ≤ 1.5 km. Includes self-loops.

**Model parameters:**

| Hyperparameter | Value |
|---|---|
| Input features (`n_features`) | 25 |
| Timesteps (`n_timesteps`) | 6 |
| GAT hidden dim (`gat_hidden`) | 64 |
| Attention heads (`n_heads`) | 4 |
| LSTM hidden dim (`lstm_hidden`) | 128 |
| LSTM layers | 2 |
| Dropout | 0.2 |
| Loss function | BCELoss |
| Optimiser | AdamW (lr=1e-3, wd=1e-4) |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Gradient clipping | max_norm=1.0 |
| Train/val split | 80 / 20 |
| Validation metric | ROC-AUC |

**Ground-truth fire_risk label formula** (used to generate labels during simulation):

```
fire_risk = 0.20 × clamp((temp − 25) / 45)
          + 0.20 × clamp((100 − humidity) / 95)
          + 0.15 × clamp((surface_temp − 30) / 70)
          + 0.25 × clamp(smoke_index / 3.5)
          + 0.10 × clamp(co_ppm / 20)
          + 0.10 × clamp(wind_speed / 60)
```

Training:
```bash
python -m simulator.run_simulator --fast --export csv --days 30
python -m data.preprocess
python -m model.train --epochs 50 --batch-size 16
```

---

### Real-Time Pipeline

**`pipeline/aggregator.py`** — NodeBuffer (60-second tumbling window per node):

```
Per window produces:
  {sensor}_mean, {sensor}_max, {sensor}_std   for all 8 sensors (24 values)
  count, max_fire_risk, any_fire_event
  latitude, longitude, window_end

History: last 6 completed windows stored → get_history_feature_vector() → (6, 8) numpy array
```

**`pipeline/stream_processor.py`** — WildfireStreamProcessor:

```
Startup:
  Connect Kafka (12 retries, exponential backoff max 30s)
  Subscribe to sensor.raw  (group: wildfire-pipeline)
  Load GAT-LSTM checkpoint

Main loop (consumes sensor.raw):
  aggregator.ingest(message)
  track throughput (log every 500 msgs)
  track latency (last 100 msg timestamps)

Every 60 seconds (_flush_and_infer):
  flush() → aggregates + feature_tensors
  publish aggregates → sensor.aggregated + InfluxDB
  model.predict_batch(feature_tensors) → {node_id: risk_score}
  for risk_score ≥ 0.80:
    alert_level = "CRITICAL" if ≥ 0.90 else "HIGH"
    publish → fire.alerts + InfluxDB
```

---

### Data Ingestion (Kafka)

| Topic | Partitions | Key | Approx. Rate |
|---|---|---|---|
| `sensor.raw` | 4 | `node_id` | ~3.3 msg/s |
| `sensor.aggregated` | 2 | `node_id` | ~1.7 msg/s |
| `fire.alerts` | 1 | `node_id` | Fire events only |
| `drone.commands` | 1 | — | Reserved |

**Producer settings:** `acks=all`, `linger_ms=10`, `compression=gzip`, `retries=5`  
**Consumer settings:** `auto_offset_reset=latest`, `auto_commit_interval=1000ms`, `max_poll_records=500`  
**Partitioning by `node_id`** guarantees ordering per sensor node.

---

### Storage Layer

**SQLite** (`data/wildfire.db`) — primary local fallback:

| Table | Key Columns |
|---|---|
| `sensor_readings` | node_id, timestamp, all 8 sensors, fire_risk, battery, RSSI |
| `sensor_aggregates` | node_id, window_end, data_json (full aggregate as JSON blob) |
| `fire_alerts` | node_id, timestamp, fire_risk_score, alert_level, acknowledged |

All operations are thread-safe (mutex Lock). Indexed on `node_id` and `timestamp`.

**InfluxDB** (`localhost:8086`) — time-series analytics:

| Measurement | Tags | Fields |
|---|---|---|
| `sensor_reading` | node_id | temperature, humidity, smoke, CO, wind, fire_risk, lat, lon |
| `sensor_aggregate` | node_id | 24 stats (mean/max/std × 8 sensors) + any_fire_event, count |
| `fire_alert` | node_id, alert_level | fire_risk_score |

Gracefully disabled if InfluxDB is unreachable — system continues without it.

---

### API Server

FastAPI on port 8000. Background Kafka consumer thread updates in-memory state. WebSocket broadcasts a snapshot every 5 seconds to all connected dashboard clients.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/sensors/nodes` | All 100 NodeStatus objects |
| GET | `/sensors/nodes/{node_id}` | Single node or 404 |
| GET | `/sensors/readings/recent?limit=100` | Last N raw readings (max 500) |
| GET | `/sensors/risk-map` | `[{node_id, lat, lon, fire_risk, is_online}]` |
| GET | `/alerts/active` | Unacknowledged FireAlert list |
| POST | `/alerts/acknowledge` | Mark alert as handled |
| GET | `/alerts/history?limit=50` | Historical alerts from SQLite |
| GET | `/predictions/system-status` | Kafka / InfluxDB / Model health + uptime |
| GET | `/predictions/risk/{node_id}` | 10-min + 30-min forecast, alert_level |
| WS | `/ws/live` | Live 5-second node snapshots |
| GET | `/docs` | Swagger UI |

**WebSocket message shape** (every 5 seconds):
```json
{
  "nodes": [{"node_id": "NODE_042", "latitude": 44.0023, "longitude": -78.9441,
              "fire_risk": 0.87, "is_online": true, "battery_pct": 83.2}],
  "active_alerts": [...],
  "timestamp": "2026-06-10T17:00:00Z"
}
```

---

### React Dashboard

**Layout** — 3-column dark theme (`#0f1117` background):

```
┌─────────────────────────────────────────────────────────────────┐
│ Header  — Title · Last Updated · Kafka/InfluxDB/Model status    │
├───────────────┬─────────────────────────┬───────────────────────┤
│ Sensor Panel  │     Leaflet Map          │   Alert Panel         │
│  Nodes Online │  CartoDB dark tiles      │  Active Alerts (N)    │
│  High Risk    │  CircleMarker per node   │  Per-alert card:      │
│  Active Alerts│  colour = risk level     │  - Level + timestamp  │
│  Kafka badge  │  size = risk magnitude   │  - Node + risk%       │
│  InfluxDB     │  dimmed = offline        │  - GPS coords         │
│  AI Model     │  popup on click          │  - Acknowledge btn    │
│  Top 8 Risk   │                          │                       │
│  RiskGauges   │                          │                       │
└───────────────┴─────────────────────────┴───────────────────────┘
│ Footer — Ontario Tech University · Group: Dev Dhruv Priyanka... │
└─────────────────────────────────────────────────────────────────┘
```

**Node marker colour:**

| Score | Colour | Level |
|---|---|---|
| ≥ 0.90 | `#ff1744` 🔴 | CRITICAL |
| ≥ 0.80 | `#ff6d00` 🟠 | HIGH |
| ≥ 0.65 | `#ffd600` 🟡 | MODERATE |
| < 0.65 | `#00e676` 🟢 | LOW |

Offline nodes rendered at 0.2 opacity. Data refreshed via HTTP poll every 10 s and WebSocket push every 5 s.

---

### Training Dataset

Two sources merged in `data/preprocess.py`:

**Source 1 — Real-world data** (`data/build_training_data.py`):
1. Load NFDB fire records (2015–2024, ≥10 ha, Canadian coordinates)
2. Fetch real hourly weather per fire event from Open-Meteo archive API
3. Compute FWI scores via Van Wagner (1987) algorithm
4. Generate 10×10 sensor node grid around each fire location
5. Simulate 48 timesteps (24 h window at 30-min intervals) per event
6. Non-fire baselines sampled from non-fire months/locations

**Source 2 — Simulator data** (`simulator/run_simulator.py`):
- 100 nodes, 5 fire scenarios, configurable duration

**Preprocessing steps:**
1. Merge real + simulated DataFrames
2. Add temporal features: `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `is_afternoon`
3. Clip outliers at 1st / 99th percentile
4. Z-score normalise numeric columns (excludes labels)
5. Output → `data/processed/features.csv`

---

## Dataset Features

**Raw sensor features (8 — model inputs):**

| Feature | Description | Unit |
|---|---|---|
| `temperature_c` | Air temperature | °C |
| `humidity_pct` | Relative humidity | % |
| `surface_temp_c` | Ground/surface IR temperature | °C |
| `smoke_index` | Smoke particle concentration | 0–5 scale |
| `co_ppm` | Carbon monoxide | PPM |
| `voc_index` | Volatile organic compounds | 0–400 index |
| `wind_speed_kmh` | Wind speed | km/h |
| `wind_direction_deg` | Wind direction | 0–360° |

**Derived temporal features (5 — added in preprocessing):**

| Feature | Description |
|---|---|
| `hour_sin` / `hour_cos` | Cyclical encoding of hour of day |
| `month_sin` / `month_cos` | Cyclical encoding of month |
| `is_afternoon` | 1 if 12:00–18:00 (peak fire window) |

**Labels & metadata:**

| Column | Description |
|---|---|
| `fire_risk` | Float [0, 1] — prediction target |
| `is_fire_event` | Boolean — `fire_risk ≥ 0.55` (training) |
| `node_id` | Sensor identifier |
| `latitude` / `longitude` | Node GPS position |
| `battery_pct` | IoT device battery level |
| `signal_rssi` | LoRaWAN signal strength (dBm) |

---

## Alert Thresholds

| Level | Threshold | Trigger |
|---|---|---|
| LOW | < 0.65 | No action |
| MODERATE | ≥ 0.65 | Dashboard highlight |
| HIGH | ≥ 0.80 | Alert published to Kafka + stored in SQLite |
| CRITICAL | ≥ 0.90 | Alert published with CRITICAL level |
| `is_fire_event` (training label) | ≥ 0.55 | Used only during dataset generation |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker Desktop

### 1. Clone and configure

```bash
git clone <repository-url>
cd "Main Project"
cp .env.example .env
# Edit .env if needed (defaults work out of the box)
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install dashboard dependencies

```bash
cd dashboard
npm install
cd ..
```

### 4. (Optional) Build training data and train model

```bash
# Generate synthetic training data
python -m simulator.run_simulator --fast --export csv --days 30

# Preprocess
python -m data.preprocess

# Train model (~50 epochs, saves checkpoint)
python -m model.train --epochs 50
```

> A pre-trained checkpoint is already included at `model/checkpoints/gat_lstm_best.pt`.

---

## Running the Project

Start each component in a **separate terminal**, in this order:

**Terminal 1 — Docker infrastructure**
```bash
docker-compose up -d
```

**Terminal 2 — FastAPI server**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 3 — Stream processor**
```bash
python -m pipeline.stream_processor
```

**Terminal 4 — Sensor simulator**
```bash
python -m simulator.run_simulator
```

**Terminal 5 — React dashboard**
```bash
cd dashboard
npm start
```

### Access Points

| Service | URL |
|---|---|
| React Dashboard | http://localhost:3000 |
| API + Swagger Docs | http://localhost:8000/docs |
| Kafka UI | http://localhost:8080 |
| InfluxDB UI | http://localhost:8086 |

> Default InfluxDB credentials: username `admin`, password `wildfire123`

---

## API Reference

### Sensor Endpoints

```
GET  /sensors/nodes                    → List all 100 NodeStatus objects
GET  /sensors/nodes/{node_id}          → Single node status
GET  /sensors/readings/recent          → Recent raw readings (?limit=100)
GET  /sensors/risk-map                 → Node positions + fire_risk for map
```

### Alert Endpoints

```
GET  /alerts/active                    → Unacknowledged fire alerts
POST /alerts/acknowledge               → { "alert_id": "..." }
GET  /alerts/history                   → Historical alerts (?limit=50)
```

### Prediction Endpoints

```
GET  /predictions/system-status        → Kafka/InfluxDB/model health + uptime
GET  /predictions/risk/{node_id}       → 10-min + 30-min risk forecast
```

### WebSocket

```
WS   /ws/live                          → 5-second live node snapshots
```

---

## Configuration

All settings are read from `.env`:

```env
# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:29092
KAFKA_TOPIC_SENSOR_RAW=sensor.raw
KAFKA_TOPIC_SENSOR_AGG=sensor.aggregated
KAFKA_TOPIC_FIRE_ALERTS=fire.alerts

# InfluxDB
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=wildfire-super-secret-token
INFLUXDB_ORG=wildfire-org
INFLUXDB_BUCKET=sensor_data
INFLUXDB_USER=admin
INFLUXDB_PASSWORD=wildfire123

# Model
MODEL_CHECKPOINT_PATH=model/checkpoints/gat_lstm_best.pt
FIRE_RISK_THRESHOLD=0.65
ALERT_THRESHOLD=0.80

# Simulator
NUM_SENSOR_NODES=100
SIMULATION_AREA_KM=10
SENSOR_INTERVAL_SECONDS=30

# Pipeline
AGG_WINDOW_SECONDS=60

# Dashboard
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000/ws/live
```

---

## Docker Services

| Container | Image | Port | Purpose |
|---|---|---|---|
| `zookeeper` | confluentinc/cp-zookeeper:7.5.0 | 2181 | Kafka coordination |
| `kafka` | confluentinc/cp-kafka:7.5.0 | 9092 / 29092 | Message broker |
| `kafka-ui` | provectuslabs/kafka-ui:latest | 8080 | Broker monitoring UI |
| `influxdb` | influxdb:2.7 | 8086 | Time-series storage |
| `api` | Dockerfile.api | 8000 | FastAPI server (containerised) |

Stop all services:
```bash
docker-compose down
```

Stop and wipe all data volumes (clean slate):
```bash
docker-compose down -v
```

---

## Key Numbers

| Parameter | Value |
|---|---|
| Sensor nodes | 100 |
| Simulation grid | 10 × 10 km |
| Sensor reading interval | 30 seconds |
| Aggregation window | 60 seconds |
| LSTM history depth | 6 minutes |
| Model input shape | (batch, 100 nodes, 6 timesteps, 25 features) |
| GAT attention heads | 4 |
| Node connectivity radius | 1.5 km |
| Dashboard HTTP poll | Every 10 seconds |
| WebSocket push rate | Every 5 seconds |

---

*Real-Time Data Analytics with IoT · Ontario Tech University · 2026*
