"""
FastAPI application — REST + WebSocket server for the wildfire dashboard.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from threading import Thread
from typing import Dict, Set

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

load_dotenv()

from api.routes import sensors as sensor_routes
from api.routes import alerts as alert_routes
from api.routes import predictions as prediction_routes
from api.schemas import NodeStatus, FireAlert

# Shared application state (in-memory; InfluxDB/SQLite are the persistent stores)
state: Dict = {
    "node_status": {},      # node_id → NodeStatus
    "live_readings": {},    # node_id → latest full sensor reading dict
    "active_alerts": {},    # alert_id → FireAlert
    "kafka_ok": False,
    "influx_ok": False,
    "model_ready": False,
    "start_time": time.time(),
    "sqlite": None,
    "influx": None,
}

# WebSocket connection manager
_ws_clients: Set[WebSocket] = set()


async def broadcast(payload: dict):
    disconnected = set()
    for ws in _ws_clients:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            disconnected.add(ws)
    _ws_clients.difference_update(disconnected)


# ------------------------------------------------------------------
# Demo / fallback simulator: uses SensorNetwork to produce real sensor
# readings (temp, humidity, smoke, CO, wind, etc.) every 5 s.
# Populates state["node_status"] AND state["live_readings"] so the
# full sensor data is available via /sensors/live-readings.
# ------------------------------------------------------------------
async def _demo_simulator_task():
    """
    Ticks the full SensorNetwork every 5 s, storing complete sensor
    readings in state["live_readings"] and updating node_status.
    Writes each reading to SQLite for historical queries too.
    Real Kafka data will overwrite entries if Docker comes back up.
    """
    from simulator.network import SensorNetwork
    from simulator.fire_scenario import default_scenarios

    start_time = datetime(2024, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    scenarios = default_scenarios(44.0, -78.95, start_time)
    network = SensorNetwork(
        n_nodes=int(os.getenv("NUM_SENSOR_NODES", 100)),
        area_km=float(os.getenv("SIMULATION_AREA_KM", 10)),
        fire_scenarios=scenarios,
        seed=42,
    )

    # Seed all nodes immediately so the map is populated at startup
    readings = network.tick(sim_time=datetime.now(timezone.utc))
    for r in readings:
        payload = r.to_dict()
        state["live_readings"][r.node_id] = payload
        ns = NodeStatus(node_id=r.node_id, latitude=r.latitude, longitude=r.longitude)
        ns.fire_risk = r.fire_risk
        ns.last_seen = r.timestamp
        ns.is_online = True
        ns.battery_pct = r.battery_pct
        state["node_status"][r.node_id] = ns

    state["kafka_ok"] = True
    logger.info(f"Demo mode: {len(readings)} nodes seeded with real sensor readings.")

    while True:
        await asyncio.sleep(5)
        readings = network.tick(sim_time=datetime.now(timezone.utc))
        db = state.get("sqlite")
        for r in readings:
            payload = r.to_dict()
            state["live_readings"][r.node_id] = payload

            ns = state["node_status"].get(r.node_id)
            if not ns:
                ns = NodeStatus(node_id=r.node_id, latitude=r.latitude, longitude=r.longitude)
            ns.fire_risk = r.fire_risk
            ns.last_seen = r.timestamp
            ns.is_online = True
            ns.battery_pct = r.battery_pct
            state["node_status"][r.node_id] = ns

            if db:
                try:
                    db.write_raw(payload)
                except Exception:
                    pass


# ------------------------------------------------------------------
# Background Kafka consumer thread
# ------------------------------------------------------------------
def _kafka_consumer_thread():
    import time as _time
    from kafka import KafkaConsumer
    from ingestion.topics import TOPIC_SENSOR_RAW, TOPIC_FIRE_ALERTS
    from dotenv import load_dotenv
    load_dotenv(override=True)

    servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")

    while True:
        consumer = None
        for attempt in range(99999):
            try:
                consumer = KafkaConsumer(
                    TOPIC_SENSOR_RAW, TOPIC_FIRE_ALERTS,
                    bootstrap_servers=servers,
                    group_id="api-server",
                    value_deserializer=lambda b: json.loads(b.decode()),
                    auto_offset_reset="latest",
                    enable_auto_commit=True,
                    request_timeout_ms=30000,
                    session_timeout_ms=6000,
                )
                state["kafka_ok"] = True
                logger.info(f"API Kafka consumer connected (attempt {attempt + 1}).")
                break
            except Exception as exc:
                wait = min(2 ** attempt, 30)
                logger.warning(f"Kafka connect attempt {attempt + 1} failed: {exc}. Retrying in {wait}s…")
                _time.sleep(wait)

        if consumer is None:
            logger.warning("Kafka consumer could not connect. Will keep retrying…")
            _time.sleep(30)
            continue

        try:
            for msg in consumer:
                topic = msg.topic
                data = msg.value

                if topic == TOPIC_SENSOR_RAW:
                    node_id = data.get("node_id", "")
                    ns = state["node_status"].get(node_id)
                    if not ns:
                        ns = NodeStatus(
                            node_id=node_id,
                            latitude=data.get("latitude", 0.0),
                            longitude=data.get("longitude", 0.0),
                        )
                    ns.fire_risk = data.get("fire_risk", 0.0)
                    ns.last_seen = data.get("timestamp", "")
                    ns.is_online = True
                    ns.battery_pct = data.get("battery_pct", 100.0)
                    state["node_status"][node_id] = ns

                    db = state.get("sqlite")
                    if db:
                        try:
                            db.write_raw(data)
                        except Exception:
                            pass

                elif topic == TOPIC_FIRE_ALERTS:
                    node_id = data.get("node_id", "")
                    alert_id = f"{node_id}_{data.get('timestamp', '')}"
                    alert = FireAlert(
                        node_id=node_id,
                        timestamp=data.get("timestamp", ""),
                        fire_risk_score=data.get("fire_risk_score", 0.0),
                        alert_level=data.get("alert_level", "HIGH"),
                    )
                    state["active_alerts"][alert_id] = alert

                    db = state.get("sqlite")
                    if db:
                        try:
                            db.write_alert(data)
                        except Exception:
                            pass

                    logger.warning(f"ALERT received: {alert.model_dump()}")

        except Exception as exc:
            logger.warning(f"Kafka consumer lost connection: {exc}. Reconnecting…")
            state["kafka_ok"] = False
            _time.sleep(5)


# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    from storage.sqlite_writer import SQLiteWriter
    state["sqlite"] = SQLiteWriter()
    logger.info("SQLite storage initialised.")

    try:
        from storage.influxdb_writer import InfluxDBWriter
        influx = InfluxDBWriter()
        state["influx"] = influx
        state["influx_ok"] = influx._connected
    except Exception as exc:
        logger.warning(f"InfluxDB not available: {exc}")

    try:
        from model.predict import FireRiskPredictor
        predictor = FireRiskPredictor()
        state["predictor"] = predictor
        state["model_ready"] = predictor.is_ready
    except Exception as exc:
        logger.warning(f"Model not loaded: {exc}")

    t = Thread(target=_kafka_consumer_thread, daemon=True)
    t.start()

    asyncio.create_task(_demo_simulator_task())

    yield

    # Shutdown
    if state.get("influx"):
        state["influx"].close()
    logger.info("API server shutting down.")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
app = FastAPI(
    title="Wildfire IoT Early Warning API",
    version="1.0.0",
    description="Real-time wildfire risk prediction via IoT sensor network and GAT-LSTM model.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sensor_routes.router)
app.include_router(alert_routes.router)
app.include_router(prediction_routes.router)


@app.get("/")
async def root():
    return {"message": "Wildfire IoT API is running.", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok", "nodes": len(state["node_status"]), "alerts": len(state["active_alerts"])}


# ------------------------------------------------------------------
# WebSocket — live push to dashboard
# ------------------------------------------------------------------
@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.add(websocket)
    logger.info(f"WebSocket client connected ({len(_ws_clients)} total)")
    try:
        while True:
            # Send live snapshot every 5 seconds
            payload = {
                "type": "snapshot",
                "nodes": [
                    {
                        "node_id": nid,
                        "lat": ns.latitude,
                        "lon": ns.longitude,
                        "fire_risk": ns.fire_risk,
                        "is_online": ns.is_online,
                        "last_seen": ns.last_seen,
                    }
                    for nid, ns in state["node_status"].items()
                ],
                "active_alerts": len(state["active_alerts"]),
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        _ws_clients.discard(websocket)
        logger.info(f"WebSocket client disconnected ({len(_ws_clients)} remaining)")
