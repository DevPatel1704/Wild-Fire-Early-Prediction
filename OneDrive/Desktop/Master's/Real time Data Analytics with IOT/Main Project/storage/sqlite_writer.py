"""
SQLite fallback writer — stores readings and alerts locally when InfluxDB is unavailable.
Useful for development and demo runs without Docker.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

DB_PATH = os.getenv("SQLITE_PATH", "data/wildfire.db")


class SQLiteWriter:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_tables()

    def _conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_tables(self):
        with self._lock, self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    temperature_c REAL,
                    humidity_pct REAL,
                    surface_temp_c REAL,
                    smoke_index REAL,
                    co_ppm REAL,
                    voc_index REAL,
                    wind_speed_kmh REAL,
                    wind_direction_deg REAL,
                    fire_risk REAL,
                    is_fire_event INTEGER,
                    battery_pct REAL,
                    signal_rssi INTEGER
                );

                CREATE TABLE IF NOT EXISTS sensor_aggregates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS fire_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    fire_risk_score REAL,
                    alert_level TEXT,
                    acknowledged INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_readings_node ON sensor_readings(node_id);
                CREATE INDEX IF NOT EXISTS idx_readings_ts ON sensor_readings(timestamp);
                CREATE INDEX IF NOT EXISTS idx_alerts_ts ON fire_alerts(timestamp);
            """)

    def write_raw(self, reading: dict):
        sensors = reading.get("sensors", {})
        with self._lock, self._conn() as conn:
            conn.execute("""
                INSERT INTO sensor_readings
                (node_id, timestamp, latitude, longitude,
                 temperature_c, humidity_pct, surface_temp_c, smoke_index,
                 co_ppm, voc_index, wind_speed_kmh, wind_direction_deg,
                 fire_risk, is_fire_event, battery_pct, signal_rssi)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                reading.get("node_id"), reading.get("timestamp"),
                reading.get("latitude"), reading.get("longitude"),
                sensors.get("temperature_c"), sensors.get("humidity_pct"),
                sensors.get("surface_temp_c"), sensors.get("smoke_index"),
                sensors.get("co_ppm"), sensors.get("voc_index"),
                sensors.get("wind_speed_kmh"), sensors.get("wind_direction_deg"),
                reading.get("fire_risk"), int(reading.get("is_fire_event", False)),
                reading.get("battery_pct"), reading.get("signal_rssi"),
            ))

    def write_aggregate(self, agg: dict):
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO sensor_aggregates (node_id, window_end, data_json) VALUES (?,?,?)",
                (agg.get("node_id"), agg.get("window_end"), json.dumps(agg))
            )

    def write_alert(self, alert: dict):
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO fire_alerts (node_id, timestamp, fire_risk_score, alert_level) VALUES (?,?,?,?)",
                (alert.get("node_id"), alert.get("timestamp"),
                 alert.get("fire_risk_score"), alert.get("alert_level"))
            )

    def get_recent_readings(self, limit: int = 200) -> list:
        with self._lock, self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_active_alerts(self) -> list:
        with self._lock, self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM fire_alerts WHERE acknowledged=0 ORDER BY timestamp DESC LIMIT 50"
            ).fetchall()
        return [dict(r) for r in rows]

    def acknowledge_alert(self, alert_id: int):
        with self._lock, self._conn() as conn:
            conn.execute("UPDATE fire_alerts SET acknowledged=1 WHERE id=?", (alert_id,))
