"""
InfluxDB 2.x writer for sensor aggregates and fire alerts.
Falls back gracefully if InfluxDB is unavailable.
"""

import os
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    _INFLUX_AVAILABLE = True
except ImportError:
    _INFLUX_AVAILABLE = False
    logger.warning("influxdb-client not installed. InfluxDB writer disabled.")


class InfluxDBWriter:
    def __init__(
        self,
        url: str = None,
        token: str = None,
        org: str = None,
        bucket: str = None,
    ):
        self.url = url or os.getenv("INFLUXDB_URL", "http://localhost:8086")
        self.token = token or os.getenv("INFLUXDB_TOKEN", "wildfire-super-secret-token")
        self.org = org or os.getenv("INFLUXDB_ORG", "wildfire-org")
        self.bucket = bucket or os.getenv("INFLUXDB_BUCKET", "sensor_data")
        self._client: Optional[object] = None
        self._write_api: Optional[object] = None
        self._connected = False

        if _INFLUX_AVAILABLE:
            self._connect()

    def _connect(self):
        try:
            self._client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
            self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
            self._connected = True
            logger.info(f"InfluxDB connected: {self.url} | org={self.org} | bucket={self.bucket}")
        except Exception as exc:
            logger.warning(f"InfluxDB connection failed: {exc}")
            self._connected = False

    def write_raw(self, reading: dict):
        """Write a single raw sensor reading."""
        if not self._connected:
            return
        try:
            p = (
                Point("sensor_reading")
                .tag("node_id", reading.get("node_id", "unknown"))
                .field("temperature_c", reading["sensors"].get("temperature_c", 0.0))
                .field("humidity_pct", reading["sensors"].get("humidity_pct", 0.0))
                .field("smoke_index", reading["sensors"].get("smoke_index", 0.0))
                .field("co_ppm", reading["sensors"].get("co_ppm", 0.0))
                .field("wind_speed_kmh", reading["sensors"].get("wind_speed_kmh", 0.0))
                .field("fire_risk", reading.get("fire_risk", 0.0))
                .field("latitude", reading.get("latitude", 0.0))
                .field("longitude", reading.get("longitude", 0.0))
                .time(reading.get("timestamp", datetime.now(timezone.utc).isoformat()),
                      WritePrecision.NANOSECOND)
            )
            self._write_api.write(bucket=self.bucket, org=self.org, record=p)
        except Exception as exc:
            logger.debug(f"InfluxDB write_raw error: {exc}")

    def write_aggregate(self, agg: dict):
        """Write a 1-minute aggregate record."""
        if not self._connected:
            return
        try:
            p = Point("sensor_aggregate").tag("node_id", agg.get("node_id", "unknown"))
            for key, val in agg.items():
                if key not in ("node_id", "window_end", "any_fire_event", "imputed") and isinstance(val, (int, float)):
                    p = p.field(key, float(val))
            p = p.field("any_fire_event", int(agg.get("any_fire_event", False)))
            p = p.time(agg.get("window_end", datetime.now(timezone.utc).isoformat()),
                       WritePrecision.NANOSECOND)
            self._write_api.write(bucket=self.bucket, org=self.org, record=p)
        except Exception as exc:
            logger.debug(f"InfluxDB write_aggregate error: {exc}")

    def write_alert(self, alert: dict):
        """Write a fire alert event."""
        if not self._connected:
            return
        try:
            p = (
                Point("fire_alert")
                .tag("node_id", alert.get("node_id", "unknown"))
                .tag("alert_level", alert.get("alert_level", "HIGH"))
                .field("fire_risk_score", float(alert.get("fire_risk_score", 0.0)))
                .time(alert.get("timestamp", datetime.now(timezone.utc).isoformat()),
                      WritePrecision.NANOSECOND)
            )
            self._write_api.write(bucket=self.bucket, org=self.org, record=p)
        except Exception as exc:
            logger.debug(f"InfluxDB write_alert error: {exc}")

    def query(self, flux_query: str) -> list:
        if not self._connected:
            return []
        try:
            query_api = self._client.query_api()
            tables = query_api.query(flux_query, org=self.org)
            results = []
            for table in tables:
                for record in table.records:
                    results.append(record.values)
            return results
        except Exception as exc:
            logger.warning(f"InfluxDB query error: {exc}")
            return []

    def close(self):
        if self._client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
