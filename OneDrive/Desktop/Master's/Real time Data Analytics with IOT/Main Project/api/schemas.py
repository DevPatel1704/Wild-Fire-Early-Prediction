"""Pydantic models for API request/response validation."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SensorData(BaseModel):
    temperature_c: float
    humidity_pct: float
    surface_temp_c: float
    smoke_index: float
    co_ppm: float
    voc_index: float
    wind_speed_kmh: float
    wind_direction_deg: float


class SensorReading(BaseModel):
    node_id: str
    timestamp: str
    latitude: float
    longitude: float
    sensors: SensorData
    fire_risk: float = Field(ge=0.0, le=1.0)
    is_fire_event: bool
    battery_pct: float
    signal_rssi: int


class NodeStatus(BaseModel):
    node_id: str
    latitude: float
    longitude: float
    last_seen: Optional[str] = None
    fire_risk: float = 0.0
    is_online: bool = True
    battery_pct: float = 100.0


class FireAlert(BaseModel):
    id: Optional[int] = None
    node_id: str
    timestamp: str
    fire_risk_score: float = Field(ge=0.0, le=1.0)
    alert_level: str  # "HIGH" or "CRITICAL"
    acknowledged: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class PredictionRequest(BaseModel):
    node_id: str
    feature_vector: List[float]


class PredictionResponse(BaseModel):
    node_id: str
    fire_risk_10min: float
    fire_risk_30min: float
    alert_level: str
    model_version: str = "gat_lstm_v1"


class AlertAcknowledge(BaseModel):
    alert_id: int


class SystemStatus(BaseModel):
    kafka_connected: bool
    influxdb_connected: bool
    model_loaded: bool
    nodes_online: int
    active_alerts: int
    uptime_seconds: float
