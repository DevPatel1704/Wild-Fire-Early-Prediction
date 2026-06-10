"""
Simulates a single ESP32-based sensor node in the forest.
Each node carries 6 sensors: MQ-2, MLX90614, DHT22, BME680, anemometer, GPS.
"""

import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class SensorReading:
    node_id: str
    timestamp: str
    latitude: float
    longitude: float
    temperature_c: float       # DHT22 air temperature
    humidity_pct: float        # DHT22 relative humidity
    surface_temp_c: float      # MLX90614 infrared surface temp
    smoke_index: float         # MQ-2 smoke density (0–5 normalised)
    co_ppm: float              # BME680 carbon monoxide (ppm)
    voc_index: float           # BME680 VOC index (0–500)
    wind_speed_kmh: float      # Anemometer
    wind_direction_deg: float  # 0–360°
    fire_risk: float           # Ground truth label 0–1 (for training)
    is_fire_event: bool
    battery_pct: float
    signal_rssi: int

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "sensors": {
                "temperature_c": round(self.temperature_c, 2),
                "humidity_pct": round(self.humidity_pct, 2),
                "surface_temp_c": round(self.surface_temp_c, 2),
                "smoke_index": round(self.smoke_index, 4),
                "co_ppm": round(self.co_ppm, 4),
                "voc_index": round(self.voc_index, 2),
                "wind_speed_kmh": round(self.wind_speed_kmh, 2),
                "wind_direction_deg": round(self.wind_direction_deg, 1),
            },
            "fire_risk": round(self.fire_risk, 4),
            "is_fire_event": self.is_fire_event,
            "battery_pct": round(self.battery_pct, 1),
            "signal_rssi": self.signal_rssi,
        }


class SensorNode:
    """Simulates one ESP32 sensor node with realistic noise and drift."""

    BASELINE = {
        "temperature_c": 22.0,
        "humidity_pct": 65.0,
        "surface_temp_c": 24.0,
        "smoke_index": 0.08,
        "co_ppm": 0.4,
        "voc_index": 50.0,
        "wind_speed_kmh": 8.0,
        "wind_direction_deg": 180.0,
    }

    def __init__(self, node_id: str, latitude: float, longitude: float, seed: int = None):
        self.node_id = node_id
        self.latitude = latitude
        self.longitude = longitude
        self._rng = random.Random(seed or id(self))
        self._state = dict(self.BASELINE)
        self._fire_influence = 0.0  # 0 = no fire influence, 1 = full fire
        self._battery = self._rng.uniform(70, 100)
        self._offline = False
        self._offline_countdown = 0

    # ------------------------------------------------------------------
    def set_fire_influence(self, influence: float):
        """Set how strongly an approaching fire affects this node (0–1)."""
        self._fire_influence = max(0.0, min(1.0, influence))

    # ------------------------------------------------------------------
    def _drift(self, key: str, amount: float, sigma: float) -> float:
        """Gradually move a state variable with Gaussian noise."""
        self._state[key] += amount + self._rng.gauss(0, sigma)
        return self._state[key]

    def _clamp(self, val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))

    # ------------------------------------------------------------------
    def _simulate_offline(self) -> bool:
        """Randomly go offline to simulate LoRaWAN packet loss (~3% chance)."""
        if self._offline:
            self._offline_countdown -= 1
            if self._offline_countdown <= 0:
                self._offline = False
            return True
        if self._rng.random() < 0.03:
            self._offline = True
            self._offline_countdown = self._rng.randint(1, 4)
            return True
        return False

    # ------------------------------------------------------------------
    def read(self, sim_time: Optional[datetime] = None) -> Optional[SensorReading]:
        """Generate one sensor reading. Returns None if node is offline."""
        if self._simulate_offline():
            return None

        fi = self._fire_influence
        ts = (sim_time or datetime.now(timezone.utc)).isoformat()

        # Drift baseline with fire influence
        temp = self._clamp(
            self._drift("temperature_c", 0.02 + fi * 1.2, 0.3 + fi * 0.5),
            -10, 70,
        )
        hum = self._clamp(
            self._drift("humidity_pct", -0.01 - fi * 0.8, 0.4 + fi * 0.3),
            5, 100,
        )
        surf = self._clamp(
            self._drift("surface_temp_c", 0.03 + fi * 1.5, 0.4 + fi * 0.6),
            -5, 100,
        )
        smoke = self._clamp(
            self._drift("smoke_index", fi * 0.15, 0.01 + fi * 0.05),
            0, 5,
        )
        co = self._clamp(
            self._drift("co_ppm", fi * 0.08, 0.01 + fi * 0.04),
            0, 100,
        )
        voc = self._clamp(
            self._drift("voc_index", fi * 5.0, 1.0 + fi * 2.0),
            0, 500,
        )
        wind_spd = self._clamp(
            self._drift("wind_speed_kmh", fi * 0.3, 0.5),
            0, 120,
        )
        wind_dir = self._state["wind_direction_deg"] + self._rng.gauss(0, 5)
        wind_dir = wind_dir % 360

        # Composite fire risk score (ground truth for supervised learning)
        fire_risk = self._compute_risk(temp, hum, surf, smoke, co, wind_spd)

        self._battery -= self._rng.uniform(0.001, 0.003)
        rssi = self._rng.randint(-110, -60)

        return SensorReading(
            node_id=self.node_id,
            timestamp=ts,
            latitude=self.latitude + self._rng.gauss(0, 0.00001),
            longitude=self.longitude + self._rng.gauss(0, 0.00001),
            temperature_c=temp,
            humidity_pct=hum,
            surface_temp_c=surf,
            smoke_index=smoke,
            co_ppm=co,
            voc_index=voc,
            wind_speed_kmh=wind_spd,
            wind_direction_deg=wind_dir,
            fire_risk=fire_risk,
            is_fire_event=fire_risk >= 0.80,
            battery_pct=self._clamp(self._battery, 0, 100),
            signal_rssi=rssi,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_risk(temp, hum, surf, smoke, co, wind) -> float:
        """Weighted composite risk score — mirrors what the ML model learns."""
        score = (
            0.20 * min(max((temp - 25) / 45, 0), 1) +
            0.20 * min(max((100 - hum) / 95, 0), 1) +
            0.15 * min(max((surf - 30) / 70, 0), 1) +
            0.25 * min(smoke / 3.5, 1) +
            0.10 * min(co / 20, 1) +
            0.10 * min(wind / 60, 1)
        )
        return round(min(score, 1.0), 4)
