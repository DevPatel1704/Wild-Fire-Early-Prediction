"""
1-minute tumbling window aggregator.
Collects readings per node and produces per-node summary statistics
that feed into the GAT-LSTM model feature vector.
"""

import os
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


WINDOW_SIZE = int(os.getenv("AGG_WINDOW_SECONDS", 60))
HISTORY_STEPS = 6  # 6 x 1-min windows = 6 minutes of history per node


class NodeBuffer:
    """Rolling buffer of raw readings for one sensor node."""

    SENSOR_KEYS = [
        "temperature_c", "humidity_pct", "surface_temp_c",
        "smoke_index", "co_ppm", "voc_index",
        "wind_speed_kmh", "wind_direction_deg",
    ]

    def __init__(self, node_id: str, max_history: int = HISTORY_STEPS):
        self.node_id = node_id
        self.max_history = max_history
        self._readings: deque = deque()
        self._aggregates: deque = deque(maxlen=max_history)  # rolling window aggs

    def add(self, reading: dict):
        self._readings.append(reading)

    def aggregate(self) -> Optional[dict]:
        """Compute stats over current window, push to history, return agg dict."""
        if not self._readings:
            return None

        readings = list(self._readings)
        self._readings.clear()

        agg = {"node_id": self.node_id, "window_end": datetime.now(timezone.utc).isoformat()}
        sensors = readings[0].get("sensors", {})

        for key in self.SENSOR_KEYS:
            vals = [r["sensors"][key] for r in readings if key in r.get("sensors", {})]
            if vals:
                agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
                agg[f"{key}_max"] = round(float(np.max(vals)), 4)
                agg[f"{key}_std"] = round(float(np.std(vals)), 4)
            else:
                agg[f"{key}_mean"] = agg[f"{key}_max"] = agg[f"{key}_std"] = 0.0

        agg["count"] = len(readings)
        agg["max_fire_risk"] = round(max(r.get("fire_risk", 0) for r in readings), 4)
        agg["any_fire_event"] = any(r.get("is_fire_event", False) for r in readings)
        agg["latitude"] = readings[-1].get("latitude", 0.0)
        agg["longitude"] = readings[-1].get("longitude", 0.0)

        self._aggregates.append(agg)
        return agg

    def get_history_feature_vector(self) -> Optional[np.ndarray]:
        """
        Returns a (history_steps, n_features) array for the LSTM.
        Uses only mean values (8 features) to match the training dataset format.
        Returns None if not enough history.
        """
        if len(self._aggregates) < self.max_history:
            return None

        feature_keys = [f"{key}_mean" for key in self.SENSOR_KEYS]

        matrix = []
        for agg in self._aggregates:
            row = [agg.get(k, 0.0) for k in feature_keys]
            matrix.append(row)
        return np.array(matrix, dtype=np.float32)


class SensorAggregator:
    """
    Maintains a NodeBuffer per node.
    Call .ingest(reading) for each raw message.
    Call .flush() every WINDOW_SIZE seconds to get aggregated records.
    """

    def __init__(self):
        self._buffers: Dict[str, NodeBuffer] = defaultdict(
            lambda: NodeBuffer("unknown")
        )

    def ingest(self, reading: dict):
        node_id = reading.get("node_id", "unknown")
        if node_id not in self._buffers:
            self._buffers[node_id] = NodeBuffer(node_id)
        self._buffers[node_id].add(reading)

    def flush(self) -> Tuple[List[dict], List[Tuple[str, np.ndarray]]]:
        """
        Returns:
            aggregates: list of per-node aggregate dicts (→ Kafka agg topic, InfluxDB)
            feature_tensors: list of (node_id, history_array) for GAT-LSTM inference
        """
        aggregates = []
        feature_tensors = []

        for node_id, buf in self._buffers.items():
            agg = buf.aggregate()
            if agg:
                aggregates.append(agg)
                vec = buf.get_history_feature_vector()
                if vec is not None:
                    feature_tensors.append((node_id, vec))

        return aggregates, feature_tensors

    def impute_missing(self, node_id: str, neighbour_aggs: List[dict]) -> dict:
        """
        If a node has no readings in the window (went offline),
        impute its values from the mean of neighbouring nodes.
        """
        if not neighbour_aggs:
            return {}

        keys = [k for k in neighbour_aggs[0] if k.endswith(("_mean", "_max", "_std"))]
        imputed = {}
        for k in keys:
            vals = [n[k] for n in neighbour_aggs if k in n]
            imputed[k] = round(float(np.mean(vals)), 4) if vals else 0.0

        imputed["node_id"] = node_id
        imputed["imputed"] = True
        imputed["count"] = 0
        return imputed
