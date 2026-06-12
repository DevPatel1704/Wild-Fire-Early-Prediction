"""
PyTorch dataset that loads the simulated CSV and builds
(x, adj, y) tensors for training the GAT-LSTM.
"""

import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from loguru import logger


SENSOR_FEATURE_COLS = [
    "temperature_c", "humidity_pct", "surface_temp_c",
    "smoke_index", "co_ppm", "voc_index",
    "wind_speed_kmh", "wind_direction_deg",
]


def build_adjacency(positions: pd.DataFrame, radius_km: float = 1.5) -> np.ndarray:
    """Build adjacency matrix from node GPS positions."""
    n = len(positions)
    adj = np.zeros((n, n), dtype=np.float32)
    lats = positions["latitude"].values
    lons = positions["longitude"].values

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(lats[i], lons[i], lats[j], lons[j])
            if d <= radius_km:
                adj[i, j] = adj[j, i] = 1.0
    np.fill_diagonal(adj, 1.0)  # self-loops
    return adj


def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(d_lon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class WildfireDataset(Dataset):
    """
    Loads simulated sensor CSV and produces sliding-window samples.

    Each sample: (x, adj, y)
      x:   (N, T, F) float tensor  — N nodes, T time steps, F features
      adj: (N, N)    float tensor  — adjacency matrix (shared across all samples)
      y:   (N,)      float tensor  — fire risk label per node at step T
    """

    def __init__(
        self,
        csv_path: str = "data/raw/simulated_readings.csv",
        n_timesteps: int = 6,
        normalize: bool = True,
        nrows: int = 0,
    ):
        self.n_timesteps = n_timesteps
        df = pd.read_csv(csv_path, nrows=nrows if nrows > 0 else None)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["timestamp", "node_id"]).reset_index(drop=True)

        self.node_ids = sorted(df["node_id"].unique())
        self.n_nodes = len(self.node_ids)
        self.node_idx = {n: i for i, n in enumerate(self.node_ids)}

        # Build adjacency from last known positions
        positions = df.groupby("node_id")[["latitude", "longitude"]].last().reset_index()
        positions = positions.sort_values("node_id").reset_index(drop=True)
        self.adj = torch.tensor(build_adjacency(positions), dtype=torch.float32)

        # Pivot: (T_total, N, F)
        feature_cols = SENSOR_FEATURE_COLS + ["fire_risk"]
        pivoted = self._pivot(df, feature_cols)
        logger.info(f"Dataset shape: {pivoted.shape} (timesteps, nodes, features)")

        if normalize:
            self._mean = pivoted[:, :, :-1].mean(axis=(0, 1), keepdims=True)
            self._std = pivoted[:, :, :-1].std(axis=(0, 1), keepdims=True) + 1e-6
            pivoted[:, :, :-1] = (pivoted[:, :, :-1] - self._mean) / self._std

        self._data = pivoted.astype(np.float32)  # (T_total, N, F)

    def _pivot(self, df: pd.DataFrame, feature_cols) -> np.ndarray:
        """Reshape df into (timesteps, n_nodes, n_features) using vectorized ops."""
        timestamps = sorted(df["timestamp"].unique())
        T = len(timestamps)
        F = len(feature_cols)
        N = self.n_nodes
        arr = np.zeros((T, N, F), dtype=np.float64)

        ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
        # Map indices without copying the full DataFrame
        t_idx = df["timestamp"].map(ts_to_idx).values.astype(np.intp)
        n_idx = df["node_id"].map(self.node_idx).values.astype(np.intp)
        valid_mask = ~(np.isnan(t_idx.astype(float)) | np.isnan(n_idx.astype(float)))
        t_idx = t_idx[valid_mask]
        n_idx = n_idx[valid_mask]

        for f_idx, col in enumerate(feature_cols):
            if col in df.columns:
                vals = df[col].values[valid_mask]
                arr[t_idx, n_idx, f_idx] = vals

        return arr

    def __len__(self) -> int:
        return max(0, self._data.shape[0] - self.n_timesteps)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window = self._data[idx: idx + self.n_timesteps]   # (T, N, F)
        x = torch.tensor(window[:, :, :-1], dtype=torch.float32)  # (T, N, F-1)
        x = x.permute(1, 0, 2)                                    # (N, T, F-1)
        y = torch.tensor(self._data[idx + self.n_timesteps, :, -1], dtype=torch.float32)  # (N,)
        return x, self.adj, y
