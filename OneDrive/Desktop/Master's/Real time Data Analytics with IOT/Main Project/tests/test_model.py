"""Tests for the GAT-LSTM model components."""

import pytest
import torch
import numpy as np

from model.gat_layer import GATLayer, MultiHeadGAT
from model.gat_lstm import GATLSTM
from pipeline.aggregator import SensorAggregator, NodeBuffer


def make_adj(n: int) -> torch.Tensor:
    adj = torch.eye(n)
    # Connect first and second node
    if n >= 2:
        adj[0, 1] = adj[1, 0] = 1.0
    return adj


def test_gat_layer_forward():
    layer = GATLayer(in_features=8, out_features=16)
    h = torch.randn(5, 8)
    adj = make_adj(5)
    out = layer(h, adj)
    assert out.shape == (5, 16)


def test_multi_head_gat():
    gat = MultiHeadGAT(in_features=8, out_features=32, n_heads=4)
    h = torch.randn(5, 8)
    adj = make_adj(5)
    out = gat(h, adj)
    assert out.shape == (5, 32)


def test_gat_lstm_forward():
    model = GATLSTM(n_features=8, n_timesteps=6, gat_hidden=32, lstm_hidden=64, n_heads=2)
    batch, n_nodes, t_steps, n_feat = 2, 5, 6, 8
    x = torch.randn(batch, n_nodes, t_steps, n_feat)
    adj = make_adj(n_nodes)
    out = model(x, adj)
    assert out.shape == (batch, n_nodes, 1)
    assert (out >= 0).all() and (out <= 1).all(), "Output should be in [0, 1]"


def test_gat_lstm_no_nans():
    model = GATLSTM(n_features=8, n_timesteps=6, gat_hidden=32, lstm_hidden=64, n_heads=2)
    x = torch.randn(1, 10, 6, 8)
    adj = make_adj(10)
    out = model(x, adj)
    assert not torch.isnan(out).any(), "Model output contains NaNs"


def test_aggregator_ingest_and_flush():
    agg = SensorAggregator()
    for i in range(5):
        reading = {
            "node_id": "NODE_000",
            "timestamp": f"2024-07-01T10:0{i}:00Z",
            "latitude": 44.0, "longitude": -78.95,
            "sensors": {
                "temperature_c": 25.0 + i,
                "humidity_pct": 60.0,
                "surface_temp_c": 26.0,
                "smoke_index": 0.1,
                "co_ppm": 0.5,
                "voc_index": 50.0,
                "wind_speed_kmh": 10.0,
                "wind_direction_deg": 180.0,
            },
            "fire_risk": 0.1,
            "is_fire_event": False,
        }
        agg.ingest(reading)
    aggregates, feature_tensors = agg.flush()
    assert len(aggregates) == 1
    assert aggregates[0]["node_id"] == "NODE_000"
    assert aggregates[0]["count"] == 5


def test_node_buffer_history():
    buf = NodeBuffer("NODE_001", max_history=3)
    sample_reading = {
        "node_id": "NODE_001", "timestamp": "t", "latitude": 44.0, "longitude": -78.95,
        "sensors": {k: 1.0 for k in NodeBuffer.SENSOR_KEYS},
        "fire_risk": 0.5, "is_fire_event": False,
    }
    for _ in range(3):
        for _ in range(5):
            buf.add(sample_reading)
        buf.aggregate()

    vec = buf.get_history_feature_vector()
    assert vec is not None
    assert vec.shape == (3, len(NodeBuffer.SENSOR_KEYS) * 3 + 1)
