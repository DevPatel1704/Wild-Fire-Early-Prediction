"""Tests for the sensor simulator."""

import pytest
from datetime import datetime, timezone

from simulator.sensor_node import SensorNode
from simulator.network import SensorNetwork
from simulator.fire_scenario import FireScenario


def test_sensor_node_produces_reading():
    node = SensorNode("NODE_000", 44.0, -78.95, seed=42)
    reading = node.read()
    assert reading is not None
    assert reading.node_id == "NODE_000"
    assert 0.0 <= reading.fire_risk <= 1.0
    assert -90 <= reading.temperature_c <= 70
    assert 0 <= reading.humidity_pct <= 100
    assert reading.smoke_index >= 0


def test_fire_influence_raises_risk():
    node = SensorNode("NODE_001", 44.0, -78.95, seed=7)
    r_normal = node.read()
    node.set_fire_influence(1.0)
    # Run several ticks to let state drift
    r_fire = None
    for _ in range(10):
        r_fire = node.read()
    assert r_fire is not None
    assert r_fire.smoke_index >= r_normal.smoke_index or r_fire.temperature_c >= r_normal.temperature_c


def test_fire_scenario_influence_at_epicentre():
    now = datetime(2024, 7, 1, 10, 0, tzinfo=timezone.utc)
    ignition = datetime(2024, 7, 1, 9, 0, tzinfo=timezone.utc)
    scenario = FireScenario("fire_test", 44.0, -78.95, ignition_time=ignition)
    influence = scenario.influence_at(44.0, -78.95, now)
    assert influence > 0.5, f"Expected high influence at epicentre, got {influence}"


def test_fire_scenario_no_influence_before_ignition():
    before = datetime(2024, 7, 1, 8, 0, tzinfo=timezone.utc)
    ignition = datetime(2024, 7, 1, 9, 0, tzinfo=timezone.utc)
    scenario = FireScenario("fire_test", 44.0, -78.95, ignition_time=ignition)
    influence = scenario.influence_at(44.0, -78.95, before)
    assert influence == 0.0


def test_sensor_network_tick_returns_readings():
    net = SensorNetwork(n_nodes=10, area_km=5, seed=0)
    readings = net.tick()
    assert len(readings) > 0
    assert len(readings) <= 10


def test_sensor_network_adjacency():
    net = SensorNetwork(n_nodes=10, area_km=5, seed=0)
    adj = net.adjacency_list(radius_km=2.0)
    assert len(adj) == 10
    # Most nodes should have at least one neighbour in a 10-node 5km grid
    neighbours = sum(len(v) for v in adj.values())
    assert neighbours > 0
