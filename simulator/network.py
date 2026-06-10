"""
Manages a grid of 100 sensor nodes deployed across a 10x10 km forest area.
"""

import math
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .sensor_node import SensorNode, SensorReading
from .fire_scenario import FireScenario


class SensorNetwork:
    """
    Deploys N sensor nodes on a grid with slight random offsets (realistic placement).
    Propagates fire scenario influence to each node at each tick.
    """

    # Ontario Tech field site approximate centre
    DEFAULT_CENTRE = (44.0000, -78.9500)

    def __init__(
        self,
        n_nodes: int = 100,
        area_km: float = 10.0,
        centre: Tuple[float, float] = None,
        fire_scenarios: Optional[List[FireScenario]] = None,
        seed: int = 42,
    ):
        self.n_nodes = n_nodes
        self.area_km = area_km
        self.centre_lat, self.centre_lon = centre or self.DEFAULT_CENTRE
        self.fire_scenarios = fire_scenarios or []
        self._rng = random.Random(seed)

        self.nodes: Dict[str, SensorNode] = {}
        self._build_grid()

    # ------------------------------------------------------------------
    def _build_grid(self):
        """Place nodes on an approx. square grid with random jitter."""
        side = math.ceil(math.sqrt(self.n_nodes))
        spacing_deg = (self.area_km / 111.0) / side  # ~111 km per degree lat

        for i in range(self.n_nodes):
            row = i // side
            col = i % side
            lat = self.centre_lat + (row - side / 2) * spacing_deg
            lat += self._rng.uniform(-spacing_deg * 0.3, spacing_deg * 0.3)
            lon_spacing = spacing_deg / math.cos(math.radians(self.centre_lat))
            lon = self.centre_lon + (col - side / 2) * lon_spacing
            lon += self._rng.uniform(-lon_spacing * 0.3, lon_spacing * 0.3)

            node_id = f"NODE_{i:03d}"
            self.nodes[node_id] = SensorNode(node_id, lat, lon, seed=self._rng.randint(0, 99999))

    # ------------------------------------------------------------------
    def _update_fire_influence(self, current_time: datetime):
        for node in self.nodes.values():
            total_influence = 0.0
            for scenario in self.fire_scenarios:
                infl = scenario.influence_at(node.latitude, node.longitude, current_time)
                total_influence = max(total_influence, infl)
            node.set_fire_influence(min(total_influence, 1.0))

    # ------------------------------------------------------------------
    def tick(self, sim_time: Optional[datetime] = None) -> List[SensorReading]:
        """
        Generate one round of readings from all nodes.
        Returns only online nodes (None readings are dropped).
        """
        now = sim_time or datetime.now(timezone.utc)
        self._update_fire_influence(now)

        readings = []
        for node in self.nodes.values():
            reading = node.read(sim_time=now)
            if reading is not None:
                readings.append(reading)
        return readings

    # ------------------------------------------------------------------
    def node_positions(self) -> List[Dict]:
        """Returns all node GPS positions — used to build the graph edges."""
        return [
            {"node_id": nid, "lat": n.latitude, "lon": n.longitude}
            for nid, n in self.nodes.items()
        ]

    def adjacency_list(self, radius_km: float = 1.5) -> Dict[str, List[str]]:
        """
        Returns a dict mapping each node to its neighbours within radius_km.
        Used to build the graph for GAT.
        """
        positions = {nid: (n.latitude, n.longitude) for nid, n in self.nodes.items()}
        adj: Dict[str, List[str]] = {nid: [] for nid in self.nodes}

        node_ids = list(self.nodes.keys())
        for i, a in enumerate(node_ids):
            for b in node_ids[i + 1:]:
                lat_a, lon_a = positions[a]
                lat_b, lon_b = positions[b]
                d = self._haversine(lat_a, lon_a, lat_b, lon_b)
                if d <= radius_km:
                    adj[a].append(b)
                    adj[b].append(a)
        return adj

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> float:
        R = 6371.0
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (math.sin(d_lat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(d_lon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
