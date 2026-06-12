"""
Defines a fire ignition scenario that spreads influence across nearby sensor nodes.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List


@dataclass
class FireScenario:
    scenario_id: str
    ignition_lat: float
    ignition_lon: float
    ignition_time: datetime
    spread_rate_km_per_min: float = 0.05   # ~3 km/h spread
    wind_direction_deg: float = 270.0      # wind pushes fire east
    max_radius_km: float = 2.0
    peak_intensity: float = 1.0

    def influence_at(self, lat: float, lon: float, current_time: datetime) -> float:
        """
        Returns fire influence (0–1) for a sensor at (lat, lon) at current_time.
        Influence grows as the fire front approaches, peaks when it passes,
        then decays.
        """
        if current_time < self.ignition_time:
            return 0.0

        elapsed_min = (current_time - self.ignition_time).total_seconds() / 60.0
        fire_radius_km = min(elapsed_min * self.spread_rate_km_per_min, self.max_radius_km)

        dist_km = self._haversine(self.ignition_lat, self.ignition_lon, lat, lon)

        if dist_km > fire_radius_km + 1.0:
            return 0.0

        if dist_km <= fire_radius_km:
            # Inside fire front — full influence decaying slowly
            decay = max(0.0, 1.0 - (fire_radius_km - dist_km) / self.max_radius_km)
            return round(self.peak_intensity * max(decay, 0.5), 4)
        else:
            # Approaching fire front — pre-fire smoke/heat signal
            proximity = 1.0 - (dist_km - fire_radius_km) / 1.0
            return round(self.peak_intensity * proximity * 0.6, 4)

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> float:
        """Returns distance in km between two lat/lon points."""
        R = 6371.0
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (math.sin(d_lat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(d_lon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def default_scenarios(area_lat: float, area_lon: float, start_time: datetime) -> List[FireScenario]:
    """Five realistic fire scenarios injected at different times and locations."""
    return [
        FireScenario(
            scenario_id="fire_001",
            ignition_lat=area_lat + 0.02,
            ignition_lon=area_lon + 0.015,
            ignition_time=start_time + timedelta(hours=6),
            spread_rate_km_per_min=0.06,
            wind_direction_deg=260,
        ),
        FireScenario(
            scenario_id="fire_002",
            ignition_lat=area_lat - 0.03,
            ignition_lon=area_lon + 0.030,
            ignition_time=start_time + timedelta(hours=18),
            spread_rate_km_per_min=0.04,
            wind_direction_deg=310,
        ),
        FireScenario(
            scenario_id="fire_003",
            ignition_lat=area_lat + 0.04,
            ignition_lon=area_lon - 0.025,
            ignition_time=start_time + timedelta(hours=30),
            spread_rate_km_per_min=0.07,
            wind_direction_deg=240,
        ),
        FireScenario(
            scenario_id="fire_004",
            ignition_lat=area_lat - 0.01,
            ignition_lon=area_lon - 0.020,
            ignition_time=start_time + timedelta(hours=50),
            spread_rate_km_per_min=0.05,
            wind_direction_deg=290,
        ),
        FireScenario(
            scenario_id="fire_005",
            ignition_lat=area_lat + 0.035,
            ignition_lon=area_lon + 0.040,
            ignition_time=start_time + timedelta(hours=68),
            spread_rate_km_per_min=0.08,
            wind_direction_deg=270,
        ),
    ]
