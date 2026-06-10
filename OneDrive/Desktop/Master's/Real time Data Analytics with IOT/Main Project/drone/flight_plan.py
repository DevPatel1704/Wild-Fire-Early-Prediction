"""
Generates autonomous drone flight plans (waypoints) for a given fire alert location.
Uses a lawnmower survey pattern centred on the alert GPS coordinate.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Waypoint:
    latitude: float
    longitude: float
    altitude_m: float
    loiter_seconds: float = 0.0
    action: str = "fly"   # fly | photo | hover | rtl


@dataclass
class FlightPlan:
    plan_id: str
    target_lat: float
    target_lon: float
    home_lat: float
    home_lon: float
    waypoints: List[Waypoint] = field(default_factory=list)
    total_distance_km: float = 0.0
    estimated_duration_min: float = 0.0


class FlightPlanner:
    """Creates a survey flight plan over the fire alert zone."""

    DEFAULT_ALTITUDE = float(120)    # metres AGL (FAA max for hobby, PX4 default)
    SURVEY_SPEED_MS = float(15)      # m/s cruise speed
    SURVEY_RADIUS_KM = float(0.5)    # survey area radius around alert point
    PHOTO_INTERVAL_M = float(50)     # take photo every 50m along path

    def __init__(self, home_lat: float = 44.0, home_lon: float = -78.95):
        self.home_lat = home_lat
        self.home_lon = home_lon

    def build_survey_plan(
        self,
        plan_id: str,
        target_lat: float,
        target_lon: float,
        altitude_m: float = None,
    ) -> FlightPlan:
        alt = altitude_m or self.DEFAULT_ALTITUDE
        plan = FlightPlan(
            plan_id=plan_id,
            target_lat=target_lat,
            target_lon=target_lon,
            home_lat=self.home_lat,
            home_lon=self.home_lon,
        )

        # 1. Takeoff
        plan.waypoints.append(Waypoint(self.home_lat, self.home_lon, alt, action="fly"))

        # 2. Fly to area and do lawnmower survey
        survey_wps = self._lawnmower_pattern(target_lat, target_lon, alt)
        plan.waypoints.extend(survey_wps)

        # 3. Return to launch
        plan.waypoints.append(Waypoint(self.home_lat, self.home_lon, alt, loiter_seconds=5, action="rtl"))

        plan.total_distance_km = self._total_distance(plan.waypoints)
        plan.estimated_duration_min = round(
            (plan.total_distance_km * 1000 / self.SURVEY_SPEED_MS) / 60, 1
        )
        return plan

    def _lawnmower_pattern(self, center_lat: float, center_lon: float, alt: float) -> List[Waypoint]:
        """
        Generates a lawnmower grid pattern covering SURVEY_RADIUS_KM around the centre.
        Each row is offset by PHOTO_INTERVAL_M to ensure overlap.
        """
        wps = []
        radius_deg_lat = self.SURVEY_RADIUS_KM / 111.0
        radius_deg_lon = self.SURVEY_RADIUS_KM / (111.0 * math.cos(math.radians(center_lat)))
        step_deg_lat = (self.PHOTO_INTERVAL_M / 1000.0) / 111.0

        n_rows = max(3, int(2 * self.SURVEY_RADIUS_KM * 1000 / self.PHOTO_INTERVAL_M))
        for row in range(n_rows):
            lat = center_lat - radius_deg_lat + row * step_deg_lat
            if row % 2 == 0:
                wps.append(Waypoint(lat, center_lon - radius_deg_lon, alt, action="photo"))
                wps.append(Waypoint(lat, center_lon + radius_deg_lon, alt, action="photo"))
            else:
                wps.append(Waypoint(lat, center_lon + radius_deg_lon, alt, action="photo"))
                wps.append(Waypoint(lat, center_lon - radius_deg_lon, alt, action="photo"))

        # Hover at centre for 10s thermal scan
        wps.append(Waypoint(center_lat, center_lon, alt, loiter_seconds=10, action="hover"))
        return wps

    @staticmethod
    def _total_distance(waypoints: List[Waypoint]) -> float:
        dist = 0.0
        for i in range(1, len(waypoints)):
            a, b = waypoints[i - 1], waypoints[i]
            dist += FlightPlanner._haversine(a.latitude, a.longitude, b.latitude, b.longitude)
        return round(dist, 3)

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> float:
        R = 6371.0
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (math.sin(d_lat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(d_lon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
