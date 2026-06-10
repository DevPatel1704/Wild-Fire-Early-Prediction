"""
Drone Dispatcher: listens for fire alerts from Kafka and sends flight commands
to the PX4 drone via MAVSDK (or simulates them when MAVSDK is unavailable).

Run:
    python -m drone.dispatcher
"""

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from loguru import logger

from drone.flight_plan import FlightPlanner, FlightPlan

ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", 0.80))
DRONE_CONNECTION = os.getenv("DRONE_CONNECTION_STRING", "udp://:14540")
HOME_LAT = float(os.getenv("HOME_LAT", 44.0))
HOME_LON = float(os.getenv("HOME_LON", -78.95))


class DroneDispatcher:
    def __init__(self):
        self._planner = FlightPlanner(home_lat=HOME_LAT, home_lon=HOME_LON)
        self._active_missions: Dict[str, FlightPlan] = {}
        self._drone = None

    # ------------------------------------------------------------------
    async def connect_drone(self) -> bool:
        """Try to connect to PX4 via MAVSDK. Returns True if successful."""
        try:
            from mavsdk import System
            self._drone = System()
            await self._drone.connect(system_address=DRONE_CONNECTION)
            async for state in self._drone.core.connection_state():
                if state.is_connected:
                    logger.info(f"Drone connected via {DRONE_CONNECTION}")
                    return True
                break
        except ImportError:
            logger.warning("mavsdk not installed — drone commands will be simulated.")
        except Exception as exc:
            logger.warning(f"Drone connection failed: {exc} — running in simulation mode.")
        return False

    # ------------------------------------------------------------------
    async def dispatch(self, alert: dict) -> Optional[FlightPlan]:
        """Generate a flight plan and execute it (or simulate it)."""
        node_id = alert.get("node_id", "unknown")
        risk = float(alert.get("fire_risk_score", 0.0))

        if risk < ALERT_THRESHOLD:
            return None

        plan_id = f"mission_{uuid.uuid4().hex[:8]}"
        lat = alert.get("latitude", HOME_LAT + 0.02)
        lon = alert.get("longitude", HOME_LON + 0.01)

        plan = self._planner.build_survey_plan(plan_id, lat, lon)
        self._active_missions[plan_id] = plan

        logger.warning(
            f"DRONE DISPATCH | plan={plan_id} | node={node_id} | risk={risk:.3f} "
            f"| target=({lat:.4f},{lon:.4f}) | waypoints={len(plan.waypoints)} "
            f"| est. {plan.estimated_duration_min} min"
        )

        if self._drone:
            await self._execute_mavlink(plan)
        else:
            await self._simulate_mission(plan)

        return plan

    # ------------------------------------------------------------------
    async def _execute_mavlink(self, plan: FlightPlan):
        """Send waypoints to the real PX4 drone via MAVSDK."""
        from mavsdk.mission import MissionItem, MissionPlan

        mission_items = []
        for wp in plan.waypoints:
            mission_items.append(MissionItem(
                latitude_deg=wp.latitude,
                longitude_deg=wp.longitude,
                relative_altitude_m=wp.altitude_m,
                speed_m_s=15.0,
                is_fly_through=True,
                gimbal_pitch_deg=float("nan"),
                gimbal_yaw_deg=float("nan"),
                camera_action=MissionItem.CameraAction.NONE,
                loiter_time_s=wp.loiter_seconds,
                camera_photo_interval_s=2.0 if wp.action == "photo" else float("nan"),
            ))

        await self._drone.mission.set_return_to_launch_after_mission(True)
        await self._drone.mission.upload_mission(MissionPlan(mission_items))
        await self._drone.action.arm()
        await self._drone.mission.start_mission()
        logger.info(f"Mission {plan.plan_id} started on PX4 drone.")

    # ------------------------------------------------------------------
    async def _simulate_mission(self, plan: FlightPlan):
        """Log the mission in simulation mode (no physical drone required)."""
        logger.info(f"[SIM] Mission {plan.plan_id} — {len(plan.waypoints)} waypoints:")
        for i, wp in enumerate(plan.waypoints):
            logger.info(
                f"  WP{i:02d}: ({wp.latitude:.5f}, {wp.longitude:.5f}) "
                f"alt={wp.altitude_m}m action={wp.action} loiter={wp.loiter_seconds}s"
            )
        logger.info(
            f"[SIM] Total distance: {plan.total_distance_km:.2f} km | "
            f"Est. duration: {plan.estimated_duration_min} min"
        )

    # ------------------------------------------------------------------
    async def listen_alerts(self):
        """Consume fire alerts from Kafka and dispatch drones."""
        try:
            from kafka import KafkaConsumer
            from ingestion.topics import TOPIC_FIRE_ALERTS

            servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
            consumer = KafkaConsumer(
                TOPIC_FIRE_ALERTS,
                bootstrap_servers=servers,
                group_id="drone-dispatcher",
                value_deserializer=lambda b: json.loads(b.decode()),
                auto_offset_reset="latest",
            )
            logger.info("Drone dispatcher listening for fire alerts...")

            for msg in consumer:
                alert = msg.value
                await self.dispatch(alert)

        except Exception as exc:
            logger.error(f"Drone dispatcher error: {exc}")


async def main():
    dispatcher = DroneDispatcher()
    await dispatcher.connect_drone()
    await dispatcher.listen_alerts()


if __name__ == "__main__":
    logger.add("logs/drone.log", rotation="10 MB")
    asyncio.run(main())
