"""
Entry point: runs the sensor network simulator and pushes readings to Kafka.
Usage:
    python -m simulator.run_simulator              # real-time mode (30s interval)
    python -m simulator.run_simulator --fast       # accelerated (no sleep)
    python -m simulator.run_simulator --export csv # write CSV to data/raw/
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from simulator.network import SensorNetwork
from simulator.fire_scenario import default_scenarios


def get_producer():
    from kafka import KafkaProducer
    servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
    for attempt in range(8):
        try:
            producer = KafkaProducer(
                bootstrap_servers=servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3,
                request_timeout_ms=10000,
            )
            logger.info(f"Kafka producer connected (attempt {attempt + 1}).")
            return producer
        except Exception as exc:
            wait = min(2 ** attempt, 20)
            logger.warning(f"Kafka connect attempt {attempt + 1} failed: {exc}. Retrying in {wait}s…")
            time.sleep(wait)
    logger.warning("Kafka unavailable after retries, running in log-only mode.")
    return None


def run(fast: bool = False, export_csv: bool = False, days: int = 3):
    start_time = datetime(2024, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
    scenarios = default_scenarios(44.0000, -78.9500, start_time)

    network = SensorNetwork(
        n_nodes=int(os.getenv("NUM_SENSOR_NODES", 100)),
        area_km=float(os.getenv("SIMULATION_AREA_KM", 10)),
        fire_scenarios=scenarios,
        seed=42,
    )

    topic_raw = os.getenv("KAFKA_TOPIC_SENSOR_RAW", "sensor.raw")
    producer = get_producer()
    interval = int(os.getenv("SENSOR_INTERVAL_SECONDS", 30))
    total_ticks = (days * 24 * 3600) // interval

    csv_writer = None
    csv_file = None
    if export_csv:
        out_path = Path("data/raw/simulated_readings.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(out_path, "w", newline="")
        fieldnames = [
            "node_id", "timestamp", "latitude", "longitude",
            "temperature_c", "humidity_pct", "surface_temp_c",
            "smoke_index", "co_ppm", "voc_index",
            "wind_speed_kmh", "wind_direction_deg",
            "fire_risk", "is_fire_event", "battery_pct", "signal_rssi",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        logger.info(f"Exporting CSV to {out_path}")

    logger.info(f"Starting simulation: {total_ticks} ticks over {days} days.")

    try:
        for tick in range(total_ticks):
            sim_time = start_time + timedelta(seconds=tick * interval)
            readings = network.tick(sim_time=sim_time)

            fire_count = sum(1 for r in readings if r.is_fire_event)
            if tick % 120 == 0 or fire_count > 0:
                logger.info(
                    f"Tick {tick}/{total_ticks} | {sim_time.strftime('%Y-%m-%d %H:%M')} "
                    f"| {len(readings)} nodes online | {fire_count} fire alerts"
                )

            for reading in readings:
                payload = reading.to_dict()
                if producer:
                    producer.send(topic_raw, value=payload)
                if csv_writer:
                    row = {
                        "node_id": reading.node_id,
                        "timestamp": reading.timestamp,
                        "latitude": reading.latitude,
                        "longitude": reading.longitude,
                        **{k: v for k, v in payload["sensors"].items()},
                        "fire_risk": reading.fire_risk,
                        "is_fire_event": reading.is_fire_event,
                        "battery_pct": reading.battery_pct,
                        "signal_rssi": reading.signal_rssi,
                    }
                    csv_writer.writerow(row)

            if producer:
                producer.flush()
            if not fast:
                time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Simulation stopped by user.")
    finally:
        if csv_file:
            csv_file.close()
            logger.info("CSV export complete.")
        if producer:
            producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wildfire IoT Sensor Simulator")
    parser.add_argument("--fast", action="store_true", help="No sleep between ticks")
    parser.add_argument("--export", choices=["csv"], help="Also export to CSV")
    parser.add_argument("--days", type=int, default=3, help="Simulation duration in days")
    args = parser.parse_args()

    logger.add("logs/simulator.log", rotation="50 MB")
    run(fast=args.fast, export_csv=(args.export == "csv"), days=args.days)
