"""
Real-time stream processor.
Reads from Kafka sensor.raw → aggregates → runs GAT-LSTM inference →
writes alerts to Kafka fire.alerts and InfluxDB.

Run with:
    python -m pipeline.stream_processor
"""

import json
import os
import time
from threading import Thread
from typing import Optional

from loguru import logger

from ingestion.topics import TOPIC_SENSOR_RAW, TOPIC_SENSOR_AGG, TOPIC_FIRE_ALERTS
from pipeline.aggregator import SensorAggregator

WINDOW_SECONDS = int(os.getenv("AGG_WINDOW_SECONDS", 60))
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", 0.80))


class WildfireStreamProcessor:
    """
    Orchestrates the streaming pipeline:
    1. Consumes raw sensor messages from Kafka
    2. Aggregates them in 1-minute tumbling windows
    3. Runs GAT-LSTM inference (if model is loaded)
    4. Publishes fire alerts back to Kafka
    5. Writes aggregates to InfluxDB
    """

    def __init__(self, bootstrap_servers: str = None, model=None, influx_writer=None):
        from kafka import KafkaConsumer, KafkaProducer

        servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")

        for attempt in range(12):
            try:
                self._consumer = KafkaConsumer(
                    TOPIC_SENSOR_RAW,
                    bootstrap_servers=servers,
                    group_id="wildfire-pipeline",
                    value_deserializer=lambda b: json.loads(b.decode()),
                    auto_offset_reset="latest",
                    enable_auto_commit=True,
                    request_timeout_ms=30000,
                    session_timeout_ms=6000,
                )
                self._producer = KafkaProducer(
                    bootstrap_servers=servers,
                    value_serializer=lambda v: json.dumps(v).encode(),
                )
                logger.info(f"Kafka connected (attempt {attempt + 1})")
                break
            except Exception as exc:
                wait = min(2 ** attempt, 30)
                logger.warning(f"Kafka connect attempt {attempt + 1} failed: {exc}. Retrying in {wait}s…")
                time.sleep(wait)
        else:
            raise RuntimeError("Could not connect to Kafka after 12 attempts.")
        self._aggregator = SensorAggregator()
        self._model = model
        self._influx = influx_writer
        self._running = False

    # ------------------------------------------------------------------
    def _flush_and_infer(self):
        """Called every WINDOW_SECONDS to aggregate and run inference."""
        aggregates, feature_tensors = self._aggregator.flush()
        if not aggregates:
            return

        # Write aggregates to Kafka agg topic and InfluxDB
        for agg in aggregates:
            self._producer.send(TOPIC_SENSOR_AGG, value=agg)
            if self._influx:
                self._influx.write_aggregate(agg)

        # Run model inference if available, else fall back to raw fire_risk threshold
        if self._model and feature_tensors:
            try:
                predictions = self._model.predict_batch(feature_tensors)
            except Exception as exc:
                logger.error(f"Inference error: {exc}")
                predictions = {}
        else:
            # No model loaded — use max_fire_risk from aggregated window
            predictions = {
                agg["node_id"]: agg.get("max_fire_risk", 0.0)
                for agg in aggregates
            }

        for node_id, risk_score in predictions.items():
            if risk_score >= ALERT_THRESHOLD:
                alert = {
                    "node_id": node_id,
                    "fire_risk_score": round(risk_score, 4),
                    "alert_level": "CRITICAL" if risk_score >= 0.90 else "HIGH",
                    "timestamp": aggregates[0].get("window_end", ""),
                }
                self._producer.send(TOPIC_FIRE_ALERTS, value=alert)
                logger.warning(
                    f"FIRE ALERT: node={node_id} risk={risk_score:.3f} "
                    f"level={alert['alert_level']}"
                )
                if self._influx:
                    self._influx.write_alert(alert)

        self._producer.flush()
        logger.debug(f"Flushed {len(aggregates)} node aggregates.")

    # ------------------------------------------------------------------
    def _window_timer(self):
        """Background thread that triggers a flush every WINDOW_SECONDS."""
        while self._running:
            time.sleep(WINDOW_SECONDS)
            if self._running:
                self._flush_and_infer()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def run(self):
        """Start consuming. Blocks until stopped."""
        self._running = True
        timer_thread = Thread(target=self._window_timer, daemon=True)
        timer_thread.start()
        logger.info(f"Stream processor started (window={WINDOW_SECONDS}s)")

        msg_count = 0
        latency_samples = []
        try:
            for msg in self._consumer:
                self._aggregator.ingest(msg.value)
                msg_count += 1
                # Use Kafka produce timestamp so simulated historical sensor
                # timestamps don't skew the metric.
                if msg.timestamp:
                    latency_ms = (time.time() * 1000) - msg.timestamp
                    if 0 < latency_ms < 60_000:
                        latency_samples.append(latency_ms)
                if msg_count % 500 == 0:
                    if latency_samples:
                        avg_lat = sum(latency_samples[-100:]) / len(latency_samples[-100:])
                        lat_str = f"avg ingestion latency (last 100): {avg_lat:.0f} ms"
                    else:
                        lat_str = "latency: n/a (msgs older than 60s)"
                    logger.info(f"Throughput checkpoint: {msg_count} msgs total | {lat_str}")
        except KeyboardInterrupt:
            logger.info("Stream processor stopped.")
        finally:
            self._running = False
            self._consumer.close()
            self._producer.close()

    def stop(self):
        self._running = False


if __name__ == "__main__":
    logger.add("logs/pipeline.log", rotation="50 MB")
    processor = WildfireStreamProcessor()
    processor.run()
