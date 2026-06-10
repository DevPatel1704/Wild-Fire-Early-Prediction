"""
Kafka producer that serialises sensor readings to JSON and sends them
to the raw sensor topic with a keyed partition strategy (node_id → partition).
"""

import json
import os
from typing import Dict, Any

from kafka import KafkaProducer
from kafka.errors import KafkaError
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from .topics import TOPIC_SENSOR_RAW, ensure_topics


class SensorProducer:
    def __init__(self, bootstrap_servers: str = None):
        servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
        ensure_topics(servers)
        self._producer = KafkaProducer(
            bootstrap_servers=servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=5,
            linger_ms=10,          # small batch wait for throughput
            compression_type="gzip",
        )
        logger.info(f"SensorProducer connected to {servers}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def send(self, reading: Dict[str, Any], topic: str = None):
        topic = topic or TOPIC_SENSOR_RAW
        node_id = reading.get("node_id", "unknown")
        future = self._producer.send(topic, key=node_id, value=reading)
        future.add_errback(lambda exc: logger.error(f"Kafka send error for {node_id}: {exc}"))

    def flush(self):
        self._producer.flush()

    def close(self):
        self._producer.flush()
        self._producer.close()
        logger.info("SensorProducer closed.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
