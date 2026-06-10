"""
Kafka consumer that reads raw sensor readings and hands them to a callback.
Used by the Faust pipeline and the FastAPI WebSocket broadcaster.
"""

import json
import os
from typing import Callable, List, Optional

from kafka import KafkaConsumer
from loguru import logger

from .topics import TOPIC_SENSOR_RAW, TOPIC_FIRE_ALERTS


class SensorConsumer:
    def __init__(
        self,
        topics: List[str] = None,
        group_id: str = "wildfire-pipeline",
        bootstrap_servers: str = None,
        auto_offset_reset: str = "latest",
    ):
        servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
        topics = topics or [TOPIC_SENSOR_RAW]
        self._consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=servers,
            group_id=group_id,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,
            max_poll_records=500,
        )
        logger.info(f"SensorConsumer subscribed to {topics} on {servers}")

    def consume(self, callback: Callable[[dict], None], max_messages: Optional[int] = None):
        """Blocking consume loop. Calls callback for every message received."""
        count = 0
        try:
            for msg in self._consumer:
                callback(msg.value)
                count += 1
                if max_messages and count >= max_messages:
                    break
        except KeyboardInterrupt:
            logger.info("Consumer stopped.")
        finally:
            self.close()

    def close(self):
        self._consumer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
