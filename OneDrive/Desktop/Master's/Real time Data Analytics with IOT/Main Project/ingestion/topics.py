"""Kafka topic definitions and admin utilities."""

import os
from kafka.admin import KafkaAdminClient, NewTopic
from loguru import logger

TOPIC_SENSOR_RAW = os.getenv("KAFKA_TOPIC_SENSOR_RAW", "sensor.raw")
TOPIC_SENSOR_AGG = os.getenv("KAFKA_TOPIC_SENSOR_AGG", "sensor.aggregated")
TOPIC_FIRE_ALERTS = os.getenv("KAFKA_TOPIC_FIRE_ALERTS", "fire.alerts")
TOPIC_DRONE_COMMANDS = "drone.commands"

ALL_TOPICS = [
    NewTopic(name=TOPIC_SENSOR_RAW, num_partitions=4, replication_factor=1),
    NewTopic(name=TOPIC_SENSOR_AGG, num_partitions=2, replication_factor=1),
    NewTopic(name=TOPIC_FIRE_ALERTS, num_partitions=1, replication_factor=1),
    NewTopic(name=TOPIC_DRONE_COMMANDS, num_partitions=1, replication_factor=1),
]


def ensure_topics(bootstrap_servers: str = None):
    servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
    try:
        admin = KafkaAdminClient(bootstrap_servers=servers)
        existing = set(admin.list_topics())
        new_topics = [t for t in ALL_TOPICS if t.name not in existing]
        if new_topics:
            admin.create_topics(new_topics=new_topics, validate_only=False)
            logger.info(f"Created Kafka topics: {[t.name for t in new_topics]}")
        else:
            logger.info("All Kafka topics already exist.")
        admin.close()
    except Exception as exc:
        logger.warning(f"Could not create topics: {exc}")
