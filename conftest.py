import pytest


@pytest.fixture(scope="session", autouse=True)
def set_test_env(tmp_path_factory):
    import os
    os.environ.setdefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
    os.environ.setdefault("INFLUXDB_URL", "http://localhost:8086")
    os.environ.setdefault("SQLITE_PATH", str(tmp_path_factory.mktemp("db") / "test.db"))
    os.environ.setdefault("MODEL_CHECKPOINT_PATH", "model/checkpoints/gat_lstm_best.pt")
