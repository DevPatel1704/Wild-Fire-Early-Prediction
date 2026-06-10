"""Basic API endpoint tests using httpx + ASGI transport (no live server needed)."""

import pytest
from httpx import AsyncClient, ASGITransport


@pytest.fixture
async def client():
    from api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_root(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "nodes" in data


@pytest.mark.asyncio
async def test_risk_map_empty(client):
    resp = await client.get("/sensors/risk-map")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_active_alerts_empty(client):
    resp = await client.get("/alerts/active")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_system_status(client):
    resp = await client.get("/predictions/system-status")
    assert resp.status_code == 200
    data = resp.json()
    assert "kafka_connected" in data
    assert "model_loaded" in data
