from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from api.schemas import NodeStatus, SensorReading

router = APIRouter(prefix="/sensors", tags=["sensors"])


@router.get("/nodes", response_model=List[NodeStatus])
async def get_all_nodes():
    """Return status of every sensor node in the network."""
    from api.main import state
    return list(state["node_status"].values())


@router.get("/nodes/{node_id}", response_model=Optional[NodeStatus])
async def get_node(node_id: str):
    """Return status of a single sensor node."""
    from api.main import state
    return state["node_status"].get(node_id)


@router.get("/readings/recent", response_model=List[dict])
async def get_recent_readings(limit: int = Query(100, le=500)):
    """Return the most recent raw sensor readings from SQLite."""
    from api.main import state
    db = state.get("sqlite")
    if db:
        return db.get_recent_readings(limit=limit)
    return []


@router.get("/readings/node/{node_id}", response_model=List[dict])
async def get_node_readings(node_id: str, limit: int = Query(100, le=500)):
    """Return historical readings for a specific node from SQLite (newest first)."""
    from api.main import state
    db = state.get("sqlite")
    if db:
        return db.get_node_readings(node_id=node_id, limit=limit)
    return []


@router.get("/live-readings", response_model=List[dict])
async def get_live_readings():
    """Return the latest full sensor reading for every online node.
    Includes temperature, humidity, smoke index, CO ppm, VOC, wind, fire risk."""
    from api.main import state
    return list(state["live_readings"].values())


@router.get("/live-readings/{node_id}", response_model=dict)
async def get_live_reading_node(node_id: str):
    """Return the latest full sensor reading for a single node."""
    from api.main import state
    reading = state["live_readings"].get(node_id)
    if not reading:
        raise HTTPException(status_code=404, detail=f"No live reading for {node_id}")
    return reading


@router.get("/risk-map", response_model=List[dict])
async def get_risk_map():
    """Return all node positions with their current fire risk scores — used to colour the map."""
    from api.main import state
    return [
        {
            "node_id": nid,
            "lat": ns.latitude,
            "lon": ns.longitude,
            "fire_risk": ns.fire_risk,
            "is_online": ns.is_online,
        }
        for nid, ns in state["node_status"].items()
    ]
