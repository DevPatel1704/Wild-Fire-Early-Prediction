from typing import List, Optional
from fastapi import APIRouter, Query
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
