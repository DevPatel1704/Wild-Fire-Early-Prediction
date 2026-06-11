from fastapi import APIRouter, HTTPException
from api.schemas import PredictionResponse, SystemStatus
import asyncio
from functools import partial

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/system-status", response_model=SystemStatus)
async def get_system_status():
    from api.main import state
    import time
    return SystemStatus(
        kafka_connected=state.get("kafka_ok", False),
        influxdb_connected=state.get("influx_ok", False),
        model_loaded=state.get("model_ready", False),
        nodes_online=sum(1 for n in state["node_status"].values() if n.is_online),
        active_alerts=len([
            a for a in state.get("active_alerts", {}).values()
            if not a.acknowledged
        ]),
        uptime_seconds=round(time.time() - state.get("start_time", time.time()), 1),
    )


@router.get("/evaluation")
async def get_evaluation():
    """Run model evaluation on stored SQLite readings and return ROC/PR curves + metrics."""
    from model.evaluate import compute_all
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, partial(compute_all, 100_000))
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/risk/{node_id}", response_model=PredictionResponse)
async def get_node_prediction(node_id: str):
    from api.main import state
    ns = state["node_status"].get(node_id)
    if not ns:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    risk = ns.fire_risk
    alert_level = "CRITICAL" if risk >= 0.90 else "HIGH" if risk >= 0.80 else "MODERATE" if risk >= 0.65 else "LOW"
    return PredictionResponse(
        node_id=node_id,
        fire_risk_10min=round(risk, 4),
        fire_risk_30min=round(min(risk * 1.15, 1.0), 4),
        alert_level=alert_level,
    )
