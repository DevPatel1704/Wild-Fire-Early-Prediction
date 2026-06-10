from typing import List
from fastapi import APIRouter, HTTPException
from api.schemas import FireAlert, AlertAcknowledge

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("/active", response_model=List[FireAlert])
async def get_active_alerts():
    """Return all unacknowledged fire alerts."""
    from api.main import state
    db = state.get("sqlite")
    if db:
        rows = db.get_active_alerts()
        alerts = []
        for r in rows:
            ns = state["node_status"].get(r.get("node_id", ""))
            alerts.append(FireAlert(
                id=r.get("id"),
                node_id=r.get("node_id", ""),
                timestamp=r.get("timestamp", ""),
                fire_risk_score=r.get("fire_risk_score", 0.0),
                alert_level=r.get("alert_level", "HIGH"),
                acknowledged=bool(r.get("acknowledged", 0)),
                latitude=ns.latitude if ns else None,
                longitude=ns.longitude if ns else None,
            ))
        return alerts
    return list(state.get("active_alerts", {}).values())


@router.post("/acknowledge")
async def acknowledge_alert(body: AlertAcknowledge):
    """Mark an alert as acknowledged."""
    from api.main import state
    db = state.get("sqlite")
    if db:
        db.acknowledge_alert(body.alert_id)
        return {"status": "ok", "alert_id": body.alert_id}
    raise HTTPException(status_code=503, detail="Storage unavailable")


@router.get("/history", response_model=List[dict])
async def get_alert_history(limit: int = 50):
    from api.main import state
    db = state.get("sqlite")
    if db:
        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM fire_alerts ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    return []
