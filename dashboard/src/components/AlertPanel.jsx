import React from "react";
import { acknowledgeAlert } from "../services/api";

const levelColor = { CRITICAL: "#ff1744", HIGH: "#ff6d00", MODERATE: "#ffd600" };

export default function AlertPanel({ alerts = [], onRefresh }) {
  const handle = async (id) => {
    try {
      await acknowledgeAlert(id);
      onRefresh();
    } catch {}
  };

  return (
    <div style={{ background: "#161b2e", borderRadius: 8, padding: 16, height: "100%", overflowY: "auto" }}>
      <h3 style={{ color: "#ff6d00", marginBottom: 12 }}>
        Active Alerts ({alerts.length})
      </h3>
      {alerts.length === 0 && (
        <p style={{ color: "#4caf50", fontSize: 13 }}>No active fire alerts.</p>
      )}
      {alerts.map((a, i) => (
        <div
          key={a.id || i}
          style={{
            background: "#1e2130",
            borderLeft: `4px solid ${levelColor[a.alert_level] || "#ff6d00"}`,
            borderRadius: 4,
            padding: "10px 12px",
            marginBottom: 10,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ color: levelColor[a.alert_level] || "#ff6d00", fontWeight: "bold", fontSize: 13 }}>
              {a.alert_level}
            </span>
            <span style={{ color: "#aaa", fontSize: 11 }}>
              {a.timestamp ? new Date(a.timestamp).toLocaleTimeString() : ""}
            </span>
          </div>
          <div style={{ color: "#e0e0e0", fontSize: 13, marginTop: 4 }}>
            Node: <strong>{a.node_id}</strong> &nbsp;|&nbsp; Risk: <strong>{(a.fire_risk_score * 100).toFixed(1)}%</strong>
          </div>
          {a.latitude && (
            <div style={{ color: "#aaa", fontSize: 11 }}>
              ({a.latitude.toFixed(4)}, {a.longitude.toFixed(4)})
            </div>
          )}
          {a.id && (
            <button
              onClick={() => handle(a.id)}
              style={{
                marginTop: 8, fontSize: 11, background: "#263040", color: "#90caf9",
                border: "1px solid #456", borderRadius: 4, padding: "3px 10px", cursor: "pointer",
              }}
            >
              Acknowledge
            </button>
          )}
        </div>
      ))}
    </div>
  );
}
