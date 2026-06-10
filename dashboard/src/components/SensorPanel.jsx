import React from "react";
import RiskGauge from "./RiskGauge";

export default function SensorPanel({ nodes = [], status = {} }) {
  const online = nodes.filter((n) => n.is_online !== false).length;
  const highRisk = nodes.filter((n) => n.fire_risk >= 0.65).length;

  return (
    <div style={{ background: "#161b2e", borderRadius: 8, padding: 16, height: "100%" }}>
      <h3 style={{ color: "#90caf9", marginBottom: 12 }}>Network Status</h3>

      <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
        {[
          { label: "Nodes Online", value: online, color: "#00e676" },
          { label: "High Risk", value: highRisk, color: "#ff6d00" },
          { label: "Active Alerts", value: status.active_alerts ?? "—", color: "#ff1744" },
        ].map(({ label, value, color }) => (
          <div
            key={label}
            style={{
              flex: "1 1 80px", background: "#1e2130", borderRadius: 6,
              padding: "10px 12px", textAlign: "center",
            }}
          >
            <div style={{ color, fontSize: 22, fontWeight: "bold" }}>{value}</div>
            <div style={{ color: "#aaa", fontSize: 11 }}>{label}</div>
          </div>
        ))}
      </div>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {[
          { label: "Kafka", key: "kafka_connected" },
          { label: "InfluxDB", key: "influxdb_connected" },
          { label: "AI Model", key: "model_loaded" },
        ].map(({ label, key }) => (
          <div
            key={key}
            style={{
              fontSize: 12, padding: "4px 10px", borderRadius: 12,
              background: status[key] ? "#1b3a2a" : "#2a1b1b",
              color: status[key] ? "#00e676" : "#f44336",
              border: `1px solid ${status[key] ? "#00e676" : "#f44336"}`,
            }}
          >
            {label}: {status[key] ? "OK" : "OFF"}
          </div>
        ))}
      </div>

      <h4 style={{ color: "#90caf9", marginTop: 16, marginBottom: 8 }}>Top Risk Nodes</h4>
      <div style={{ overflowY: "auto", maxHeight: 260 }}>
        {[...nodes]
          .sort((a, b) => b.fire_risk - a.fire_risk)
          .slice(0, 8)
          .map((n) => (
            <div
              key={n.node_id}
              style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
                padding: "6px 8px", marginBottom: 4,
                background: "#1e2130", borderRadius: 4,
              }}
            >
              <span style={{ fontSize: 12, color: "#e0e0e0" }}>{n.node_id}</span>
              <RiskGauge risk={n.fire_risk} />
            </div>
          ))}
      </div>
    </div>
  );
}
