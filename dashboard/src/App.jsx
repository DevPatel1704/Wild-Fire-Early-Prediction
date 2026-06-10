import React, { useEffect, useRef, useState } from "react";
import FireMap from "./components/Map";
import AlertPanel from "./components/AlertPanel";
import SensorPanel from "./components/SensorPanel";
import { fetchAlerts, fetchRiskMap, fetchSystemStatus, connectWebSocket } from "./services/api";

const POLL_MS = 10000;

export default function App() {
  const [nodes, setNodes] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [status, setStatus] = useState({});
  const [lastUpdate, setLastUpdate] = useState(null);
  const wsRef = useRef(null);

  const refresh = async () => {
    try {
      const [riskMap, alertData, sysStatus] = await Promise.all([
        fetchRiskMap(),
        fetchAlerts(),
        fetchSystemStatus(),
      ]);
      setNodes(riskMap);
      setAlerts(alertData);
      setStatus(sysStatus);
      setLastUpdate(new Date().toLocaleTimeString());
    } catch (err) {
      console.warn("API poll failed:", err.message);
    }
  };

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, POLL_MS);

    // Prefer WebSocket for live updates
    wsRef.current = connectWebSocket((msg) => {
      if (msg.type === "snapshot" && msg.nodes) {
        setNodes(msg.nodes);
      }
    });

    return () => {
      clearInterval(interval);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "#0f1117" }}>
      {/* Header */}
      <header style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "10px 20px", background: "#161b2e", borderBottom: "1px solid #263040",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 22 }}>🔥</span>
          <div>
            <div style={{ color: "#ff6d00", fontWeight: "bold", fontSize: 16 }}>
              Wildfire IoT Early Warning System
            </div>
            <div style={{ color: "#aaa", fontSize: 11 }}>
              Ontario Forest Sensor Network — Real-Time GAT-LSTM Risk Detection
            </div>
          </div>
        </div>
        <div style={{ color: "#aaa", fontSize: 12 }}>
          {lastUpdate ? `Updated: ${lastUpdate}` : "Loading..."}
          &nbsp;|&nbsp;
          <span style={{ color: status.kafka_connected ? "#00e676" : "#f44336" }}>
            {status.kafka_connected ? "● LIVE" : "○ OFFLINE"}
          </span>
        </div>
      </header>

      {/* Main layout */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden", gap: 8, padding: 8 }}>
        {/* Left: sensor panel */}
        <div style={{ width: 280, minWidth: 240, overflowY: "auto" }}>
          <SensorPanel nodes={nodes} status={status} />
        </div>

        {/* Centre: map */}
        <div style={{ flex: 1 }}>
          <FireMap nodes={nodes} />
        </div>

        {/* Right: alerts */}
        <div style={{ width: 300, minWidth: 240, overflowY: "auto" }}>
          <AlertPanel alerts={alerts} onRefresh={refresh} />
        </div>
      </div>

      {/* Footer */}
      <footer style={{
        padding: "6px 20px", background: "#161b2e",
        borderTop: "1px solid #263040", color: "#555", fontSize: 11,
        display: "flex", justifyContent: "space-between",
      }}>
        <span>Real-Time Data Analytics with IoT — Ontario Tech University</span>
        <span>Group: Dev · Dhruv · Priyanka · Slesha · Rashmi</span>
      </footer>
    </div>
  );
}
