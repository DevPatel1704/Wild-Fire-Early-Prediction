import axios from "axios";

const BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";
const WS_URL = process.env.REACT_APP_WS_URL || "ws://localhost:8000/ws/live";

const api = axios.create({ baseURL: BASE, timeout: 10000 });

export const fetchNodes = () => api.get("/sensors/nodes").then((r) => r.data);
export const fetchRiskMap = () => api.get("/sensors/risk-map").then((r) => r.data);
export const fetchAlerts = () => api.get("/alerts/active").then((r) => r.data);
export const fetchSystemStatus = () => api.get("/predictions/system-status").then((r) => r.data);
export const acknowledgeAlert = (alertId) =>
  api.post("/alerts/acknowledge", { alert_id: alertId }).then((r) => r.data);

export function connectWebSocket(onMessage) {
  const ws = new WebSocket(WS_URL);
  ws.onmessage = (e) => {
    try {
      onMessage(JSON.parse(e.data));
    } catch {}
  };
  ws.onerror = () => console.warn("WebSocket error — falling back to polling.");
  return ws;
}
