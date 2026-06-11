import React, { useEffect, useRef, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import { Line } from "react-chartjs-2";
import { fetchLiveReading, fetchNodeHistory, fetchRiskMap } from "../services/api";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

const NODES = Array.from({ length: 100 }, (_, i) => `NODE_${String(i).padStart(3, "0")}`);

function riskColor(r) {
  if (r >= 0.9) return "#f44336";
  if (r >= 0.8) return "#ff6d00";
  if (r >= 0.65) return "#ffb300";
  if (r >= 0.4) return "#ffee58";
  return "#00e676";
}
function riskLabel(r) {
  if (r >= 0.9) return "CRITICAL";
  if (r >= 0.8) return "HIGH";
  if (r >= 0.65) return "MODERATE";
  return "LOW";
}
function fmt(v, d = 1) {
  return v !== undefined && v !== null ? Number(v).toFixed(d) : "—";
}

function MetricCard({ label, value, unit, icon, color = "#4fc3f7", alert = false }) {
  return (
    <div style={{
      background: alert ? color + "18" : "#1a2235",
      border: `1px solid ${alert ? color : "#2a3550"}`,
      borderRadius: 10,
      padding: "14px 18px",
      minWidth: 130,
      flex: "1 1 130px",
      transition: "border-color 0.3s",
    }}>
      <div style={{ color: "#666", fontSize: 11, marginBottom: 6, display: "flex", alignItems: "center", gap: 5 }}>
        <span>{icon}</span>
        <span style={{ letterSpacing: 0.5 }}>{label.toUpperCase()}</span>
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
        <span style={{ color: alert ? color : "#e8eaf6", fontSize: 24, fontWeight: 700, fontFamily: "monospace" }}>
          {value}
        </span>
        <span style={{ color: "#555", fontSize: 12 }}>{unit}</span>
      </div>
    </div>
  );
}

function WindCompass({ deg }) {
  const dirs = ["N","NE","E","SE","S","SW","W","NW"];
  const label = dirs[Math.round(deg / 45) % 8];
  return (
    <span style={{ color: "#80deea" }}>
      {fmt(deg, 0)}° <span style={{ color: "#555" }}>({label})</span>
    </span>
  );
}

export default function SensorReadingsPage({ onBack }) {
  const [selectedNode, setSelectedNode] = useState("NODE_000");
  const [live, setLive] = useState(null);
  const [history, setHistory] = useState([]);
  const [riskMap, setRiskMap] = useState({});
  const [search, setSearch] = useState("");
  const [lastTick, setLastTick] = useState(null);
  const timerRef = useRef(null);

  const loadNode = async (nid) => {
    try {
      const [reading, hist] = await Promise.all([
        fetchLiveReading(nid),
        fetchNodeHistory(nid, 100),
      ]);
      setLive(reading);
      setHistory([...hist].reverse());   // oldest→newest for chart
      setLastTick(new Date());
    } catch (e) {
      console.warn("loadNode failed", e);
    }
  };

  const loadRiskMap = async () => {
    try {
      const data = await fetchRiskMap();
      const m = {};
      data.forEach((n) => { m[n.node_id] = n.fire_risk; });
      setRiskMap(m);
    } catch {}
  };

  useEffect(() => {
    loadRiskMap();
    loadNode(selectedNode);
    timerRef.current = setInterval(() => {
      loadRiskMap();
      loadNode(selectedNode);
    }, 5000);
    return () => clearInterval(timerRef.current);
  }, [selectedNode]);

  const s = live?.sensors ?? {};
  const risk = live?.fire_risk ?? 0;
  const rc = riskColor(risk);

  // Build chart
  const chartLabels = history.map((r) => {
    const d = new Date(r.timestamp);
    return `${String(d.getHours()).padStart(2,"0")}:${String(d.getMinutes()).padStart(2,"0")}:${String(d.getSeconds()).padStart(2,"0")}`;
  });

  const chartData = {
    labels: chartLabels,
    datasets: [
      {
        label: "Fire Risk",
        data: history.map((r) => r.fire_risk),
        borderColor: "#f44336",
        backgroundColor: "rgba(244,67,54,0.08)",
        borderWidth: 2,
        tension: 0.4,
        fill: true,
        pointRadius: 0,
        yAxisID: "yRisk",
      },
      {
        label: "Temperature (°C)",
        data: history.map((r) => r.temperature_c),
        borderColor: "#ff9800",
        backgroundColor: "transparent",
        borderWidth: 1.5,
        tension: 0.4,
        pointRadius: 0,
        yAxisID: "yVal",
      },
      {
        label: "Humidity (%)",
        data: history.map((r) => r.humidity_pct),
        borderColor: "#4fc3f7",
        backgroundColor: "transparent",
        borderWidth: 1.5,
        tension: 0.4,
        pointRadius: 0,
        yAxisID: "yVal",
      },
      {
        label: "Smoke Index",
        data: history.map((r) => r.smoke_index),
        borderColor: "#ef9a9a",
        backgroundColor: "transparent",
        borderWidth: 1.5,
        tension: 0.4,
        pointRadius: 0,
        yAxisID: "yRisk",
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        labels: { color: "#888", boxWidth: 12, font: { size: 11 } },
      },
    },
    scales: {
      x: {
        ticks: { color: "#555", maxTicksLimit: 10, font: { size: 10 } },
        grid: { color: "#1e2535" },
      },
      yRisk: {
        type: "linear",
        position: "left",
        min: 0,
        max: 1,
        ticks: { color: "#888", font: { size: 10 } },
        grid: { color: "#1e2535" },
        title: { display: true, text: "Risk / Smoke", color: "#666", font: { size: 10 } },
      },
      yVal: {
        type: "linear",
        position: "right",
        ticks: { color: "#888", font: { size: 10 } },
        grid: { drawOnChartArea: false },
        title: { display: true, text: "Temp / Humidity", color: "#666", font: { size: 10 } },
      },
    },
  };

  const filtered = NODES.filter((n) => n.includes(search.toUpperCase().trim() || ""));

  return (
    <div style={{
      display: "flex", flexDirection: "column", height: "100vh",
      background: "#0d1117", color: "#e8eaf6", fontFamily: "'Segoe UI', monospace",
    }}>
      {/* ── Header ── */}
      <header style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "10px 20px", background: "#161b2e",
        borderBottom: "1px solid #263040", flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <button
            onClick={onBack}
            style={{
              background: "#1e2d42", border: "1px solid #2a3550", color: "#aaa",
              padding: "6px 14px", borderRadius: 6, cursor: "pointer", fontSize: 12,
              transition: "background 0.2s",
            }}
          >
            ← Dashboard
          </button>
          <div>
            <div style={{ color: "#ff6d00", fontWeight: "bold", fontSize: 15 }}>
              Live Sensor Readings
            </div>
            <div style={{ color: "#444", fontSize: 11 }}>
              All 100 nodes · real-time sensor data · updates every 5 s
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          {lastTick && (
            <span style={{ color: "#555", fontSize: 11 }}>
              Last refresh: {lastTick.toLocaleTimeString()}
            </span>
          )}
          <span style={{ color: "#00e676", fontSize: 12 }}>● LIVE</span>
        </div>
      </header>

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* ── Sidebar ── */}
        <aside style={{
          width: 210, minWidth: 160, background: "#161b2e",
          borderRight: "1px solid #263040",
          display: "flex", flexDirection: "column", flexShrink: 0,
        }}>
          <div style={{ padding: "10px 10px 8px", borderBottom: "1px solid #1e2535" }}>
            <input
              type="text"
              placeholder="Search node…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{
                width: "100%", boxSizing: "border-box",
                background: "#0d1117", border: "1px solid #2a3550",
                color: "#ccc", padding: "7px 10px", borderRadius: 6, fontSize: 12,
                outline: "none",
              }}
            />
            <div style={{ color: "#444", fontSize: 10, marginTop: 5, paddingLeft: 2 }}>
              {filtered.length} / {NODES.length} nodes
            </div>
          </div>
          <div style={{ overflowY: "auto", flex: 1 }}>
            {filtered.map((node) => {
              const r = riskMap[node] ?? 0;
              const isSelected = node === selectedNode;
              const color = riskColor(r);
              return (
                <div
                  key={node}
                  onClick={() => setSelectedNode(node)}
                  style={{
                    padding: "9px 12px",
                    cursor: "pointer",
                    background: isSelected ? "#1e2d42" : "transparent",
                    borderLeft: `3px solid ${color}`,
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    borderBottom: "1px solid #0f1520",
                  }}
                >
                  <span style={{ fontSize: 12, color: isSelected ? "#fff" : "#bbb", letterSpacing: 0.3 }}>
                    {node}
                  </span>
                  <span style={{
                    fontSize: 10, padding: "2px 6px", borderRadius: 3,
                    background: color + "22", color, fontWeight: 700,
                  }}>
                    {(r * 100).toFixed(0)}%
                  </span>
                </div>
              );
            })}
          </div>
        </aside>

        {/* ── Main ── */}
        <main style={{ flex: 1, overflowY: "auto", padding: "18px 22px", display: "flex", flexDirection: "column", gap: 18 }}>

          {/* Node title row */}
          <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
            <div style={{
              background: rc + "18", border: `2px solid ${rc}`,
              borderRadius: 10, padding: "10px 22px",
            }}>
              <span style={{ color: rc, fontWeight: 700, fontSize: 20, letterSpacing: 1 }}>{selectedNode}</span>
            </div>
            <div>
              <div style={{ color: "#aaa", fontSize: 12 }}>
                📍 {fmt(live?.latitude, 5)}, {fmt(live?.longitude, 5)}
              </div>
              <div style={{ color: "#555", fontSize: 11, marginTop: 2 }}>
                {live?.timestamp ? `Updated: ${new Date(live.timestamp).toLocaleTimeString()}` : "Waiting for data…"}
              </div>
            </div>
            <div style={{ marginLeft: "auto", textAlign: "right" }}>
              <div style={{ color: rc, fontWeight: 700, fontSize: 32, lineHeight: 1 }}>
                {(risk * 100).toFixed(1)}%
              </div>
              <div style={{
                fontSize: 11, color: rc, background: rc + "22",
                padding: "2px 8px", borderRadius: 4, marginTop: 4,
                display: "inline-block", fontWeight: 700,
              }}>
                {riskLabel(risk)}
              </div>
            </div>
          </div>

          {/* ── Current sensor cards ── */}
          <section>
            <div style={{ color: "#555", fontSize: 10, letterSpacing: 1, marginBottom: 10 }}>
              CURRENT SENSOR READINGS
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
              <MetricCard
                label="Air Temperature"  icon="🌡"  unit="°C"  color="#ff9800"
                value={fmt(s.temperature_c)}
                alert={s.temperature_c > 35}
              />
              <MetricCard
                label="Humidity"  icon="💧"  unit="%"  color="#4fc3f7"
                value={fmt(s.humidity_pct)}
                alert={s.humidity_pct < 20}
              />
              <MetricCard
                label="Surface Temp"  icon="🔆"  unit="°C"  color="#ff5722"
                value={fmt(s.surface_temp_c)}
                alert={s.surface_temp_c > 50}
              />
              <MetricCard
                label="Smoke Index"  icon="💨"  unit=""  color="#ef9a9a"
                value={fmt(s.smoke_index, 3)}
                alert={s.smoke_index > 0.5}
              />
              <MetricCard
                label="CO"  icon="☁"  unit="ppm"  color="#ce93d8"
                value={fmt(s.co_ppm, 2)}
                alert={s.co_ppm > 5}
              />
              <MetricCard
                label="VOC Index"  icon="🧪"  unit=""  color="#a5d6a7"
                value={fmt(s.voc_index)}
                alert={s.voc_index > 150}
              />
              <MetricCard
                label="Wind Speed"  icon="🌬"  unit="km/h"  color="#80cbc4"
                value={fmt(s.wind_speed_kmh)}
              />
              <MetricCard
                label="Wind Dir"  icon="🧭"  unit=""  color="#80deea"
                value={<WindCompass deg={s.wind_direction_deg} />}
              />
              <MetricCard
                label="Battery"  icon="🔋"  unit="%"  color="#c5e1a5"
                value={fmt(live?.battery_pct, 0)}
                alert={live?.battery_pct < 20}
              />
              <MetricCard
                label="Signal RSSI"  icon="📡"  unit="dBm"  color="#b0bec5"
                value={live?.signal_rssi ?? "—"}
              />
            </div>
          </section>

          {/* ── Trend chart ── */}
          <section style={{
            background: "#161b2e", border: "1px solid #263040",
            borderRadius: 10, padding: "16px 18px",
          }}>
            <div style={{ color: "#555", fontSize: 10, letterSpacing: 1, marginBottom: 12 }}>
              HISTORICAL TREND — FIRE RISK · TEMPERATURE · HUMIDITY · SMOKE &nbsp;
              <span style={{ color: "#444" }}>({history.length} readings)</span>
            </div>
            {history.length === 0 ? (
              <div style={{
                height: 200, display: "flex", alignItems: "center",
                justifyContent: "center", color: "#333", fontSize: 13,
              }}>
                Accumulating readings — check back in a few seconds…
              </div>
            ) : (
              <div style={{ height: 220 }}>
                <Line data={chartData} options={chartOptions} />
              </div>
            )}
          </section>

          {/* ── History table ── */}
          <section style={{
            background: "#161b2e", border: "1px solid #263040",
            borderRadius: 10, overflow: "hidden",
          }}>
            <div style={{
              padding: "10px 18px", borderBottom: "1px solid #1e2535",
              color: "#555", fontSize: 10, letterSpacing: 1,
              display: "flex", justifyContent: "space-between", alignItems: "center",
            }}>
              <span>READING HISTORY</span>
              <span style={{ color: "#444" }}>{history.length} records stored</span>
            </div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr style={{ background: "#111827" }}>
                    {[
                      ["Timestamp",       "left"],
                      ["Temp °C",         "right"],
                      ["Humidity %",      "right"],
                      ["Surface °C",      "right"],
                      ["Smoke",           "right"],
                      ["CO ppm",          "right"],
                      ["VOC",             "right"],
                      ["Wind km/h",       "right"],
                      ["Dir °",           "right"],
                      ["Fire Risk",       "right"],
                      ["Event",           "center"],
                    ].map(([h, a]) => (
                      <th key={h} style={{
                        padding: "9px 14px", color: "#555", textAlign: a,
                        borderBottom: "1px solid #1e2535", whiteSpace: "nowrap",
                        fontWeight: 600, letterSpacing: 0.3,
                      }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[...history].reverse().slice(0, 100).map((r, i) => {
                    const rc2 = riskColor(r.fire_risk);
                    return (
                      <tr key={i} style={{
                        background: i % 2 === 0 ? "#161b2e" : "#101520",
                        borderBottom: "1px solid #1a2030",
                      }}>
                        <td style={{ padding: "7px 14px", color: "#555", whiteSpace: "nowrap" }}>
                          {new Date(r.timestamp).toLocaleTimeString()}
                        </td>
                        <td style={{ padding: "7px 14px", color: r.temperature_c > 35 ? "#ff9800" : "#ccc", textAlign: "right" }}>
                          {fmt(r.temperature_c)}
                        </td>
                        <td style={{ padding: "7px 14px", color: r.humidity_pct < 20 ? "#4fc3f7" : "#ccc", textAlign: "right" }}>
                          {fmt(r.humidity_pct)}
                        </td>
                        <td style={{ padding: "7px 14px", color: "#ff7043", textAlign: "right" }}>
                          {fmt(r.surface_temp_c)}
                        </td>
                        <td style={{ padding: "7px 14px", color: r.smoke_index > 0.5 ? "#ef9a9a" : "#ccc", textAlign: "right" }}>
                          {fmt(r.smoke_index, 3)}
                        </td>
                        <td style={{ padding: "7px 14px", color: "#ce93d8", textAlign: "right" }}>
                          {fmt(r.co_ppm, 2)}
                        </td>
                        <td style={{ padding: "7px 14px", color: "#a5d6a7", textAlign: "right" }}>
                          {fmt(r.voc_index)}
                        </td>
                        <td style={{ padding: "7px 14px", color: "#80cbc4", textAlign: "right" }}>
                          {fmt(r.wind_speed_kmh)}
                        </td>
                        <td style={{ padding: "7px 14px", color: "#80deea", textAlign: "right" }}>
                          {fmt(r.wind_direction_deg, 0)}
                        </td>
                        <td style={{ padding: "7px 14px", textAlign: "right" }}>
                          <span style={{
                            color: rc2, fontWeight: 700,
                            background: rc2 + "18", padding: "2px 7px", borderRadius: 4,
                          }}>
                            {(r.fire_risk * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td style={{ padding: "7px 14px", textAlign: "center" }}>
                          {r.is_fire_event
                            ? <span style={{ color: "#f44336" }}>🔥</span>
                            : <span style={{ color: "#2a3550" }}>—</span>}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {history.length === 0 && (
                <div style={{ padding: 30, textAlign: "center", color: "#333" }}>
                  No history yet. Readings accumulate every 5 seconds.
                </div>
              )}
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
