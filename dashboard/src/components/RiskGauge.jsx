import React from "react";
import { Doughnut } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip } from "chart.js";
ChartJS.register(ArcElement, Tooltip);

const riskColor = (r) => {
  if (r >= 0.90) return "#ff1744";
  if (r >= 0.80) return "#ff6d00";
  if (r >= 0.65) return "#ffd600";
  return "#00e676";
};

const riskLabel = (r) => {
  if (r >= 0.90) return "CRITICAL";
  if (r >= 0.80) return "HIGH";
  if (r >= 0.65) return "MODERATE";
  return "LOW";
};

export default function RiskGauge({ risk = 0, nodeId }) {
  const pct = Math.round(risk * 100);
  const color = riskColor(risk);
  const data = {
    datasets: [
      {
        data: [pct, 100 - pct],
        backgroundColor: [color, "#1e2130"],
        borderWidth: 0,
        cutout: "75%",
      },
    ],
  };
  return (
    <div style={{ textAlign: "center", padding: "8px" }}>
      <div style={{ position: "relative", width: 120, margin: "0 auto" }}>
        <Doughnut data={data} options={{ plugins: { tooltip: { enabled: false } }, animation: false }} />
        <div
          style={{
            position: "absolute", top: "50%", left: "50%",
            transform: "translate(-50%, -50%)",
            color, fontWeight: "bold", fontSize: 18,
          }}
        >
          {pct}%
        </div>
      </div>
      <div style={{ color, fontWeight: "bold", fontSize: 12, marginTop: 4 }}>
        {riskLabel(risk)}
      </div>
      {nodeId && <div style={{ color: "#aaa", fontSize: 11 }}>{nodeId}</div>}
    </div>
  );
}
