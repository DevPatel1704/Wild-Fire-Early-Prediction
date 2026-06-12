import React, { useEffect, useState } from "react";
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Title, Tooltip, Legend, Filler,
} from "chart.js";
import { Line, Scatter } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

function MetricBadge({ label, value, color = "#4fc3f7", sub }) {
  return (
    <div style={{
      background: "#1a2235", border: `1px solid ${color}44`,
      borderRadius: 10, padding: "16px 22px", textAlign: "center", minWidth: 140,
    }}>
      <div style={{ color: "#555", fontSize: 10, letterSpacing: 1, marginBottom: 8 }}>{label}</div>
      <div style={{ color, fontSize: 28, fontWeight: 700, fontFamily: "monospace" }}>{value}</div>
      {sub && <div style={{ color: "#444", fontSize: 10, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function ConfusionMatrix({ cm }) {
  if (!cm) return null;
  const [[tn, fp], [fn, tp]] = cm;
  const total = tn + fp + fn + tp;
  const cells = [
    { label: "True Negative", abbr: "TN", val: tn, color: "#00e676", bg: "#00e67618" },
    { label: "False Positive", abbr: "FP", val: fp, color: "#f44336", bg: "#f4433618" },
    { label: "False Negative", abbr: "FN", val: fn, color: "#ff9800", bg: "#ff980018" },
    { label: "True Positive",  abbr: "TP", val: tp, color: "#00e676", bg: "#00e67618" },
  ];
  return (
    <div>
      <div style={{ color: "#555", fontSize: 10, letterSpacing: 1, marginBottom: 12 }}>
        CONFUSION MATRIX <span style={{ color: "#444" }}>(threshold = 0.80)</span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2, maxWidth: 360 }}>
        <div style={{ display: "flex", justifyContent: "space-around", paddingBottom: 6 }}>
          <span style={{ color: "#555", fontSize: 11 }}></span>
          <span style={{ color: "#555", fontSize: 11 }}>Predicted: Safe</span>
          <span style={{ color: "#555", fontSize: 11 }}>Predicted: Fire</span>
        </div>
        {[["Actual: Safe", 0], ["Actual: Fire", 1]].map(([rowLabel, row]) => (
          <div key={row} style={{ display: "flex", alignItems: "center", gap: 2 }}>
            <div style={{ width: 110, color: "#555", fontSize: 11, textAlign: "right", paddingRight: 8 }}>{rowLabel}</div>
            {[0, 1].map((col) => {
              const cell = cells[row * 2 + col];
              return (
                <div key={col} style={{
                  flex: 1, background: cell.bg, border: `1px solid ${cell.color}44`,
                  borderRadius: 6, padding: "14px 8px", textAlign: "center",
                }}>
                  <div style={{ color: cell.color, fontWeight: 700, fontSize: 20, fontFamily: "monospace" }}>
                    {cell.val.toLocaleString()}
                  </div>
                  <div style={{ color: cell.color, fontSize: 11, marginTop: 2 }}>{cell.abbr}</div>
                  <div style={{ color: "#555", fontSize: 10 }}>
                    {((cell.val / total) * 100).toFixed(1)}%
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function EvaluationPage({ onBack }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("/predictions/evaluation")
      .then((r) => {
        if (!r.ok) throw new Error(`API error ${r.status}`);
        return r.json();
      })
      .then((d) => { setData(d); setLoading(false); })
      .catch((e) => { setError(e.message); setLoading(false); });
  }, []);

  const rocData = data ? {
    datasets: [
      {
        label: `ROC Curve  (AUC = ${data.auc_roc.toFixed(4)})`,
        data: data.roc_curve.fpr.map((x, i) => ({ x, y: data.roc_curve.tpr[i] })),
        borderColor: "#f44336",
        backgroundColor: "rgba(244,67,54,0.08)",
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        showLine: true,
        tension: 0.3,
      },
      {
        label: "Random Classifier",
        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        borderColor: "#444",
        borderDash: [6, 4],
        borderWidth: 1,
        pointRadius: 0,
        showLine: true,
      },
    ],
  } : null;

  const prData = data ? {
    datasets: [
      {
        label: `PR Curve  (Avg Precision = ${data.avg_precision.toFixed(4)})`,
        data: data.pr_curve.recall.map((x, i) => ({ x, y: data.pr_curve.precision[i] })),
        borderColor: "#4fc3f7",
        backgroundColor: "rgba(79,195,247,0.08)",
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        showLine: true,
        tension: 0.3,
      },
    ],
  } : null;

  const scatterOpts = (xlabel, ylabel, title) => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: { labels: { color: "#999", font: { size: 11 } } },
      title: {
        display: true, text: title,
        color: "#aaa", font: { size: 13, weight: "bold" },
      },
    },
    scales: {
      x: {
        type: "linear", min: 0, max: 1,
        title: { display: true, text: xlabel, color: "#666" },
        ticks: { color: "#555" }, grid: { color: "#1e2535" },
      },
      y: {
        type: "linear", min: 0, max: 1,
        title: { display: true, text: ylabel, color: "#666" },
        ticks: { color: "#555" }, grid: { color: "#1e2535" },
      },
    },
  });

  return (
    <div style={{
      display: "flex", flexDirection: "column", height: "100vh",
      background: "#0d1117", color: "#e8eaf6", fontFamily: "'Segoe UI', monospace",
    }}>
      {/* Header */}
      <header style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "10px 20px", background: "#161b2e",
        borderBottom: "1px solid #263040", flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <button onClick={onBack} style={{
            background: "#1e2d42", border: "1px solid #2a3550", color: "#aaa",
            padding: "6px 14px", borderRadius: 6, cursor: "pointer", fontSize: 12,
          }}>← Dashboard</button>
          <div>
            <div style={{ color: "#ff6d00", fontWeight: "bold", fontSize: 15 }}>
              Model Evaluation — GAT-LSTM
            </div>
            <div style={{ color: "#444", fontSize: 11 }}>
              Wildfire risk prediction · evaluated on {data ? `${data.n_samples.toLocaleString()} samples` : "…"}
            </div>
          </div>
        </div>
        {data && (
          <div style={{ color: "#00e676", fontSize: 12 }}>
            AUC = {data.auc_roc.toFixed(6)}
          </div>
        )}
      </header>

      {/* Body */}
      <div style={{ flex: 1, overflowY: "auto", padding: "22px 26px", display: "flex", flexDirection: "column", gap: 24 }}>

        {loading && (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 300, color: "#555" }}>
            Computing evaluation on {(100000).toLocaleString()} samples…
          </div>
        )}

        {error && (
          <div style={{ color: "#f44336", padding: 20 }}>Error: {error}</div>
        )}

        {data && (
          <>
            {/* Key metrics */}
            <section>
              <div style={{ color: "#555", fontSize: 10, letterSpacing: 1, marginBottom: 12 }}>
                KEY METRICS
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
                <MetricBadge label="AUC-ROC"    value={data.auc_roc.toFixed(4)}      color="#f44336" sub="Area under ROC curve" />
                <MetricBadge label="AVG PRECISION" value={data.avg_precision.toFixed(4)} color="#4fc3f7" sub="Area under PR curve" />
                <MetricBadge label="F1 SCORE"   value={data.f1.toFixed(4)}            color="#ff9800" sub={`@ threshold ${data.threshold}`} />
                <MetricBadge label="ACCURACY"   value={`${(data.accuracy*100).toFixed(2)}%`} color="#00e676" sub={`@ threshold ${data.threshold}`} />
                <MetricBadge label="PRECISION"  value={data.precision.toFixed(4)}     color="#ce93d8" sub="True fire / all predicted fire" />
                <MetricBadge label="RECALL"     value={data.recall.toFixed(4)}        color="#a5d6a7" sub="Fires caught / all actual fires" />
                <MetricBadge label="BEST THRESHOLD" value={data.best_threshold.toFixed(3)} color="#80cbc4" sub="Max Youden's J" />
                <MetricBadge label="SAMPLES" value={data.n_samples.toLocaleString()}  color="#b0bec5" sub={`${data.n_positive.toLocaleString()} fire / ${data.n_negative.toLocaleString()} safe`} />
              </div>
            </section>

            {/* ROC + PR curves side by side */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
              <div style={{ background: "#161b2e", border: "1px solid #263040", borderRadius: 10, padding: "18px 20px" }}>
                <div style={{ height: 300 }}>
                  <Scatter data={rocData} options={scatterOpts("False Positive Rate", "True Positive Rate", "ROC Curve")} />
                </div>
                <div style={{ marginTop: 10, color: "#555", fontSize: 11, textAlign: "center" }}>
                  AUC = <span style={{ color: "#f44336", fontWeight: 700 }}>{data.auc_roc.toFixed(6)}</span>
                  &nbsp;·&nbsp;Best threshold: <span style={{ color: "#f44336" }}>{data.best_threshold.toFixed(3)}</span>
                </div>
              </div>

              <div style={{ background: "#161b2e", border: "1px solid #263040", borderRadius: 10, padding: "18px 20px" }}>
                <div style={{ height: 300 }}>
                  <Scatter data={prData} options={scatterOpts("Recall", "Precision", "Precision-Recall Curve")} />
                </div>
                <div style={{ marginTop: 10, color: "#555", fontSize: 11, textAlign: "center" }}>
                  Avg Precision = <span style={{ color: "#4fc3f7", fontWeight: 700 }}>{data.avg_precision.toFixed(6)}</span>
                  &nbsp;·&nbsp;F1 @ 0.80: <span style={{ color: "#4fc3f7" }}>{data.f1.toFixed(4)}</span>
                </div>
              </div>
            </div>

            {/* Confusion matrix + interpretation */}
            <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: 24, alignItems: "start" }}>
              <div style={{ background: "#161b2e", border: "1px solid #263040", borderRadius: 10, padding: "20px 24px" }}>
                <ConfusionMatrix cm={data.confusion_matrix} />
              </div>

              <div style={{ background: "#161b2e", border: "1px solid #263040", borderRadius: 10, padding: "20px 24px" }}>
                <div style={{ color: "#555", fontSize: 10, letterSpacing: 1, marginBottom: 14 }}>
                  INTERPRETATION
                </div>
                {[
                  ["AUC-ROC", data.auc_roc.toFixed(4), "Probability that the model ranks a fire node higher than a safe node. 1.0 = perfect, 0.5 = random."],
                  ["Avg Precision", data.avg_precision.toFixed(4), "Area under the Precision-Recall curve. High value means few false alarms even at high recall."],
                  ["Precision", data.precision.toFixed(4), "Of all nodes the model flagged as fire, this fraction actually had fire. Higher = fewer false alarms."],
                  ["Recall", data.recall.toFixed(4), "Of all actual fire nodes, this fraction was caught. Higher = fewer missed fires (critical for safety)."],
                  ["F1 Score", data.f1.toFixed(4), "Harmonic mean of Precision and Recall. Balanced metric for the binary fire/safe decision."],
                  ["Best Threshold", data.best_threshold.toFixed(3), `Threshold that maximises Youden's J (TPR − FPR). Current operating threshold: ${data.threshold}.`],
                ].map(([name, val, desc]) => (
                  <div key={name} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: "1px solid #1e2535" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                      <span style={{ color: "#ccc", fontSize: 13, fontWeight: 600 }}>{name}</span>
                      <span style={{ color: "#ff6d00", fontFamily: "monospace", fontSize: 13 }}>{val}</span>
                    </div>
                    <div style={{ color: "#555", fontSize: 11, lineHeight: 1.5 }}>{desc}</div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
