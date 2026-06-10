import React from "react";
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";

const riskToColor = (r) => {
  if (r >= 0.90) return "#ff1744";
  if (r >= 0.80) return "#ff6d00";
  if (r >= 0.65) return "#ffd600";
  if (r >= 0.40) return "#ffab40";
  return "#00e676";
};

const riskLabel = (r) => {
  if (r >= 0.90) return "CRITICAL";
  if (r >= 0.80) return "HIGH";
  if (r >= 0.65) return "MODERATE";
  return "LOW";
};

export default function FireMap({ nodes = [] }) {
  const centre = nodes.length > 0
    ? [
        nodes.reduce((s, n) => s + n.lat, 0) / nodes.length,
        nodes.reduce((s, n) => s + n.lon, 0) / nodes.length,
      ]
    : [44.0, -78.95];

  return (
    <MapContainer
      center={centre}
      zoom={13}
      style={{ width: "100%", height: "100%", borderRadius: 8, background: "#161b2e" }}
    >
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        attribution='&copy; <a href="https://carto.com/">CARTO</a>'
      />
      {nodes.map((n) => (
        <CircleMarker
          key={n.node_id}
          center={[n.lat, n.lon]}
          radius={n.fire_risk >= 0.65 ? 10 : 6}
          pathOptions={{
            color: riskToColor(n.fire_risk),
            fillColor: riskToColor(n.fire_risk),
            fillOpacity: n.is_online === false ? 0.2 : 0.8,
            weight: n.fire_risk >= 0.80 ? 2 : 1,
          }}
        >
          <Popup>
            <div style={{ minWidth: 160 }}>
              <strong>{n.node_id}</strong><br />
              Risk: <strong style={{ color: riskToColor(n.fire_risk) }}>
                {(n.fire_risk * 100).toFixed(1)}% — {riskLabel(n.fire_risk)}
              </strong><br />
              Status: {n.is_online !== false ? "Online" : "Offline"}<br />
              {n.last_seen && <>Last seen: {new Date(n.last_seen).toLocaleTimeString()}<br /></>}
              ({n.lat?.toFixed(4)}, {n.lon?.toFixed(4)})
            </div>
          </Popup>
        </CircleMarker>
      ))}
    </MapContainer>
  );
}
