const dashboardUrl = "/api/dashboard?alert_limit=8&timeline_points=24";
const streamUrl = "/api/stream/dashboard?alert_limit=8&timeline_points=24";

const elements = {
  livePill: document.getElementById("live-pill"),
  rowsScored: document.getElementById("rows-scored"),
  latestScoreFoot: document.getElementById("latest-score-foot"),
  avgAqi: document.getElementById("avg-aqi"),
  maxRisk: document.getElementById("max-risk"),
  activeAlerts: document.getElementById("active-alerts"),
  latestTimestamp: document.getElementById("latest-timestamp"),
  latestScoredAt: document.getElementById("latest-scored-at"),
  stationsSeen: document.getElementById("stations-seen"),
  avgAlerts: document.getElementById("avg-alerts"),
  timelineShell: document.getElementById("timeline-shell"),
  labelGrid: document.getElementById("label-grid"),
  stationGrid: document.getElementById("station-grid"),
  alertsRail: document.getElementById("alerts-rail"),
  diagnosticsGrid: document.getElementById("diagnostics-grid"),
  thresholdStrip: document.getElementById("threshold-strip"),
};

let pollHandle = null;
let eventSource = null;

function formatPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function formatCompact(value, digits = 0) {
  return Number(value || 0).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function formatDate(value) {
  if (!value) {
    return "-";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function setLiveState(mode, text) {
  elements.livePill.textContent = text;
  elements.livePill.classList.toggle("live", mode === "live");
}

function renderPlaceholder(container, message, className = "placeholder") {
  container.innerHTML = `<div class="${className}">${message}</div>`;
}

function renderOverview(snapshot) {
  const overview = snapshot.overview || {};
  elements.rowsScored.textContent = formatCompact(overview.readings_scored);
  elements.latestScoreFoot.textContent = overview.latest_scored_at
    ? `Updated ${formatDate(overview.latest_scored_at)}`
    : "Waiting for predictions";
  elements.avgAqi.textContent = formatCompact(overview.avg_aqi, 1);
  elements.maxRisk.textContent = formatPercent(overview.max_risk_prob);
  elements.activeAlerts.textContent = formatCompact(overview.active_alert_rows);
  elements.latestTimestamp.textContent = formatDate(overview.latest_timestamp);
  elements.latestScoredAt.textContent = formatDate(overview.latest_scored_at);
  elements.stationsSeen.textContent = formatCompact(overview.stations_seen);
  elements.avgAlerts.textContent = formatCompact(overview.avg_total_alerts, 2);
}

function createTimelinePath(series, width, height, accessor) {
  const values = series.map(accessor);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;

  return series.map((point, index) => {
    const x = (index / Math.max(series.length - 1, 1)) * width;
    const y = height - ((accessor(point) - min) / span) * height;
    return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(" ");
}

function renderTimeline(timeline) {
  if (!timeline?.length) {
    renderPlaceholder(elements.timelineShell, "No scoring history available yet.");
    return;
  }

  const width = 920;
  const height = 180;
  const aqiPath = createTimelinePath(timeline, width, height, (point) => point.avg_aqi);
  const riskPath = createTimelinePath(timeline, width, height, (point) => point.max_risk_prob * 100);
  const alertPath = createTimelinePath(timeline, width, height, (point) => point.avg_total_alerts * 25);
  const lastPoint = timeline[timeline.length - 1];

  elements.timelineShell.innerHTML = `
    <svg viewBox="0 0 ${width} ${height + 30}" preserveAspectRatio="none" aria-label="AQI and risk timeline">
      <line x1="0" y1="${height}" x2="${width}" y2="${height}" stroke="rgba(0,0,0,0.12)" />
      <path d="${aqiPath}" fill="none" stroke="#111111" stroke-width="3.4" />
      <path d="${riskPath}" fill="none" stroke="#d72616" stroke-width="2.5" stroke-dasharray="10 10" />
      <path d="${alertPath}" fill="none" stroke="#6d6d6d" stroke-width="2" stroke-dasharray="4 7" />
    </svg>
    <div class="timeline-meta">
      <div>
        <span class="timeline-label">Latest Avg AQI</span>
        <strong>${formatCompact(lastPoint.avg_aqi, 1)}</strong>
      </div>
      <div>
        <span class="timeline-label">Peak Probability</span>
        <strong>${formatPercent(lastPoint.max_risk_prob)}</strong>
      </div>
      <div>
        <span class="timeline-label">Active Rows</span>
        <strong>${formatCompact(lastPoint.active_rows)}</strong>
      </div>
    </div>
  `;
}

function renderLabels(labels) {
  if (!labels?.length) {
    renderPlaceholder(elements.labelGrid, "No label telemetry yet.");
    return;
  }

  elements.labelGrid.innerHTML = labels.map((label) => `
    <article class="label-card">
      <span class="label-name">${label.display_name}</span>
      <strong class="label-rate">${formatPercent(label.current_rate)}</strong>
      <div class="label-track">
        <div class="label-fill" style="width: ${Math.max(label.current_rate * 100, 4)}%"></div>
      </div>
      <div class="label-drift">
        Train ${formatPercent(label.training_rate)} / Gap ${label.rate_gap >= 0 ? "+" : "-"}${formatPercent(Math.abs(label.rate_gap))}
      </div>
    </article>
  `).join("");
}

function renderStations(stations) {
  if (!stations?.length) {
    renderPlaceholder(elements.stationGrid, "No station summary available.");
    return;
  }

  elements.stationGrid.innerHTML = stations.slice(0, 6).map((station) => `
    <article class="station-card ${station.max_risk_prob >= 0.8 ? "high" : ""}">
      <div class="station-top">
        <div>
          <div class="station-name">${station.station_id}</div>
          <span class="station-meta">${station.station_type} / ${station.dominant_risk}</span>
        </div>
        <strong class="station-score">${formatPercent(station.max_risk_prob)}</strong>
      </div>
      <div class="station-copy">
        AQI ${formatCompact(station.aqi, 1)} with ${formatCompact(station.total_alerts)} active flags.
        Quality markers observed: ${formatCompact(station.quality_flags)}.
      </div>
    </article>
  `).join("");
}

function renderAlerts(alerts) {
  if (!alerts?.length) {
    renderPlaceholder(elements.alertsRail, "No active escalations in the current file.");
    return;
  }

  elements.alertsRail.innerHTML = alerts.map((alert) => `
    <article class="alert-item">
      <div class="alert-top">
        <div>
          <div class="alert-name">${alert.station_id}</div>
          <span class="alert-meta">${alert.station_type} / ${formatDate(alert.timestamp)}</span>
        </div>
        <strong class="alert-score">${formatPercent(alert.max_risk_prob)}</strong>
      </div>
      <div class="alert-copy">
        ${alert.dominant_risk} dominates this row with AQI ${formatCompact(alert.aqi, 1)}
        and ${formatCompact(alert.total_alerts)} simultaneous alerts.
      </div>
    </article>
  `).join("");
}

function renderDiagnostics(snapshot) {
  const monitoring = snapshot.monitoring || {};
  const model = snapshot.model || {};
  const metrics = model.metrics || {};

  const cards = [
    {
      label: "Selected Model",
      value: model.selected_model || "Unknown",
      copy: `${formatCompact(model.feature_count || 0)} engineered inputs carried into production.`,
    },
    {
      label: "Macro F1",
      value: formatCompact(metrics.f1_macro, 4),
      copy: "Balanced label performance after threshold tuning.",
    },
    {
      label: "Hamming Loss",
      value: formatCompact(metrics.hamming_loss, 4),
      copy: "Lower is better across all label decisions.",
    },
    {
      label: "Co-occurrence Gap",
      value: formatCompact(monitoring.cooccurrence_gap, 4),
      copy: "Distance between live alert structure and training label structure.",
    },
    {
      label: "Quality Flag Rate",
      value: formatPercent(monitoring.quality_flag_rate),
      copy: "Rows carrying sensor repair or missingness markers.",
    },
    {
      label: "Exact Match",
      value: formatCompact(metrics.exact_match, 4),
      copy: "Share of readings with all five labels predicted correctly.",
    },
  ];

  elements.diagnosticsGrid.innerHTML = cards.map((card) => `
    <article class="diag-card">
      <div class="diag-top">
        <div>
          <span class="diag-label">${card.label}</span>
          <strong class="diag-value">${card.value}</strong>
        </div>
      </div>
      <div class="diag-copy">${card.copy}</div>
    </article>
  `).join("");

  const thresholds = model.thresholds || {};
  elements.thresholdStrip.innerHTML = Object.keys(thresholds).length
    ? Object.entries(thresholds).map(([label, value]) => `
        <span class="threshold-pill">${label.replace("label_", "").replaceAll("_", " ")} / ${value}</span>
      `).join("")
    : '<span class="threshold-pill">Thresholds unavailable</span>';
}

function renderSnapshot(snapshot) {
  renderOverview(snapshot);
  renderTimeline(snapshot.timeline);
  renderLabels(snapshot.labels);
  renderStations(snapshot.stations);
  renderAlerts(snapshot.alerts);
  renderDiagnostics(snapshot);
}

async function fetchDashboard() {
  const response = await fetch(dashboardUrl, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Dashboard request failed with status ${response.status}`);
  }

  return response.json();
}

function startPolling() {
  if (pollHandle) {
    return;
  }

  pollHandle = window.setInterval(async () => {
    try {
      const snapshot = await fetchDashboard();
      renderSnapshot(snapshot);
      setLiveState("live", "Polling live feed");
    } catch (_error) {
      setLiveState("idle", "Polling retrying");
    }
  }, 15000);
}

function connectStream() {
  if (!window.EventSource) {
    startPolling();
    return;
  }

  eventSource = new EventSource(streamUrl);
  setLiveState("idle", "Opening live stream");

  eventSource.addEventListener("dashboard", (event) => {
    const snapshot = JSON.parse(event.data);
    renderSnapshot(snapshot);
    setLiveState("live", "Live stream active");
  });

  eventSource.onerror = () => {
    setLiveState("idle", "Stream interrupted, falling back");
    eventSource?.close();
    startPolling();
  };
}

async function bootstrap() {
  try {
    const snapshot = await fetchDashboard();
    renderSnapshot(snapshot);
    setLiveState("live", "Initial snapshot loaded");
  } catch (_error) {
    setLiveState("idle", "Initial load failed");
    renderPlaceholder(elements.timelineShell, "Dashboard data is unavailable right now.");
  }

  connectStream();
}

bootstrap();
