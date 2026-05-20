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


class PredictionManager {
  constructor() {
    this.batchRows = [];
    this.initializeEventListeners();
    this.loadHistory();
  }

  initializeEventListeners() {
    // Manual form
    document.getElementById("manual-form").addEventListener("submit", (e) => this.handleManualSubmit(e));

    // Batch controls
    document.getElementById("batch-add-row").addEventListener("click", () => this.addBatchRow());
    document.getElementById("batch-generate-sample").addEventListener("click", () => this.generateSampleData());
    document.getElementById("batch-clear").addEventListener("click", () => this.clearBatch());
    document.getElementById("batch-submit").addEventListener("click", () => this.submitBatch());
  }

  async handleManualSubmit(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const payload = Object.fromEntries(formData);

    // Convert to numbers
    payload.pm25 = parseFloat(payload.pm25);
    payload.pm10 = parseFloat(payload.pm10);
    payload.aqi = parseInt(payload.aqi);
    payload.temp_c = parseFloat(payload.temp_c);
    payload.humidity_pct = parseFloat(payload.humidity_pct);
    payload.wind_speed_ms = parseFloat(payload.wind_speed_ms);
    payload.pressure_hpa = 1013;
    payload.precipitation_mm = 0;
    payload.visibility_km = 10;
    payload.temp_inversion = 0;
    payload.elevation_m = 100;
    payload.near_highway = 0;
    payload.near_industry = 0;
    payload.station_id = "USR_MANUAL";
    payload.co = 1;
    payload.no2 = 30;
    payload.o3 = 50;
    payload.so2 = 5;
    payload.benzene = 1;
    payload.wind_dir_deg = 180;

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const result = await response.json();

      if (response.ok && result.status === "success") {
        this.showResult("manual-result", true, result.prediction);
        this.loadHistory();
        form.reset();
      } else {
        this.showResult("manual-result", false, result.detail || "Prediction failed");
      }
    } catch (error) {
      this.showResult("manual-result", false, error.message);
    }
  }

  addBatchRow() {
    if (this.batchRows.length >= 100) {
      alert("Maximum 100 rows reached");
      return;
    }
    this.batchRows.push({
      pm25: "",
      pm10: "",
      aqi: "",
      temp_c: "",
      humidity_pct: "",
      wind_speed_ms: "",
      season: "",
      station_type: "",
    });
    this.renderBatchTable();
  }

  clearBatch() {
    this.batchRows = [];
    this.renderBatchTable();
  }

  removeBatchRow(index) {
    this.batchRows.splice(index, 1);
    this.renderBatchTable();
  }

  updateBatchRow(index, field, value) {
    this.batchRows[index][field] = value;
  }

  renderBatchTable() {
    const tbody = document.getElementById("batch-tbody");
    const count = document.getElementById("batch-count");

    tbody.innerHTML = this.batchRows
      .map(
        (row, index) => `
      <tr>
        <td>${index + 1}</td>
        <td><input type="number" step="0.1" value="${row.pm25}" data-field="pm25" data-index="${index}" class="batch-input"></td>
        <td><input type="number" step="0.1" value="${row.pm10}" data-field="pm10" data-index="${index}" class="batch-input"></td>
        <td><input type="number" step="1" value="${row.aqi}" data-field="aqi" data-index="${index}" class="batch-input"></td>
        <td><input type="number" step="0.1" value="${row.temp_c}" data-field="temp_c" data-index="${index}" class="batch-input"></td>
        <td><input type="number" step="0.1" value="${row.humidity_pct}" data-field="humidity_pct" data-index="${index}" class="batch-input"></td>
        <td><input type="number" step="0.1" value="${row.wind_speed_ms}" data-field="wind_speed_ms" data-index="${index}" class="batch-input"></td>
        <td>
          <select data-field="season" data-index="${index}" class="batch-select">
            <option value="${row.season}">${row.season || "--"}</option>
            <option value="Winter">Winter</option>
            <option value="Spring">Spring</option>
            <option value="Summer">Summer</option>
            <option value="Fall">Fall</option>
          </select>
        </td>
        <td>
          <select data-field="station_type" data-index="${index}" class="batch-select">
            <option value="${row.station_type}">${row.station_type || "--"}</option>
            <option value="Urban">Urban</option>
            <option value="Suburban">Suburban</option>
            <option value="Industrial">Industrial</option>
            <option value="Rural">Rural</option>
          </select>
        </td>
        <td>
          <div class="batch-row-actions">
            <button type="button" data-index="${index}" class="batch-delete">Delete</button>
          </div>
        </td>
      </tr>
    `
      )
      .join("");

    // Attach event listeners
    document.querySelectorAll(".batch-input, .batch-select").forEach((el) => {
      el.addEventListener("change", (e) => {
        const index = parseInt(e.target.dataset.index);
        const field = e.target.dataset.field;
        this.updateBatchRow(index, field, e.target.value);
      });
    });

    document.querySelectorAll(".batch-delete").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        this.removeBatchRow(parseInt(e.target.dataset.index));
      });
    });

    count.textContent = `${this.batchRows.length} / 100 rows`;
  }

  async generateSampleData() {
    const btn = document.getElementById("batch-generate-sample");
    btn.disabled = true;
    btn.textContent = "Generating...";

    try {
      const response = await fetch("/api/generate-sample-data?count=10");
      const data = await response.json();

      if (data.status === "success") {
        // Populate batch rows with generated data
        this.batchRows = data.rows;
        this.renderBatchTable();
        
        // Show success message
        this.showResult("batch-result", true, `✓ Generated ${data.count} sample rows. Ready to submit!`);
      } else {
        this.showResult("batch-result", false, data.detail || "Generation failed");
      }
    } catch (error) {
      this.showResult("batch-result", false, error.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Generate Sample (10)";
    }
  }

  async submitBatch() {
    if (this.batchRows.length === 0) {
      alert("Add at least one row");
      return;
    }

    const payload = {
      rows: this.batchRows.map((row) => ({
        pm25: parseFloat(row.pm25) || 0,
        pm10: parseFloat(row.pm10) || 0,
        aqi: parseInt(row.aqi) || 0,
        temp_c: parseFloat(row.temp_c) || 20,
        humidity_pct: parseFloat(row.humidity_pct) || 50,
        wind_speed_ms: parseFloat(row.wind_speed_ms) || 2,
        season: row.season || "Spring",
        station_type: row.station_type || "Urban",
        pressure_hpa: 1013,
        precipitation_mm: 0,
        visibility_km: 10,
        temp_inversion: 0,
        elevation_m: 100,
        near_highway: 0,
        near_industry: 0,
        station_id: "BATCH_AUTO",
        co: 1,
        no2: 30,
        o3: 50,
        so2: 5,
        benzene: 1,
        wind_dir_deg: 180,
      })),
    };

    try {
      const response = await fetch("/api/predict-batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const result = await response.json();

      if (response.ok && result.status === "success") {
        this.showResult("batch-result", true, `✓ Processed ${result.rows_processed} rows successfully`);
        this.clearBatch();
        this.loadHistory();
      } else {
        this.showResult("batch-result", false, result.detail || "Batch processing failed");
      }
    } catch (error) {
      this.showResult("batch-result", false, error.message);
    }
  }

  showResult(elementId, isSuccess, message) {
    const el = document.getElementById(elementId);
    el.classList.remove("hidden", "success", "error");
    el.classList.add(isSuccess ? "success" : "error");

    if (typeof message === "object") {
      el.innerHTML = `
        <div class="result-row">
          <span class="result-label">Total Alerts:</span>
          <span class="result-value">${message.total_alerts || 0}</span>
        </div>
        <div class="result-row">
          <span class="result-label">Max Risk:</span>
          <span class="result-value">${formatPercent(message.max_risk_prob)}</span>
        </div>
        <div class="result-row">
          <span class="result-label">Dominant Risk:</span>
          <span class="result-value">${message.dominant_risk || "—"}</span>
        </div>
        <div class="result-row">
          <span class="result-label">Scored At:</span>
          <span class="result-value">${formatDate(message.scored_at)}</span>
        </div>
      `;
    } else {
      el.textContent = message;
    }
  }

  async loadHistory() {
    try {
      const response = await fetch("/api/predictions-history?limit=10");
      const data = await response.json();

      if (data.status === "success" && data.records.length > 0) {
        this.renderHistory(data.records);
      } else {
        document.getElementById("history-tbody").innerHTML = "";
        document.getElementById("history-message").style.display = "block";
      }
    } catch (error) {
      console.error("Error loading history:", error);
    }
  }

  renderHistory(records) {
    const tbody = document.getElementById("history-tbody");
    const message = document.getElementById("history-message");

    message.style.display = "none";
    tbody.innerHTML = records
      .reverse()
      .map(
        (rec) => `
      <tr>
        <td>${rec.reading_id}</td>
        <td>${rec.station_id || "—"}</td>
        <td>${rec.total_alerts || 0}</td>
        <td class="${rec.max_risk_prob >= 0.7 ? "high-risk" : ""}">${formatPercent(rec.max_risk_prob)}</td>
        <td>${rec.dominant_risk || "—"}</td>
        <td>${formatDate(rec.scored_at)}</td>
      </tr>
    `
      )
      .join("");
  }
}

// Initialize prediction manager on page load
document.addEventListener("DOMContentLoaded", () => {
  new PredictionManager();
});
