const DATA_URLS = {
  spreadCompact: "./data/spread_daily_compact.json",
  trajectoryCompact: "./data/spread_trajectory_compact.json",
  dailySummary: "./data/daily_risk_summary.json",
  tractRisk: "./data/tract_risk.geojson",
};

const RISK_COLORS = ["#fff7bc", "#fee391", "#fec44f", "#fb6a4a", "#cb181d"];
const CALIFORNIA_BOUNDS = L.latLngBounds([32.35, -124.55], [42.1, -114.05]);

const state = {
  dates: [],
  currentIndex: 0,
  mode: "spread",
  speedMs: 800,
  playing: false,
  timerId: null,
  pointsByDate: new Map(),
  dailyByDate: new Map(),
  centroids: [],
  pointBreaks: [],
  tractBreaks: [],
  map: null,
  canvasRenderer: null,
  spreadLayer: null,
  trajectoryLayer: null,
  tractLayer: null,
  tractLoaded: false,
  tractFeatureCount: 0,
  chart: null,
};

const ui = {
  viewMode: document.getElementById("viewMode"),
  playBtn: document.getElementById("playBtn"),
  speedSelect: document.getElementById("speedSelect"),
  dateSlider: document.getElementById("dateSlider"),
  dateLabel: document.getElementById("dateLabel"),
  statusText: document.getElementById("statusText"),
  mapLegend: document.getElementById("mapLegend"),
  metricSamples: document.getElementById("metricSamples"),
  metricHazard: document.getElementById("metricHazard"),
  metricRisk: document.getElementById("metricRisk"),
  metricEal: document.getElementById("metricEal"),
};

const formatInt = new Intl.NumberFormat("en-US");
const formatMoney = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

boot();

async function boot() {
  try {
    setStatus("Loading compact spread and timeline assets...");
    const [spreadCompact, trajectoryCompact, dailySummary] = await Promise.all([
      loadJson(DATA_URLS.spreadCompact),
      loadJson(DATA_URLS.trajectoryCompact),
      loadJson(DATA_URLS.dailySummary),
    ]);

    prepareData(spreadCompact, trajectoryCompact, dailySummary);
    initMap();
    initChart();
    bindEvents();
    setDateIndex(0, true);
    setStatus(
      `Loaded ${formatInt.format(state.dates.length)} days and ${formatInt.format(
        totalSpreadPoints(),
      )} spread points. Tract risk map loads on demand.`,
    );
  } catch (error) {
    console.error(error);
    setStatus(
      `Failed to load map assets. Re-run build_frontend_assets.py. ${error.message}`,
      true,
    );
  }
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} -> HTTP ${response.status}`);
  }
  return response.json();
}

function prepareData(spreadCompact, trajectoryCompact, dailySummary) {
  const byDate = spreadCompact?.points_by_date || {};
  for (const [date, rows] of Object.entries(byDate)) {
    const cleanRows = [];
    for (const row of rows || []) {
      const lon = Number(row?.[0]);
      const lat = Number(row?.[1]);
      if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
        continue;
      }
      cleanRows.push([
        lon,
        lat,
        safeNum(row?.[2]),
        safeNum(row?.[3]),
        safeNum(row?.[4]),
      ]);
    }
    state.pointsByDate.set(date, cleanRows);
  }

  for (const row of dailySummary || []) {
    if (!row?.sample_date) {
      continue;
    }
    state.dailyByDate.set(row.sample_date, row);
  }

  const dateSet = new Set([
    ...(spreadCompact?.dates || []),
    ...Object.keys(byDate),
    ...state.dailyByDate.keys(),
  ]);
  state.dates = Array.from(dateSet).sort();

  const pointRiskValues = [];
  for (const rows of state.pointsByDate.values()) {
    for (const row of rows) {
      pointRiskValues.push(row[3]);
    }
  }
  state.pointBreaks = computeBreaks(pointRiskValues);

  const centroids = trajectoryCompact?.centroids || [];
  state.centroids = centroids
    .map((c) => ({
      date: String(c.sample_date || ""),
      lon: Number(c.lon),
      lat: Number(c.lat),
    }))
    .filter(
      (c) => c.date && Number.isFinite(c.lon) && Number.isFinite(c.lat),
    )
    .sort((a, b) => a.date.localeCompare(b.date));
}

function initMap() {
  state.map = L.map("map", {
    preferCanvas: true,
    maxBounds: CALIFORNIA_BOUNDS,
    maxBoundsViscosity: 1.0,
    minZoom: 5.5,
    zoomSnap: 0.25,
  });
  state.map.fitBounds(CALIFORNIA_BOUNDS, { padding: [8, 8] });

  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; CARTO',
    subdomains: "abcd",
    maxZoom: 18,
    noWrap: true,
    bounds: CALIFORNIA_BOUNDS,
  }).addTo(state.map);

  state.canvasRenderer = L.canvas({ padding: 0.35 });
  state.spreadLayer = L.layerGroup().addTo(state.map);
  state.trajectoryLayer = L.layerGroup().addTo(state.map);
}

function bindEvents() {
  ui.dateSlider.max = String(Math.max(state.dates.length - 1, 0));
  ui.dateSlider.value = "0";
  ui.dateSlider.addEventListener("input", () => {
    setDateIndex(Number(ui.dateSlider.value), true);
  });

  ui.viewMode.addEventListener("change", async () => {
    state.mode = ui.viewMode.value;
    if (state.mode === "risk") {
      await ensureTractLayer();
      state.map.removeLayer(state.spreadLayer);
      state.map.removeLayer(state.trajectoryLayer);
      if (state.tractLayer) {
        state.map.addLayer(state.tractLayer);
      }
    } else {
      if (state.tractLayer) {
        state.map.removeLayer(state.tractLayer);
      }
      state.map.addLayer(state.spreadLayer);
      state.map.addLayer(state.trajectoryLayer);
    }
    setDateIndex(state.currentIndex, false);
  });

  ui.playBtn.addEventListener("click", () => {
    if (state.playing) {
      stopPlayback();
    } else {
      startPlayback();
    }
  });

  ui.speedSelect.addEventListener("change", () => {
    state.speedMs = Number(ui.speedSelect.value);
    if (state.playing) {
      stopPlayback();
      startPlayback();
    }
  });
}

async function ensureTractLayer() {
  if (state.tractLoaded) {
    return;
  }
  setStatus("Loading California tract risk map...");
  const tractFc = await loadJson(DATA_URLS.tractRisk);

  const tractRiskValues = (tractFc.features || [])
    .map((f) => Number(f?.properties?.risk_score_mean))
    .filter((v) => Number.isFinite(v));
  state.tractBreaks = computeBreaks(tractRiskValues);

  state.tractLayer = L.geoJSON(tractFc, {
    renderer: state.canvasRenderer,
    style: (feature) => styleTract(feature),
    onEachFeature: (feature, layer) => {
      const p = feature.properties || {};
      layer.bindPopup(
        `<strong>Tract ${p.GEOID ?? "N/A"}</strong><br/>
        Mean risk: ${safeNum(p.risk_score_mean).toFixed(4)}<br/>
        Mean hazard: ${safeNum(p.hazard_index_mean).toFixed(4)}<br/>
        Mean exposure: ${safeNum(p.exposure_index_mean).toFixed(4)}<br/>
        Mean vulnerability: ${safeNum(p.vulnerability_index_mean).toFixed(4)}<br/>
        EAL sum: ${formatMoney.format(safeNum(p.risk_eal_usd_sum))}`,
      );
    },
  });
  state.tractLoaded = true;
  state.tractFeatureCount = tractFc.features?.length || 0;
  setStatus(`California tract risk layer loaded (${formatInt.format(state.tractFeatureCount)} tracts).`);
}

function startPlayback() {
  if (!state.dates.length) {
    return;
  }
  if (state.currentIndex >= state.dates.length - 1) {
    setDateIndex(0, true);
  }
  state.playing = true;
  ui.playBtn.textContent = "Pause";
  state.timerId = window.setInterval(() => {
    if (state.currentIndex >= state.dates.length - 1) {
      stopPlayback();
      return;
    }
    setDateIndex(state.currentIndex + 1, true);
  }, state.speedMs);
}

function stopPlayback() {
  state.playing = false;
  ui.playBtn.textContent = "Play";
  if (state.timerId) {
    window.clearInterval(state.timerId);
    state.timerId = null;
  }
}

function setDateIndex(nextIndex, panToDate) {
  if (!state.dates.length) {
    return;
  }
  state.currentIndex = Math.max(0, Math.min(nextIndex, state.dates.length - 1));
  const date = state.dates[state.currentIndex];
  ui.dateSlider.value = String(state.currentIndex);
  ui.dateLabel.textContent = `Date: ${date}`;

  if (state.mode === "spread") {
    renderSpreadForDate(date);
    renderTrajectory(date, panToDate);
    renderSpreadLegend();
  } else {
    renderRiskLegend();
  }

  updateDailyMetrics(date);
  updateChartSelection();
}

function renderSpreadForDate(date) {
  state.spreadLayer.clearLayers();
  const rows = state.pointsByDate.get(date) || [];
  for (const row of rows) {
    const [lon, lat, hazard, risk, eal] = row;
    const marker = L.circleMarker([lat, lon], {
      renderer: state.canvasRenderer,
      radius: 3 + clamp(hazard * 8, 0, 8),
      color: "#522012",
      weight: 0.65,
      fillColor: colorFromBreaks(risk, state.pointBreaks),
      fillOpacity: 0.84,
    });
    marker.bindPopup(
      `Date: ${date}<br/>
      Hazard index: ${hazard.toFixed(4)}<br/>
      Risk score: ${risk.toFixed(4)}<br/>
      EAL: ${formatMoney.format(eal)}`,
    );
    marker.addTo(state.spreadLayer);
  }
}

function renderTrajectory(date, panToDate) {
  state.trajectoryLayer.clearLayers();
  const upto = [];
  for (const point of state.centroids) {
    if (point.date <= date) {
      upto.push(point);
    } else {
      break;
    }
  }
  if (!upto.length) {
    return;
  }
  if (upto.length >= 2) {
    L.polyline(
      upto.map((c) => [c.lat, c.lon]),
      {
        renderer: state.canvasRenderer,
        color: "#1f6f8b",
        weight: 2.8,
        opacity: 0.9,
        dashArray: "8 6",
      },
    ).addTo(state.trajectoryLayer);
  }

  const latest = upto[upto.length - 1];
  for (const item of upto) {
    L.circleMarker([item.lat, item.lon], {
      renderer: state.canvasRenderer,
      radius: item === latest ? 5.6 : 2.4,
      color: "#0b4a5f",
      fillColor: item === latest ? "#17a2b8" : "#0b4a5f",
      fillOpacity: 0.95,
      weight: 1,
    })
      .bindTooltip(item.date)
      .addTo(state.trajectoryLayer);
  }

  if (panToDate && latest) {
    state.map.panTo([latest.lat, latest.lon], { animate: true, duration: 0.5 });
  }
}

function updateDailyMetrics(date) {
  const row = state.dailyByDate.get(date);
  if (!row) {
    ui.metricSamples.textContent = "-";
    ui.metricHazard.textContent = "-";
    ui.metricRisk.textContent = "-";
    ui.metricEal.textContent = "-";
    return;
  }
  ui.metricSamples.textContent = formatInt.format(Number(row.samples || 0));
  ui.metricHazard.textContent = safeNum(row.hazard_index_mean).toFixed(4);
  ui.metricRisk.textContent = safeNum(row.risk_score_mean).toFixed(4);
  ui.metricEal.textContent = formatMoney.format(safeNum(row.risk_eal_usd_sum));
}

function initChart() {
  const labels = state.dates;
  const risk = labels.map((d) => safeNum(state.dailyByDate.get(d)?.risk_score_mean));
  const hazard = labels.map((d) => safeNum(state.dailyByDate.get(d)?.hazard_index_mean));
  const ealM = labels.map((d) => safeNum(state.dailyByDate.get(d)?.risk_eal_usd_sum) / 1_000_000);

  const ctx = document.getElementById("riskChart");
  state.chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Risk (H×E×V)",
          data: risk,
          yAxisID: "yRisk",
          borderColor: "#c0392b",
          backgroundColor: "rgba(192,57,43,0.12)",
          borderWidth: 2,
          tension: 0.22,
          pointRadius: 0,
        },
        {
          label: "Hazard Index",
          data: hazard,
          yAxisID: "yRisk",
          borderColor: "#1f6f8b",
          backgroundColor: "rgba(31,111,139,0.12)",
          borderWidth: 1.8,
          tension: 0.22,
          pointRadius: 0,
        },
        {
          label: "EAL (USD millions)",
          data: ealM,
          yAxisID: "yEal",
          borderColor: "#a45a00",
          backgroundColor: "rgba(164,90,0,0.13)",
          borderWidth: 1.8,
          tension: 0.22,
          pointRadius: 0,
        },
        {
          label: "Selected Day",
          data: labels.map((_, idx) => (idx === state.currentIndex ? risk[idx] : null)),
          yAxisID: "yRisk",
          borderColor: "#111",
          backgroundColor: "#111",
          pointRadius: 4.2,
          pointHoverRadius: 4.2,
          showLine: false,
        },
      ],
    },
    options: {
      animation: false,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 14 } },
      },
      scales: {
        yRisk: {
          type: "linear",
          position: "left",
          title: { display: true, text: "Risk / Hazard Index" },
        },
        yEal: {
          type: "linear",
          position: "right",
          title: { display: true, text: "EAL (USD millions)" },
          grid: { drawOnChartArea: false },
        },
        x: {
          ticks: {
            maxRotation: 0,
            callback: (_, idx) => {
              if (idx % 20 === 0 || idx === state.currentIndex) {
                return labels[idx];
              }
              return "";
            },
          },
        },
      },
    },
  });
}

function updateChartSelection() {
  if (!state.chart) {
    return;
  }
  const riskData = state.chart.data.datasets[0].data;
  state.chart.data.datasets[3].data = state.dates.map((_, idx) =>
    idx === state.currentIndex ? riskData[idx] : null,
  );
  state.chart.update("none");
}

function renderSpreadLegend() {
  const [b1, b2, b3, b4] = state.pointBreaks;
  ui.mapLegend.innerHTML = [
    legendChip(RISK_COLORS[0], `≤ ${b1.toFixed(3)}`),
    legendChip(RISK_COLORS[1], `${b1.toFixed(3)} - ${b2.toFixed(3)}`),
    legendChip(RISK_COLORS[2], `${b2.toFixed(3)} - ${b3.toFixed(3)}`),
    legendChip(RISK_COLORS[3], `${b3.toFixed(3)} - ${b4.toFixed(3)}`),
    legendChip(RISK_COLORS[4], `> ${b4.toFixed(3)}`),
  ].join("");
}

function renderRiskLegend() {
  if (!state.tractLoaded || state.tractBreaks.length < 4) {
    ui.mapLegend.innerHTML = legendChip("#d4d8d7", "Loading tract risk layer...");
    return;
  }
  const [b1, b2, b3, b4] = state.tractBreaks;
  ui.mapLegend.innerHTML = [
    legendChip(RISK_COLORS[0], `Tract risk ≤ ${b1.toFixed(3)}`),
    legendChip(RISK_COLORS[1], `${b1.toFixed(3)} - ${b2.toFixed(3)}`),
    legendChip(RISK_COLORS[2], `${b2.toFixed(3)} - ${b3.toFixed(3)}`),
    legendChip(RISK_COLORS[3], `${b3.toFixed(3)} - ${b4.toFixed(3)}`),
    legendChip(RISK_COLORS[4], `> ${b4.toFixed(3)}`),
  ].join("");
}

function styleTract(feature) {
  const risk = Number(feature?.properties?.risk_score_mean);
  return {
    color: "#592f23",
    weight: 0.45,
    fillColor: colorFromBreaks(risk, state.tractBreaks),
    fillOpacity: 0.72,
  };
}

function legendChip(color, text) {
  return `<span class="legend-item"><i class="swatch" style="background:${color}"></i>${text}</span>`;
}

function colorFromBreaks(value, breaks) {
  if (!Number.isFinite(value) || breaks.length < 4) {
    return "#d4d8d7";
  }
  if (value <= breaks[0]) return RISK_COLORS[0];
  if (value <= breaks[1]) return RISK_COLORS[1];
  if (value <= breaks[2]) return RISK_COLORS[2];
  if (value <= breaks[3]) return RISK_COLORS[3];
  return RISK_COLORS[4];
}

function computeBreaks(values) {
  const clean = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (!clean.length) {
    return [0.2, 0.4, 0.6, 0.8];
  }
  return [0.2, 0.4, 0.6, 0.8].map((q) => quantile(clean, q));
}

function quantile(sorted, q) {
  if (!sorted.length) {
    return 0;
  }
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (base + 1 < sorted.length) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

function totalSpreadPoints() {
  let count = 0;
  for (const rows of state.pointsByDate.values()) {
    count += rows.length;
  }
  return count;
}

function setStatus(message, isError = false) {
  ui.statusText.textContent = message;
  ui.statusText.style.color = isError ? "#8f2f1b" : "";
}

function safeNum(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : 0;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
