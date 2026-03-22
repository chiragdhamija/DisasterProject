const API = {
  meta: "/api/meta",
  window: "/api/window",
  tractRisk: "/api/tract-risk",
};

const RISK_COLORS = ["#fff7bc", "#fee391", "#fec44f", "#fb6a4a", "#cb181d"];
const DEFAULT_BOUNDS = L.latLngBounds([32.35, -124.55], [42.1, -114.05]);

const state = {
  allDates: [],
  selectedBaseDate: "",
  windowDates: [],
  currentOffset: 0,
  mode: "spread",
  speedMs: 800,
  playing: false,
  timerId: null,
  sliderDebounceId: null,

  pointsByDate: new Map(), // date -> [ [lon,lat,hazard,risk,eal], ... ]
  dailyByDate: new Map(), // date -> summary object
  centroids: [], // [{sample_date, lon, lat}, ...]
  trajectories: [], // [{trajectory_id, points:[{sample_date, lon, lat, weight_sum, samples}]}]

  pointBreaks: [0.2, 0.4, 0.6, 0.8],
  tractBreaks: [0.2, 0.4, 0.6, 0.8],

  windowCache: new Map(), // base_date -> window payload
  tractCache: new Map(), // date -> tract payload
  riskFetchSeq: 0,

  map: null,
  canvasRenderer: null,
  spreadLayer: null,
  trajectoryLayer: null,
  tractLayer: null,
  activeTractDate: "",
  chart: null,
};

const ui = {
  baseDateSelect: document.getElementById("baseDateSelect"),
  viewMode: document.getElementById("viewMode"),
  playBtn: document.getElementById("playBtn"),
  speedSelect: document.getElementById("speedSelect"),
  dateSlider: document.getElementById("dateSlider"),
  dateLabel: document.getElementById("dateLabel"),
  statusText: document.getElementById("statusText"),
  mapLegend: document.getElementById("mapLegend"),
  metricHazard: document.getElementById("metricHazard"),
  metricTotalRisk: document.getElementById("metricTotalRisk"),
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
    initMap();
    bindEvents();
    setStatus("Loading API metadata...");

    const meta = await loadJson(API.meta);
    const dates = Array.isArray(meta.dates) ? meta.dates : [];
    if (!dates.length) {
      throw new Error("No dates available from API.");
    }

    state.allDates = dates;
    state.pointBreaks = normalizeBreaks(meta.point_breaks);
    const bounds = parseBounds(meta.california_bounds);
    if (bounds) {
      state.map.setMaxBounds(bounds);
      state.map.fitBounds(bounds, { padding: [10, 10] });
    } else {
      state.map.fitBounds(DEFAULT_BOUNDS, { padding: [10, 10] });
    }

    populateAvailableDateOptions(dates, meta.default_date || dates[0]);
    const defaultDate = ui.baseDateSelect.value || dates[0];

    await loadBaseDateWindow(defaultDate, true);
  } catch (error) {
    console.error(error);
    setStatus(`Failed to load API data: ${error.message}`, true);
  }
}

function initMap() {
  state.map = L.map("map", {
    preferCanvas: true,
    maxBounds: DEFAULT_BOUNDS,
    maxBoundsViscosity: 1.0,
    minZoom: 5.5,
    zoomSnap: 0.25,
  });
  state.map.fitBounds(DEFAULT_BOUNDS, { padding: [10, 10] });

  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; CARTO',
    subdomains: "abcd",
    maxZoom: 18,
    noWrap: true,
    bounds: DEFAULT_BOUNDS,
  }).addTo(state.map);

  state.canvasRenderer = L.canvas({ padding: 0.35 });
  state.spreadLayer = L.layerGroup().addTo(state.map);
  state.trajectoryLayer = L.layerGroup().addTo(state.map);
}

function bindEvents() {
  ui.baseDateSelect.addEventListener("change", async () => {
    const nextDate = ui.baseDateSelect.value;
    if (!nextDate) {
      return;
    }
    stopPlayback();
    await loadBaseDateWindow(nextDate, true);
  });

  ui.dateSlider.addEventListener("input", () => {
    const nextOffset = Number(ui.dateSlider.value);
    const previewDate =
      state.windowDates[Math.max(0, Math.min(nextOffset, state.windowDates.length - 1))] || "-";
    ui.dateLabel.textContent = `Date: ${previewDate}`;
    if (state.sliderDebounceId) {
      clearTimeout(state.sliderDebounceId);
    }
    const delayMs = state.mode === "risk" ? 220 : 70;
    state.sliderDebounceId = setTimeout(() => {
      void setDateOffset(nextOffset, true);
    }, delayMs);
  });

  ui.dateSlider.addEventListener("change", () => {
    if (state.sliderDebounceId) {
      clearTimeout(state.sliderDebounceId);
      state.sliderDebounceId = null;
    }
    void setDateOffset(Number(ui.dateSlider.value), true);
  });

  ui.viewMode.addEventListener("change", () => {
    state.mode = ui.viewMode.value;
    if (state.mode === "risk") {
      state.map.removeLayer(state.spreadLayer);
      state.map.removeLayer(state.trajectoryLayer);
      void renderRiskForDate(currentDate());
    } else {
      if (state.tractLayer) {
        state.map.removeLayer(state.tractLayer);
      }
      state.map.addLayer(state.spreadLayer);
      state.map.addLayer(state.trajectoryLayer);
      renderSpreadForDate(currentDate());
      renderTrajectoryForDate(currentDate(), false);
      renderSpreadLegend();
    }
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

async function loadBaseDateWindow(baseDate, panToDate) {
  if (!state.allDates.includes(baseDate)) {
    setStatus(`Base date ${baseDate} is not in available dataset dates.`, true);
    return;
  }
  setStatus(`Loading available-date window for ${baseDate} (t..t+2 available dates) from backend API...`);

  let payload = state.windowCache.get(baseDate);
  if (!payload) {
    const query = new URLSearchParams({ date: baseDate, horizon: "2" });
    payload = await loadJson(`${API.window}?${query.toString()}`);
    state.windowCache.set(baseDate, payload);
  }

  state.selectedBaseDate = payload.base_date || baseDate;
  state.windowDates = Array.isArray(payload.window_dates) ? payload.window_dates : [];
  state.currentOffset = 0;
  state.pointsByDate.clear();
  state.dailyByDate.clear();
  state.centroids = [];
  state.trajectories = [];
  state.activeTractDate = "";

  const pbd = payload.points_by_date || {};
  for (const d of state.windowDates) {
    state.pointsByDate.set(d, Array.isArray(pbd[d]) ? pbd[d] : []);
  }
  for (const row of payload.daily || []) {
    if (row?.sample_date) {
      state.dailyByDate.set(row.sample_date, row);
    }
  }
  state.centroids = (payload.centroids || [])
    .map((c) => ({
      sample_date: String(c.sample_date || ""),
      lon: Number(c.lon),
      lat: Number(c.lat),
    }))
    .filter((c) => c.sample_date && Number.isFinite(c.lon) && Number.isFinite(c.lat));
  state.trajectories = (payload.trajectories || [])
    .map((t, idx) => {
      const rawPoints = Array.isArray(t.points) ? t.points : [];
      const points = rawPoints
        .map((p) => ({
          sample_date: String(p.sample_date || ""),
          lon: Number(p.lon),
          lat: Number(p.lat),
          weight_sum: safeNum(p.weight_sum),
          samples: Math.max(0, Math.round(safeNum(p.samples))),
        }))
        .filter((p) => p.sample_date && Number.isFinite(p.lon) && Number.isFinite(p.lat))
        .sort((a, b) => a.sample_date.localeCompare(b.sample_date));
      const tid = Number.isFinite(Number(t.trajectory_id)) ? Number(t.trajectory_id) : idx + 1;
      return { trajectory_id: tid, points };
    })
    .filter((t) => t.points.length > 0);

  state.pointBreaks = normalizeBreaks(payload.point_breaks || state.pointBreaks);

  ui.baseDateSelect.value = state.selectedBaseDate;
  ui.dateSlider.min = "0";
  ui.dateSlider.max = String(Math.max(0, state.windowDates.length - 1));
  ui.dateSlider.value = "0";

  renderWindowChart();
  await setDateOffset(0, panToDate);

  const windowLabel = state.windowDates.join(", ");
  setStatus(`Loaded window: ${windowLabel}`);
}

function currentDate() {
  if (!state.windowDates.length) {
    return "";
  }
  const idx = Math.max(0, Math.min(state.currentOffset, state.windowDates.length - 1));
  return state.windowDates[idx];
}

async function setDateOffset(nextOffset, panToDate) {
  if (!state.windowDates.length) {
    return;
  }
  state.currentOffset = Math.max(0, Math.min(nextOffset, state.windowDates.length - 1));
  ui.dateSlider.value = String(state.currentOffset);

  const date = currentDate();
  ui.dateLabel.textContent = `Date: ${date}`;
  updateDailyMetrics(date);
  updateChartSelection();

  if (state.mode === "spread") {
    renderSpreadForDate(date);
    renderTrajectoryForDate(date, panToDate);
    renderSpreadLegend();
  } else {
    await renderRiskForDate(date);
  }
}

function renderSpreadForDate(date) {
  state.spreadLayer.clearLayers();
  const rows = state.pointsByDate.get(date) || [];
  const isBaseDay = date === state.selectedBaseDate;
  for (const row of rows) {
    const lon = Number(row[0]);
    const lat = Number(row[1]);
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
      continue;
    }
    const hazard = safeNum(row[2]);
    const risk = safeNum(row[3]);
    const eal = safeNum(row[4]);
    const predFireFrac = safeNum(row[5]);
    const gtFireFrac = safeNum(row[6]);

    if (isBaseDay) {
      if (gtFireFrac <= 0) {
        continue;
      }
      const marker = L.circleMarker([lat, lon], {
        renderer: state.canvasRenderer,
        radius: 3 + clamp(Math.sqrt(gtFireFrac) * 16, 0, 9),
        color: "#5c1306",
        weight: 0.7,
        fillColor: observedFireColor(gtFireFrac),
        fillOpacity: 0.9,
      });
      marker.bindPopup(
        `Date: ${date}<br/>Observed fire fraction (ground truth): ${gtFireFrac.toFixed(4)}`,
      );
      marker.addTo(state.spreadLayer);
      continue;
    }

    if (predFireFrac <= 0) {
      continue;
    }

    const marker = L.circleMarker([lat, lon], {
      renderer: state.canvasRenderer,
      radius: 3 + clamp(Math.sqrt(predFireFrac) * 16, 0, 8),
      color: "#522012",
      weight: 0.65,
      fillColor: colorFromBreaks(risk, state.pointBreaks),
      fillOpacity: 0.84,
    });
    marker.bindPopup(
      `Date: ${date}<br/>Hazard index: ${hazard.toFixed(4)}<br/>Risk (H×E×V): ${formatMoney.format(
        risk,
      )}<br/>Predicted fire fraction: ${predFireFrac.toFixed(4)}<br/>EAL: ${formatMoney.format(eal)}`,
    );
    marker.addTo(state.spreadLayer);
  }
}

function renderTrajectoryForDate(date, panToDate) {
  state.trajectoryLayer.clearLayers();
  if (!state.selectedBaseDate || date <= state.selectedBaseDate) {
    return;
  }
  if (state.trajectories.length) {
    const latestCandidates = [];
    for (const traj of state.trajectories) {
      const upto = traj.points.filter(
        (p) => p.sample_date > state.selectedBaseDate && p.sample_date <= date,
      );
      if (!upto.length) {
        continue;
      }
      const color = trajectoryColor(traj.trajectory_id);
      if (upto.length >= 2) {
        L.polyline(
          upto.map((p) => [p.lat, p.lon]),
          {
            renderer: state.canvasRenderer,
            color,
            weight: 2.3,
            opacity: 0.82,
          },
        ).addTo(state.trajectoryLayer);
      }

      const latest = upto[upto.length - 1];
      latestCandidates.push(latest);
      for (const p of upto) {
        const isLatest = p === latest;
        L.circleMarker([p.lat, p.lon], {
          renderer: state.canvasRenderer,
          radius: isLatest ? 4.8 : 2.2,
          color: "#0b2e3a",
          fillColor: color,
          fillOpacity: isLatest ? 0.95 : 0.75,
          weight: isLatest ? 1.2 : 0.7,
        })
          .bindTooltip(`T${traj.trajectory_id} | ${p.sample_date}`)
          .addTo(state.trajectoryLayer);
      }
    }

    if (panToDate && latestCandidates.length) {
      const latest = latestCandidates.reduce((best, c) => {
        if (!best) {
          return c;
        }
        if (c.sample_date > best.sample_date) {
          return c;
        }
        if (c.sample_date < best.sample_date) {
          return best;
        }
        return safeNum(c.weight_sum) > safeNum(best.weight_sum) ? c : best;
      }, null);
      if (latest) {
        state.map.panTo([latest.lat, latest.lon], { animate: true, duration: 0.5 });
      }
    }
    return;
  }

  // Legacy fallback: single weighted centroid trajectory.
  const upto = state.centroids.filter(
    (c) => c.sample_date > state.selectedBaseDate && c.sample_date <= date,
  );
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
  for (const c of upto) {
    L.circleMarker([c.lat, c.lon], {
      renderer: state.canvasRenderer,
      radius: c === latest ? 5.6 : 2.4,
      color: "#0b4a5f",
      fillColor: c === latest ? "#17a2b8" : "#0b4a5f",
      fillOpacity: 0.95,
      weight: 1,
    })
      .bindTooltip(c.sample_date)
      .addTo(state.trajectoryLayer);
  }

  if (panToDate && latest) {
    state.map.panTo([latest.lat, latest.lon], { animate: true, duration: 0.5 });
  }
}

async function renderRiskForDate(date) {
  if (!date) {
    return;
  }
  if (state.activeTractDate === date && state.tractLayer) {
    if (!state.map.hasLayer(state.tractLayer) && state.mode === "risk") {
      state.tractLayer.addTo(state.map);
    }
    renderRiskLegend();
    return;
  }
  state.riskFetchSeq += 1;
  const fetchId = state.riskFetchSeq;

  let payload = state.tractCache.get(date);
  if (!payload) {
    const query = new URLSearchParams({ date });
    setStatus(`Loading tract risk map for ${date}...`);
    payload = await loadJson(`${API.tractRisk}?${query.toString()}`);
    state.tractCache.set(date, payload);
  }
  if (fetchId !== state.riskFetchSeq) {
    return;
  }

  const fc = payload.feature_collection || { type: "FeatureCollection", features: [] };
  state.tractBreaks = normalizeBreaks(payload.risk_breaks || state.tractBreaks);

  if (state.tractLayer) {
    state.map.removeLayer(state.tractLayer);
  }
  state.tractLayer = L.geoJSON(fc, {
    renderer: state.canvasRenderer,
    style: (feature) => styleTract(feature),
  }).addTo(state.map);
  state.activeTractDate = date;

  renderRiskLegend();
  setStatus(
    `Loaded tract risk for ${date} (${formatInt.format(fc.features?.length || 0)} tracts).`,
  );
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

function renderWindowChart() {
  const labels = state.windowDates;
  const totalRiskM = labels.map((d) => safeNum(state.dailyByDate.get(d)?.risk_eal_usd_sum) / 1_000_000);

  if (state.chart) {
    state.chart.destroy();
    state.chart = null;
  }

  const ctx = document.getElementById("riskChart");
  state.chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Total Risk (USD millions/day)",
          data: totalRiskM,
          borderColor: "#c0392b",
          backgroundColor: "rgba(192,57,43,0.12)",
          borderWidth: 2,
          tension: 0.2,
          pointRadius: 2,
        },
        {
          label: "Selected Day",
          data: labels.map((_, idx) => (idx === state.currentOffset ? totalRiskM[idx] : null)),
          borderColor: "#111",
          backgroundColor: "#111",
          pointRadius: 5,
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
        y: {
          type: "linear",
          title: { display: true, text: "USD millions/day" },
        },
      },
    },
  });
}

function updateChartSelection() {
  if (!state.chart) {
    return;
  }
  const totalRisk = state.chart.data.datasets[0].data;
  state.chart.data.datasets[1].data = state.windowDates.map((_, idx) =>
    idx === state.currentOffset ? totalRisk[idx] : null,
  );
  state.chart.update("none");
}

function updateDailyMetrics(date) {
  const row = state.dailyByDate.get(date);
  if (!row) {
    ui.metricHazard.textContent = "-";
    ui.metricTotalRisk.textContent = "-";
    return;
  }
  ui.metricHazard.textContent = safeNum(row.hazard_index_mean).toFixed(4);
  ui.metricTotalRisk.textContent = formatMoney.format(safeNum(row.risk_eal_usd_sum));
}

function renderSpreadLegend() {
  if (currentDate() === state.selectedBaseDate) {
    ui.mapLegend.innerHTML = [
      legendChip("#fee5d9", "Observed fire frac ≤ 0.01"),
      legendChip("#fcae91", "0.01 - 0.05"),
      legendChip("#fb6a4a", "0.05 - 0.15"),
      legendChip("#cb181d", "> 0.15"),
    ].join("");
    return;
  }
  const [b1, b2, b3, b4] = state.pointBreaks;
  ui.mapLegend.innerHTML = [
    legendChip(RISK_COLORS[0], `Predicted risk ≤ ${formatRiskValue(b1)}`),
    legendChip(RISK_COLORS[1], `${formatRiskValue(b1)} - ${formatRiskValue(b2)}`),
    legendChip(RISK_COLORS[2], `${formatRiskValue(b2)} - ${formatRiskValue(b3)}`),
    legendChip(RISK_COLORS[3], `${formatRiskValue(b3)} - ${formatRiskValue(b4)}`),
    legendChip(RISK_COLORS[4], `> ${formatRiskValue(b4)}`),
  ].join("");
}

function renderRiskLegend() {
  const [b1, b2, b3, b4] = state.tractBreaks;
  ui.mapLegend.innerHTML = [
    legendChip(RISK_COLORS[0], `Tract risk ≤ ${formatRiskValue(b1)}`),
    legendChip(RISK_COLORS[1], `${formatRiskValue(b1)} - ${formatRiskValue(b2)}`),
    legendChip(RISK_COLORS[2], `${formatRiskValue(b2)} - ${formatRiskValue(b3)}`),
    legendChip(RISK_COLORS[3], `${formatRiskValue(b3)} - ${formatRiskValue(b4)}`),
    legendChip(RISK_COLORS[4], `> ${formatRiskValue(b4)}`),
  ].join("");
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

function trajectoryColor(id) {
  const palette = [
    "#205072",
    "#329d9c",
    "#56c596",
    "#9c6f44",
    "#8c4f91",
    "#d96c06",
    "#2f5d50",
    "#7f4f24",
    "#3d405b",
    "#4a7c59",
  ];
  const raw = Number(id);
  const idx = Number.isFinite(raw) ? Math.abs(Math.trunc(raw) - 1) % palette.length : 0;
  return palette[idx];
}

function observedFireColor(value) {
  if (value <= 0.01) return "#fee5d9";
  if (value <= 0.05) return "#fcae91";
  if (value <= 0.15) return "#fb6a4a";
  return "#cb181d";
}

function startPlayback() {
  if (state.windowDates.length <= 1) {
    return;
  }
  if (state.currentOffset >= state.windowDates.length - 1) {
    void setDateOffset(0, true);
  }
  state.playing = true;
  ui.playBtn.textContent = "Pause";
  state.timerId = window.setInterval(async () => {
    if (state.currentOffset >= state.windowDates.length - 1) {
      stopPlayback();
      return;
    }
    await setDateOffset(state.currentOffset + 1, true);
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

function normalizeBreaks(values) {
  if (!Array.isArray(values) || values.length < 4) {
    return [0.2, 0.4, 0.6, 0.8];
  }
  return values.slice(0, 4).map((v) => safeNum(v));
}

function parseBounds(raw) {
  if (!Array.isArray(raw) || raw.length !== 2) {
    return null;
  }
  const sw = raw[0];
  const ne = raw[1];
  if (!Array.isArray(sw) || !Array.isArray(ne) || sw.length !== 2 || ne.length !== 2) {
    return null;
  }
  const b = L.latLngBounds([Number(sw[0]), Number(sw[1])], [Number(ne[0]), Number(ne[1])]);
  if (!b.isValid()) {
    return null;
  }
  return b;
}

function populateAvailableDateOptions(dates, defaultDate) {
  ui.baseDateSelect.innerHTML = "";
  for (const d of dates) {
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = d;
    ui.baseDateSelect.appendChild(opt);
  }
  if (dates.includes(defaultDate)) {
    ui.baseDateSelect.value = defaultDate;
  } else if (dates.length) {
    ui.baseDateSelect.value = dates[0];
  }
}

function setStatus(message, isError = false) {
  ui.statusText.textContent = message;
  ui.statusText.style.color = isError ? "#8f2f1b" : "";
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} -> HTTP ${response.status}`);
  }
  return response.json();
}

function safeNum(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : 0;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function formatRiskValue(value) {
  const num = safeNum(value);
  if (num >= 1_000_000_000) return `$${(num / 1_000_000_000).toFixed(2)}B`;
  if (num >= 1_000_000) return `$${(num / 1_000_000).toFixed(2)}M`;
  if (num >= 1_000) return `$${(num / 1_000).toFixed(1)}K`;
  return `$${num.toFixed(0)}`;
}

function totalSpreadPoints() {
  let count = 0;
  for (const rows of state.pointsByDate.values()) {
    count += rows.length;
  }
  return count;
}
