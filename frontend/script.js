/* ==========================================================
   DentCalib UI — Confidence Calibration Benchmark for
   Impacted Tooth Detection (YOLOv8 / RT-DETR / DentalCalib-Net).

   Data sources: precomputed calibration metrics, reliability
   bins, per-image predictions, and NLL values, exported from the
   thesis's Stage 5 pipeline (see /data and /images). Pages 1-3
   (Live OPG Auditor, Benchmark Explorer, Reliability Viewer) read
   real data via the loaders below (loadCalibrationData,
   loadReliabilityBins, loadPerImagePredictions, loadNllData,
   loadTestImageList).

   Page 4 (OOD Degradation Curves) also reads real data -- see
   renderDegradationChart() and renderDegTable() below. Its
   Spearman rho and Status columns come from a hardcoded
   DECOUPLING_LOOKUP table (Temperature-Scaling-only; see the note
   above that constant), while the chart and ECE@S5/mAP@S5 numbers
   are computed live and respond to the Pre/Post toggle.

   The Live OPG Auditor only accepts uploads matching filenames
   from the 201-image DENTEX test partition; it does not run live
   inference on arbitrary images (see handleUploadedFile).
   ========================================================== */

// ---------- Navigation ----------
const DATA_ROOT = new URL(
  document.currentScript?.dataset.dataRoot || "data/",
  document.currentScript?.src || window.location.href
);
const dataUrl = path => new URL(path, DATA_ROOT);

const navItems = document.querySelectorAll(".nav-item");
const pages = document.querySelectorAll(".page");

navItems.forEach(btn => {
  btn.addEventListener("click", () => {
    navItems.forEach(b => b.classList.remove("active"));
    pages.forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`page-${btn.dataset.page}`).classList.add("active");
  });
});

// ---------- Small helpers ----------
const fmt = n => n.toFixed(3);

// ---------- Chart.js shared config ----------
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.color = "#4C5A54";

function diagonalDataset() {
  return {
    label: "Perfect calibration",
    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    borderColor: "#C7C2B2",
    borderDash: [4, 4],
    borderWidth: 1,
    pointRadius: 0,
    fill: false,
  };
}

const chartRegistry = {};

function reliabilityChart(canvasId, bins, color) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  if (chartRegistry[canvasId]) {
    chartRegistry[canvasId].destroy();
  }
  const chart = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        diagonalDataset(),
        {
          label: "Observed",
          data: bins,
          borderColor: color,
          backgroundColor: color,
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.15,
        },
      ],
    },
  options: {
      responsive: true,
      scales: {
        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "Confidence" }, grid: { color: "#EEEBE1" } },
        y: { min: 0, max: 1, title: { display: true, text: "Accuracy" }, grid: { color: "#EEEBE1" } },
      },
      plugins: { legend: { display: false } },
    },
  });
  chartRegistry[canvasId] = chart;
  return chart;
}

let calibrationData = null;

async function loadCalibrationData() {
  if (calibrationData) return calibrationData;
  try {
    const res = await fetch(dataUrl("calibration_results.json"));
    if (!res.ok) throw new Error(`fetch failed: ${res.status}`);
    calibrationData = await res.json();
  } catch (err) {
    console.error("Failed to load calibration_results.json:", err);
    calibrationData = [];
  }
  return calibrationData;
}

reliabilityChart("chart-rel-yolo", [], "#C4501E");
reliabilityChart("chart-rel-rtdetr", [], "#2F7D4F");

// Page 3: reliability viewer chart objects (created once, updated by refreshReliabilityViewer)
let rvRawChart = reliabilityChart("chart-rv-raw", [], "#C4501E");
let rvRecalChart = reliabilityChart("chart-rv-recal", [], "#1E6E63");

// ---------- Reliability bin data (REAL DATA) ----------
let reliabilityBinData = null;

async function loadReliabilityBins() {
  if (reliabilityBinData) return reliabilityBinData;
  try {
    const res = await fetch(dataUrl("reliability_bins.json"));
    if (!res.ok) throw new Error(`fetch failed: ${res.status}`);
    reliabilityBinData = await res.json();
  } catch (err) {
    console.error("Failed to load reliability_bins.json:", err);
    reliabilityBinData = {};
  }
  return reliabilityBinData;
}

function conditionKeyFromLabel(label, severity) {
  const map = {
    "Gaussian noise": "gaussian_noise",
    "Motion blur": "motion_blur",
    "Brightness variation": "brightness_variation",
    "JPEG compression": "jpeg_compression",
  };
  const base = map[label];
  return base ? `${base}_S${severity}` : "clean";
}

function binsToChartPoints(binEntry) {
  if (!binEntry) return [];
  return binEntry.confidence.map((c, i) => ({ x: c, y: binEntry.accuracy[i] }));
}

function lookupBins(model, condition, method) {
  const key = `${model}__${condition}__${method}`;
  return reliabilityBinData[key] || null;
}

async function refreshReliabilityViewer() {
  await loadReliabilityBins();

  const model = document.getElementById("rv-model").value === "YOLOv8" ? "yolov8" : "rtdetr";
  const condLabel = document.getElementById("rv-condition").value;
  const severity = document.getElementById("rv-severity").value;
  const condition = conditionKeyFromLabel(condLabel, severity);
  const methodLabel = document.getElementById("rv-method").value;
  const recalMethod = methodLabel === "DentalCalib-Net" ? "dcn" : "ts";

  document.getElementById("rv-recal-title").textContent = `${methodLabel} — after recalibration`;

  const rawBins = lookupBins(model, condition, "raw");
  const recalBins = lookupBins(model, condition, recalMethod);

  if (!rawBins || !recalBins) {
    console.warn(`No bin data for ${model} / ${condition} -- check the export covers this combination.`);
    return;
  }

  rvRawChart.data.datasets[1].data = binsToChartPoints(rawBins);
  rvRawChart.update();
  rvRecalChart.data.datasets[1].data = binsToChartPoints(recalBins);
  rvRecalChart.update();

  const calRow = calibrationData
    ? calibrationData.find(r => r.model === model && r.condition === condition && r.confidence_type === "raw")
    : null;
  const calRowRecal = calibrationData
    ? calibrationData.find(r => r.model === model && r.condition === condition && r.confidence_type === recalMethod)
    : null;

  await loadNllData();
  const nllKey = `${model}__${condition}`;
  const nllEntry = nllData[nllKey];

  document.getElementById("rv-raw-ece").textContent = calRow ? fmt(calRow.ece) : "—";
  document.getElementById("rv-raw-mce").textContent = calRow ? fmt(calRow.mce) : "—";
  document.getElementById("rv-raw-nll").textContent = nllEntry ? fmt(nllEntry.nll_raw) : "—";

  document.getElementById("rv-recal-ece").textContent = calRowRecal ? fmt(calRowRecal.ece) : "—";
  document.getElementById("rv-recal-mce").textContent = calRowRecal ? fmt(calRowRecal.mce) : "—";
  document.getElementById("rv-recal-nll").textContent =
    recalMethod === "ts" && nllEntry ? fmt(nllEntry.nll_ts) : "N/A";

  const rmsdRow = calRow ? fmt(calRow.rmsd ?? 0) : "—";
  const rmsdRecal = calRowRecal ? fmt(calRowRecal.rmsd ?? 0) : "—";
  const mapNote = calRow && calRowRecal
    ? (calRow.map50 === calRowRecal.map50 ? "mAP@0.50 unchanged" : `mAP@0.50 ${calRow.map50.toFixed(3)} → ${calRowRecal.map50.toFixed(3)}`)
    : "";
  document.getElementById("rv-summary").textContent =
    `RD deviation RMSD: raw ${rmsdRow} → post-${methodLabel} ${rmsdRecal} · ${mapNote}`;
}

["rv-model", "rv-condition", "rv-severity", "rv-method"].forEach(id => {
  document.getElementById(id)?.addEventListener("change", refreshReliabilityViewer);
});

// initial render once both data sources are ready
loadCalibrationData().then(() => refreshReliabilityViewer());

// ---------- Page 4: OOD Degradation Curves (REAL DATA) ----------

// Ground truth from Tables 11, 12, 18 -- computed under Temperature Scaling
// only, since that's the only per-corruption-type Spearman correlation your
// thesis actually computed. No equivalent DCN correlation exists in your
// documented results, so the rho/status columns always reflect TS, even
// when the chart and ECE@S5/mAP@S5 numbers are toggled to show DCN.
const DECOUPLING_LOOKUP = {
  yolov8: {
    gaussian_noise: { rho: 0.7071, status: "Stable" },
    motion_blur: { rho: 0.9000, status: "Decoupled" },
    brightness_variation: { rho: 0.7000, status: "Stable" },
    jpeg_compression: { rho: 1.0000, status: "Decoupled" },
  },
  rtdetr: {
    gaussian_noise: { rho: 0.0000, status: "Stable" },
    motion_blur: { rho: 0.7000, status: "Stable" },
    brightness_variation: { rho: -0.6000, status: "Stable" },
    jpeg_compression: { rho: 0.3000, status: "Stable" },
  },
};

function degCorruptionKey(label) {
  const map = {
    "Gaussian noise": "gaussian_noise",
    "Motion blur": "motion_blur",
    "Brightness variation": "brightness_variation",
    "JPEG compression": "jpeg_compression",
  };
  return map[label];
}

function degModelsSelected(label) {
  if (label === "Both") return ["yolov8", "rtdetr"];
  return [label === "YOLOv8" ? "yolov8" : "rtdetr"];
}

function degRecalMethod() {
  const active = document.querySelector("#deg-recalib-toggle .toggle-opt.active");
  return active?.dataset.mode === "post" ? "dcn" : "raw";
}

let degChart;

// Finds the severity step where a model's ECE and mAP stop moving in
// opposite-and-proportional directions (the "decoupling point"). Not a
// formula specified in the proposal -- our own construction, flagged for
// documentation. Normalizes both series to 0-1, then finds the largest
// deviation from the expected delta_ece = -delta_map relationship between
// consecutive severities. Returns the LATER severity of that pair (where
// the divergence becomes visible), or null if there isn't enough valid data.
function computeDecouplingSeverity(eceSeries, mapSeries, severities) {
  const valid = severities.map((s, i) => ({ s, ece: eceSeries[i], map: mapSeries[i] }))
    .filter(p => p.ece != null && p.map != null);
  if (valid.length < 2) return null;

  const eceVals = valid.map(p => p.ece);
  const mapVals = valid.map(p => p.map);
  const eceMin = Math.min(...eceVals), eceMax = Math.max(...eceVals);
  const mapMin = Math.min(...mapVals), mapMax = Math.max(...mapVals);
  if (eceMax === eceMin || mapMax === mapMin) return null;

  const normEce = eceVals.map(v => (v - eceMin) / (eceMax - eceMin));
  const normMap = mapVals.map(v => (v - mapMin) / (mapMax - mapMin));

  let worstIdx = -1;
  let worstVal = -Infinity;
  for (let i = 0; i < valid.length - 1; i++) {
    const dEce = normEce[i + 1] - normEce[i];
    const dMap = normMap[i + 1] - normMap[i];
    const divergence = Math.abs(dEce + dMap); // expected: dEce ≈ -dMap, so sum ≈ 0
    if (divergence > worstVal) {
      worstVal = divergence;
      worstIdx = i;
    }
  }
  return worstIdx >= 0 ? valid[worstIdx + 1].s : null;
}

const decouplingLinePlugin = {
  id: "decouplingLine",
  afterDraw(chart) {
    const severity = chart.$decouplingSeverity;
    const color = chart.$decouplingColor;
    if (severity == null) return;
    const { ctx, chartArea, scales } = chart;
    const x = scales.x.getPixelForValue(chart.data.labels.indexOf(severity));

    ctx.save();
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = color || "#C4501E";
    ctx.lineWidth = 2;
    ctx.moveTo(x, chartArea.top);
    ctx.lineTo(x, chartArea.bottom);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = color || "#C4501E";
    ctx.font = "11px sans-serif";
    const label = `Decoupling point (S${severity})`;
    const labelWidth = ctx.measureText(label).width;
    const padding = 6;
    const fitsOnRight = x + padding + labelWidth <= chartArea.right;

    if (fitsOnRight) {
      ctx.textAlign = "left";
      ctx.fillText(label, x + padding, chartArea.top + 12);
    } else {
      ctx.textAlign = "right";
      ctx.fillText(label, x - padding, chartArea.top + 12);
    }
    ctx.restore();
  },
};

async function renderDegradationChart() {
  await loadCalibrationData();

  const models = degModelsSelected(document.getElementById("deg-model").value);
  const corruption = degCorruptionKey(document.getElementById("deg-corruption").value);
  const method = degRecalMethod();
  const severities = [1, 2, 3, 4, 5];
  const colors = { yolov8: "#C4501E", rtdetr: "#1E6E63" };
  const datasets = [];
  let decouplingSeverity = null;
  let decouplingColor = null;

  models.forEach(model => {
    const eceSeries = [];
    const mapSeries = [];
    severities.forEach(s => {
      const row = calibrationData.find(
        r => r.model === model && r.condition === `${corruption}_S${s}` && r.confidence_type === method
      );
      eceSeries.push(row ? row.ece : null);
      mapSeries.push(row ? row.map50 : null);
    });
    const label = model === "yolov8" ? "YOLOv8" : "RT-DETR";
    datasets.push({
      label: `${label} ECE`,
      data: eceSeries,
      borderColor: colors[model],
      backgroundColor: colors[model],
      yAxisID: "y",
      tension: 0.2,
    });
    datasets.push({
      label: `${label} mAP@0.50`,
      data: mapSeries,
      borderColor: colors[model],
      backgroundColor: colors[model],
      borderDash: [5, 3],
      yAxisID: "y1",
      tension: 0.2,
    });

    // Decoupling marker only computed/shown in single-model view -- see
    // note in documentation on why "Both" mode has no combined marker.
    if (models.length === 1) {
      decouplingSeverity = computeDecouplingSeverity(eceSeries, mapSeries, severities);
      decouplingColor = colors[model];
    }
  });

  if (decouplingSeverity != null) {
    datasets.push({
      label: `Decoupling point (S${decouplingSeverity})`,
      data: severities.map(() => null),
      borderColor: decouplingColor,
      borderDash: [4, 4],
      borderWidth: 2,
      pointRadius: 0,
      yAxisID: "y",
    });
  }

  const ctx = document.getElementById("chart-degradation");
  if (degChart) degChart.destroy();
  degChart = new Chart(ctx, {
    type: "line",
    data: { labels: severities, datasets },
    options: {
      responsive: true,
      scales: {
        x: { title: { display: true, text: "OOD severity" }, grid: { display: false } },
        y: { position: "left", title: { display: true, text: "ECE" }, grid: { color: "#EEEBE1" } },
        y1: { position: "right", title: { display: true, text: "mAP@0.50" }, grid: { display: false } },
      },
      plugins: { legend: { position: "top", align: "end" } },
    },
    plugins: [decouplingLinePlugin],
  });
  degChart.$decouplingSeverity = decouplingSeverity;
  degChart.$decouplingColor = decouplingColor;
  degChart.update();
}

async function renderDegTable() {
  await loadCalibrationData();

  const models = degModelsSelected(document.getElementById("deg-model").value);
  const corruption = degCorruptionKey(document.getElementById("deg-corruption").value);
  const method = degRecalMethod();
  const severities = [1, 2, 3, 4, 5];

  const rows = models.map(model => {
    const eceSeries = [];
    const mapSeries = [];
    severities.forEach(s => {
      const row = calibrationData.find(
        r => r.model === model && r.condition === `${corruption}_S${s}` && r.confidence_type === method
      );
      eceSeries.push(row ? row.ece : null);
      mapSeries.push(row ? row.map50 : null);
    });
    const s5Row = calibrationData.find(
      r => r.model === model && r.condition === `${corruption}_S5` && r.confidence_type === method
    );
    const lookup = DECOUPLING_LOOKUP[model][corruption];
    const decSeverity = computeDecouplingSeverity(eceSeries, mapSeries, severities);
    return {
      model: model === "yolov8" ? "YOLOv8" : "RT-DETR",
      rho: lookup.rho,
      status: lookup.status,
      decSeverity,
      ece5: s5Row ? s5Row.ece : null,
      map5: s5Row ? s5Row.map50 : null,
      bad: lookup.status === "Decoupled",
    };
  });

  const tbody = document.querySelector("#deg-table tbody");
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.model}</td>
      <td>${r.rho.toFixed(4)}<span style="color:#8a8a8a;font-size:0.85em"> (TS)</span></td>
      <td>${r.bad && r.decSeverity != null ? "S" + r.decSeverity : "—"}<span style="color:#8a8a8a;font-size:0.85em"> (live, ${method === "dcn" ? "post-DCN" : "pre-recal"})</span></td>
      <td>${r.ece5 != null ? fmt(r.ece5) : "—"}</td>
      <td>${r.map5 != null ? r.map5.toFixed(3) : "—"}</td>
      <td class="${r.bad ? "status-bad" : "status-good"}">${r.bad ? "Decoupled — dangerous" : "Stable / Coupled"}<span style="color:#8a8a8a;font-size:0.85em"> (TS)</span></td>
    </tr>`).join("");
}

function updateDegradationPage() {
  renderDegradationChart();
  renderDegTable();
}

document.getElementById("deg-update")?.addEventListener("click", updateDegradationPage);

document.getElementById("deg-recalib-toggle")?.addEventListener("click", e => {
  const opt = e.target.closest(".toggle-opt");
  if (!opt) return;
  document.querySelectorAll("#deg-recalib-toggle .toggle-opt").forEach(o => o.classList.remove("active"));
  opt.classList.add("active");
});

updateDegradationPage();

// ---------- Page 2: benchmark explorer table (REAL DATA, pivoted) ----------
function conditionLabel(row) {
  if (row.condition === "clean") return "Clean (baseline)";
  return row.corruption_type
    .split("_")
    .map(w => w[0].toUpperCase() + w.slice(1))
    .join(" ");
}

// Group the flat 126-row dataset into one row per (model, condition),
// carrying raw / ts / dcn ECE side by side. MCE, D-ECE, mAP@0.50 are
// taken from the raw confidence_type row, consistent with the rest of
// this study's tables (e.g. Table 9/10), which report those three
// against the uncalibrated baseline.
function pivotByCondition(rows) {
  const groups = {};
  for (const r of rows) {
    const key = `${r.model}__${r.condition}`;
    if (!groups[key]) {
      groups[key] = {
        model: r.model,
        condition: r.condition,
        corruption_type: r.corruption_type,
        severity: r.severity,
      };
    }
    groups[key][`ece_${r.confidence_type}`] = r.ece;
    if (r.confidence_type === "raw") {
      groups[key].mce = r.mce;
      groups[key].dece = r.dece;
      groups[key].map50 = r.map50;
    }
  }
  return Object.values(groups);
}

function applyExplorerFilters(pivoted) {
  const model = document.getElementById("f-model").value;
  const corruption = document.getElementById("f-corruption").value;
  const severity = document.getElementById("f-severity").value;

  return pivoted.filter(r => {
    if (model === "YOLOv8" && r.model !== "yolov8") return false;
    if (model === "RT-DETR" && r.model !== "rtdetr") return false;

    if (corruption !== "All types") {
      const wanted = corruption.toLowerCase().replace(/\s+/g, "_");
      if (r.condition === "clean" || r.corruption_type !== wanted) return false;
    }

    if (severity !== "All") {
      if (r.condition === "clean" || String(r.severity) !== severity) return false;
    }

    return true;
  });
}

// The Recalibration filter narrows WHICH ece column(s) are shown,
// rather than which rows -- since each pivoted row already carries
// all three methods for direct comparison, matching how Tables 14/15
// present this data in the thesis itself.
function visibleMethods() {
  const recal = document.getElementById("f-recal").value;
  const map = { "Raw": ["raw"], "Temp. scaling": ["ts"], "DentalCalib-Net": ["dcn"] };
  return map[recal] || ["raw", "ts", "dcn"];
}

function computeBaselineAvg(pivoted, methods) {
  const model = document.getElementById("f-model").value;
  const cleanRows = pivoted.filter(r => {
    if (r.condition !== "clean") return false;
    if (model === "YOLOv8" && r.model !== "yolov8") return false;
    if (model === "RT-DETR" && r.model !== "rtdetr") return false;
    return true;
  });
  const vals = [];
  cleanRows.forEach(r => methods.forEach(m => {
    if (r[`ece_${m}`] != null) vals.push(r[`ece_${m}`]);
  }));
  return {
    ece: vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null,
    mce: cleanRows.length ? cleanRows.reduce((s, r) => s + r.mce, 0) / cleanRows.length : null,
    dece: cleanRows.length ? cleanRows.reduce((s, r) => s + r.dece, 0) / cleanRows.length : null,
    map50: cleanRows.length ? cleanRows.reduce((s, r) => s + r.map50, 0) / cleanRows.length : null,
  };
}

function deltaCaption(current, baseline, metricName, higherIsWorse = true) {
  if (baseline == null) return { text: `No baseline for ${metricName}`, cls: "" };
  const diff = current - baseline;
  const cls = (diff > 0) === higherIsWorse ? "bad" : "good";
  const sign = diff >= 0 ? "+" : "";
  return { text: `${sign}${diff.toFixed(3)} vs baseline`, cls };
}

async function renderResultsTable() {
  const raw = await loadCalibrationData();
  const pivoted = pivotByCondition(raw);
  const filtered = applyExplorerFilters(pivoted);
  const methods = visibleMethods();
  const tbody = document.querySelector("#results-table tbody");

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9">No rows match the selected filters.</td></tr>`;
    return;
  }

  const cell = (row, method) =>
    methods.includes(method) && row[`ece_${method}`] != null
      ? fmt(row[`ece_${method}`])
      : "—";

  tbody.innerHTML = filtered.map(r => `
    <tr>
      <td>${r.model === "yolov8" ? "YOLOv8" : "RT-DETR"}</td>
      <td>${conditionLabel(r)}</td>
      <td>${r.condition === "clean" ? "—" : "S" + r.severity}</td>
      <td>${cell(r, "raw")}</td>
      <td>${cell(r, "ts")}</td>
      <td>${cell(r, "dcn")}</td>
      <td>${fmt(r.mce)}</td>
      <td>${fmt(r.dece)}</td>
      <td>${r.map50.toFixed(2)}</td>
    </tr>`).join("");

  // Stat cards: average ECE across the currently visible method(s) only
  const eceVals = [];
  filtered.forEach(r => methods.forEach(m => {
    if (r[`ece_${m}`] != null) eceVals.push(r[`ece_${m}`]);
  }));
  const avg = arr => arr.reduce((s, v) => s + v, 0) / arr.length;

  const baseline = computeBaselineAvg(pivoted, methods);
  const avgEce = eceVals.length ? avg(eceVals) : null;
  const avgMce = avg(filtered.map(r => r.mce));
  const avgDece = avg(filtered.map(r => r.dece));
  const avgMap = avg(filtered.map(r => r.map50));

  document.getElementById("avg-ece").textContent = avgEce != null ? fmt(avgEce) : "—";
  document.getElementById("avg-mce").textContent = fmt(avgMce);
  document.getElementById("avg-dece").textContent = fmt(avgDece);
  document.getElementById("avg-map").textContent = fmt(avgMap);

  const eceDelta = avgEce != null ? deltaCaption(avgEce, baseline.ece, "ECE") : { text: "—", cls: "" };
  const mceDelta = deltaCaption(avgMce, baseline.mce, "MCE");
  const deceDelta = deltaCaption(avgDece, baseline.dece, "D-ECE");
  const mapDelta = deltaCaption(avgMap, baseline.map50, "mAP@0.50", false); // higher mAP is better, so higherIsWorse=false

  document.getElementById("ece-delta").textContent = eceDelta.text;
  document.getElementById("ece-delta").className = `stat-delta ${eceDelta.cls}`;
  document.getElementById("mce-delta").textContent = mceDelta.text;
  document.getElementById("mce-delta").className = `stat-delta ${mceDelta.cls}`;
  document.getElementById("dece-delta").textContent = deceDelta.text;
  document.getElementById("dece-delta").className = `stat-delta ${deceDelta.cls}`;
  document.getElementById("map-delta").textContent = mapDelta.text;
  document.getElementById("map-delta").className = `stat-delta ${mapDelta.cls}`;
}

renderResultsTable();
document.getElementById("applyFilters")?.addEventListener("click", renderResultsTable);

// -------- NLL data (REAL DATA, raw/TS only -- DCN excluded per Stage 4 methodology) --------
let nllData = null;

async function loadNllData() {
  if (nllData) return nllData;
  try {
    const res = await fetch(dataUrl("nll_results.json"));
    if (!res.ok) throw new Error(`fetch failed: ${res.status}`);
    nllData = await res.json();
  } catch (err) {
    console.error("Failed to load nll_results.json:", err);
    nllData = {};
  }
  return nllData;
}

// ---------- Page 1: upload + run inference ----------
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");

dropzone.addEventListener("click", () => fileInput.click());

["dragenter", "dragover"].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  })
);
dropzone.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];
  if (file) handleUploadedFile(file);
});
fileInput.addEventListener("change", e => {
  const file = e.target.files[0];
  if (file) handleUploadedFile(file);
});

let selectedFilename = null;

async function handleUploadedFile(file) {
  try {
    await loadTestImageList();
  } catch (error) {
    selectedFilename = null;
    resetAuditorResults();
    console.error("Auditor data is unavailable:", error);
    document.getElementById("upload-status").textContent =
      "Auditor data is missing. Copy frontend/data and frontend/images from the shared dataset, then open this page over HTTP.";
    return;
  }

  if (!testImageList.includes(file.name)) {
    selectedFilename = null;
    resetAuditorResults();
    document.getElementById("upload-status").textContent =
      `No saved results for "${file.name}". Use an image from the 201-image DENTEX test set with its original filename.`;
    return;
  }
  selectedFilename = file.name;
  document.getElementById("upload-status").textContent = `✓ Uploaded: ${file.name}`;
  resetAuditorResults();
}

function resetAuditorResults() {
  ["canvas-yolo", "canvas-rtdetr"].forEach(id => {
    const canvas = document.getElementById(id);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  });
  document.querySelectorAll(".opg-preview .placeholder").forEach(p => (p.style.display = ""));

  [
    "yolo-ece", "yolo-mce", "yolo-dece", "rtdetr-ece", "rtdetr-mce", "rtdetr-dece",
    "s1-yolo", "s1-rt", "s2-yolo", "s2-rt", "s3-yolo", "s3-rt",
    "s1-delta", "s2-delta", "s3-delta", "yolo-chart-note", "rtdetr-chart-note",
  ].forEach(id => {
    const element = document.getElementById(id);
    element.textContent = "—";
    element.className = element.className.replace(/\s(good|bad)\b/g, "");
  });

  reliabilityChart("chart-rel-yolo", [], "#C4501E");
  reliabilityChart("chart-rel-rtdetr", [], "#2F7D4F");
}

// ---------- Page 1: Live OPG Auditor (REAL per-image data) ----------
let perImageCache = {};
let testImageList = null;
let testImageListPromise = null;

async function loadPerImagePredictions(model, condition) {
  const cacheKey = `${model}__${condition}`;
  if (cacheKey in perImageCache) return perImageCache[cacheKey];
  const res = await fetch(dataUrl(`per_image/${cacheKey}.json`), { cache: "no-cache" });
  if (!res.ok) throw new Error(`Missing per_image/${cacheKey}.json (HTTP ${res.status})`);
  perImageCache[cacheKey] = await res.json();
  return perImageCache[cacheKey];
}

function loadTestImageList() {
  if (!testImageListPromise) {
    testImageListPromise = loadPerImagePredictions("yolov8", "clean").then(clean => {
      testImageList = Object.keys(clean).sort();
      if (!testImageList.length) throw new Error("The clean YOLOv8 per-image data is empty.");
      return testImageList;
    });
  }
  return testImageListPromise;
}

function drawImageWithBoxes(canvasId, imagePath, predictions, threshold, confField) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const img = new Image();
  img.onload = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    predictions.forEach(pred => {
      const conf = pred[confField];
      if (conf < threshold) return;
      const [x1, y1, x2, y2] = pred.bbox;
      const isCorrect = pred.label === 1;
      ctx.strokeStyle = isCorrect ? "#2F7D4F" : "#C4501E";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = isCorrect ? "#2F7D4F" : "#C4501E";
      ctx.font = "12px monospace";
      ctx.fillText(conf.toFixed(2), x1, Math.max(10, y1 - 4));
    });
  };
  img.onerror = () => console.error(`Failed to load image: ${imagePath}`);
  img.src = imagePath;
}

function setupImageModal() {
  const modal = document.getElementById("image-modal");
  const modalImg = document.getElementById("image-modal-img");

  function openModal(canvas) {
    modalImg.src = canvas.toDataURL();
    modal.classList.add("active");
  }
  function closeModal() {
    modal.classList.remove("active");
  }

  ["canvas-yolo", "canvas-rtdetr"].forEach(id => {
    document.getElementById(id)?.addEventListener("click", () => {
      const canvas = document.getElementById(id);
      openModal(canvas);
    });
  });

  modal.addEventListener("click", closeModal);
  document.querySelector(".image-modal-close")?.addEventListener("click", e => {
    e.stopPropagation();
    closeModal();
  });
}

function auditorImagePath(condition, filename) {
  return `images/${condition}/${filename}`;
}

function auditorConditionKey(label, severity) {
  const map = {
    "None (clean)": null,
    "Gaussian noise": "gaussian_noise",
    "Motion blur": "motion_blur",
    "JPEG compression": "jpeg_compression",
    "Pixel exposure": "brightness_variation",
  };
  const base = map[label];
  return base ? `${base}_S${severity}` : "clean";
}

function auditorRecalMethod(label) {
  return { "None (raw)": "raw", "Temperature scaling": "ts", "DentalCalib-Net": "dcn" }[label] || "raw";
}

function calibrationInterpretation(binEntry) {
  if (!binEntry?.confidence.length) return { text: "No detections for this image", cls: "" };
  const gaps = binEntry.confidence.map((c, i) => c - binEntry.accuracy[i]);
  const meanGap = gaps.reduce((s, g) => s + g, 0) / gaps.length;
  const maxGap = Math.max(...gaps.map(Math.abs));

  if (maxGap < 0.05) return { text: "Closely follows the diagonal", cls: "good" };
  if (meanGap > 0.05) return { text: "Overconfident: predicted confidence exceeds observed accuracy", cls: "bad" };
  if (meanGap < -0.05) return { text: "Underconfident: predicted confidence trails observed accuracy", cls: "bad" };
  return { text: "Roughly calibrated, with some bin-level deviation", cls: "good" };
}

function betterModelDelta(yoloVal, rtVal, label) {
  if (!Number.isFinite(yoloVal) || !Number.isFinite(rtVal)) {
    return { text: `${label} unavailable per image`, cls: "" };
  }
  const diff = rtVal - yoloVal;
  const better = diff < 0 ? "RT-DETR" : diff > 0 ? "YOLOv8" : null;
  if (better === null) return { text: `Tied (${label})`, cls: "" };
  return { text: `${better} ${diff.toFixed(3)} better`, cls: "good" };
}

function summarizeImagePredictions(predictions, confidenceField, nBins = 15) {
  const counts = Array(nBins).fill(0);
  const confidenceSums = Array(nBins).fill(0);
  const labelSums = Array(nBins).fill(0);
  let total = 0;

  // 2D confidence x IoU grid for D-ECE, matching calibration_metrics.py's compute_dece
  const nCells = nBins * nBins;
  const cellCounts = Array(nCells).fill(0);
  const cellLabelSums = Array(nCells).fill(0);
  const cellConfSums = Array(nCells).fill(0);
  let deceTotal = 0;

  const binIndex = value => Math.min(nBins - 1, Math.max(0, Math.ceil(value * nBins) - 1));

  predictions.forEach(prediction => {
    const confidence = Number(prediction[confidenceField]);
    const label = Number(prediction.label);
    if (!Number.isFinite(confidence) || ![0, 1].includes(label)) return;

    const bin = binIndex(confidence);
    counts[bin] += 1;
    confidenceSums[bin] += confidence;
    labelSums[bin] += label;
    total += 1;

    const iou = Number(prediction.iou);
    if (Number.isFinite(iou) && iou >= 0 && iou <= 1) {
      const ci = binIndex(confidence);
      const ii = binIndex(iou);
      const flat = ci * nBins + ii;
      cellCounts[flat] += 1;
      cellLabelSums[flat] += label;
      cellConfSums[flat] += confidence;
      deceTotal += 1;
    }
  });

  if (!total) return { ece: null, mce: null, dece: null, bins: { confidence: [], accuracy: [] } };

  const populated = counts.map((count, index) => ({ count, index })).filter(bin => bin.count > 0);
  const gaps = populated.map(({ count, index }) => {
    const confidence = confidenceSums[index] / count;
    const accuracy = labelSums[index] / count;
    return { confidence, accuracy, gap: Math.abs(accuracy - confidence), weight: count / total };
  });

  let dece = null;
  if (deceTotal > 0) {
    dece = 0;
    for (let cell = 0; cell < nCells; cell++) {
      const c = cellCounts[cell];
      if (c === 0) continue;
      const acc = cellLabelSums[cell] / c;
      const avgConf = cellConfSums[cell] / c;
      dece += (c / deceTotal) * Math.abs(acc - avgConf);
    }
  }

  return {
    ece: gaps.reduce((sum, bin) => sum + bin.weight * bin.gap, 0),
    mce: Math.max(...gaps.map(bin => bin.gap)),
    dece,
    bins: {
      confidence: gaps.map(bin => bin.confidence),
      accuracy: gaps.map(bin => bin.accuracy),
    },
  };
}

async function runAuditorInference() {
  await loadTestImageList();

  if (!selectedFilename) return; // nothing uploaded yet
  document.querySelectorAll(".opg-preview .placeholder").forEach(p => (p.style.display = "none"));
  const filename = selectedFilename;
  const condLabel = document.getElementById("oodCondition").value;
  const severity = document.getElementById("severity").value;
  const recalLabel = document.getElementById("recalMethod").value;
  const condition = auditorConditionKey(condLabel, severity);
  const method = auditorRecalMethod(recalLabel);
  const imagePath = auditorImagePath(condition, filename);

  const confField = { raw: "confidence_raw", ts: "confidence_ts", dcn: "confidence_dcn" }[method];
  const yoloPreds = (await loadPerImagePredictions("yolov8", condition))[filename] || [];
  const rtPreds = (await loadPerImagePredictions("rtdetr", condition))[filename] || [];
  const yoloMetrics = summarizeImagePredictions(yoloPreds, confField);
  const rtMetrics = summarizeImagePredictions(rtPreds, confField);
  const showMetric = value => value == null ? "—" : fmt(value);

  document.getElementById("yolo-ece").textContent = showMetric(yoloMetrics.ece);
  document.getElementById("yolo-mce").textContent = showMetric(yoloMetrics.mce);
  document.getElementById("yolo-dece").textContent = showMetric(yoloMetrics.dece);
  document.getElementById("rtdetr-ece").textContent = showMetric(rtMetrics.ece);
  document.getElementById("rtdetr-mce").textContent = showMetric(rtMetrics.mce);
  document.getElementById("rtdetr-dece").textContent = showMetric(rtMetrics.dece);
  document.getElementById("s1-yolo").textContent = showMetric(yoloMetrics.ece);
  document.getElementById("s1-rt").textContent = showMetric(rtMetrics.ece);
  document.getElementById("s2-yolo").textContent = showMetric(yoloMetrics.mce);
  document.getElementById("s2-rt").textContent = showMetric(rtMetrics.mce);
  document.getElementById("s3-yolo").textContent = showMetric(yoloMetrics.dece);
  document.getElementById("s3-rt").textContent = showMetric(rtMetrics.dece);

  const eceDelta = betterModelDelta(yoloMetrics.ece, rtMetrics.ece, "ECE");
  const mceDelta = betterModelDelta(yoloMetrics.mce, rtMetrics.mce, "MCE");
  const deceDelta = betterModelDelta(yoloMetrics.dece, rtMetrics.dece, "D-ECE");
  document.getElementById("s1-delta").textContent = eceDelta.text;
  document.getElementById("s1-delta").className = `stat-delta ${eceDelta.cls}`;
  document.getElementById("s2-delta").textContent = mceDelta.text;
  document.getElementById("s2-delta").className = `stat-delta ${mceDelta.cls}`;
  document.getElementById("s3-delta").textContent = deceDelta.text;
  document.getElementById("s3-delta").className = `stat-delta ${deceDelta.cls}`;

  reliabilityChart("chart-rel-yolo", binsToChartPoints(yoloMetrics.bins), "#C4501E");
  reliabilityChart("chart-rel-rtdetr", binsToChartPoints(rtMetrics.bins), "#2F7D4F");

  const yoloNote = calibrationInterpretation(yoloMetrics.bins);
  const rtNote = calibrationInterpretation(rtMetrics.bins);
  const yoloNoteEl = document.getElementById("yolo-chart-note");
  const rtNoteEl = document.getElementById("rtdetr-chart-note");
  yoloNoteEl.textContent = yoloNote.text;
  yoloNoteEl.className = `chart-note ${yoloNote.cls}`;
  rtNoteEl.textContent = rtNote.text;
  rtNoteEl.className = `chart-note ${rtNote.cls}`;

  drawImageWithBoxes("canvas-yolo", imagePath, yoloPreds, 0.1, confField);
  drawImageWithBoxes("canvas-rtdetr", imagePath, rtPreds, 0.1, confField);
}

document.getElementById("runInference")?.addEventListener("click", () => {
  const btn = document.getElementById("runInference");
  const originalText = btn.textContent;
  btn.textContent = "Loading…";
  btn.disabled = true;
  runAuditorInference().catch(error => {
    console.error("Auditor results could not be loaded:", error);
    document.getElementById("upload-status").textContent =
      `Could not load the selected condition's saved results: ${error.message}`;
  }).finally(() => {
    btn.textContent = originalText;
    btn.disabled = false;
    document.querySelector(".card-title")?.scrollIntoView({ behavior: "smooth", block: "start" });
  });
});

loadTestImageList().catch(error => {
  console.error("Auditor data is unavailable:", error);
  document.getElementById("upload-status").textContent =
    "Auditor data is missing. Copy frontend/data and frontend/images from the shared dataset, then open this page over HTTP.";
});
setupImageModal();