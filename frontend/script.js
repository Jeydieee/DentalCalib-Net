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

   Page 4 (OOD Degradation Curves) is NOT yet wired to real data --
   degradationSeries() and renderDegTable() below still use
   Math.random() and hardcoded rows. Do not treat Page 4's numbers
   as real findings until that page is integrated the same way as
   Pages 1-3.

   The Live OPG Auditor only accepts uploads matching filenames
   from the 201-image DENTEX test partition; it does not run live
   inference on arbitrary images (see handleUploadedFile).
   ========================================================== */

// ---------- Navigation ----------
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
const rand = (min, max) => Math.random() * (max - min) + min;
const fmt = n => n.toFixed(3);

function reliabilityBins(overconfident = true) {
  // 10 confidence bins from 0.05 to 0.95
  const bins = [];
  for (let i = 0; i < 10; i++) {
    const conf = (i + 0.5) / 10;
    let acc;
    if (overconfident) {
      acc = conf - (0.05 + conf * 0.25 * Math.random());
    } else {
      acc = conf - (0.02 + (1 - conf) * 0.08 * Math.random());
    }
    bins.push({ x: conf, y: Math.max(0, Math.min(1, acc)) });
  }
  return bins;
}

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
    const res = await fetch("data/calibration_results.json");
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
    const res = await fetch("data/reliability_bins.json");
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
      <td>${r.decSeverity != null ? "S" + r.decSeverity : "—"}</td>
      <td>${r.ece5 != null ? fmt(r.ece5) : "—"}</td>
      <td>${r.map5 != null ? r.map5.toFixed(3) : "—"}</td>
      <td class="${r.bad ? "status-bad" : "status-good"}">${r.bad ? "Decoupled — dangerous" : "Stable / Coupled"}</td>
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
  const cleanRows = pivoted.filter(r => r.condition === "clean");
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
    const res = await fetch("data/nll_results.json");
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
  await loadTestImageList();
  if (!testImageList.includes(file.name)) {
    alert(
      `"${file.name}" is not part of the 201-image DENTEX test partition this benchmark was computed on.\n\n` +
      `Only images actually run through YOLOv8/RT-DETR during this study have precomputed results.`
    );
    return;
  }
  selectedFilename = file.name;
  document.getElementById("upload-status").textContent = `✓ Uploaded: ${file.name}`;
  document.querySelectorAll(".opg-preview .placeholder").forEach(p => (p.style.display = "none"));
  // Inference no longer runs automatically here -- the user sets OOD
  // condition/severity/recalibration next, then explicitly clicks
  // "Run inference" to populate results and trigger the scroll below.
}

// ---------- Page 1: Live OPG Auditor (REAL per-image data) ----------
let perImageCache = {};
let testImageList = null;

async function loadPerImagePredictions(model, condition) {
  const cacheKey = `${model}__${condition}`;
  if (perImageCache[cacheKey]) return perImageCache[cacheKey];
  try {
    const res = await fetch(`data/per_image/${cacheKey}.json`);
    if (!res.ok) throw new Error(`fetch failed: ${res.status}`);
    perImageCache[cacheKey] = await res.json();
  } catch (err) {
    console.error(`Failed to load per_image/${cacheKey}.json:`, err);
    perImageCache[cacheKey] = {};
  }
  return perImageCache[cacheKey];
}

async function loadTestImageList() {
  if (testImageList) return testImageList;
  const clean = await loadPerImagePredictions("yolov8", "clean");
  testImageList = Object.keys(clean).sort();
  return testImageList;
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
  if (!binEntry) return { text: "No data for this condition", cls: "" };
  const gaps = binEntry.confidence.map((c, i) => c - binEntry.accuracy[i]);
  const meanGap = gaps.reduce((s, g) => s + g, 0) / gaps.length;
  const maxGap = Math.max(...gaps.map(Math.abs));

  if (maxGap < 0.05) return { text: "Closely follows the diagonal", cls: "good" };
  if (meanGap > 0.05) return { text: "Overconfident: predicted confidence exceeds observed accuracy", cls: "bad" };
  if (meanGap < -0.05) return { text: "Underconfident: predicted confidence trails observed accuracy", cls: "bad" };
  return { text: "Roughly calibrated, with some bin-level deviation", cls: "good" };
}

function betterModelDelta(yoloVal, rtVal, label) {
  const diff = rtVal - yoloVal;
  const better = diff < 0 ? "RT-DETR" : diff > 0 ? "YOLOv8" : null;
  if (better === null) return { text: `Tied (${label})`, cls: "" };
  return { text: `${better} ${diff.toFixed(3)} better`, cls: "good" };
}

async function runAuditorInference() {
  await Promise.all([loadCalibrationData(), loadReliabilityBins(), loadTestImageList()]);

  if (!selectedFilename) return; // nothing uploaded yet
  const filename = selectedFilename;
  const condLabel = document.getElementById("oodCondition").value;
  const severity = document.getElementById("severity").value;
  const recalLabel = document.getElementById("recalMethod").value;
  const condition = auditorConditionKey(condLabel, severity);
  const method = auditorRecalMethod(recalLabel);
  const imagePath = auditorImagePath(condition, filename);

  const yoloRow = calibrationData.find(r => r.model === "yolov8" && r.condition === condition && r.confidence_type === method);
  const rtRow = calibrationData.find(r => r.model === "rtdetr" && r.condition === condition && r.confidence_type === method);
  const yoloBins = lookupBins("yolov8", condition, method);
  const rtBins = lookupBins("rtdetr", condition, method);

  if (!yoloRow || !rtRow) {
    console.warn(`No aggregate data for condition=${condition} method=${method}`);
    return;
  }

  document.getElementById("yolo-ece").textContent = fmt(yoloRow.ece);
  document.getElementById("yolo-mce").textContent = fmt(yoloRow.mce);
  document.getElementById("yolo-dece").textContent = fmt(yoloRow.dece);
  document.getElementById("rtdetr-ece").textContent = fmt(rtRow.ece);
  document.getElementById("rtdetr-mce").textContent = fmt(rtRow.mce);
  document.getElementById("rtdetr-dece").textContent = fmt(rtRow.dece);
  document.getElementById("s1-yolo").textContent = fmt(yoloRow.ece);
  document.getElementById("s1-rt").textContent = fmt(rtRow.ece);
  document.getElementById("s2-yolo").textContent = fmt(yoloRow.mce);
  document.getElementById("s2-rt").textContent = fmt(rtRow.mce);
  document.getElementById("s3-yolo").textContent = fmt(yoloRow.dece);
  document.getElementById("s3-rt").textContent = fmt(rtRow.dece);

  const eceDelta = betterModelDelta(yoloRow.ece, rtRow.ece, "ECE");
  const mceDelta = betterModelDelta(yoloRow.mce, rtRow.mce, "MCE");
  const deceDelta = betterModelDelta(yoloRow.dece, rtRow.dece, "D-ECE");
  document.getElementById("s1-delta").textContent = eceDelta.text;
  document.getElementById("s1-delta").className = `stat-delta ${eceDelta.cls}`;
  document.getElementById("s2-delta").textContent = mceDelta.text;
  document.getElementById("s2-delta").className = `stat-delta ${mceDelta.cls}`;
  document.getElementById("s3-delta").textContent = deceDelta.text;
  document.getElementById("s3-delta").className = `stat-delta ${deceDelta.cls}`;

  reliabilityChart("chart-rel-yolo", binsToChartPoints(yoloBins), "#C4501E");
  reliabilityChart("chart-rel-rtdetr", binsToChartPoints(rtBins), "#2F7D4F");

  const yoloNote = calibrationInterpretation(yoloBins);
  const rtNote = calibrationInterpretation(rtBins);
  const yoloNoteEl = document.getElementById("yolo-chart-note");
  const rtNoteEl = document.getElementById("rtdetr-chart-note");
  yoloNoteEl.textContent = yoloNote.text;
  yoloNoteEl.className = `chart-note ${yoloNote.cls}`;
  rtNoteEl.textContent = rtNote.text;
  rtNoteEl.className = `chart-note ${rtNote.cls}`;

  const confField = { raw: "confidence_raw", ts: "confidence_ts", dcn: "confidence_dcn" }[method];
  const yoloPreds = (await loadPerImagePredictions("yolov8", condition))[filename] || [];
  const rtPreds = (await loadPerImagePredictions("rtdetr", condition))[filename] || [];
  drawImageWithBoxes("canvas-yolo", imagePath, yoloPreds, 0.1, confField);
  drawImageWithBoxes("canvas-rtdetr", imagePath, rtPreds, 0.1, confField);
}

document.getElementById("runInference")?.addEventListener("click", () => {
  const btn = document.getElementById("runInference");
  const originalText = btn.textContent;
  btn.textContent = "Loading…";
  btn.disabled = true;
  runAuditorInference().finally(() => {
    btn.textContent = originalText;
    btn.disabled = false;
    document.querySelector(".card-title")?.scrollIntoView({ behavior: "smooth", block: "start" });
  });
});

loadTestImageList();
setupImageModal();