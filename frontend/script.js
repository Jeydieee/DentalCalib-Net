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
   renderDegradationChart() and renderDegTable() below. Spearman
   rho, p-values and the Coupled / Decoupled status are computed
   live from calibration_results.json for the selected method
   (Raw / TS / DCN); the chart and ECE@S5 / mAP@S5 numbers use
   the same rows.

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
      maintainAspectRatio: !ctx.closest(".chart-wrap"),
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
  const isClean = condLabel === "Clean (baseline)";
  document.getElementById("rv-severity").disabled = isClean;
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

  const showMetric = value => (value == null ? "—" : fmt(value));

  document.getElementById("rv-raw-ece").textContent = calRow ? showMetric(calRow.ece) : "—";
  document.getElementById("rv-raw-mce").textContent = calRow ? showMetric(calRow.mce) : "—";
  document.getElementById("rv-raw-dece").textContent = calRow ? showMetric(calRow.dece) : "—";

  document.getElementById("rv-recal-ece").textContent = calRowRecal ? showMetric(calRowRecal.ece) : "—";
  document.getElementById("rv-recal-mce").textContent = calRowRecal ? showMetric(calRowRecal.mce) : "—";
  document.getElementById("rv-recal-dece").textContent = calRowRecal ? showMetric(calRowRecal.dece) : "—";

  const rmsdRow = calRow && calRow.rmsd != null ? fmt(calRow.rmsd) : "—";
  const rmsdRecal = calRowRecal && calRowRecal.rmsd != null ? fmt(calRowRecal.rmsd) : "—";
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
// Spearman rho, its p-value and the Coupled / Decoupled status are computed
// live from calibration_results.json for the selected method (Raw / TS / DCN),
// so the chart, the table and the status always describe the same rows.
// The p-value matches scipy.stats.spearmanr (two-sided, t-approximation, which
// for n = 5 severity levels has 3 degrees of freedom).

const ALPHA = 0.05;
const DEG_SEVERITIES = [1, 2, 3, 4, 5];
const DEG_METHOD_NAMES = {
  raw: "Raw (no recalibration)",
  ts: "Temperature Scaling",
  dcn: "DentalCalib-Net",
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
  return active?.dataset.mode || "raw";
}

// ECE and mAP@0.50 at S1..S5 for one model / corruption / method
function degSeries(model, corruption, method) {
  const ece = [];
  const map = [];
  DEG_SEVERITIES.forEach(s => {
    const row = calibrationData.find(
      r => r.model === model && r.condition === `${corruption}_S${s}` && r.confidence_type === method
    );
    ece.push(row ? row.ece : null);
    map.push(row ? row.map50 : null);
  });
  return { ece, map };
}

// ---- Spearman rank correlation (average ranks for ties, like scipy) ----
function averageRanks(values) {
  const order = values.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const ranks = new Array(values.length);
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j + 1 < order.length && order[j + 1].v === order[i].v) j++;
    const rank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[order[k].i] = rank;
    i = j + 1;
  }
  return ranks;
}

function pearson(a, b) {
  const n = a.length;
  const ma = a.reduce((s, v) => s + v, 0) / n;
  const mb = b.reduce((s, v) => s + v, 0) / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return da > 0 && db > 0 ? num / Math.sqrt(da * db) : null;
}

// Two-sided p-value of a t statistic with 3 degrees of freedom (n = 5).
function pFromT3(t) {
  const s = Math.abs(t) / Math.sqrt(3);
  const cdf = 0.5 + (s / (1 + s * s) + Math.atan(s)) / Math.PI;
  return 2 * (1 - cdf);
}

function spearman(x, y) {
  if (x.length !== 5) return { rho: null, p: null }; // p-value formula above is for n = 5
  const rho = pearson(averageRanks(x), averageRanks(y));
  if (rho == null) return { rho: null, p: null };
  if (Math.abs(rho) >= 1) return { rho, p: 0 };
  const t = rho * Math.sqrt(3 / (1 - rho * rho));
  return { rho, p: pFromT3(t) };
}

// mAP@0.50 vs ECE across S1-S5; needs all five severity levels
function degSpearman(eceSeries, mapSeries) {
  if (eceSeries.some(v => v == null) || mapSeries.some(v => v == null)) return { rho: null, p: null };
  return spearman(mapSeries, eceSeries);
}

// Proposal (Stage 3): a negative rho means accuracy and calibration degrade
// together (Coupled); a near-zero or positive rho means calibration degrades
// independently of accuracy (Decoupled, the dangerous failure mode).
// "Stable" appears in the proposal's Table 18 without a definition, so it is
// not produced here. "Significant" uses alpha = 0.05 from the proposal.
function decouplingStatus(rho, p) {
  if (rho == null) return { decoupled: false, significant: false, label: "—" };
  const decoupled = rho >= 0;
  const significant = p != null && p < ALPHA;
  const base = decoupled ? "Decoupled — dangerous" : "Coupled";
  return { decoupled, significant, label: significant ? base : `${base} (n.s.)` };
}

let degChart;

// Visual aid only (not a measure defined in the proposal): the severity step
// where the normalized ECE and mAP trends diverge most from the expected
// "ECE up, mAP down" relationship. Returns the LATER severity of that step,
// or null if there is not enough valid data.
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

// Draws one dashed vertical line per marker passed in the chart's
// plugin options: { markers: [{ severity, color, text }] }
const decouplingLinePlugin = {
  id: "decouplingLine",
  afterDraw(chart, args, opts) {
    const markers = opts?.markers || [];
    if (!markers.length) return;
    const { ctx, chartArea, scales } = chart;

    markers.forEach((m, idx) => {
      const x = scales.x.getPixelForValue(chart.data.labels.indexOf(m.severity));
      const color = m.color || "#C4501E";

      ctx.save();
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = color;
      ctx.font = "11px sans-serif";
      const labelWidth = ctx.measureText(m.text).width;
      const padding = 6;
      const y = chartArea.top + 12 + idx * 14;
      if (x + padding + labelWidth <= chartArea.right) {
        ctx.textAlign = "left";
        ctx.fillText(m.text, x + padding, y);
      } else {
        ctx.textAlign = "right";
        ctx.fillText(m.text, x - padding, y);
      }
      ctx.restore();
    });
  },
};

async function renderDegradationChart() {
  await loadCalibrationData();

  const models = degModelsSelected(document.getElementById("deg-model").value);
  const corruption = degCorruptionKey(document.getElementById("deg-corruption").value);
  const method = degRecalMethod();
  const colors = { yolov8: "#C4501E", rtdetr: "#1E6E63" };
  const datasets = [];
  const markers = [];

  models.forEach(model => {
    const { ece, map } = degSeries(model, corruption, method);
    const label = model === "yolov8" ? "YOLOv8" : "RT-DETR";

    datasets.push({
      label: `${label} ECE`,
      data: ece,
      borderColor: colors[model],
      backgroundColor: colors[model],
      yAxisID: "y",
      tension: 0.2,
    });
    datasets.push({
      label: `${label} mAP@0.50`,
      data: map,
      borderColor: colors[model],
      backgroundColor: colors[model],
      borderDash: [5, 3],
      yAxisID: "y1",
      tension: 0.2,
    });

    // The decoupling marker is drawn only for curves whose status is Decoupled.
    const { rho, p } = degSpearman(ece, map);
    if (decouplingStatus(rho, p).decoupled) {
      const severity = computeDecouplingSeverity(ece, map, DEG_SEVERITIES);
      if (severity != null) {
        markers.push({ severity, color: colors[model], text: `${label}: decoupling point (S${severity})` });
      }
    }
  });

  const ctx = document.getElementById("chart-degradation");
  if (degChart) degChart.destroy();
  degChart = new Chart(ctx, {
    type: "line",
    data: { labels: DEG_SEVERITIES, datasets },
    options: {
      responsive: true,
      scales: {
        x: { title: { display: true, text: "OOD severity" }, grid: { display: false } },
        // Both axes start at zero so neither series is visually exaggerated.
        y: { position: "left", min: 0, title: { display: true, text: "ECE (left axis)" }, grid: { color: "#EEEBE1" } },
        y1: { position: "right", min: 0, max: 1, title: { display: true, text: "mAP@0.50 (right axis)" }, grid: { display: false } },
      },
      plugins: {
        legend: { position: "top", align: "end" },
        decouplingLine: { markers },
      },
    },
    plugins: [decouplingLinePlugin],
  });
}

async function renderDegTable() {
  await loadCalibrationData();

  const models = degModelsSelected(document.getElementById("deg-model").value);
  const corruption = degCorruptionKey(document.getElementById("deg-corruption").value);
  const method = degRecalMethod();

  const rows = models.map(model => {
    const { ece, map } = degSeries(model, corruption, method);
    const { rho, p } = degSpearman(ece, map);
    const status = decouplingStatus(rho, p);
    return {
      model: model === "yolov8" ? "YOLOv8" : "RT-DETR",
      rho,
      p,
      status,
      decSeverity: status.decoupled ? computeDecouplingSeverity(ece, map, DEG_SEVERITIES) : null,
      ece5: ece[4],
      map5: map[4],
    };
  });

  const fmtP = p => (p == null ? "—" : p < 0.0005 ? "<0.001" : p.toFixed(4));

  const tbody = document.querySelector("#deg-table tbody");
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.model}</td>
      <td>${r.rho != null ? r.rho.toFixed(4) : "—"}</td>
      <td>${fmtP(r.p)}</td>
      <td>${r.rho == null ? "—" : r.status.significant ? "Significant" : "Not significant"}</td>
      <td>${r.decSeverity != null ? "S" + r.decSeverity : "—"}</td>
      <td>${r.ece5 != null ? fmt(r.ece5) : "—"}</td>
      <td>${r.map5 != null ? r.map5.toFixed(3) : "—"}</td>
      <td class="${r.status.decoupled ? "status-bad" : "status-good"}">${r.status.label}</td>
    </tr>`).join("");

  document.getElementById("deg-note").innerHTML = `
    <b>Basis:</b> all values use <b>${DEG_METHOD_NAMES[method]}</b> confidences.
    ρ is the Spearman rank correlation between mAP@0.50 and ECE across S1–S5;
    p is two-sided (scipy.stats.spearmanr). With five severity levels,
    |ρ| ≥ 0.878 is needed for significance at α = ${ALPHA}; "n.s." means not significant.<br />
    <b>Status (proposal rule):</b> negative ρ = Coupled (accuracy and calibration
    degrade together); near-zero or positive ρ = Decoupled (calibration degrades
    independently of accuracy).<br />
    <b>Decoupling severity</b> is a visual aid, not a measure defined in the proposal:
    the severity step where the normalized ECE and mAP trends diverge most. It is
    shown only for Decoupled curves.`;
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
// Every row of calibration_results.json is (model, condition, confidence_type).
// The pivot below folds the three confidence types (raw / ts / dcn) into one
// row per (model, condition) so each metric can be shown side by side.
// The clean baselines are pinned at the top of the table and are never
// filtered out (Appendix 4); averages and deltas use OOD rows only.
const METRIC_LABELS = { ece: "ECE", mce: "MCE", dece: "D-ECE", map50: "mAP@0.50" };
const METHODS = ["raw", "ts", "dcn"];
const METHOD_NAMES = { raw: "Raw", ts: "TS", dcn: "DCN" };

function conditionLabel(row) {
  if (row.condition === "clean") return "Clean (baseline)";
  return row.corruption_type
    .split("_")
    .map(w => w[0].toUpperCase() + w.slice(1))
    .join(" ");
}

function modelName(key) {
  return key === "yolov8" ? "YOLOv8" : "RT-DETR";
}

// Optimized Temperature Scaling parameter, one per model (learned on the
// validation logits in Stage 4). Fill in from the training log.
const TEMPERATURE_LOOKUP = { yolov8: null, rtdetr: null };

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
    const g = groups[key];
    Object.keys(METRIC_LABELS).forEach(m => {
      g[`${m}_${r.confidence_type}`] = r[m];
    });
    // Optimized temperature: only meaningful on the TS row. Requires the
    // export to include a "temperature" field; otherwise the column shows "—".
    if (r.confidence_type === "ts") {
      g.temperature = r.temperature ?? TEMPERATURE_LOOKUP[r.model] ?? null;
    }
  }
  return Object.values(groups);
}

function selectedMetric() {
  const label = document.getElementById("f-metric").value;
  return Object.keys(METRIC_LABELS).find(k => METRIC_LABELS[k] === label) || "ece";
}

function modelMatches(r) {
  const model = document.getElementById("f-model").value;
  if (model === "YOLOv8") return r.model === "yolov8";
  if (model === "RT-DETR") return r.model === "rtdetr";
  return true;
}

function applyExplorerFilters(pivoted) {
  const corruption = document.getElementById("f-corruption").value;
  const severity = document.getElementById("f-severity").value;

  return pivoted.filter(r => {
    if (r.condition === "clean") return false; // clean baselines are pinned separately
    if (!modelMatches(r)) return false;

    if (corruption !== "All types") {
      const wanted = corruption.toLowerCase().replace(/\s+/g, "_");
      if (r.corruption_type !== wanted) return false;
    }
    if (severity !== "All" && String(r.severity) !== severity) return false;
    return true;
  });
}

// The Recalibration filter narrows WHICH method columns are populated,
// not which rows exist; "All" shows raw, TS and DCN together.
function visibleMethods() {
  const recal = document.getElementById("f-recal").value;
  const map = { "Raw": ["raw"], "Temp. scaling": ["ts"], "DentalCalib-Net": ["dcn"] };
  return map[recal] || METHODS;
}

function deltaCaption(current, baseline, metricName, higherIsWorse = true) {
  if (baseline == null) return { text: `No baseline for ${metricName}`, cls: "" };
  const diff = current - baseline;
  const cls = (diff > 0) === higherIsWorse ? "bad" : "good";
  const sign = diff >= 0 ? "+" : "";
  return { text: `${sign}${diff.toFixed(3)} vs clean baseline`, cls };
}

const showVal = v => (v == null ? "—" : fmt(v));

// ΔECE / ΔMCE / ΔD-ECE = |OOD - clean| for the same model and method, as
// defined in the proposal. mAP is not a calibration metric and has no
// proposal-defined delta, so it is shown as the signed change (OOD - clean).
function deltaOf(metric, value, base) {
  if (value == null || base == null) return null;
  return metric === "map50" ? value - base : Math.abs(value - base);
}

function showDelta(metric, d) {
  if (d == null) return "—";
  if (metric === "map50") return `${d >= 0 ? "+" : ""}${d.toFixed(3)}`;
  return fmt(d);
}

function explorerRowHtml(r, metric, shownMethods, base, pinnedClass) {
  const cls = pinnedClass ? ` class="pinned-row ${pinnedClass}"` : "";
  const val = m => (shownMethods.includes(m) ? showVal(r[`${metric}_${m}`]) : "—");
  const dlt = m => {
    if (pinnedClass || !base || !shownMethods.includes(m)) return "—";
    return showDelta(metric, deltaOf(metric, r[`${metric}_${m}`], base[`${metric}_${m}`]));
  };
  const t = r.temperature != null ? r.temperature.toFixed(3) : "—";
  return `
    <tr${cls}>
      <td>${modelName(r.model)}</td>
      <td>${conditionLabel(r)}</td>
      <td>${r.condition === "clean" ? "—" : "S" + r.severity}</td>
      <td>${val("raw")}</td>
      <td>${val("ts")}</td>
      <td>${val("dcn")}</td>
      <td>${dlt("raw")}</td>
      <td>${dlt("ts")}</td>
      <td>${dlt("dcn")}</td>
      <td>${t}</td>
    </tr>`;
}

async function renderResultsTable() {
  const raw = await loadCalibrationData();
  const pivoted = pivotByCondition(raw);
  const filtered = applyExplorerFilters(pivoted);
  const methods = visibleMethods();
  const metric = selectedMetric();
  const metricLabel = METRIC_LABELS[metric];
  const tbody = document.querySelector("#results-table tbody");

  const cleanByModel = {};
  pivoted.filter(r => r.condition === "clean").forEach(r => { cleanByModel[r.model] = r; });

  // Column headers follow the selected metric
  METHODS.forEach(m => {
    document.getElementById(`th-${m}`).textContent = `${metricLabel} (${METHOD_NAMES[m]})`;
    document.getElementById(`th-d${m}`).textContent = `Δ${metricLabel} (${METHOD_NAMES[m]})`;
  });

  // Pinned clean-baseline rows: always visible, all three methods shown
  const pinned = ["yolov8", "rtdetr"]
    .filter(m => cleanByModel[m])
    .map((m, i) => explorerRowHtml(cleanByModel[m], metric, METHODS, null, `p${i + 1}`))
    .join("");

  const body = filtered.length
    ? filtered.map(r => explorerRowHtml(r, metric, methods, cleanByModel[r.model], "")).join("")
    : `<tr><td colspan="10">No OOD rows match the selected filters.</td></tr>`;

  tbody.innerHTML = pinned + body;

  // Summary cards: average over the filtered OOD rows for the selected method.
  // With Recalibration = All the main value is Raw, and TS / DCN averages are
  // listed underneath. Deltas compare against the clean baseline of the same
  // method (and the same model filter).
  const avg = arr => (arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : null);
  const cardMethod = methods.length === 1 ? methods[0] : "raw";

  [
    { m: "ece", v: "avg-ece", d: "ece-delta", worse: true },
    { m: "mce", v: "avg-mce", d: "mce-delta", worse: true },
    { m: "dece", v: "avg-dece", d: "dece-delta", worse: true },
    { m: "map50", v: "avg-map", d: "map-delta", worse: false },
  ].forEach(({ m, v, d, worse }) => {
    const mean = key => avg(filtered.map(r => r[`${m}_${key}`]).filter(x => x != null));
    const main = mean(cardMethod);
    const base = avg(
      pivoted
        .filter(r => r.condition === "clean" && modelMatches(r))
        .map(r => r[`${m}_${cardMethod}`])
        .filter(x => x != null)
    );

    let text = "—";
    let cls = "";
    if (main != null) {
      if (methods.length === 1) {
        const c = deltaCaption(main, base, METRIC_LABELS[m], worse);
        text = `${METHOD_NAMES[cardMethod]}: ${c.text}`;
        cls = c.cls;
      } else {
        text = `Raw · TS ${showVal(mean("ts"))} · DCN ${showVal(mean("dcn"))}`;
      }
    }
    document.getElementById(v).textContent = main != null ? fmt(main) : "—";
    document.getElementById(d).textContent = text;
    document.getElementById(d).className = `stat-delta ${cls}`;
  });
}

renderResultsTable();
document.getElementById("applyFilters")?.addEventListener("click", renderResultsTable);

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
    "yolo-n", "yolo-vs-raw", "rtdetr-n", "rtdetr-vs-raw",
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
    "Brightness variation": "brightness_variation",
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

// Appendix 4: delta indicator showing ECE and MCE improvement relative to raw,
// plus the detection count, because per-image metrics on a handful of boxes
// are noisy and should not be read like the test-set benchmark numbers.
// Display-only: boxes below this score are not drawn. The metrics always use
// every saved detection, so they stay comparable to the benchmark numbers.
const DISPLAY_MIN_CONF = 0.25;

function showRecalDelta(prefix, method, raw, current, nDetections, nShown) {
  const nEl = document.getElementById(`${prefix}-n`);
  const dEl = document.getElementById(`${prefix}-vs-raw`);

  nEl.textContent = nDetections
    ? `Metrics use all ${nDetections} detections; ${nShown} drawn (confidence ≥ ${DISPLAY_MIN_CONF})`
    : "No detections in this image";

  if (method === "raw") {
    dEl.textContent = "Raw confidences (reference)";
    dEl.className = "stat-delta";
    return;
  }
  if (raw.ece == null || current.ece == null) {
    dEl.textContent = "—";
    dEl.className = "stat-delta";
    return;
  }
  const phrase = (name, rawV, curV) => {
    const d = rawV - curV; // positive = error went down
    if (Math.abs(d) < 0.0005) return `${name} unchanged`;
    return `${name} ${d > 0 ? "improved" : "worsened"} by ${Math.abs(d).toFixed(3)}`;
  };
  const dE = raw.ece - current.ece;
  const dM = raw.mce - current.mce;
  dEl.textContent = `${phrase("ECE", raw.ece, current.ece)} · ${phrase("MCE", raw.mce, current.mce)} vs raw`;
  dEl.className = `stat-delta ${dE > 0 && dM > 0 ? "good" : dE < 0 && dM < 0 ? "bad" : ""}`;
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

  drawImageWithBoxes("canvas-yolo", imagePath, yoloPreds, DISPLAY_MIN_CONF, confField);
  drawImageWithBoxes("canvas-rtdetr", imagePath, rtPreds, DISPLAY_MIN_CONF, confField);

  const rawYolo = method === "raw" ? yoloMetrics : summarizeImagePredictions(yoloPreds, "confidence_raw");
  const rawRt = method === "raw" ? rtMetrics : summarizeImagePredictions(rtPreds, "confidence_raw");
  const shown = preds => preds.filter(p => p[confField] >= DISPLAY_MIN_CONF).length;
  showRecalDelta("yolo", method, rawYolo, yoloMetrics, yoloPreds.length, shown(yoloPreds));
  showRecalDelta("rtdetr", method, rawRt, rtMetrics, rtPreds.length, shown(rtPreds));
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

// ---------- Auditor: condition description panel (Appendix 4) ----------
const CONDITION_INFO = {
  "None (clean)": {
    what: "Original test image with no corruption applied. This is the in-distribution baseline.",
    analog: "A well-exposed, correctly positioned, properly processed panoramic radiograph.",
  },
  "Gaussian noise": {
    what: "Random per-pixel intensity noise added to the image.",
    analog: "Quantum mottle from low-dose exposure, or electronic noise from an aging or poorly maintained sensor.",
  },
  "Motion blur": {
    what: "Directional smearing of the image along one axis.",
    analog: "Patient movement during the scan, or an unstable head position.",
  },
  "JPEG compression": {
    what: "Lossy re-compression that introduces blocking and ringing artifacts.",
    analog: "Images re-saved or transmitted at low quality between clinics, PACS exports or messaging apps.",
  },
  "Brightness variation": {
    what: "Global shift in image brightness.",
    analog: "Over- or under-exposure from wrong exposure settings or inconsistent machine calibration across clinics.",
  },
};

function updateConditionInfo() {
  const label = document.getElementById("oodCondition").value;
  const info = CONDITION_INFO[label];
  const box = document.getElementById("cond-info");
  const slider = document.getElementById("severity");
  slider.disabled = label === "None (clean)";
  if (!info) { box.innerHTML = ""; return; }
  box.innerHTML = `<b>${label}</b>${info.what}<div class="analog">Clinical analog: ${info.analog}</div>`;
}
document.getElementById("oodCondition")?.addEventListener("change", updateConditionInfo);
updateConditionInfo();