/* ==========================================================
   DentCalib UI — Confidence Calibration Benchmark for
   Impacted Tooth Detection (YOLOv8 / RT-DETR / DentalCalib-Net).

   Data sources: precomputed calibration metrics, reliability
   bins and per-image predictions, exported from the thesis's
   Stage 5 pipeline (see /data and /images). Pages 1-3
   (Live OPG Auditor, Benchmark Explorer, Reliability Viewer) read
   real data via the loaders below (loadCalibrationData,
   loadReliabilityBins, loadPerImagePredictions,
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

// ---------- Collapsible sidebar ----------
(() => {
  const app = document.getElementById("app");
  const toggle = document.getElementById("sidebarToggle");
  const KEY = "dentcalib.sidebarCollapsed";

  const setCollapsed = collapsed => {
    app.classList.toggle("sidebar-collapsed", collapsed);
    const label = collapsed ? "Expand sidebar" : "Collapse sidebar";
    toggle.setAttribute("aria-expanded", String(!collapsed));
    toggle.setAttribute("aria-label", label);
    toggle.title = label;
    try { localStorage.setItem(KEY, collapsed ? "1" : "0"); } catch {}
  };

  let saved = false;
  try { saved = localStorage.getItem(KEY) === "1"; } catch {}
  setCollapsed(saved);

  toggle.addEventListener("click", () =>
    setCollapsed(!app.classList.contains("sidebar-collapsed"))
  );
  // let Chart.js re-measure once the width transition finishes
  document.getElementById("sidebar").addEventListener("transitionend", e => {
    if (e.propertyName === "width") window.dispatchEvent(new Event("resize"));
  });
})();

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

// One colour per recalibration method, used on every page (keep in sync with
// --m-raw / --m-ts / --m-dcn in style.css).
const METHOD_COLORS = { raw: "#C4501E", ts: "#3F5F8F", dcn: "#1E6E63" };

// Mix a hex colour towards white (amount 0..1); used to tell RT-DETR from YOLOv8
// on charts where both models share the selected method's colour.
function tint(hex, amount) {
  const n = parseInt(hex.slice(1), 16);
  const mix = c => Math.round(c + (255 - c) * amount);
  return `rgb(${mix(n >> 16)}, ${mix((n >> 8) & 255)}, ${mix(n & 255)})`;
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

reliabilityChart("chart-rel-yolo", [], METHOD_COLORS.raw);
reliabilityChart("chart-rel-rtdetr", [], METHOD_COLORS.raw);

// Page 3: reliability viewer chart objects (created once, updated by refreshReliabilityViewer)
let rvRawChart = reliabilityChart("chart-rv-raw", [], METHOD_COLORS.raw);
let rvRecalChart = reliabilityChart("chart-rv-recal", [], METHOD_COLORS.dcn);

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

  const recalTitle = document.getElementById("rv-recal-title");
  recalTitle.textContent = `${methodLabel} — after recalibration`;
  recalTitle.dataset.method = recalMethod;
  const recalSeries = rvRecalChart.data.datasets[1];
  recalSeries.borderColor = recalSeries.backgroundColor = METHOD_COLORS[recalMethod];

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

// Status follows Table 18 (data_prep/build_table_18.py):
//   p >= alpha           -> Stable
//   p <  alpha, rho <  0 -> Coupled
//   p <  alpha, rho >= 0 -> Decoupled
// "Stable" is the authors' own label, not defined in the proposal text.
function decouplingStatus(rho, p) {
  if (rho == null) return { decoupled: false, significant: false, stable: false, label: "—" };
  const significant = p != null && p < ALPHA;
  if (!significant) return { decoupled: false, significant, stable: true, label: "Stable" };
  const decoupled = rho >= 0;
  return { decoupled, significant, stable: false, label: decoupled ? "Decoupled" : "Coupled" };
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

// Severity-range filter (Appendix 4). Affects the plotted points only.
// Returns indices into DEG_SEVERITIES; swaps the ends if From > To.
function degRange() {
  const a = DEG_SEVERITIES.indexOf(Number(document.getElementById("deg-sev-from")?.value ?? DEG_SEVERITIES[0]));
  const b = DEG_SEVERITIES.indexOf(Number(document.getElementById("deg-sev-to")?.value ?? DEG_SEVERITIES[DEG_SEVERITIES.length - 1]));
  const lo = Math.max(0, Math.min(a, b));
  const hi = Math.max(a, b) < 0 ? DEG_SEVERITIES.length - 1 : Math.max(a, b);
  return { lo, hi };
}

async function renderDegradationChart() {
  await loadCalibrationData();

  const models = degModelsSelected(document.getElementById("deg-model").value);
  const corruption = degCorruptionKey(document.getElementById("deg-corruption").value);
  const method = degRecalMethod();
  const colors = { yolov8: METHOD_COLORS[method], rtdetr: tint(METHOD_COLORS[method], 0.5) };
  const pointStyles = { yolov8: "circle", rtdetr: "rectRot" };
  const datasets = [];
  const markers = [];
  const { lo, hi } = degRange();
  const minSev = DEG_SEVERITIES[lo], maxSev = DEG_SEVERITIES[hi];

  models.forEach(model => {
    const { ece, map } = degSeries(model, corruption, method);
    const label = model === "yolov8" ? "YOLOv8" : "RT-DETR";

    datasets.push({
      label: `${label} ECE`,
      data: ece.slice(lo, hi + 1),
      borderColor: colors[model],
      backgroundColor: colors[model],
      pointStyle: pointStyles[model],
      yAxisID: "y",
      tension: 0.2,
    });
    datasets.push({
      label: `${label} mAP@0.50`,
      data: map.slice(lo, hi + 1),
      borderColor: colors[model],
      backgroundColor: colors[model],
      borderDash: [5, 3],
      pointStyle: pointStyles[model],
      yAxisID: "y1",
      tension: 0.2,
    });

    // The decoupling marker is drawn only for curves whose status is Decoupled,
    // and only if its severity is inside the plotted range.
    const { rho, p } = degSpearman(ece, map);
    if (decouplingStatus(rho, p).decoupled) {
      const severity = computeDecouplingSeverity(ece, map, DEG_SEVERITIES);
      if (severity != null && severity >= minSev && severity <= maxSev) {
        markers.push({ severity, color: colors[model], text: `${label}: decoupling point (S${severity})` });
      }
    }
  });

  const ctx = document.getElementById("chart-degradation");
  if (degChart) degChart.destroy();
  degChart = new Chart(ctx, {
    type: "line",
    data: { labels: DEG_SEVERITIES.slice(lo, hi + 1), datasets },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      onHover: (e, items) => { e.native.target.style.cursor = items.length ? "pointer" : "default"; },
      onClick: (e, items, chart) => {
        const hit = chart.getElementsAtEventForMode(e, "index", { intersect: false }, true)[0];
        if (!hit) return;
        const severity = chart.data.labels[hit.index];
        const corruptionLabel = document.getElementById("deg-corruption").value;
        const recal = { raw: "None (raw)", ts: "Temperature scaling", dcn: "DentalCalib-Net" }[degRecalMethod()];
        location.hash = pageHash("auditor", {
          oodCondition: corruptionLabel,
          severity,
          recalMethod: recal,
        });
      },
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
      <td class="${r.status.decoupled ? "status-bad" : r.status.stable ? "" : "status-good"}">${r.status.label}</td>
    </tr>`).join("");

  document.getElementById("deg-note").innerHTML = `
    <b>Basis:</b> all values use <b>${DEG_METHOD_NAMES[method]}</b> confidences.
    ρ is the Spearman rank correlation between mAP@0.50 and ECE across S1–S5;
    p is two-sided (scipy.stats.spearmanr). With five severity levels,
    |ρ| ≥ 0.878 is needed for significance at α = ${ALPHA}.
    The severity range filter changes only the plotted curves; ρ, p and the
    status above always use S1–S5.<br />
    <b>Status (as in Table 18):</b> p ≥ α is <b>Stable</b> (no statistically detectable
    mAP–ECE relationship; this label is the authors' own and is not defined in the
    proposal text). With p &lt; α, a negative ρ is <b>Coupled</b> (accuracy and calibration
    degrade together) and a non-negative ρ is <b>Decoupled</b> (calibration degrades
    independently of accuracy, the dangerous failure mode).<br />
    <b>Decoupling severity</b> is a visual aid, not a measure defined in the proposal:
    the severity step where the normalized ECE and mAP trends diverge most. It is
    shown only for Decoupled curves.`;
}

function updateDegradationPage() {
  renderDegradationChart();
  renderDegTable();
}

// Filters apply as soon as they change; there is no separate "Update chart" button.
["deg-model", "deg-corruption", "deg-sev-from", "deg-sev-to"].forEach(id => {
  document.getElementById(id)?.addEventListener("change", updateDegradationPage);
});

document.getElementById("deg-recalib-toggle")?.addEventListener("click", e => {
  const opt = e.target.closest(".toggle-opt");
  if (!opt) return;
  document.querySelectorAll("#deg-recalib-toggle .toggle-opt").forEach(o => o.classList.remove("active"));
  opt.classList.add("active");
  updateDegradationPage();
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

// Optimized Temperature Scaling parameter, one per model, learned on the
// validation logits (stage4_outputs/temperature_scaling_params.json).
const TEMPERATURE_LOOKUP = { yolov8: 1.5669029187255183, rtdetr: 0.9256808418850526 };

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

// Method name with its colour dot (same colours as the table headers and charts)
const methodTag = (m, value) =>
  `<span class="m-tag" data-method="${m}">${METHOD_NAMES[m]}${value != null ? " " + value : ""}</span>`;

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

// Best value per row: lowest for the calibration errors, highest for mAP.
// Ties across every shown method are not highlighted (nothing to compare).
function bestMethods(r, metric, shownMethods) {
  const vals = shownMethods
    .map(m => [m, r[`${metric}_${m}`]])
    .filter(([, v]) => v != null);
  if (vals.length < 2) return [];
  const pick = metric === "map50" ? Math.max : Math.min;
  const best = pick(...vals.map(([, v]) => v));
  const winners = vals.filter(([, v]) => v === best).map(([m]) => m);
  return winners.length === vals.length ? [] : winners;
}

const RV_CONDITION_LABELS = {
  gaussian_noise: "Gaussian noise",
  motion_blur: "Motion blur",
  brightness_variation: "Brightness variation",
  jpeg_compression: "JPEG compression",
};

const RV_METHOD_LABELS = { ts: "Temperature scaling", dcn: "DentalCalib-Net" };

// Reliability Viewer URL for this row's model / condition. The Viewer would
// otherwise fall back to Temperature scaling, so the method is always named:
// the clicked column's method, else the Explorer's recalibration filter, else
// DentalCalib-Net (the method this project is about).
function reliabilityHref(r, method) {
  const recal = document.getElementById("f-recal").value;
  const chosen =
    method ||
    (recal === "Temp. scaling" ? "ts" : "dcn");
  const params = {
    "rv-model": modelName(r.model),
    "rv-condition": r.condition === "clean" ? "Clean (baseline)" : RV_CONDITION_LABELS[r.corruption_type],
  };
  if (r.condition !== "clean") params["rv-severity"] = r.severity;
  params["rv-method"] = RV_METHOD_LABELS[chosen];
  return pageHash("reliability", params).replace(/&/g, "&amp;");
}

function reliabilityLink(r) {
  const href = reliabilityHref(r);
  const name = `${modelName(r.model)} ${conditionLabel(r)}${r.condition === "clean" ? "" : " S" + r.severity}`;
  return `<a class="row-link" href="${href}" title="Open the reliability diagram" aria-label="Open the reliability diagram for ${name}">↗</a>`;
}

function explorerRowHtml(r, metric, shownMethods, base, pinnedClass) {
  const cls = pinnedClass ? ` class="pinned-row ${pinnedClass}"` : "";
  const best = pinnedClass ? [] : bestMethods(r, metric, shownMethods);
  const val = m => {
    let text = shownMethods.includes(m) ? showVal(r[`${metric}_${m}`]) : "—";
    // TS and DCN values open that method's reliability diagram
    if (text !== "—" && m !== "raw") {
      text = `<a class="cell-link" href="${reliabilityHref(r, m)}" title="Open the ${METHOD_NAMES[m]} reliability diagram">${text}</a>`;
    }
    return best.includes(m) ? `<td class="best" title="Best ${METRIC_LABELS[metric]} in this row">${text}</td>` : `<td>${text}</td>`;
  };
  // The displayed delta follows the proposal (|OOD - clean|, signed for mAP);
  // its colour uses the signed change: orange = worse than clean, green = better.
  const dlt = m => {
    if (pinnedClass || !base || !shownMethods.includes(m)) return "<td>—</td>";
    const cur = r[`${metric}_${m}`];
    const ref = base[`${metric}_${m}`];
    const d = deltaOf(metric, cur, ref);
    if (d == null) return "<td>—</td>";
    const signed = cur - ref;
    const worse = metric === "map50" ? signed < 0 : signed > 0;
    const tone = signed === 0 ? "" : worse ? "d-bad" : "d-good";
    const tip = `${METRIC_LABELS[metric]} ${fmt(ref)} (clean) → ${fmt(cur)}`;
    return `<td class="${tone}" title="${tip}">${showDelta(metric, d)}</td>`;
  };
  const t = r.temperature != null ? r.temperature.toFixed(3) : "—";
  return `
    <tr${cls}>
      <td>${modelName(r.model)}</td>
      <td class="cond-cell">${conditionLabel(r)} ${reliabilityLink(r)}</td>
      <td>${r.condition === "clean" ? "—" : "S" + r.severity}</td>
      ${val("raw")}
      ${val("ts")}
      ${val("dcn")}
      ${dlt("raw")}
      ${dlt("ts")}
      ${dlt("dcn")}
      <td>${t}</td>
    </tr>`;
}

// Keep both tables at the same scroll position (vertical and horizontal).
function syncPaneScroll(cards) {
  cards.forEach(card => {
    if (card.dataset.synced) return;
    card.dataset.synced = "1";
    card.addEventListener("scroll", () => {
      // Writing an equal value fires no scroll event, so this cannot loop.
      cards.forEach(other => {
        if (other === card) return;
        if (other.scrollTop !== card.scrollTop) other.scrollTop = card.scrollTop;
        if (other.scrollLeft !== card.scrollLeft) other.scrollLeft = card.scrollLeft;
      });
    });
  });
}

async function renderResultsTable() {
  const raw = await loadCalibrationData();
  const pivoted = pivotByCondition(raw);
  const filtered = applyExplorerFilters(pivoted);
  const methods = visibleMethods();
  const metric = selectedMetric();
  const metricLabel = METRIC_LABELS[metric];
  const cleanByModel = {};
  pivoted.filter(r => r.condition === "clean").forEach(r => { cleanByModel[r.model] = r; });

  // "All models" splits YOLOv8 and RT-DETR into two side-by-side tables so
  // they can be compared row by row; a single model keeps one full-width table.
  const split = document.getElementById("f-model").value === "All models";

  // Column headers follow the selected metric (every pane has its own header row).
  // Split tables are narrow, so they use short headers (Raw / TS / DCN) and the
  // metric name moves into each table's title; the full name stays as a tooltip.
  METHODS.forEach(m => {
    const full = `${metricLabel} (${METHOD_NAMES[m]})`;
    const fullDelta = `Δ${metricLabel} (${METHOD_NAMES[m]})`;
    document.querySelectorAll(`.th-${m}`).forEach(th => {
      th.textContent = split ? METHOD_NAMES[m] : full;
      th.title = full;
    });
    document.querySelectorAll(`.th-d${m}`).forEach(th => {
      th.textContent = split ? `Δ${METHOD_NAMES[m]}` : fullDelta;
      th.title = fullDelta;
    });
  });

  // Pinned clean-baseline rows: always visible, all three methods shown
  const pinnedFor = models => models
    .filter(m => cleanByModel[m])
    .map((m, i) => explorerRowHtml(cleanByModel[m], metric, METHODS, null, `p${i + 1}`))
    .join("");
  const bodyFor = rows => rows.length
    ? rows.map(r => explorerRowHtml(r, metric, methods, cleanByModel[r.model], "")).join("")
    : `<tr><td colspan="10">No OOD rows match the selected filters.</td></tr>`;

  const panes = [
    { el: document.getElementById("pane-a"), models: split ? ["yolov8"] : ["yolov8", "rtdetr"], rows: split ? filtered.filter(r => r.model === "yolov8") : filtered },
    { el: document.getElementById("pane-b"), models: ["rtdetr"], rows: filtered.filter(r => r.model === "rtdetr") },
  ];
  document.getElementById("results-split").classList.toggle("is-split", split);
  panes[1].el.hidden = !split;
  panes.forEach(p => {
    const title = p.el.querySelector(".table-pane-title");
    title.hidden = !split;
    title.textContent = `${modelName(p.el.id === "pane-a" ? "yolov8" : "rtdetr")} — ${metricLabel}`;
    p.el.querySelector("tbody").innerHTML = pinnedFor(p.models) + bodyFor(p.rows);
  });
  syncPaneScroll(panes.map(p => p.el.querySelector(".table-card")));

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
        text = `${methodTag(cardMethod)}: ${c.text}`;
        cls = c.cls;
      } else {
        text = `${methodTag("raw")} · ${methodTag("ts", showVal(mean("ts")))} · ${methodTag("dcn", showVal(mean("dcn")))}`;
      }
    }
    document.getElementById(v).textContent = main != null ? fmt(main) : "—";
    document.getElementById(d).innerHTML = text;
    document.getElementById(d).className = `stat-delta ${cls}`;
  });
}

renderResultsTable();
["f-model", "f-corruption", "f-severity", "f-recal", "f-metric"].forEach(id => {
  document.getElementById(id)?.addEventListener("change", renderResultsTable);
});

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

// kind: "info" (default), "ok", "warn" or "error"; colours live in style.css
function setUploadStatus(text, kind = "info") {
  const status = document.getElementById("upload-status");
  status.textContent = text;
  status.className = `upload-sub upload-status ${kind}`;
}

// Auditor empty / loading / ready states ------------------------------
const modelCards = () => document.querySelectorAll(".model-card");
function setPreviewMessage(text) {
  document.querySelectorAll(".opg-preview .placeholder").forEach(p => (p.textContent = text));
}

async function handleUploadedFile(file) {
  try {
    await loadTestImageList();
  } catch (error) {
    selectedFilename = null;
    resetAuditorResults();
    setPreviewMessage("Auditor data unavailable");
    console.error("Auditor data is unavailable:", error);
    setUploadStatus(
      "Auditor data is missing. Copy frontend/data and frontend/images from the shared dataset, then open this page over HTTP.",
      "error"
    );
    return;
  }

  if (!testImageList.includes(file.name)) {
    selectedFilename = null;
    resetAuditorResults();
    setUploadStatus(
      `No saved results for "${file.name}". Use an image from the 201-image DENTEX test set with its original filename.`,
      "error"
    );
    setPreviewMessage("No saved results for this image");
    return;
  }
  selectedFilename = file.name;
  setUploadStatus(`✓ Uploaded: ${file.name}`, "ok");
  resetAuditorResults();
  setPreviewMessage("Ready — press “Show results”");
}

function resetAuditorResults() {
  modelCards().forEach(c => c.classList.remove("has-results", "is-loading"));
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

  reliabilityChart("chart-rel-yolo", [], METHOD_COLORS.raw);
  reliabilityChart("chart-rel-rtdetr", [], METHOD_COLORS.raw);
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
      const tag = pred.quadrant ? `Q${pred.quadrant} ` : "";
      ctx.fillText(`${tag}${conf.toFixed(2)}`, x1, Math.max(10, y1 - 4));
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

// Signed difference only (RT-DETR minus YOLOv8). No "better" verdict: per-image
// values are dominated by low-score detections, so they are not a model ranking.
function modelDifference(yoloVal, rtVal, label) {
  if (!Number.isFinite(yoloVal) || !Number.isFinite(rtVal)) {
    return { text: `${label} unavailable per image`, cls: "" };
  }
  const diff = rtVal - yoloVal;
  const sign = diff >= 0 ? "+" : "−";
  return { text: `RT-DETR − YOLOv8: ${sign}${Math.abs(diff).toFixed(3)} (this image only)`, cls: "" };
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

  const eceDelta = modelDifference(yoloMetrics.ece, rtMetrics.ece, "ECE");
  const mceDelta = modelDifference(yoloMetrics.mce, rtMetrics.mce, "MCE");
  const deceDelta = modelDifference(yoloMetrics.dece, rtMetrics.dece, "D-ECE");
  document.getElementById("s1-delta").textContent = eceDelta.text;
  document.getElementById("s1-delta").className = `stat-delta ${eceDelta.cls}`;
  document.getElementById("s2-delta").textContent = mceDelta.text;
  document.getElementById("s2-delta").className = `stat-delta ${mceDelta.cls}`;
  document.getElementById("s3-delta").textContent = deceDelta.text;
  document.getElementById("s3-delta").className = `stat-delta ${deceDelta.cls}`;

  reliabilityChart("chart-rel-yolo", binsToChartPoints(yoloMetrics.bins), METHOD_COLORS[method]);
  reliabilityChart("chart-rel-rtdetr", binsToChartPoints(rtMetrics.bins), METHOD_COLORS[method]);

  const yoloNote = calibrationInterpretation(yoloMetrics.bins);
  const rtNote = calibrationInterpretation(rtMetrics.bins);
  const yoloNoteEl = document.getElementById("yolo-chart-note");
  const rtNoteEl = document.getElementById("rtdetr-chart-note");
  yoloNoteEl.textContent = yoloNote.text;
  yoloNoteEl.className = `chart-note ${yoloNote.cls}`;
  rtNoteEl.textContent = rtNote.text;
  rtNoteEl.className = `chart-note ${rtNote.cls}`;

  drawImageWithBoxes("canvas-yolo", imagePath, yoloPreds, DISPLAY_MIN_CONF, "confidence_raw");
  drawImageWithBoxes("canvas-rtdetr", imagePath, rtPreds, DISPLAY_MIN_CONF, "confidence_raw");

  const rawYolo = method === "raw" ? yoloMetrics : summarizeImagePredictions(yoloPreds, "confidence_raw");
  const rawRt = method === "raw" ? rtMetrics : summarizeImagePredictions(rtPreds, "confidence_raw");
  const shown = preds => preds.filter(p => p.confidence_raw >= DISPLAY_MIN_CONF).length;
  showRecalDelta("yolo", method, rawYolo, yoloMetrics, yoloPreds.length, shown(yoloPreds));
  showRecalDelta("rtdetr", method, rawRt, rtMetrics, rtPreds.length, shown(rtPreds));
}

document.getElementById("runInference")?.addEventListener("click", () => {
  const btn = document.getElementById("runInference");

  // Nothing chosen yet: say so and point at the drop zone instead of doing nothing.
  if (!selectedFilename) {
    setUploadStatus("Choose an image from the DENTEX test set first.", "warn");
    dropzone.classList.remove("nudge");
    void dropzone.offsetWidth; // restart the animation
    dropzone.classList.add("nudge");
    dropzone.scrollIntoView({ behavior: "smooth", block: "center" });
    return;
  }

  const originalText = btn.textContent;
  btn.textContent = "Loading…";
  btn.disabled = true;
  modelCards().forEach(c => c.classList.add("is-loading"));
  runAuditorInference().then(() => {
    modelCards().forEach(c => c.classList.add("has-results"));
  }).catch(error => {
    console.error("Auditor results could not be loaded:", error);
    setUploadStatus(`Could not load the selected condition's saved results: ${error.message}`, "error");
  }).finally(() => {
    btn.textContent = originalText;
    btn.disabled = false;
    modelCards().forEach(c => c.classList.remove("is-loading"));
    document.querySelector(".card-title")?.scrollIntoView({ behavior: "smooth", block: "start" });
  });
});

loadTestImageList().catch(error => {
  console.error("Auditor data is unavailable:", error);
  setUploadStatus(
    "Auditor data is missing. Copy frontend/data and frontend/images from the shared dataset, then open this page over HTTP.",
    "error"
  );
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
// Build an in-app link: pageHash("reliability", { "rv-model": "YOLOv8" }) -> "#reliability?rv-model=YOLOv8"
function pageHash(page, params = {}) {
  const q = new URLSearchParams(params).toString();
  return `#${page}${q ? "?" + q : ""}`;
}

// ---------- Navigation: page + filter state live in the URL hash ----------
// Format: #explorer?f-model=YOLOv8&f-metric=MCE. Only values that differ from
// the page defaults are written, so links stay short. Back/forward, reloads and
// shared links all restore the same page and filters.
(() => {
  const PAGE_CONTROLS = {
    auditor: ["oodCondition", "severity", "recalMethod"],
    explorer: ["f-model", "f-corruption", "f-severity", "f-recal", "f-metric"],
    reliability: ["rv-model", "rv-condition", "rv-severity", "rv-method"],
    degradation: ["deg-model", "deg-corruption", "deg-sev-from", "deg-sev-to"],
    pipeline: [],
  };
  const REFRESH = {
    auditor: params => {
      updateConditionInfo();
      // arriving from a link (e.g. a Degradation point) with an image already
      // chosen: show that condition straight away
      if (params.toString() && selectedFilename) el("runInference")?.click();
      else if (params.toString()) {
        setUploadStatus("Choose an image to view the selected condition.");
      }
    },
    explorer: () => renderResultsTable(),
    reliability: () => refreshReliabilityViewer(),
    degradation: () => updateDegradationPage(),
  };
  const DEFAULT_PAGE = "auditor";
  const DEG_MODE = "deg-mode"; // the Raw / TS / DCN toggle, not a form control

  const el = id => document.getElementById(id);
  const degMode = () =>
    document.querySelector("#deg-recalib-toggle .toggle-opt.active")?.dataset.mode || "raw";

  const defaults = {};
  Object.values(PAGE_CONTROLS).flat().forEach(id => { defaults[id] = el(id)?.value; });
  defaults[DEG_MODE] = degMode();

  let currentPage = DEFAULT_PAGE;

  function showPage(name) {
    currentPage = name;
    navItems.forEach(b => {
      const on = b.dataset.page === name;
      b.classList.toggle("active", on);
      if (on) b.setAttribute("aria-current", "page");
      else b.removeAttribute("aria-current");
    });
    pages.forEach(p => p.classList.toggle("active", p.id === `page-${name}`));
  }

  function hashFor(page) {
    const params = new URLSearchParams();
    PAGE_CONTROLS[page].forEach(id => {
      const v = el(id)?.value;
      if (v != null && v !== defaults[id]) params.set(id, v);
    });
    if (page === "degradation" && degMode() !== defaults[DEG_MODE]) params.set(DEG_MODE, degMode());
    const q = params.toString();
    return `#${page}${q ? "?" + q : ""}`;
  }

  function writeHash(push) {
    const hash = hashFor(currentPage);
    if (hash === location.hash || (hash === `#${DEFAULT_PAGE}` && !location.hash)) return;
    history[push ? "pushState" : "replaceState"](null, "", hash);
  }

  function parseHash() {
    const [rawPage, query = ""] = location.hash.slice(1).split("?");
    const page = rawPage in PAGE_CONTROLS ? rawPage : DEFAULT_PAGE;
    return { page, params: new URLSearchParams(query) };
  }

  function applyHash() {
    const { page, params } = parseHash();
    showPage(page);
    PAGE_CONTROLS[page].forEach(id => {
      const node = el(id);
      if (!node) return;
      const wanted = params.get(id) ?? defaults[id];
      // ignore values that are not valid options (stale or hand-edited links)
      if (node.tagName === "SELECT" && ![...node.options].some(o => o.value === wanted)) return;
      node.value = wanted;
    });
    if (page === "degradation") {
      const mode = params.get(DEG_MODE) ?? defaults[DEG_MODE];
      document.querySelectorAll("#deg-recalib-toggle .toggle-opt").forEach(o =>
        o.classList.toggle("active", o.dataset.mode === mode)
      );
    }
    // data-backed pages need their saved results before they can draw
    loadCalibrationData().catch(() => {}).finally(() => REFRESH[page]?.(params));
  }

  navItems.forEach(btn => {
    btn.addEventListener("click", () => {
      showPage(btn.dataset.page);
      writeHash(true);
    });
  });

  // keep the URL in step with the filters (replace, so Back still means "previous page")
  ["change", "input"].forEach(evt =>
    document.addEventListener(evt, e => {
      if (PAGE_CONTROLS[currentPage].includes(e.target.id)) writeHash(false);
    })
  );
  el("deg-recalib-toggle")?.addEventListener("click", () => writeHash(false));

  window.addEventListener("popstate", applyHash);
  window.addEventListener("hashchange", applyHash);

  if (location.hash) applyHash();
})();
