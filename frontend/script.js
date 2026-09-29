/* ==========================================================
   DentCalib UI — demo interactivity + mock data.

   IMPORTANT: every number in this file is fabricated with
   Math.random(). There is no trained YOLOv8 / RT-DETR model
   behind this UI, and the thesis has no completed results yet
   (Chapters 1-3 only — Methodology, no Results chapter, and the
   result tables in the appendix are unfilled templates). Do not
   treat any value rendered by this file as a real finding.

   When real results exist, replace the generator functions below
   (reliabilityBins, degradationSeries, generateResultsRows, the
   hardcoded rows in renderDegTable) with a fetch() call to a JSON
   file or API that serves your actual computed metrics.
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
  const postMode = document.querySelector("#rv-toggle .toggle-opt.active")?.dataset.mode === "post";

  const rawBins = lookupBins(model, condition, "raw");
  const recalMethod = postMode ? "dcn" : "raw";
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
    `RD deviation RMSD: raw ${rmsdRow} → post-${recalMethod.toUpperCase()} ${rmsdRecal} · ${mapNote}`;
}

document.getElementById("rv-toggle")?.addEventListener("click", e => {
  const opt = e.target.closest(".toggle-opt");
  if (!opt) return;
  document.querySelectorAll("#rv-toggle .toggle-opt").forEach(o => o.classList.remove("active"));
  opt.classList.add("active");
  refreshReliabilityViewer();
});

["rv-model", "rv-condition", "rv-severity"].forEach(id => {
  document.getElementById(id)?.addEventListener("change", refreshReliabilityViewer);
});

// initial render once both data sources are ready
loadCalibrationData().then(() => refreshReliabilityViewer());

// ---------- Page 4: OOD degradation curves ----------
function degradationSeries() {
  const severities = [1, 2, 3, 4, 5];
  let ece = 0.09, map = 0.42;
  const eceSeries = [], mapSeries = [];
  severities.forEach(s => {
    ece += rand(0.02, 0.06);
    map -= rand(0.03, 0.07);
    eceSeries.push(ece);
    mapSeries.push(Math.max(0.05, map));
  });
  return { severities, eceSeries, mapSeries };
}

let degChart;
function renderDegradationChart() {
  const { severities, eceSeries, mapSeries } = degradationSeries();
  const ctx = document.getElementById("chart-degradation");
  if (degChart) degChart.destroy();
  degChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: severities,
      datasets: [
        {
          label: "ECE",
          data: eceSeries,
          borderColor: "#C4501E",
          backgroundColor: "#C4501E",
          yAxisID: "y",
          tension: 0.2,
        },
        {
          label: "mAP@0.50",
          data: mapSeries,
          borderColor: "#1E6E63",
          backgroundColor: "#1E6E63",
          borderDash: [5, 3],
          yAxisID: "y1",
          tension: 0.2,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { title: { display: true, text: "OOD severity" }, grid: { display: false } },
        y: { position: "left", title: { display: true, text: "ECE" }, grid: { color: "#EEEBE1" } },
        y1: { position: "right", title: { display: true, text: "mAP@0.50" }, grid: { display: false } },
      },
      plugins: { legend: { position: "top", align: "end" } },
    },
  });
}
renderDegradationChart();
document.getElementById("deg-update")?.addEventListener("click", renderDegradationChart);

function renderDegTable() {
  const rows = [
    { model: "YOLOv8", rho: -0.94, sev: "S3", ece: 0.512, map: 0.198, bad: true },
    { model: "RT-DETR", rho: -0.81, sev: "S4", ece: 0.274, map: 0.224, bad: false },
  ];
  const tbody = document.querySelector("#deg-table tbody");
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.model}</td>
      <td>${r.rho.toFixed(2)}</td>
      <td>${r.sev}</td>
      <td>${r.ece.toFixed(3)}</td>
      <td>${r.map.toFixed(3)}</td>
      <td class="${r.bad ? "status-bad" : "status-good"}">${r.bad ? "Decoupled — dangerous" : "Coupled degradation"}</td>
    </tr>`).join("");
}
renderDegTable();

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

  document.getElementById("avg-ece").textContent = eceVals.length ? fmt(avg(eceVals)) : "—";
  document.getElementById("avg-mce").textContent = fmt(avg(filtered.map(r => r.mce)));
  document.getElementById("avg-dece").textContent = fmt(avg(filtered.map(r => r.dece)));
  document.getElementById("avg-map").textContent = fmt(avg(filtered.map(r => r.map50)));
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
  document.querySelectorAll(".opg-preview .placeholder").forEach(p => (p.style.display = "none"));
  runAuditorInference();
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
  });
});

loadTestImageList();

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