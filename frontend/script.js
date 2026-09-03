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

// Page 1: reliability diagrams
reliabilityChart("chart-rel-yolo", reliabilityBins(true), "#C4501E");
reliabilityChart("chart-rel-rtdetr", reliabilityBins(false), "#2F7D4F");

// Page 3: reliability viewer (raw vs recalibrated)
let rvRawChart = reliabilityChart("chart-rv-raw", reliabilityBins(true), "#C4501E");
let rvRecalChart = reliabilityChart("chart-rv-recal", reliabilityBins(false), "#1E6E63");

document.getElementById("rv-toggle")?.addEventListener("click", e => {
  const opt = e.target.closest(".toggle-opt");
  if (!opt) return;
  document.querySelectorAll("#rv-toggle .toggle-opt").forEach(o => o.classList.remove("active"));
  opt.classList.add("active");
  // Regenerate curves to simulate a refresh
  rvRawChart.data.datasets[1].data = reliabilityBins(true);
  rvRawChart.update();
  rvRecalChart.data.datasets[1].data = reliabilityBins(opt.dataset.mode === "post" ? false : true);
  rvRecalChart.update();
});

["rv-model", "rv-condition", "rv-severity"].forEach(id => {
  document.getElementById(id)?.addEventListener("change", () => {
    rvRawChart.data.datasets[1].data = reliabilityBins(true);
    rvRawChart.update();
    rvRecalChart.data.datasets[1].data = reliabilityBins(false);
    rvRecalChart.update();
    document.getElementById("rv-raw-ece").textContent = fmt(rand(0.14, 0.24));
    document.getElementById("rv-recal-ece").textContent = fmt(rand(0.04, 0.09));
  });
});

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

// ---------- Page 2: benchmark explorer table ----------
const CONDITIONS = ["Gaussian noise", "Motion blur", "JPEG compression", "Pixel exposure"];
const MODELS = ["YOLOv8", "RT-DETR"];

function generateResultsRows(n = 10) {
  const rows = [];
  for (let i = 0; i < n; i++) {
    const model = MODELS[i % 2];
    const cond = CONDITIONS[i % CONDITIONS.length];
    const sev = "S" + (1 + (i % 5));
    const eceRaw = rand(0.08, 0.25);
    const eceTS = eceRaw - rand(0.02, 0.06);
    const ecePre = eceRaw + rand(0.0, 0.03);
    const eceDCN = eceRaw - rand(0.06, 0.12);
    rows.push({
      model, cond, sev,
      eceRaw, eceTS, ecePre, eceDCN,
      mce: eceRaw + rand(0.05, 0.15),
      dece: eceRaw + rand(0.0, 0.05),
      map: rand(0.18, 0.36),
      optT: rand(1.0, 2.2),
    });
  }
  return rows;
}

function renderResultsTable() {
  const tbody = document.querySelector("#results-table tbody");
  const rows = generateResultsRows();
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.model}</td>
      <td>${r.cond}</td>
      <td>${r.sev}</td>
      <td>${fmt(r.eceRaw)}</td>
      <td>${fmt(r.eceTS)}</td>
      <td>${fmt(r.ecePre)}</td>
      <td>${fmt(r.eceDCN)}</td>
      <td>${fmt(r.mce)}</td>
      <td>${fmt(r.dece)}</td>
      <td>${r.map.toFixed(2)}</td>
      <td>${r.optT.toFixed(2)}</td>
    </tr>`).join("");
}
renderResultsTable();

document.getElementById("applyFilters")?.addEventListener("click", () => {
  renderResultsTable();
  document.getElementById("avg-ece").textContent = fmt(rand(0.1, 0.2));
  document.getElementById("avg-mce").textContent = fmt(rand(0.2, 0.32));
  document.getElementById("avg-dece").textContent = fmt(rand(0.14, 0.24));
  document.getElementById("avg-map").textContent = fmt(rand(0.22, 0.4));
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
  if (file) loadPreview(file);
});
fileInput.addEventListener("change", e => {
  const file = e.target.files[0];
  if (file) loadPreview(file);
});

function loadPreview(file) {
  const url = URL.createObjectURL(file);
  ["img-yolo", "img-rtdetr"].forEach(id => {
    const img = document.getElementById(id);
    img.src = url;
    img.hidden = false;
    img.previousElementSibling?.remove; // no-op, placeholder stays hidden via CSS below
  });
  document.querySelectorAll(".opg-preview .placeholder").forEach(p => (p.style.display = "none"));
}

document.getElementById("runInference")?.addEventListener("click", () => {
  const btn = document.getElementById("runInference");
  const originalText = btn.textContent;
  btn.textContent = "Running…";
  btn.disabled = true;

  setTimeout(() => {
    const yoloECE = rand(0.1, 0.22);
    const rtECE = rand(0.05, 0.14);
    const yoloMCE = yoloECE + rand(0.08, 0.16);
    const rtMCE = rtECE + rand(0.05, 0.1);
    const yoloDECE = yoloECE + rand(0.02, 0.06);
    const rtDECE = rtECE + rand(0.02, 0.05);

    document.getElementById("yolo-ece").textContent = fmt(yoloECE);
    document.getElementById("yolo-mce").textContent = fmt(yoloMCE);
    document.getElementById("yolo-dece").textContent = fmt(yoloDECE);
    document.getElementById("rtdetr-ece").textContent = fmt(rtECE);
    document.getElementById("rtdetr-mce").textContent = fmt(rtMCE);
    document.getElementById("rtdetr-dece").textContent = fmt(rtDECE);

    document.getElementById("s1-yolo").textContent = fmt(yoloECE);
    document.getElementById("s1-rt").textContent = fmt(rtECE);
    document.getElementById("s2-yolo").textContent = fmt(yoloMCE);
    document.getElementById("s2-rt").textContent = fmt(rtMCE);
    document.getElementById("s3-yolo").textContent = fmt(yoloDECE);
    document.getElementById("s3-rt").textContent = fmt(rtDECE);

    reliabilityChart("chart-rel-yolo", reliabilityBins(true), "#C4501E");
    reliabilityChart("chart-rel-rtdetr", reliabilityBins(false), "#2F7D4F");

    btn.textContent = originalText;
    btn.disabled = false;
  }, 700);
});