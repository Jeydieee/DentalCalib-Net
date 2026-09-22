from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from calibration_metrics import (
    compute_ece,
    compute_mce,
    compute_dece,
    compute_rmsd,
    bin_statistics,
)

MODELS = ["yolov8", "rtdetr"]
CONFIDENCE_TYPES = {
    "raw": "confidence_raw",
    "ts": "confidence_ts",
    "dcn": "confidence_dcn",
}
QUADRANTS = [1, 2, 3, 4]
N_BINS = 15
MIN_BIN_COUNT = 1

QUADRANT_LABELED_DIR = r"D:\DentalCalib-Net\stage4_outputs\quadrant_labeled_predictions"
OUT_JSON = r"D:\DentalCalib-Net\stage5_outputs\table_10_per_quadrant.json"
OUT_CSV = r"D:\DentalCalib-Net\stage5_outputs\table_10_per_quadrant.csv"

COLUMNS = [
    "model", "quadrant", "confidence_type",
    "ece", "mce", "dece", "rmsd",
    "n_predictions", "n_bins_populated",
]


def load_quadrant_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_by_quadrant(data):
    """Flatten all predictions across images, grouped by quadrant."""
    by_quadrant = {q: {"conf": {k: [] for k in CONFIDENCE_TYPES.values()},
                        "labels": [], "ious": []} for q in QUADRANTS}

    for entry in data:
        for pred in entry["predictions"]:
            q = pred["quadrant"]
            if q not in by_quadrant:
                continue
            label = pred.get("label")
            iou = pred.get("best_iou")
            if label is None or iou is None:
                continue
            for k in CONFIDENCE_TYPES.values():
                by_quadrant[q]["conf"][k].append(float(pred[k]))
            by_quadrant[q]["labels"].append(int(label))
            by_quadrant[q]["ious"].append(float(iou))

    return by_quadrant


def score_quadrant(conf_dict, labels, ious):
    rows = []
    for short, key in CONFIDENCE_TYPES.items():
        conf = np.asarray(conf_dict[key], dtype=np.float64)
        lab = np.asarray(labels, dtype=np.int64)
        iou = np.asarray(ious, dtype=np.float64)

        stats = bin_statistics(conf, lab, N_BINS, MIN_BIN_COUNT)
        ece = compute_ece(conf, lab, N_BINS, MIN_BIN_COUNT)
        mce = compute_mce(conf, lab, N_BINS, MIN_BIN_COUNT)
        dece = compute_dece(conf, lab, iou, N_BINS, MIN_BIN_COUNT)
        rmsd = compute_rmsd(conf, lab, N_BINS, MIN_BIN_COUNT)

        rows.append({
            "confidence_type": short,
            "ece": ece,
            "mce": mce,
            "dece": dece,
            "rmsd": rmsd,
            "n_predictions": int(lab.size),
            "n_bins_populated": stats["n_bins_populated"],
        })
    return rows


def main():
    results = []
    failures = []

    for model in MODELS:
        path = f"{QUADRANT_LABELED_DIR}/{model}_clean_quadrant.json"
        print(f"\n=== {model} (clean, per-quadrant) ===")

        try:
            data = load_quadrant_file(path)
        except FileNotFoundError:
            print(f"  MISSING: {path}")
            failures.append((model, "file not found"))
            continue

        by_quadrant = collect_by_quadrant(data)

        for q in QUADRANTS:
            q_data = by_quadrant[q]
            n = len(q_data["labels"])
            if n == 0:
                print(f"  quadrant {q}: 0 predictions, skipping")
                failures.append((f"{model} quadrant {q}", "no predictions"))
                continue

            try:
                rows = score_quadrant(q_data["conf"], q_data["labels"], q_data["ious"])
            except Exception as exc:
                print(f"  quadrant {q}: FAILED ({type(exc).__name__}: {exc})")
                failures.append((f"{model} quadrant {q}", str(exc)))
                continue

            for r in rows:
                results.append({"model": model, "quadrant": q, **r})
                sparse_flag = " <-- SPARSE" if r["n_bins_populated"] < N_BINS // 2 else ""
                print(f"  quadrant {q} [{r['confidence_type']}]: "
                      f"n={r['n_predictions']:4d}  ece={r['ece']:.4f}  "
                      f"mce={r['mce']:.4f}  dece={r['dece']:.4f}  "
                      f"rmsd={r['rmsd']:.4f}  bins={r['n_bins_populated']}/{N_BINS}{sparse_flag}")

    if results:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k) for k in COLUMNS})
        print(f"\nwrote {OUT_JSON}")
        print(f"wrote {OUT_CSV}")

    print(f"\nrows computed: {len(results)} / {len(MODELS) * len(QUADRANTS) * len(CONFIDENCE_TYPES)}")
    if failures:
        print(f"\nFAILURES/WARNINGS ({len(failures)}):")
        for f_, why in failures:
            print(f"  {f_}: {why}")


if __name__ == "__main__":
    main()