CSV_LABELED_DIR = r"D:\DentalCalib-Net\stage4_outputs\labeled_predictions"
CSV_RECAL_DIR = r"D:\DentalCalib-Net\stage4_outputs\recalibrated_predictions"
OUT_PATH = r"D:\DentalCalib-Net\stage5_outputs\nll_results.csv"

import json
import csv
import numpy as np
from pathlib import Path

CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness_variation", "jpeg_compression"]
SEVERITIES = [1, 2, 3, 4, 5]
MODELS = ["yolov8", "rtdetr"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def compute_nll(confidences, labels, eps=1e-7):
    p = np.clip(np.asarray(confidences, dtype=np.float64), eps, 1 - eps)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def get_condition_files(model, condition):
    if condition == "clean":
        if model == "yolov8":
            labeled = f"{CSV_LABELED_DIR}/yolov8_predictions_clean_labeled.json"
        else:
            labeled = f"{CSV_LABELED_DIR}/rtdetr_predictions_clean_filtered_labeled.json"
    else:
        if model == "yolov8":
            labeled = f"{CSV_LABELED_DIR}/yolov8_{condition}_labeled.json"
        else:
            labeled = f"{CSV_LABELED_DIR}/rtdetr_{condition}_filtered_labeled.json"
    recal = f"{CSV_RECAL_DIR}/{model}_{condition.replace('_S', '_S') if condition != 'clean' else 'clean'}_recalibrated.json"
    if condition != "clean":
        recal = f"{CSV_RECAL_DIR}/{model}_{condition}_recalibrated.json"
    return labeled, recal


def process_condition(model, condition):
    labeled_path, recal_path = get_condition_files(model, condition)
    labeled = load_json(labeled_path)
    recal = load_json(recal_path)
    recal_by_img = {e["image_path"]: e["predictions"] for e in recal}

    raw_conf, ts_conf, labels = [], [], []
    for entry in labeled:
        img = entry["image_path"]
        rec_preds = recal_by_img.get(img)
        if rec_preds is None:
            continue
        for lp, rp in zip(entry["predictions"], rec_preds):
            raw_conf.append(rp["confidence_raw"])
            ts_conf.append(rp["confidence_ts"])
            labels.append(lp["label"])

    return {
        "model": model,
        "condition": condition,
        "n_predictions": len(labels),
        "nll_raw": compute_nll(raw_conf, labels),
        "nll_ts": compute_nll(ts_conf, labels),
    }


if __name__ == "__main__":
    results = []
    for model in MODELS:
        conditions = ["clean"] + [f"{c}_S{s}" for c in CORRUPTIONS for s in SEVERITIES]
        for cond in conditions:
            r = process_condition(model, cond)
            results.append(r)
            print(f"{model:8s} {cond:25s} n={r['n_predictions']:5d}  "
                  f"NLL_raw={r['nll_raw']:.4f}  NLL_ts={r['nll_ts']:.4f}")

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "condition", "n_predictions", "nll_raw", "nll_ts"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nwrote {OUT_PATH}")