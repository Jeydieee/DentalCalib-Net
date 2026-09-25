CSV_PATH = r"D:\DentalCalib-Net\stage5_outputs\calibration_results.csv"

import csv
import numpy as np
from scipy import stats

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

d = {(r["model"], r["condition"], r["confidence_type"]): r for r in rows}

CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness_variation", "jpeg_compression"]
SEVERITIES = [1, 2, 3, 4, 5]
OOD_CONDITIONS = [f"{c}_S{s}" for c in CORRUPTIONS for s in SEVERITIES]
MODELS = ["yolov8", "rtdetr"]
METRICS = ["ece", "mce", "dece"]
METHODS = {"ts": "Temperature Scaling", "dcn": "DentalCalib-Net"}


def get(model, condition, ct, metric):
    return float(d[(model, condition, ct)][metric])


print("Table 20: Wilcoxon Signed-Rank Test -- Pre vs Post Recalibration, COCO-C OOD Conditions (n=20)")
print()

header = f'{"Model":10s} {"Method":20s} | {"ECE W":>8s}{"p":>9s}{"Sig":>6s} | {"MCE W":>8s}{"p":>9s}{"Sig":>6s} | {"D-ECE W":>8s}{"p":>9s}{"Sig":>6s}'
print(header)

for model in MODELS:
    for method_key, method_label in METHODS.items():
        row = f"{model:10s} {method_label:20s} |"
        for metric in METRICS:
            raw_vals = np.array([get(model, c, "raw", metric) for c in OOD_CONDITIONS])
            post_vals = np.array([get(model, c, method_key, metric) for c in OOD_CONDITIONS])
            diff = raw_vals - post_vals
            if np.allclose(diff, 0.0):
                stat, p = 0.0, 1.0
            else:
                stat, p = stats.wilcoxon(raw_vals, post_vals)
            sig = "Yes" if p < 0.05 else "No"
            row += f" {stat:8.2f}{p:9.4f}{sig:>6s} |"
        print(row)