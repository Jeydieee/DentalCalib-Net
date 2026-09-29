CSV_PATH = r"D:\DentalCalib-Net\stage5_outputs\calibration_results.csv"

import csv
import numpy as np

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

d = {(r["model"], r["condition"], r["confidence_type"]): r for r in rows}

CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness_variation", "jpeg_compression"]
SEVERITIES = [1, 2, 3, 4, 5]
MODELS = ["yolov8", "rtdetr"]


def get(model, condition, metric):
    return float(d[(model, condition, "ts")][metric])


print("Table 13: Reliability Diagram Deviation (RMSD) of YOLOv8 and RT-DETR Under TS")
print()

header = f'{"Model":10s} {"OOD Condition":22s} | {"Baseline":>9s} | {"S1":>7s}{"S2":>7s}{"S3":>7s}{"S4":>7s}{"S5":>7s}'
print(header)

for model in MODELS:
    baseline = get(model, "clean", "rmsd")
    print(f"--- {model} (baseline RMSD: {baseline:.4f}) ---")

    all_severity_vals = {s: [] for s in SEVERITIES}
    for corr in CORRUPTIONS:
        label = corr.replace("_", " ").title()
        row = f"{'':10s} {label:22s} | {baseline:9.4f} |"
        for s in SEVERITIES:
            val = get(model, f"{corr}_S{s}", "rmsd")
            row += f"{val:7.4f}"
            all_severity_vals[s].append(val)
        print(row)

    avg_row = f"{'':10s} {'Average (All 4 OOD)':22s} | {baseline:9.4f} |"
    for s in SEVERITIES:
        avg_row += f"{np.mean(all_severity_vals[s]):7.4f}"
    print(avg_row)
    print()