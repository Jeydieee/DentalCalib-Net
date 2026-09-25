CSV_PATH = r"D:\DentalCalib-Net\stage5_outputs\calibration_results.csv"
MODEL = "rtdetr"  # Table 11 = yolov8, Table 12 = rtdetr

import csv
import numpy as np
from scipy import stats

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

d = {(r["model"], r["condition"], r["confidence_type"]): r for r in rows}

CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness_variation", "jpeg_compression"]
SEVERITIES = [1, 2, 3, 4, 5]


def get(condition, metric):
    return float(d[(MODEL, condition, "ts")][metric])


baseline_ece = get("clean", "ece")
print(f"Table {'11' if MODEL=='yolov8' else '12'}: ECE Delta, Spearman rho, Decoupling Coefficient -- {MODEL} under TS")
print(f"Baseline (clean) ECE: {baseline_ece:.4f}")
print()

all_rhos = []
header = f'{"Corruption Type":22s} | {"Baseline":>9s} | {"S1":>7s}{"S2":>7s}{"S3":>7s}{"S4":>7s}{"S5":>7s} | {"Spearman rho":>13s} | {"Decoupling Coef":>16s}'
print(header)

for corr in CORRUPTIONS:
    d_eces = []
    maps = []
    eces = []
    for s in SEVERITIES:
        cond = f"{corr}_S{s}"
        ece = get(cond, "ece")
        map50 = get(cond, "map50")
        d_eces.append(abs(ece - baseline_ece))
        eces.append(ece)
        maps.append(map50)

    rho, p = stats.spearmanr(maps, eces)
    all_rhos.append(rho)

    label = corr.replace("_", " ").title()
    row = f"{label:22s} | {baseline_ece:9.4f} |"
    for de in d_eces:
        row += f"{de:7.4f}"
    row += f" | {rho:13.4f} | {rho:16.4f}"
    print(row)

avg_rho = np.mean(all_rhos)
print(f'\n{"Average Across All 4 OOD":22s} | {"":9s} | {"(see per-severity avg below)":>35s} | {avg_rho:13.4f} | {avg_rho:16.4f}')

# per-severity average delta ECE across the 4 corruption types, for the average row
avg_d_eces = []
for s in SEVERITIES:
    vals = [abs(get(f"{c}_S{s}", "ece") - baseline_ece) for c in CORRUPTIONS]
    avg_d_eces.append(np.mean(vals))
print(f'{"":22s} | {baseline_ece:9.4f} | ' + "".join(f"{v:7.4f}" for v in avg_d_eces))