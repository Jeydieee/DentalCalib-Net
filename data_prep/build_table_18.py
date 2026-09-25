CSV_PATH = r"D:\DentalCalib-Net\stage5_outputs\calibration_results.csv"

import csv
import numpy as np
from scipy import stats

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

d = {(r["model"], r["condition"], r["confidence_type"]): r for r in rows}

CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness_variation", "jpeg_compression"]
SEVERITIES = [1, 2, 3, 4, 5]
MODELS = ["yolov8", "rtdetr"]
ALPHA = 0.05


def get(model, condition, metric):
    return float(d[(model, condition, "ts")][metric])


def interpret(rho, p):
    if p >= ALPHA:
        return "Stable"  # no statistically detectable mAP-ECE relationship
    return "Coupled" if rho < 0 else "Decoupled"


print("Table 18: Spearman Correlation -- mAP@0.50 vs ECE, per Model per OOD Condition, TS")
print('NOTE: "Stable" = not statistically significant (p >= 0.05); this label is my')
print("      construction, not explicitly defined in the proposal text -- flag for adviser.")
print()

header = f'{"Model":10s} {"OOD Condition":22s} | {"Spearman rho":>13s} | {"p-value":>9s} | {"Sig?":>6s} | {"Decoupl. Coef":>14s} | {"Interpretation":>14s}'
print(header)

for model in MODELS:
    all_rhos = []
    all_ps = []
    for corr in CORRUPTIONS:
        maps = [get(model, f"{corr}_S{s}", "map50") for s in SEVERITIES]
        eces = [get(model, f"{corr}_S{s}", "ece") for s in SEVERITIES]
        rho, p = stats.spearmanr(maps, eces)
        all_rhos.append(rho)
        all_ps.append(p)
        sig = "Yes" if p < ALPHA else "No"
        interp = interpret(rho, p)
        label = corr.replace("_", " ").title()
        print(f"{model:10s} {label:22s} | {rho:13.4f} | {p:9.4f} | {sig:>6s} | {rho:14.4f} | {interp:>14s}")

    avg_rho = np.mean(all_rhos)
    avg_p = np.mean(all_ps)  # descriptive only, not a combined test
    print(f'{model:10s} {"Average (All 4 OOD)":22s} | {avg_rho:13.4f} | {avg_p:9.4f}* | {"--":>6s} | {avg_rho:14.4f} | {"--":>14s}')
    print("  * mean of individual p-values, not a combined significance test")
    print()