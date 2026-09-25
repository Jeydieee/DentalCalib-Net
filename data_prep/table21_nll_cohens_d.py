CSV_PATH = r"D:\DentalCalib-Net\stage5_outputs\nll_results.csv"

import csv
import numpy as np
from scipy import stats

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

d = {(r["model"], r["condition"]): r for r in rows}

CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness_variation", "jpeg_compression"]
SEVERITIES = [1, 2, 3, 4, 5]
OOD_CONDITIONS = [f"{c}_S{s}" for c in CORRUPTIONS for s in SEVERITIES]
MODELS = ["yolov8", "rtdetr"]


def cohens_dz(diff):
    sd = diff.std(ddof=1)
    return 0.0 if sd == 0 else float(diff.mean() / sd)


def effect_label(dz):
    a = abs(dz)
    if a < 0.2:
        return "Negligible"
    elif a < 0.5:
        return "Small"
    elif a < 0.8:
        return "Medium"
    return "Large"


print("Table 21 -- NLL component: Cohen's dz (raw vs TS), COCO-C OOD conditions (n=20)")
print()

for model in MODELS:
    raw_vals = np.array([float(d[(model, c)]["nll_raw"]) for c in OOD_CONDITIONS])
    ts_vals = np.array([float(d[(model, c)]["nll_ts"]) for c in OOD_CONDITIONS])
    diff = raw_vals - ts_vals

    if np.allclose(diff, 0.0):
        stat, p = 0.0, 1.0
    else:
        stat, p = stats.wilcoxon(raw_vals, ts_vals)

    dz = cohens_dz(diff)
    direction = "Improved" if dz > 0 else "Worsened"

    print(f"[{model}] NLL: mean_raw={raw_vals.mean():.4f}  mean_ts={ts_vals.mean():.4f}  "
          f"Cohen's dz={dz:.4f} ({effect_label(dz)})  Direction={direction}  "
          f"Wilcoxon p={p:.4g}")