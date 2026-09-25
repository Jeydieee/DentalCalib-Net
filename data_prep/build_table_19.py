CSV_DIR = r"D:\DentalCalib-Net\stage4_outputs\quadrant_labeled_predictions"

import json
import numpy as np
from scipy import stats
from pathlib import Path

from calibration_metrics import compute_rmsd

CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness_variation", "jpeg_compression"]
SEVERITY_LEVELS = {1: "S1 -- Mild OOD", 3: "S3 -- Moderate OOD", 5: "S5 -- Severe OOD"}
QUADRANTS = [1, 2, 3, 4]
MODELS = ["yolov8", "rtdetr"]
CONF_TYPE = "ts"


def load_quadrant_file(model, corruption, severity):
    path = f"{CSV_DIR}/{model}_{corruption}_S{severity}_quadrant.json"
    with open(path) as f:
        return json.load(f)


def quadrant_rmsd(data, quadrant):
    conf, labels = [], []
    for entry in data:
        for pred in entry["predictions"]:
            if pred["quadrant"] != quadrant:
                continue
            label = pred.get("label")
            if label is None:
                continue
            conf.append(pred[f"confidence_{CONF_TYPE}"])
            labels.append(int(label))
    if len(labels) == 0:
        return None
    return compute_rmsd(np.array(conf), np.array(labels))


def cohens_d_pooled(x, y):
    nx, ny = x.size, y.size
    pooled_sd = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled_sd == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled_sd)


def effect_label(d_val):
    a = abs(d_val)
    if a < 0.2: return "Negligible"
    elif a < 0.5: return "Small"
    elif a < 0.8: return "Medium"
    return "Large"


print("Table 19: Mann-Whitney U Test -- RMSD, YOLOv8 vs RT-DETR, Representative OOD Severities")
print("QUADRANT-BASED, pooled across 4 corruption types (n=16 per group per severity)")
print()

n_tests = len(SEVERITY_LEVELS)

for sev, sev_label in SEVERITY_LEVELS.items():
    model_values = {}
    for model in MODELS:
        vals = []
        for corr in CORRUPTIONS:
            data = load_quadrant_file(model, corr, sev)
            for q in QUADRANTS:
                v = quadrant_rmsd(data, q)
                if v is not None:
                    vals.append(v)
        model_values[model] = np.array(vals)

    x = model_values["yolov8"]
    y = model_values["rtdetr"]

    stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    p_bonf = min(p * n_tests, 1.0)
    sig = p_bonf < 0.05
    dcoh = cohens_d_pooled(x, y)

    print(f"[{sev_label}]")
    print(f"  n_yolov8={x.size}  n_rtdetr={y.size}")
    print(f"  Mean YOLOv8={x.mean():.4f}  Mean RT-DETR={y.mean():.4f}  Diff={x.mean()-y.mean():+.4f}")
    print(f"  U={stat:.2f}  p={p:.4g}  Bonferroni-adj p (n={n_tests})={p_bonf:.4g}  Sig={sig}")
    print(f"  Cohen's d={dcoh:.4f} ({effect_label(dcoh)})")
    print()

print("CAVEAT: quadrant x corruption-type pooling (n=16 per group per severity) is an")
print("extension of the quadrant-based reading used in Table 17, not independently")
print("confirmed by the approved proposal. Flag alongside Table 17's methodology note.")