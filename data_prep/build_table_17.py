CSV_PATH = r"D:\DentalCalib-Net\stage5_outputs\table_10_per_quadrant.csv"

import csv
import numpy as np
from scipy import stats

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

d = {(r["model"], int(r["quadrant"]), r["confidence_type"]): r for r in rows}

QUADRANTS = [1, 2, 3, 4]
METRICS = ["ece", "mce", "dece"]
CONF_TYPE = "ts"  # Table 17 is explicitly scoped to Temperature Scaling


def get_series(model, metric):
    return np.array([float(d[(model, q, CONF_TYPE)][metric]) for q in QUADRANTS])


def cohens_d_pooled(x, y):
    nx, ny = x.size, y.size
    pooled_sd = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled_sd == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled_sd)


def effect_label(d_val):
    a = abs(d_val)
    if a < 0.2:
        return "Negligible"
    elif a < 0.5:
        return "Small"
    elif a < 0.8:
        return "Medium"
    return "Large"


print("Table 17: Mann-Whitney U Test -- Baseline Calibration, YOLOv8 vs RT-DETR")
print("Clean DENTEX 2023 Test Set, Under Temperature Scaling")
print("QUADRANT-BASED (n=4 per group, per quadrant of Table 10) -- see caveat below")
print()

metric_label = {"ece": "Expected Calibration Error (ECE)",
                 "mce": "Maximum Calibration Error (MCE)",
                 "dece": "Detection Calibration Error (D-ECE)"}

n_tests = len(METRICS)
results = []

for metric in METRICS:
    x = get_series("yolov8", metric)
    y = get_series("rtdetr", metric)

    stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    p_bonf = min(p * n_tests, 1.0)
    sig = p_bonf < 0.05
    dcoh = cohens_d_pooled(x, y)

    results.append({
        "metric": metric_label[metric],
        "U": stat, "p": p, "p_bonf": p_bonf, "sig": sig,
        "mean_yolo": x.mean(), "mean_rt": y.mean(),
        "mean_diff": x.mean() - y.mean(),
        "cohens_d": dcoh, "effect": effect_label(dcoh),
    })

    print(f"[{metric_label[metric]}]")
    print(f"  Mean YOLOv8={x.mean():.4f}  Mean RT-DETR={y.mean():.4f}  Diff={x.mean()-y.mean():+.4f}")
    print(f"  U={stat:.2f}  p={p:.4g}  Bonferroni-adj p (n={n_tests})={p_bonf:.4g}  Sig={sig}")
    print(f"  Cohen's d={dcoh:.4f} ({effect_label(dcoh)})")
    print(f"  YOLOv8 quadrant values: {get_series('yolov8', metric).round(4).tolist()}")
    print(f"  RT-DETR quadrant values: {get_series('rtdetr', metric).round(4).tolist()}")
    print()

print("CAVEAT: n=4 per group (one value per FDI quadrant, from Table 10).")
print("This is our team's best-supported reading of an underspecified methodology --")
print("not explicitly confirmed in the approved proposal. Statistical power at n=4 is low;")
print("a non-significant result here should not be read as strong evidence of no difference.")