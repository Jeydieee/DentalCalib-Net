QUADRANT_DIR = r"D:\DentalCalib-Net\stage4_outputs\quadrant_labeled_predictions"

import json
import numpy as np
from scipy import stats
import sys
sys.path.insert(0, r"D:\DentalCalib-Net\data_prep")
from calibration_metrics import compute_rmsd

CORRUPTIONS_NO_GN = ["motion_blur", "brightness_variation", "jpeg_compression"]
QUADRANTS = [1, 2, 3, 4]
SEVERITIES = {1: "S1", 3: "S3", 5: "S5"}
MODELS = ["yolov8", "rtdetr"]
CONF_TYPE = "ts"


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


def cohens_d(x, y):
    nx, ny = x.size, y.size
    pooled_sd = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return 0.0 if pooled_sd == 0 else float((x.mean() - y.mean()) / pooled_sd)


print("Table 19 RECOMPUTED excluding Gaussian Noise (motion_blur + brightness_variation + jpeg_compression only)")
print("Same n=12 per group per severity (3 corruption types x 4 quadrants), not n=16")
print()

for sev, label in SEVERITIES.items():
    model_vals = {}
    for model in MODELS:
        vals = []
        for corr in CORRUPTIONS_NO_GN:
            with open(f"{QUADRANT_DIR}/{model}_{corr}_S{sev}_quadrant.json") as f:
                data = json.load(f)
            for q in QUADRANTS:
                v = quadrant_rmsd(data, q)
                if v is not None:
                    vals.append(v)
        model_vals[model] = np.array(vals)

    x, y = model_vals["yolov8"], model_vals["rtdetr"]
    stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    d = cohens_d(x, y)
    print(f"{label}: n_yolo={x.size} n_rt={y.size} mean_yolo={x.mean():.4f} mean_rt={y.mean():.4f} "
          f"diff={x.mean()-y.mean():+.4f} U={stat:.2f} p={p:.4g} d={d:.4f}")