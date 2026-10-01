try:
    from .paths import project_path, raw_data_path
except ImportError:
    from paths import project_path, raw_data_path

LABELED_DIR = project_path("stage4_outputs", "labeled_predictions")
RECAL_DIR = project_path("stage4_outputs", "recalibrated_predictions")
OUT_PATH = project_path("frontend", "data", "reliability_bins.json")

import json
import sys
sys.path.insert(0, project_path("data_prep"))
from calibration_metrics import bin_statistics

CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness_variation", "jpeg_compression"]
SEVERITIES = [1, 2, 3, 4, 5]
MODELS = ["yolov8", "rtdetr"]
CONF_FIELDS = {"raw": "confidence_raw", "ts": "confidence_ts", "dcn": "confidence_dcn"}
N_BINS = 15


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_files(model, condition):
    if condition == "clean":
        labeled = (f"{LABELED_DIR}/yolov8_predictions_clean_labeled.json" if model == "yolov8"
                   else f"{LABELED_DIR}/rtdetr_predictions_clean_filtered_labeled.json")
        recal = f"{RECAL_DIR}/{model}_clean_recalibrated.json"
    else:
        labeled = (f"{LABELED_DIR}/yolov8_{condition}_labeled.json" if model == "yolov8"
                   else f"{LABELED_DIR}/rtdetr_{condition}_filtered_labeled.json")
        recal = f"{RECAL_DIR}/{model}_{condition}_recalibrated.json"
    return labeled, recal


def collect(model, condition):
    labeled_path, recal_path = get_files(model, condition)
    labeled = load_json(labeled_path)
    recal = load_json(recal_path)
    recal_by_img = {e["image_path"]: e["predictions"] for e in recal}

    per_method = {"raw": ([], []), "ts": ([], []), "dcn": ([], [])}
    for entry in labeled:
        rec_preds = recal_by_img.get(entry["image_path"])
        if rec_preds is None:
            continue
        for lp, rp in zip(entry["predictions"], rec_preds):
            label = lp.get("label")
            if label is None:
                continue
            for method, field in CONF_FIELDS.items():
                per_method[method][0].append(rp[field])
                per_method[method][1].append(int(label))
    return per_method


if __name__ == "__main__":
    conditions = ["clean"] + [f"{c}_S{s}" for c in CORRUPTIONS for s in SEVERITIES]
    output = {}

    for model in MODELS:
        for condition in conditions:
            per_method = collect(model, condition)
            for method, (conf, labels) in per_method.items():
                if len(labels) == 0:
                    continue
                try:
                    stats = bin_statistics(conf, labels, N_BINS, min_bin_count=1)
                except ValueError:
                    continue
                key = f"{model}__{condition}__{method}"
                output[key] = {
                    "confidence": stats["confidence"].tolist(),
                    "accuracy": stats["accuracy"].tolist(),
                    "count": stats["count"].tolist(),
                    "n_bins_populated": stats["n_bins_populated"],
                }
            print(f"{model:8s} {condition:25s} done")

    with open(OUT_PATH, "w") as f:
        json.dump(output, f)
    print(f"\nwrote {len(output)} bin-series -> {OUT_PATH}")