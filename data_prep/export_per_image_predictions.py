QUADRANT_DIR = r"D:\DentalCalib-Net\stage4_outputs\quadrant_labeled_predictions"
OUT_PATH = r"D:\DentalCalib-Net\frontend\data\per_image_predictions.json"

import json
from pathlib import Path

CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness_variation", "jpeg_compression"]
SEVERITIES = [1, 2, 3, 4, 5]
MODELS = ["yolov8", "rtdetr"]


def load_condition_file(model, condition):
    path = f"{QUADRANT_DIR}/{model}_{condition}_quadrant.json"
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    conditions = ["clean"] + [f"{c}_S{s}" for c in CORRUPTIONS for s in SEVERITIES]
    output = {}

    for model in MODELS:
        for condition in conditions:
            data = load_condition_file(model, condition)
            for entry in data:
                filename = Path(entry["image_path"]).name
                key = f"{model}__{condition}__{filename}"
                preds = []
                for pred in entry["predictions"]:
                    preds.append({
                        "bbox": pred.get("bbox_xyxy"),
                        "confidence_raw": pred.get("confidence_raw"),
                        "confidence_ts": pred.get("confidence_ts"),
                        "confidence_dcn": pred.get("confidence_dcn"),
                        "quadrant": pred.get("quadrant"),
                        "label": pred.get("label"),
                    })
                output[key] = preds
            print(f"{model:8s} {condition:25s} done")

    with open(OUT_PATH, "w") as f:
        json.dump(output, f)
    print(f"\nwrote {len(output)} image entries -> {OUT_PATH}")