QUADRANT_DIR = r"D:\DentalCalib-Net\stage4_outputs\quadrant_labeled_predictions"
OUT_DIR = r"D:\DentalCalib-Net\frontend\data\per_image"

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
    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    conditions = ["clean"] + [f"{c}_S{s}" for c in CORRUPTIONS for s in SEVERITIES]
    total_size = 0

    for model in MODELS:
        for condition in conditions:
            data = load_condition_file(model, condition)
            output = {}
            for entry in data:
                filename = Path(entry["image_path"]).name
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
                output[filename] = preds

            out_path = out_root / f"{model}__{condition}.json"
            with open(out_path, "w") as f:
                json.dump(output, f)
            size = out_path.stat().st_size
            total_size += size
            print(f"{model:8s} {condition:25s} {len(output):4d} images  {size/1024:8.1f} KB")

    print(f"\nTotal: {total_size/1024/1024:.1f} MB across {len(MODELS)*len(conditions)} files")