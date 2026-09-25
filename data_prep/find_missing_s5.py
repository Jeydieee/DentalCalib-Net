QUADRANT_DIR = r"D:\DentalCalib-Net\stage4_outputs\quadrant_labeled_predictions"

import json

CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness_variation", "jpeg_compression"]
QUADRANTS = [1, 2, 3, 4]

for corr in CORRUPTIONS:
    with open(f"{QUADRANT_DIR}/yolov8_{corr}_S5_quadrant.json") as f:
        data = json.load(f)
    for q in QUADRANTS:
        count = sum(
            1 for entry in data for pred in entry["predictions"]
            if pred["quadrant"] == q and pred.get("label") is not None
        )
        flag = "  <-- ZERO" if count == 0 else ""
        print(f"{corr:22s} S5 quadrant {q}: {count} labeled predictions{flag}")