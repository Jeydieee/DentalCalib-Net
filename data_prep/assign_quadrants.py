import json
from pathlib import Path

def assign_quadrant(bbox_xyxy, img_width=640, img_height=640):
    """
    Assign an FDI-style quadrant (1-4) based on bounding box center position.
    Convention confirmed against DENTEX ground-truth annotations:
      - Quadrant 1 (upper-right, FDI 11-18): left half, top half
      - Quadrant 2 (upper-left,  FDI 21-28): right half, top half
      - Quadrant 3 (lower-left,  FDI 31-38): right half, bottom half
      - Quadrant 4 (lower-right, FDI 41-48): left half, bottom half
    """
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    left = cx < (img_width / 2)
    top = cy < (img_height / 2)

    if left and top:
        return 1
    elif not left and top:
        return 2
    elif not left and not top:
        return 3
    else:  # left and not top
        return 4


def load_json_list(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        for key in ("images", "results", "data", "entries"):
            if isinstance(raw.get(key), list):
                return raw[key]
        raise ValueError(f"{path}: no list found under expected keys")
    return raw


def build_quadrant_labeled_file(labeled_path, recal_path, out_path):
    """
    Merge labeled predictions (correctness/IoU) with recalibrated predictions
    (raw/ts/dcn confidence), and add a 'quadrant' field to every prediction
    based on its bounding box position. Same merge-by-image_path-and-index
    logic as run_all_calibration_metrics.py, so outputs stay consistent
    with the rest of Stage 5.
    """
    labeled = load_json_list(labeled_path)
    recal = load_json_list(recal_path)
    recal_by_image = {e["image_path"]: e["predictions"] for e in recal}

    output = []
    skipped = 0

    for entry in labeled:
        image_path = entry["image_path"]
        recal_preds = recal_by_image.get(image_path)
        if recal_preds is None:
            skipped += 1
            continue

        lab_preds = entry.get("predictions", []) or []
        if len(lab_preds) != len(recal_preds):
            raise ValueError(
                f"prediction count mismatch for {image_path!r}: "
                f"{len(lab_preds)} in {labeled_path} vs {len(recal_preds)} in {recal_path}"
            )

        new_predictions = []
        for lab_pred, recal_pred in zip(lab_preds, recal_preds):
            bbox = recal_pred.get("bbox_xyxy") or lab_pred.get("bbox_xyxy")
            quadrant = assign_quadrant(bbox)

            merged_pred = dict(recal_pred)
            merged_pred["label"] = lab_pred.get("label")
            merged_pred["best_iou"] = lab_pred.get("best_iou")
            merged_pred["quadrant"] = quadrant
            new_predictions.append(merged_pred)

        output.append({"image_path": image_path, "predictions": new_predictions})

    with open(out_path, "w") as f:
        json.dump(output, f)

    total_preds = sum(len(e["predictions"]) for e in output)
    print(f"{Path(labeled_path).name}: {total_preds} predictions quadrant-tagged, "
          f"{skipped} images skipped (no matching recal entry) -> {out_path}")


if __name__ == "__main__":
    LABELED_DIR = r"D:\DentalCalib-Net\stage4_outputs\labeled_predictions"
    RECAL_DIR = r"D:\DentalCalib-Net\stage4_outputs\recalibrated_predictions"
    OUTPUT_DIR = r"D:\DentalCalib-Net\stage4_outputs\quadrant_labeled_predictions"

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    CORRUPTION_TYPES = ["brightness_variation", "gaussian_noise", "jpeg_compression", "motion_blur"]
    SEVERITIES = [1, 2, 3, 4, 5]

    for model in ["yolov8", "rtdetr"]:
        print(f"\n=== Quadrant-tagging {model} ===")

        if model == "rtdetr":
            labeled_clean = f"{LABELED_DIR}/rtdetr_predictions_clean_filtered_labeled.json"
        else:
            labeled_clean = f"{LABELED_DIR}/yolov8_predictions_clean_labeled.json"
        recal_clean = f"{RECAL_DIR}/{model}_clean_recalibrated.json"
        out_clean = f"{OUTPUT_DIR}/{model}_clean_quadrant.json"
        build_quadrant_labeled_file(labeled_clean, recal_clean, out_clean)

        for corruption in CORRUPTION_TYPES:
            for severity in SEVERITIES:
                if model == "rtdetr":
                    labeled_path = f"{LABELED_DIR}/rtdetr_{corruption}_S{severity}_filtered_labeled.json"
                else:
                    labeled_path = f"{LABELED_DIR}/yolov8_{corruption}_S{severity}_labeled.json"
                recal_path = f"{RECAL_DIR}/{model}_{corruption}_S{severity}_recalibrated.json"
                out_path = f"{OUTPUT_DIR}/{model}_{corruption}_S{severity}_quadrant.json"

                build_quadrant_labeled_file(labeled_path, recal_path, out_path)

    print("\nDone. All predictions quadrant-tagged.")