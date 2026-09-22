import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(r"D:\DentalCalib-Net\data_prep")))
from compute_map import greedy_ap


def build_gt_lookup(gt_path, split_ids):
    """Returns {filename: [list of gt boxes in xyxy]} for images in split_ids,
    Impacted (category_id_3==0) only."""
    with open(gt_path) as f:
        merged = json.load(f)

    images_by_id = {img['id']: img for img in merged['images'] if img['id'] in split_ids}

    gt_by_filename = {}
    for img in images_by_id.values():
        gt_by_filename[img['file_name']] = []

    for ann in merged['annotations']:
        if ann['category_id_3'] != 0:
            continue
        img = images_by_id.get(ann['image_id'])
        if img is None:
            continue
        x, y, w, h = ann['bbox']
        gt_box = [x, y, x + w, y + h]
        gt_by_filename[img['file_name']].append(gt_box)

    return gt_by_filename


def build_dets(predictions_path):
    """Returns list of {'filename': str, 'bbox': [x1,y1,x2,y2], 'confidence': float}
    matching greedy_match's expected structure."""
    with open(predictions_path) as f:
        predictions_data = json.load(f)

    dets = []
    for entry in predictions_data:
        filename = Path(entry['image_path']).name
        for pred in entry['predictions']:
            dets.append({
                'filename': filename,
                'bbox': pred['bbox_xyxy'],
                'confidence': pred['confidence'],
            })
    return dets


if __name__ == "__main__":
    GT_PATH = r"D:\DentalCalib-Net\data_prep\repool_output\dentex_merged_1005_resized.json"
    STAGE2_DIR = r"D:\DentalCalib-Net\stage2_outputs"

    val_ids = set(pd.read_csv(r"D:\DentalCalib-Net\data_prep\split_output\val_ids.csv")['image_id'].tolist())

    gt_val = build_gt_lookup(GT_PATH, val_ids)
    n_gt_val = sum(len(boxes) for boxes in gt_val.values())
    print(f"Validation partition ground truth boxes (Impacted only): {n_gt_val}")

    print("\n=== YOLOv8 ===")
    dets_yolo = build_dets(f"{STAGE2_DIR}/yolov8_predictions_val.json")
    result_yolo = greedy_ap(dets_yolo, gt_val, n_gt_val, iou_threshold=0.50)
    print(f"  Validation split mAP@0.50 (Stage 5 corrected greedy matching): {result_yolo['ap']:.4f}")
    print(f"  Section 3 reported (Ultralytics training-time validation): 0.883")

    print("\n=== RT-DETR ===")
    dets_rtdetr = build_dets(f"{STAGE2_DIR}/rtdetr_predictions_val.json")
    result_rtdetr = greedy_ap(dets_rtdetr, gt_val, n_gt_val, iou_threshold=0.50)
    print(f"  Validation split mAP@0.50 (Stage 5 corrected greedy matching): {result_rtdetr['ap']:.4f}")
    print(f"  Section 3 reported (Ultralytics training-time validation): 0.887")

    print(f"\n(For reference) Table 9's test-split mAP values: YOLOv8=0.9099, RT-DETR=0.8871")