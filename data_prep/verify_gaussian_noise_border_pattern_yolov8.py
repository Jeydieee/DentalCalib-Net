import json
import numpy as np
from pathlib import Path

GT_PATH = r"D:\DentalCalib-Net\data_prep\repool_output\dentex_merged_1005_resized.json"
YOLOV8_OOD_DIR = r"D:\DentalCalib-Net\stage2_outputs\yolov8_ood_predictions"

IMAGE_SIZE = 640
BORDER_MARGIN = 20  # pixels from any edge to count as "border"
TOP_N = 5           # check up to top-N predictions per image by confidence
SEVERITIES = ['S2', 'S3', 'S4', 'S5']
SAMPLE_SIZE = 30    # number of images to check per severity (out of 201)


def load_ground_truth(json_path):
    with open(json_path) as f:
        merged = json.load(f)
    images_by_id = {img['id']: img['file_name'] for img in merged['images']}
    gt_by_filename = {}
    for ann in merged['annotations']:
        if ann['category_id_3'] != 0:
            continue
        file_name = images_by_id.get(ann['image_id'])
        if file_name is None:
            continue
        x, y, w, h = ann['bbox']
        gt_by_filename.setdefault(file_name, []).append([x, y, x + w, y + h])
    return gt_by_filename


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def is_near_border(box, img_size=IMAGE_SIZE, margin=BORDER_MARGIN):
    """True if any part of the box touches within `margin` pixels of any edge."""
    x1, y1, x2, y2 = box
    return (
        x1 <= margin or y1 <= margin or
        x2 >= img_size - margin or y2 >= img_size - margin
    )


def distance_to_nearest_gt(pred_box, gt_boxes):
    """Center-to-center distance from prediction to the nearest ground truth box.
    Returns None if there is no ground truth for this image."""
    if not gt_boxes:
        return None
    pred_center = box_center(pred_box)
    distances = []
    for gt_box in gt_boxes:
        gt_center = box_center(gt_box)
        d = np.sqrt((pred_center[0] - gt_center[0])**2 + (pred_center[1] - gt_center[1])**2)
        distances.append(d)
    return min(distances)


def analyze_severity(severity, gt_by_filename, sample_size=SAMPLE_SIZE, seed=42):
    pred_path = f"{YOLOV8_OOD_DIR}/yolov8_gaussian_noise_{severity}.json"
    with open(pred_path) as f:
        predictions_data = json.load(f)

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(predictions_data), size=min(sample_size, len(predictions_data)), replace=False)
    sampled_entries = [predictions_data[i] for i in indices]

    border_count = 0
    total_checked = 0
    distances = []
    images_with_all_border = 0
    images_with_zero_predictions = 0
    predictions_per_image = []

    for entry in sampled_entries:
        filename = Path(entry['image_path']).name
        gt_boxes = gt_by_filename.get(filename, [])

        all_preds = entry.get('predictions', []) or []
        predictions_per_image.append(len(all_preds))
        if not all_preds:
            images_with_zero_predictions += 1
            continue

        sorted_preds = sorted(all_preds, key=lambda p: p['confidence'], reverse=True)
        top_preds = sorted_preds[:TOP_N]

        image_border_flags = []
        for pred in top_preds:
            box = pred['bbox_xyxy']
            total_checked += 1

            near_border = is_near_border(box)
            image_border_flags.append(near_border)
            if near_border:
                border_count += 1

            d = distance_to_nearest_gt(box, gt_boxes)
            if d is not None:
                distances.append(d)

        if image_border_flags and all(image_border_flags):
            images_with_all_border += 1

    border_pct = 100 * border_count / total_checked if total_checked > 0 else None
    avg_distance = np.mean(distances) if distances else None
    n_with_preds = len(sampled_entries) - images_with_zero_predictions
    images_pct_all_border = (100 * images_with_all_border / n_with_preds
                              if n_with_preds > 0 else None)
    images_pct_zero_predictions = (100 * images_with_zero_predictions / len(sampled_entries)
                                    if sampled_entries else 0)

    return {
        'severity': severity,
        'images_sampled': len(sampled_entries),
        'predictions_checked': total_checked,
        'avg_predictions_per_image': float(np.mean(predictions_per_image)) if predictions_per_image else 0.0,
        'images_with_zero_predictions': images_with_zero_predictions,
        'images_with_zero_predictions_pct': images_pct_zero_predictions,
        'border_prediction_pct': border_pct,
        'images_where_all_checked_preds_are_border_pct': images_pct_all_border,
        'avg_distance_to_nearest_gt_px': avg_distance,
    }


if __name__ == "__main__":
    gt_by_filename = load_ground_truth(GT_PATH)

    print(f"Sampling {SAMPLE_SIZE} images per severity, checking up to top-{TOP_N} "
          f"predictions each (YOLOv8).\n")

    results = []
    for severity in SEVERITIES:
        result = analyze_severity(severity, gt_by_filename)
        results.append(result)

        print(f"=== YOLOv8 Gaussian Noise {severity} ===")
        print(f"  Images sampled: {result['images_sampled']}")
        print(f"  Avg predictions/image (whole image, pre-top-N): "
              f"{result['avg_predictions_per_image']:.2f}")
        print(f"  Images with ZERO predictions: "
              f"{result['images_with_zero_predictions']} "
              f"({result['images_with_zero_predictions_pct']:.1f}%)")
        print(f"  Predictions checked (top-{TOP_N} of images that had any): "
              f"{result['predictions_checked']}")
        if result['border_prediction_pct'] is not None:
            print(f"  % of checked predictions near image border (within {BORDER_MARGIN}px): "
                  f"{result['border_prediction_pct']:.1f}%")
        else:
            print("  % near border: N/A (no predictions to check)")
        if result['images_where_all_checked_preds_are_border_pct'] is not None:
            print(f"  % of (non-empty) images where ALL checked predictions are border-clustered: "
                  f"{result['images_where_all_checked_preds_are_border_pct']:.1f}%")
        else:
            print("  % of images all-border: N/A")
        avg_d = result['avg_distance_to_nearest_gt_px']
        print(f"  Avg. distance from prediction to nearest ground truth: {avg_d:.1f}px"
              if avg_d is not None else "  Avg. distance: N/A (no predictions or no ground truth)")
        print()

    output_path = r"D:\DentalCalib-Net\stage4_outputs\gaussian_noise_border_analysis_yolov8.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved full results -> {output_path}")
