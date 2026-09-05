import json
from pathlib import Path

# ── Reuse functions from iou_labeling.py ─────────────────────────────

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def bbox_xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


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
        gt_box = bbox_xywh_to_xyxy(ann['bbox'])
        gt_by_filename.setdefault(file_name, []).append(gt_box)

    return gt_by_filename


# ── Batch labeling ────────────────────────────────────────────────────

def label_predictions(predictions_data, gt_by_filename, iou_threshold=0.50):
    """predictions_data: list of {'image_path': ..., 'predictions': [...]}
    Returns the same structure, with each prediction dict augmented with
    a 'label' key (1 = correct, 0 = incorrect) and 'best_iou' for reference."""

    labeled = []
    for entry in predictions_data:
        file_name = Path(entry['image_path']).name
        gt_boxes = gt_by_filename.get(file_name, [])

        new_predictions = []
        for pred in entry['predictions']:
            pred_box = pred['bbox_xyxy']

            best_iou = 0.0
            for gt_box in gt_boxes:
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou

            label = 1 if best_iou >= iou_threshold else 0

            new_pred = dict(pred)
            new_pred['label'] = label
            new_pred['best_iou'] = best_iou
            new_predictions.append(new_pred)

        labeled.append({
            'image_path': entry['image_path'],
            'predictions': new_predictions,
        })

    return labeled


def process_prediction_file(pred_path, gt_by_filename, out_path):
    with open(pred_path) as f:
        predictions_data = json.load(f)

    labeled = label_predictions(predictions_data, gt_by_filename)

    with open(out_path, 'w') as f:
        json.dump(labeled, f)

    total_preds = sum(len(e['predictions']) for e in labeled)
    total_correct = sum(
        sum(1 for p in e['predictions'] if p['label'] == 1)
        for e in labeled
    )
    print(f"{Path(pred_path).name}: {total_preds} predictions, {total_correct} labeled correct ({100*total_correct/total_preds:.1f}%) -> {out_path}")


if __name__ == "__main__":
    GT_PATH = r"D:/DentalCalib-Net/data_prep/repool_output/dentex_merged_1005_resized.json"
    STAGE2_DIR = r"D:/DentalCalib-Net/stage2_outputs"
    OUTPUT_DIR = r"D:/DentalCalib-Net/stage4_outputs/labeled_predictions"

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    gt_by_filename = load_ground_truth(GT_PATH)

    CORRUPTION_TYPES = ['gaussian_noise', 'motion_blur', 'brightness_variation', 'jpeg_compression']
    SEVERITIES = ['S1', 'S2', 'S3', 'S4', 'S5']
    MODELS = ['yolov8', 'rtdetr']

    print("Labeling clean test predictions...")
    for model in MODELS:
        pred_path = f"{STAGE2_DIR}/{model}_predictions_clean.json"
        out_path = f"{OUTPUT_DIR}/{model}_predictions_clean_labeled.json"
        process_prediction_file(pred_path, gt_by_filename, out_path)

    print("\nLabeling OOD predictions...")
    for model in MODELS:
        for corruption in CORRUPTION_TYPES:
            for severity in SEVERITIES:
                pred_path = f"{STAGE2_DIR}/{model}_ood_predictions/{model}_{corruption}_{severity}.json"
                out_path = f"{OUTPUT_DIR}/{model}_{corruption}_{severity}_labeled.json"
                process_prediction_file(pred_path, gt_by_filename, out_path)

    print("\nDone.")