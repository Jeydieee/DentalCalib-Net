import json
import cv2
from pathlib import Path

GT_PATH = r"D:/DentalCalib-Net/data_prep/repool_output/dentex_merged_1005_resized.json"
RTDETR_OOD_DIR = r"D:/DentalCalib-Net/stage2_outputs/rtdetr_ood_predictions"
IMAGE_DIR = r"D:/DentalCalib-Net/dataset_split/ood/gaussian_noise"
OUTPUT_DIR = r"D:/DentalCalib-Net/stage4_outputs/visual_check"

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


def visualize_check(image_filename, severity, top_n=5):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    gt_by_filename = load_ground_truth(GT_PATH)
    gt_boxes = gt_by_filename.get(image_filename, [])

    pred_path = f"{RTDETR_OOD_DIR}/rtdetr_gaussian_noise_{severity}.json"
    with open(pred_path) as f:
        predictions_data = json.load(f)

    entry = None
    for e in predictions_data:
        if Path(e['image_path']).name == image_filename:
            entry = e
            break

    if entry is None:
        print(f"Image {image_filename} not found in {pred_path}")
        return

    img_path = f"{IMAGE_DIR}/{severity}/{image_filename}"
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load image: {img_path}")
        return

    print(f"\n=== {image_filename} ({severity}) ===")
    print(f"Ground truth boxes ({len(gt_boxes)}):")
    for box in gt_boxes:
        print(f"  {box}")
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green = ground truth

    sorted_preds = sorted(entry['predictions'], key=lambda p: p['confidence'], reverse=True)
    top_preds = sorted_preds[:top_n]

    print(f"\nTop {top_n} RT-DETR predictions by confidence:")
    for i, pred in enumerate(top_preds):
        box = pred['bbox_xyxy']
        conf = pred['confidence']
        print(f"  #{i+1}: conf={conf:.4f}, box={box}")
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red = prediction

    out_path = f"{OUTPUT_DIR}/{image_filename.replace('.png', '')}_{severity}_check.png"
    cv2.imwrite(out_path, img)
    print(f"\nSaved visualization -> {out_path}")


if __name__ == "__main__":
    # check a few images at S3 and S4
    test_cases = [
        ("test_0.png", "S3"),
        ("test_0.png", "S4"),
        ("test_111.png", "S3"),
    ]

    for filename, severity in test_cases:
        visualize_check(filename, severity)